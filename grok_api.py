"""自部署 grok2api：图片兜底与异步视频生成。"""

import asyncio
import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urljoin, urlsplit

import aiohttp

from .qiniu_api import (
    QiniuApiError,
    QiniuAuthError,
    QiniuImageClient,
    QiniuNotConfiguredError,
    QiniuRateLimitError,
    QiniuResponseError,
    QiniuSafetyError,
    QiniuTransientApiError,
    _config_positive_int,
    _config_string,
    _error_details,
    _looks_like_safety_error,
    _validate_http_url,
)

IMAGE_MODEL = "grok-imagine-image-2.0"
VIDEO_MODEL = "grok-imagine-video"
MAX_VIDEO_BYTES = 200 * 1024 * 1024
VIDEO_ASPECT_RATIOS = ("1:1", "2:3", "3:2", "9:16", "16:9")


class GrokVideoError(RuntimeError):
    """视频任务失败、过期或返回的媒体无效。"""


class GrokClient(QiniuImageClient):
    """复用图片编解码和会话管理；请求协议与七牛完全独立。"""

    def __init__(self, config: Any):
        super().__init__({
            "api_key": _config_string(config, "grok2api_api_key", "", allow_empty=True),
            "concurrency": _config_positive_int(config, "grok2api_concurrency", 2),
        })
        base = _config_string(config, "grok2api_base_url", "", allow_empty=True).rstrip("/")
        if base:
            _validate_http_url(base, "grok2api_base_url")
            parts = urlsplit(base)
            if parts.query or parts.fragment or parts.username or parts.password:
                raise ValueError("grok2api_base_url 不能包含查询参数、片段或账号密码")
            if not base.endswith("/v1"):
                base += "/v1"
        self.api_base = base
        self.model = IMAGE_MODEL
        self.video_timeout = _config_positive_int(config, "grok2api_video_timeout", 600)
        self.retries = 2  # 仅用于已生成图片的下载，不重新提交生成请求。
        size = str(config.get("size", "auto") or "auto")
        self.image_aspect_ratio = ""
        if "x" in size:
            width, height = (int(value) for value in size.split("x"))
            divisor = math.gcd(width, height)
            self.image_aspect_ratio = f"{width // divisor}:{height // divisor}"

    @property
    def configured(self) -> bool:
        # 自部署关闭鉴权时允许 Key 留空；地址留空才禁用。
        return bool(self.api_base)

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _media_url(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise QiniuResponseError("grok2api 未返回媒体 URL")
        try:
            return _validate_http_url(urljoin(self.api_base + "/", value.strip()), "媒体 URL")
        except ValueError:
            raise QiniuResponseError("grok2api 返回的媒体 URL 无效") from None

    def _media_headers(self, url: str) -> Dict[str, str]:
        media_parts, api_parts = urlsplit(url), urlsplit(self.api_base)
        same_origin = (media_parts.scheme, media_parts.netloc) == (api_parts.scheme, api_parts.netloc)
        return self._headers() if same_origin else {}

    async def _request_json(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.configured:
            raise QiniuNotConfiguredError("未配置 grok2api_base_url")
        session = await self._get_session()
        # POST 不自动重发，避免异步视频重复创建和重复计费。
        async with session.request(
            method, self.api_base + path, json=payload, headers=self._headers(),
            allow_redirects=False,
        ) as response:
            if response.status // 100 != 2:
                raw = await response.content.read(8192)
                code, message = _error_details(raw.decode("utf-8", errors="replace"))
                error_type = QiniuApiError
                if response.status == 429:
                    error_type = QiniuRateLimitError
                elif response.status >= 500:
                    error_type = QiniuTransientApiError
                elif _looks_like_safety_error(code, message):
                    error_type = QiniuSafetyError
                elif response.status in (401, 403):
                    error_type = QiniuAuthError
                raise error_type(response.status, message, code)
            try:
                data = json.loads(await response.text())
            except (ValueError, UnicodeError):
                raise QiniuResponseError("grok2api 返回了无效 JSON") from None
            if not isinstance(data, dict):
                raise QiniuResponseError("grok2api 响应必须是 JSON 对象")
            if data.get("error"):
                code, message = _error_details(json.dumps(data))
                error_type = QiniuSafetyError if _looks_like_safety_error(code, message) else QiniuApiError
                raise error_type(response.status, message, code)
            return data

    async def generate_image(self, prompt: str, image: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {
            "model": IMAGE_MODEL, "prompt": prompt, "n": 1,
            "response_format": "b64_json",
        }
        path = "/images/generations"
        if image:
            path = "/images/edits"
            payload["image"] = {"url": self.as_image_reference(image)}
        elif self.image_aspect_ratio:
            payload["aspect_ratio"] = self.image_aspect_ratio
        async with self._semaphore:
            data = await self._request_json("POST", path, payload)
            items = data.get("data")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("url"), str):
                        try:
                            item["url"] = self._media_url(item["url"])
                        except QiniuResponseError:
                            # 无效 URL 不应掩盖同一条目中的有效 Base64 或后续图片。
                            pass
            source = self._select_response_image(data)
            if source.b64_json:
                return source.b64_json
            session = await self._get_session()
            headers = self._media_headers(source.url)
            headers["Accept"] = "image/*, application/octet-stream"
            raw = await self._download_bytes(session, source.url, headers=headers)
            return base64.b64encode(raw).decode("ascii")

    @staticmethod
    def validate_video_options(duration: int, aspect_ratio: str, resolution: str) -> None:
        if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 15:
            raise ValueError("视频时长必须是 1～15 秒的整数")
        if aspect_ratio not in VIDEO_ASPECT_RATIOS:
            raise ValueError("视频比例支持 " + "、".join(VIDEO_ASPECT_RATIOS))
        if resolution not in ("480p", "720p"):
            raise ValueError("视频分辨率支持 480p 或 720p")

    async def generate_video(
        self, prompt: str, image: Optional[str] = None,
        duration: int = 8, aspect_ratio: str = "16:9", resolution: str = "720p",
    ) -> str:
        self.validate_video_options(duration, aspect_ratio, resolution)
        payload: Dict[str, Any] = {
            "model": VIDEO_MODEL, "prompt": prompt, "duration": duration,
            "aspect_ratio": aspect_ratio, "resolution": resolution,
        }
        if image:
            payload["image"] = {"url": self.as_image_reference(image)}
        async with self._semaphore:
            return await asyncio.wait_for(self._video_job(payload), timeout=self.video_timeout)

    async def _video_job(self, payload: Dict[str, Any]) -> str:
        created = await self._request_json("POST", "/videos/generations", payload)
        request_id = created.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise QiniuResponseError("grok2api 未返回视频 request_id")
        path = "/videos/" + quote(request_id, safe="")
        failures = 0
        while True:
            await asyncio.sleep(5)
            try:
                data = await self._request_json("GET", path)
            except (aiohttp.ClientError, asyncio.TimeoutError, QiniuRateLimitError, QiniuTransientApiError):
                failures += 1
                if failures >= 3:
                    raise
                continue
            failures = 0
            status = data.get("status")
            if status == "pending":
                continue
            if status == "failed":
                raise GrokVideoError("上游视频任务失败")
            if status != "done":
                raise QiniuResponseError("grok2api 返回了未知的视频任务状态")
            video = data.get("video")
            if not isinstance(video, dict):
                raise QiniuResponseError("grok2api 视频任务完成但没有结果")
            return self._media_url(video.get("url"))

    async def download_video(self, url: str, destination: Path) -> None:
        """流式下载到临时文件，让消息平台无需访问自部署服务地址。"""
        session = await self._get_session()
        headers = self._media_headers(url)
        headers["Accept"] = "video/*, application/octet-stream"
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=180)) as response:
            if response.status // 100 != 2:
                raise GrokVideoError(f"视频下载失败（HTTP {response.status}）")
            if response.content_length and response.content_length > MAX_VIDEO_BYTES:
                raise GrokVideoError("视频文件超过 200 MiB")
            size = 0
            prefix = bytearray()
            with destination.open("wb") as output:
                async for chunk in response.content.iter_chunked(64 * 1024):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise GrokVideoError("视频文件超过 200 MiB")
                    if len(prefix) < 12:
                        prefix.extend(chunk[:12 - len(prefix)])
                    output.write(chunk)
            if size < 12 or prefix[4:8] != b"ftyp":
                raise GrokVideoError("上游返回的文件不是 MP4 视频")
