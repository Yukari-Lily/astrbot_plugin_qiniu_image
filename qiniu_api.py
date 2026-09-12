"""七牛 Modelink bypass API 客户端。"""

import asyncio
import base64
import binascii
import json
import random
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import aiohttp

GENERATIONS_PATH = "/bypass/openai/v1/images/generations"
EDITS_PATH = "/bypass/openai/v1/images/edits"
API_BASE = "https://api.qnaigc.com"
REQUEST_ATTEMPTS = 4
REQUEST_TIMEOUT_SECONDS = 480
CONNECT_TIMEOUT_SECONDS = 30
MAX_INPUT_IMAGE_BYTES = 40_000_000
MAX_OUTPUT_IMAGE_BYTES = 80 * 1024 * 1024
OUTPUT_FORMAT = "png"
SUPPORTED_MODELS = (
    "openai/gpt-image-2.5-sunburst",
    "openai/gpt-image-2.5-flare",
    "openai/gpt-image-2",
)
SUPPORTED_QUALITIES = ("low", "medium", "high", "auto")
SUPPORTED_SIZES = (
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "2048x2048",
    "2048x1152",
    "3840x2160",
    "2160x3840",
)


class QiniuApiError(RuntimeError):
    """上游返回的确定性 API 错误。"""

    def __init__(self, status: int, message: str, code: Optional[str] = None):
        self.status = status
        self.code = code
        self.detail = _safe_excerpt(message)
        suffix = f" code={code}" if code else ""
        super().__init__(f"HTTP {status}{suffix}: {self.detail}")


class QiniuAuthError(QiniuApiError):
    """API Key 无效或没有调用权限。"""


class QiniuRateLimitError(QiniuApiError):
    """上游限流。"""

    def __init__(
        self,
        status: int,
        message: str,
        code: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        self.retry_after = retry_after
        super().__init__(status, message, code)


class QiniuSafetyError(QiniuApiError):
    """安全系统拒绝。"""


class QiniuTransientApiError(QiniuApiError):
    """上游暂时性服务错误，可重试。"""

    def __init__(
        self,
        status: int,
        message: str,
        code: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        self.retry_after = retry_after
        super().__init__(status, message, code)


class QiniuResponseError(RuntimeError):
    """上游成功响应不符合预期结构。"""


class QiniuRequestUncertainError(RuntimeError):
    """上游请求在多次尝试后仍未完成。"""


class QiniuImageDownloadError(RuntimeError):
    """下载上游返回图片时失败。"""


class QiniuInputError(ValueError):
    """用户提供的输入图片无法转换为上游支持的格式。"""


class QiniuNotConfiguredError(RuntimeError):
    """未配置 api_key。"""


class _RetryableImageDownloadError(QiniuImageDownloadError):
    """下载图片时发生可安全重试的错误。"""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(message)


def _validate_http_url(value: str, field_name: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"{field_name} 必须是有效的 HTTP(S) 地址")
    return value


def _image_mime(raw: bytes, *, allow_gif: bool = False) -> Optional[str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if allow_gif and raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _safe_excerpt(value: Any, limit: int = 200) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer ***", text)
    text = re.sub(r"(?i)data:image/[^;]+;base64,[a-z0-9+/=\s]+", "<image data>", text)
    text = re.sub(r"(?i)base64://[a-z0-9+/=\s]+", "base64://<image data>", text)
    text = re.sub(r"https?://[^\s\"'<>]+", "<url>", text)
    text = " ".join(text.split())
    return text[:limit]


def _error_details(text: str) -> Tuple[Optional[str], str]:
    try:
        body = json.loads(text)
    except (TypeError, ValueError):
        return None, "上游返回了非 JSON 错误"

    error = body.get("error", body) if isinstance(body, dict) else body
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        message = error.get("message") or error.get("detail") or "上游拒绝了请求"
        return _safe_excerpt(code) or None, _safe_excerpt(message)
    return None, _safe_excerpt(error) or "上游拒绝了请求"


def _looks_like_safety_error(code: Optional[str], message: str) -> bool:
    haystack = f"{code or ''} {message}".lower()
    markers = (
        "safety",
        "moderation",
        "content_policy",
        "policy_violation",
        "content violation",
        "内容审核",
        "违规内容",
    )
    return any(marker in haystack for marker in markers)


def _retry_after_seconds(headers: Any) -> Optional[float]:
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, min(delay, 30.0))


def _backoff_seconds(attempt: int, retry_after: Optional[float] = None) -> float:
    if retry_after is not None:
        return retry_after
    return min(6.0, 0.8 * attempt) + random.random() * 0.3


def _config_string(config: Any, name: str, default: str, allow_empty: bool = False) -> str:
    value = config.get(name, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise ValueError(f"qiniu_image 配置项 {name} 必须是字符串")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"qiniu_image 配置项 {name} 不能为空")
    return value


def _config_positive_int(config: Any, name: str, default: int) -> int:
    value = config.get(name, default)
    if value is None or value == "":
        value = default
    if isinstance(value, bool):
        raise ValueError(f"qiniu_image 配置项 {name} 必须是正整数")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"qiniu_image 配置项 {name} 必须是正整数") from None
    if value <= 0:
        raise ValueError(f"qiniu_image 配置项 {name} 必须大于 0")
    return value


def _config_choice(config: Any, name: str, default: str, choices: Tuple[str, ...]) -> str:
    value = _config_string(config, name, default)
    if value not in choices:
        raise ValueError(f"qiniu_image 配置项 {name} 必须是 {'/'.join(choices)} 之一")
    return value


class QiniuImageClient:
    def __init__(self, config: Any):
        self.api_base = API_BASE
        self.api_key = _config_string(config, "api_key", "", allow_empty=True)
        self.model = _config_choice(
            config,
            "model",
            "openai/gpt-image-2.5-sunburst",
            SUPPORTED_MODELS,
        )

        self.retries = REQUEST_ATTEMPTS
        self.image_config = {
            "quality": _config_choice(config, "quality", "auto", SUPPORTED_QUALITIES),
            "size": _config_choice(config, "size", "auto", SUPPORTED_SIZES),
            "output_format": OUTPUT_FORMAT,
        }
        self.moderation = _config_string(config, "moderation", "low").strip().lower()
        if self.moderation not in ("auto", "low"):
            raise ValueError("qiniu_image 配置项 moderation 必须是 auto 或 low")

        self._timeout = aiohttp.ClientTimeout(
            total=REQUEST_TIMEOUT_SECONDS,
            connect=CONNECT_TIMEOUT_SECONDS,
            sock_connect=CONNECT_TIMEOUT_SECONDS,
        )
        self._semaphore = asyncio.Semaphore(_config_positive_int(config, "concurrency", 3))
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        session, self._session = self._session, None
        if session and not session.closed:
            await session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=self._timeout)
            return self._session

    async def text_to_image(self, prompt: str) -> List[str]:
        """文生图，返回不带前缀的 base64 图片列表。"""
        if not self.configured:
            raise QiniuNotConfiguredError("未配置 api_key")
        async with self._semaphore:
            session = await self._get_session()
            data = await self._post_json_with_retry(session, GENERATIONS_PATH, self._base_payload(prompt))
            return await self._images_from_response(session, data)

    async def image_to_image(self, image: str, prompt: str) -> List[str]:
        """图生图（编辑），image 为 http(s) URL 或 base64:// 形式。"""
        if not self.configured:
            raise QiniuNotConfiguredError("未配置 api_key")
        payload = self._base_payload(prompt)
        payload["images"] = [{"image_url": self.as_image_reference(image)}]

        async with self._semaphore:
            session = await self._get_session()
            data = await self._post_json_with_retry(session, EDITS_PATH, payload)
            return await self._images_from_response(session, data)

    def _base_payload(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            **self.image_config,
            "n": 1,
            "stream": False,
        }
        if self.moderation != "auto":
            payload["moderation"] = self.moderation
        return payload

    def decode_base64_image(self, value: str, *, max_bytes: int = MAX_INPUT_IMAGE_BYTES) -> bytes:
        compact = re.sub(r"\s+", "", value)
        max_encoded_length = ((max_bytes + 2) // 3) * 4
        if not compact or len(compact) > max_encoded_length:
            raise ValueError("图片 Base64 为空或超过大小限制")
        try:
            raw = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("图片 Base64 格式无效") from None
        if not raw or len(raw) > max_bytes:
            raise ValueError("图片为空或超过大小限制")
        return raw

    def as_image_reference(self, image: str) -> str:
        """将平台图片表示转换为 OpenAI Images 的 image_url 值。"""
        if not isinstance(image, str) or not image:
            raise QiniuInputError("未能识别输入图片")
        if image.startswith(("http://", "https://")):
            try:
                return _validate_http_url(image, "输入图片 URL")
            except ValueError as exc:
                raise QiniuInputError(str(exc)) from None
        if image.startswith("base64://"):
            encoded = image[len("base64://"):]
            try:
                raw = self.decode_base64_image(encoded)
            except ValueError as exc:
                raise QiniuInputError(str(exc)) from None
            mime = _image_mime(raw, allow_gif=True)
            if not mime:
                raise QiniuInputError("输入图片格式无效，仅支持 PNG、JPEG、WebP 或 GIF")
            compact = re.sub(r"\s+", "", encoded)
            return f"data:{mime};base64,{compact}"
        raise QiniuInputError("未能识别输入图片（仅支持 URL 或 base64://）")

    async def _post_json_with_retry(
        self,
        session: aiohttp.ClientSession,
        path: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        attempts = self.retries

        for attempt in range(1, attempts + 1):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status // 100 == 2:
                        text = await resp.text()
                        code, message = None, ""
                    else:
                        raw_error = await resp.content.read(8192)
                        text = raw_error.decode(resp.charset or "utf-8", errors="replace")
                        code, message = _error_details(text)

                    if resp.status == 429:
                        raise QiniuRateLimitError(
                            resp.status,
                            message,
                            code,
                            _retry_after_seconds(resp.headers),
                        )

                    if 500 <= resp.status < 600:
                        raise QiniuTransientApiError(
                            resp.status,
                            message,
                            code,
                            _retry_after_seconds(resp.headers),
                        )

                    if resp.status // 100 != 2:
                        if _looks_like_safety_error(code, message):
                            raise QiniuSafetyError(resp.status, message, code)
                        if resp.status in (401, 403):
                            raise QiniuAuthError(resp.status, message, code)
                        raise QiniuApiError(resp.status, message, code)

                    try:
                        data = json.loads(text)
                    except (TypeError, ValueError):
                        raise QiniuResponseError("上游返回了无效 JSON") from None
                    if not isinstance(data, dict):
                        raise QiniuResponseError("上游响应必须是 JSON 对象")
                    self._validate_image_response(data)
                    return data

            except (QiniuRateLimitError, QiniuTransientApiError) as exc:
                if attempt < attempts:
                    await asyncio.sleep(_backoff_seconds(attempt, exc.retry_after))
                    continue
                raise
            except QiniuResponseError:
                if attempt < attempts:
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                raise
            except QiniuApiError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < attempts:
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                raise QiniuRequestUncertainError(
                    f"上游请求失败（{type(exc).__name__}），已达到最大尝试次数"
                ) from None

        raise QiniuRequestUncertainError("上游请求未完成")

    async def _download_to_b64(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            _validate_http_url(url, "响应图片 URL")
        except ValueError as exc:
            raise QiniuImageDownloadError(str(exc)) from None

        for attempt in range(1, self.retries + 1):
            try:
                async with session.get(url) as resp:
                    if resp.status == 429 or 500 <= resp.status < 600:
                        retry_after = _retry_after_seconds(resp.headers) if resp.status == 429 else None
                        raise _RetryableImageDownloadError(
                            f"图片下载暂时失败：HTTP {resp.status}",
                            retry_after,
                        )
                    if resp.status // 100 != 2:
                        raise QiniuImageDownloadError(f"图片下载失败：HTTP {resp.status}")

                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        try:
                            if int(content_length) > MAX_OUTPUT_IMAGE_BYTES:
                                raise QiniuImageDownloadError("响应图片超过大小限制")
                        except ValueError:
                            pass

                    content_type = (resp.headers.get("Content-Type") or "").split(";", 1)[0].lower()
                    if content_type and not (
                        content_type.startswith("image/") or content_type == "application/octet-stream"
                    ):
                        raise QiniuImageDownloadError(f"响应不是图片（Content-Type: {content_type}）")

                    content = bytearray()
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        content.extend(chunk)
                        if len(content) > MAX_OUTPUT_IMAGE_BYTES:
                            raise QiniuImageDownloadError("响应图片超过大小限制")
                    if not content:
                        raise QiniuImageDownloadError("响应图片为空")
                    raw_image = bytes(content)
                    if not _image_mime(raw_image):
                        raise QiniuImageDownloadError("响应内容不是支持的 PNG、JPEG 或 WebP 图片")
                    return base64.b64encode(raw_image).decode("ascii")

            except _RetryableImageDownloadError as exc:
                if attempt < self.retries:
                    await asyncio.sleep(_backoff_seconds(attempt, exc.retry_after))
                    continue
                raise
            except QiniuImageDownloadError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < self.retries:
                    await asyncio.sleep(_backoff_seconds(attempt))
                    continue
                raise QiniuImageDownloadError(
                    f"图片下载失败（{type(exc).__name__}）"
                ) from None

        raise QiniuImageDownloadError("图片下载失败")

    def _normalize_response_base64(self, value: str) -> str:
        if value.startswith("data:"):
            marker = ";base64,"
            index = value.find(marker)
            if index < 0:
                raise QiniuResponseError("上游返回了无效的图片 data URI")
            value = value[index + len(marker):]
        try:
            raw = self.decode_base64_image(value, max_bytes=MAX_OUTPUT_IMAGE_BYTES)
        except ValueError as exc:
            raise QiniuResponseError(str(exc)) from None
        if not _image_mime(raw):
            raise QiniuResponseError("上游返回的 Base64 不是支持的 PNG、JPEG 或 WebP 图片")
        return re.sub(r"\s+", "", value)

    def _validate_image_response(self, data: Dict[str, Any]) -> None:
        """验证并规范图片响应；失败时由调用方按成功率优先策略重试。"""
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise QiniuResponseError("生成成功但响应中没有图片数据")

        last_error: Optional[QiniuResponseError] = None
        for item in items:
            if not isinstance(item, dict):
                continue
            inline = item.get("b64_json")
            if isinstance(inline, str) and inline:
                try:
                    item["b64_json"] = self._normalize_response_base64(inline)
                    return
                except QiniuResponseError as exc:
                    last_error = exc
            url = item.get("url")
            if isinstance(url, str) and url:
                try:
                    _validate_http_url(url, "响应图片 URL")
                    return
                except ValueError as exc:
                    last_error = QiniuResponseError(str(exc))

        if last_error:
            raise last_error
        raise QiniuResponseError("生成成功但响应中没有可用的 b64_json 或 URL")

    async def _images_from_response(
        self,
        session: aiohttp.ClientSession,
        data: Dict[str, Any],
    ) -> List[str]:
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise QiniuResponseError("生成成功但响应中没有图片数据")

        for item in items:
            if not isinstance(item, dict):
                continue
            inline = item.get("b64_json")
            if isinstance(inline, str) and inline:
                return [self._normalize_response_base64(inline)]
            url = item.get("url")
            if isinstance(url, str) and url:
                return [await self._download_to_b64(session, url)]

        raise QiniuResponseError("生成成功但响应中没有可用的 b64_json 或 URL")
