"""Public web image retrieval. No search API, browser, or credentials required."""

import asyncio
import base64
import ipaddress
import io
import socket
import time
import warnings
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urldefrag

import aiohttp
from PIL import Image

from .qiniu_api import MAX_INPUT_IMAGE_BYTES, _image_mime

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_CANDIDATES = 6
REFERENCE_FETCH_SECONDS = 40


def validate_image(raw: bytes) -> None:
    if not _image_mime(raw, allow_gif=True):
        raise ValueError("不支持的图片格式")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as picture:
                picture.verify()
            with Image.open(io.BytesIO(raw)) as picture:
                picture.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValueError("图片损坏或像素尺寸过大") from None


def public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def public_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 8192:
        raise ValueError("图片或网页地址无效")
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password:
        raise ValueError("只支持无凭据的公开 HTTP(S) 地址")
    if p.port not in (None, 80, 443):
        raise ValueError("网页或图片端口不支持")
    host = p.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("不能访问本地或内网地址")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not public_ip(host):
            raise ValueError("不能访问本地或内网地址")
    return urldefrag(url)[0]


class PublicResolver(aiohttp.abc.AbstractResolver):
    """Validate the actual addresses used by the connector, preventing DNS rebinding."""
    def __init__(self):
        self.delegate = aiohttp.resolver.DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        rows = await self.delegate.resolve(host, port, family)
        if not rows or any(not public_ip(row["host"]) for row in rows):
            raise OSError("DNS 返回非公开地址")
        return rows

    async def close(self):
        await self.delegate.close()


class PageImages(HTMLParser):
    def __init__(self, url: str):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.images: list[dict] = []
        self.text: list[str] = []
        self._skip = 0
        self._figure: list[dict] | None = None
        self._caption = False
        self._caption_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "nav", "footer"):
            self._skip += 1
        if self._skip:
            return
        if tag == "figure":
            self._figure = []
        if tag == "figcaption":
            self._caption, self._caption_text = True, []
        if tag == "meta" and attrs.get("property", attrs.get("name", "")) in ("og:image", "twitter:image"):
            self.add_image(attrs.get("content", ""), "网页封面（需核对，可能无关）", 2)
        if tag in ("img", "source"):
            label = (attrs.get("alt") or attrs.get("title") or "")[:600]
            for key in ("data-src", "data-original", "data-lazy-src", "src"):
                if attrs.get(key):
                    self.add_image(attrs[key], label, 0 if label else 1)
                    break
            srcset = attrs.get("data-srcset") or attrs.get("srcset") or ""
            if srcset:
                self.add_image(srcset.split(",")[-1].strip().split(" ")[0], label, 0 if label else 1)

    def add_image(self, value, label, rank):
        if not value or value.startswith(("data:", "javascript:")):
            return
        row = {"url": urljoin(self.url, value), "label": label, "rank": rank, "source_url": self.url}
        self.images.append(row)
        if self._figure is not None:
            self._figure.append(row)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer"):
            self._skip = max(0, self._skip - 1)
        if tag == "figcaption":
            self._caption = False
            for row in self._figure or []:
                row["label"] = (row["label"] + " " + " ".join(self._caption_text))[:800]
                row["rank"] = 0
        if tag == "figure":
            self._figure = None

    def handle_data(self, data):
        if self._skip or not data.strip():
            return
        if len(self.text) < 1500:
            self.text.append(data.strip())
        if self._caption:
            self._caption_text.append(data.strip())


class ReferenceFetcher:
    def __init__(self):
        self._session = None
        self._resolver = None
        self._slots = asyncio.Semaphore(3)

    async def close(self):
        if self._session:
            await self._session.close()
        if self._resolver:
            await self._resolver.close()

    async def fetch(self, url: str, *, image_only=False) -> tuple[bytes, str, str]:
        async with self._slots:
            return await asyncio.wait_for(self._fetch(url, image_only=image_only), timeout=30)

    async def _fetch(self, url: str, *, image_only=False):
        if self._session is None:
            self._resolver = PublicResolver()
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(resolver=self._resolver, limit=3),
                timeout=aiohttp.ClientTimeout(total=20, connect=8), trust_env=False,
                cookie_jar=aiohttp.DummyCookieJar(),
                headers={"User-Agent": "AstrBot-QiniuImage/1.4 (reference image fetcher)"},
            )
        for _ in range(5):
            url = public_url(url)
            async with self._session.get(url, allow_redirects=False) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise ValueError("重定向缺少地址")
                    url = urljoin(url, location)
                    continue
                if response.status != 200:
                    raise ValueError(f"读取失败 HTTP {response.status}")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
                limit = MAX_INPUT_IMAGE_BYTES if image_only or content_type.startswith("image/") else MAX_PAGE_BYTES
                data = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    data.extend(chunk)
                    if len(data) > limit:
                        raise ValueError("网页或图片超过大小限制")
                raw = bytes(data)
                mime = _image_mime(raw, allow_gif=True)
                if image_only and not mime:
                    raise ValueError("内容不是支持的图片")
                if not raw:
                    raise ValueError("网页或图片为空")
                if mime:
                    await asyncio.to_thread(validate_image, raw)
                return raw, mime or content_type, url
        raise ValueError("网页或图片重定向过多")

    async def candidates(self, source_urls: list[str]) -> tuple[list[dict], list[str]]:
        if not isinstance(source_urls, list) or not 1 <= len(source_urls) <= 3:
            raise ValueError("请提供 1 至 3 个搜索结果页面或图片地址")
        rows, failures, seen = [], [], set()
        deadline = time.monotonic() + REFERENCE_FETCH_SECONDS

        async def bounded_fetch(url, *, image_only=False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(self.fetch(url, image_only=image_only), timeout=min(10, remaining))

        for source in source_urls:
            try:
                raw, mime, final_url = await bounded_fetch(source)
                if mime.startswith("image/"):
                    rows.append({"url": final_url, "source_url": final_url, "label": "图片直链",
                                 "rank": 0, "bytes": raw, "page_text": ""})
                else:
                    if mime not in ("text/html", "application/xhtml+xml", "text/plain", ""):
                        raise ValueError("来源不是网页或支持的图片")
                    parser = PageImages(final_url)
                    parser.feed(raw.decode("utf-8", errors="replace"))
                    for row in parser.images:
                        row["page_text"] = " ".join(parser.text)[:6000]
                        rows.append(row)
            except (ValueError, OSError, aiohttp.ClientError, asyncio.TimeoutError):
                failures.append("部分来源页面无法读取")
        candidates = []
        # Bound attempted downloads as well as the number of model input images.
        attempts = 0
        for row in sorted(rows, key=lambda r: r["rank"]):
            if row["url"] in seen:
                continue
            seen.add(row["url"])
            if attempts >= MAX_CANDIDATES:
                break
            attempts += 1
            try:
                if "bytes" not in row:
                    row["bytes"], _, row["url"] = await bounded_fetch(row["url"], image_only=True)
                candidates.append(row)
            except (ValueError, OSError, aiohttp.ClientError, asyncio.TimeoutError):
                failures.append("部分候选图片无法读取")
        return candidates, sorted(set(failures))


def image_data(raw: bytes) -> str:
    mime = _image_mime(raw, allow_gif=True)
    if not mime:
        raise ValueError("不支持的图片格式")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
