"""冷门角色的外观获取：下载候选图 → 一次视觉抽取 → 缓存。

搜索不在这里——沿用一直以来的做法，由主聊天模型用它已有的联网搜索（Tavily）
找到来源网页，再把网址交给本模块。插件只做搜索之后的事：安全下载、选图核对、
抽取、缓存。因此本模块不需要任何搜索密钥。

抽取结果是一份定键外观记录（见 `appearance.py`），最终由插件逐字写进提示词。
参考图本身**不会**进入绘图接口——这是刻意的：它只用来产出文字。
"""

import asyncio
import base64
import hashlib
import io
import ipaddress
import json
import re
import socket
import time
import warnings
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit, urldefrag

import aiohttp
from PIL import Image

from astrbot.api import logger

from . import appearance
from .qiniu_api import MAX_INPUT_IMAGE_BYTES, _image_mime

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_SOURCES = 4
MAX_CANDIDATES = 6
FETCH_BUDGET_SECONDS = 40
CACHE_TTL_SECONDS = 6 * 3600
CACHE_MAX_ENTRIES = 200


# --------------------------------------------------------------------------
# 图片校验与 SSRF 防护
# --------------------------------------------------------------------------

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
    """校验连接器实际使用的地址，防止 DNS 重绑定。"""

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
    """从来源页里提取候选图与页面正文。"""

    def __init__(self, url: str):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.images: List[dict] = []
        self.text: List[str] = []
        self._skip = 0
        self._figure: Optional[List[dict]] = None
        self._caption = False
        self._caption_text: List[str] = []

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


class SafeFetcher:
    """只访问公开 HTTP(S) 地址的抓取器，附带重定向与大小限制。"""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._resolver: Optional[PublicResolver] = None
        self._slots = asyncio.Semaphore(3)

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
        if self._resolver:
            await self._resolver.close()
            self._resolver = None

    async def fetch(self, url: str, *, image_only: bool = False) -> Tuple[bytes, str, str]:
        async with self._slots:
            return await asyncio.wait_for(self._fetch(url, image_only=image_only), timeout=30)

    async def _fetch(self, url: str, *, image_only: bool = False):
        if self._session is None:
            self._resolver = PublicResolver()
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(resolver=self._resolver, limit=3),
                timeout=aiohttp.ClientTimeout(total=20, connect=8), trust_env=False,
                cookie_jar=aiohttp.DummyCookieJar(),
                headers={"User-Agent": "AstrBot-QiniuImage/2.0 (character reference fetcher)"},
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


def image_data(raw: bytes) -> str:
    mime = _image_mime(raw, allow_gif=True)
    if not mime:
        raise ValueError("不支持的图片格式")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


# --------------------------------------------------------------------------
# 外观缓存
# --------------------------------------------------------------------------

def _normalize_key(*parts: Any) -> str:
    text = "|".join(re.sub(r"[\s\W_]+", "", str(part or "")).casefold() for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class AppearanceCache:
    """按内容键缓存外观记录。纯文本，进程内，不落盘。"""

    def __init__(self, *, ttl: float = CACHE_TTL_SECONDS, max_entries: int = CACHE_MAX_ENTRIES,
                 clock: Callable[[], float] = time.monotonic):
        self._ttl = ttl
        self._max = max_entries
        self._clock = clock
        self._entries: Dict[str, Tuple[float, dict]] = {}
        self._aliases: Dict[str, str] = {}

    def _prune(self, now: float) -> None:
        for key in [key for key, (expiry, _) in self._entries.items() if expiry <= now]:
            self._entries.pop(key, None)
        if len(self._entries) > self._max:
            ordered = sorted(self._entries.items(), key=lambda item: item[1][0])
            for key, _ in ordered[:len(self._entries) - self._max]:
                self._entries.pop(key, None)
        live = set(self._entries)
        for alias, key in list(self._aliases.items()):
            if key not in live:
                self._aliases.pop(alias, None)

    def key(self, name: str, work: str = "", version: str = "") -> str:
        return _normalize_key(work, name, version)

    def get(self, *keys: str) -> Optional[dict]:
        now = self._clock()
        self._prune(now)
        for key in keys:
            if not key:
                continue
            entry = self._entries.get(self._aliases.get(key, key))
            if entry and entry[0] > now:
                return entry[1]
        return None

    def put(self, key: str, record: dict, *, alias_name: str = "") -> None:
        now = time.monotonic()
        self._entries[key] = (now + self._ttl, record)
        if alias_name:
            self._aliases[_normalize_key(alias_name)] = key
        self._prune(now)

    def alias_key(self, canonical_name: str) -> str:
        return _normalize_key(canonical_name)


# --------------------------------------------------------------------------
# 抽取
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "你是角色外观提取员。只记录图中可见的外观，不做身份推断，"
    "不补充图片以外的资料，也不把该角色的既有印象写进来。"
)

_FIELD_HINT = "、".join(appearance.REQUIRED_PATHS)


def _extraction_prompt(subject: str, work: str, version: str,
                       rows: List[dict], problems: Sequence[str] = ()) -> str:
    lines = [
        "核对候选图与目标人物及形象版本，选出一张，并把可见外观逐字段填满。",
        "",
        "规则：",
        "- 只填图里看得见的像素。不要用你对该角色的既有印象补全，也不要照抄页面文字。",
        "- 不要记录动作、表情、姿势、镜头、视角、背景、光照或画风——这些不属于外观。",
        f"- 下列字段必须填满：{_FIELD_HINT}。",
        "- 真的没有该特征（例如没有头饰）时留空，不要写“未知”“无”“n/a”等占位内容。",
        "- 左右不对称的细节必须写明在哪一侧。",
        "- 候选图之间版本冲突时返回 uncertain，不要取平均。",
        "- identity_basis 只写身份判断依据，不要重复外观描述。",
    ]
    if problems:
        lines.insert(1, f"上一次抽取有这些问题：{'；'.join(problems)}。请重新看图修正，只改这些字段。")
    lines += [
        "",
        "返回 JSON：",
        '{"status":"confirmed","selected_index":1,"canonical_name":"准确名称","version":"形象版本",'
        '"identity_basis":"为什么这是该角色或该版本",'
        '"appearance":{"silhouette":"","hair":{"color":"","length":"","style":"","front":""},'
        '"eyes":{"color":"","shape":""},"skin":"","build":"",'
        '"outfit":{"pieces":[],"cut":"","trim":""},"headwear":"","accessory":"",'
        '"palette":[],"marks":"","asymmetry":"","signature":"","version":""}}',
        "",
        "身份依据不足、候选图与目标无关或是网页 logo 时返回 "
        '{"status":"uncertain","reason":"原因"}。',
        "",
        json.dumps(
            {"subject": subject, "work": work, "version": version,
             "candidates": [{"index": i + 1, **{k: row.get(k, "") for k in ("source_url", "label", "page_text")}}
                            for i, row in enumerate(rows)]},
            ensure_ascii=False,
        ),
    ]
    return "\n".join(lines)


def _valid_assessment(result: Any, total: int) -> bool:
    if not isinstance(result, dict) or result.get("status") != "confirmed":
        return False
    index = result.get("selected_index")
    if type(index) is not int or not 1 <= index <= total:
        return False
    if not str(result.get("canonical_name", "")).strip():
        return False
    if not str(result.get("identity_basis", "")).strip():
        return False
    record = result.get("appearance")
    # 只看致命问题：可选字段上有个占位词不该触发重试，那一项渲染时会被丢掉。
    return isinstance(record, dict) and not appearance.fatal_problems(record)


class CharacterReference:
    """把一个名字加一组来源网址，变成一份可用的外观记录。"""

    def __init__(self, context: Any, providers: dict, *, fetcher: SafeFetcher,
                 cache: Optional[AppearanceCache] = None):
        self.context = context
        self.providers = providers
        self.fetcher = fetcher
        self.cache = cache or AppearanceCache()

    def lookup(self, name: str, work: str = "", version: str = "") -> Optional[dict]:
        """按名字取缓存记录；画图时走这里，避免重复抓取。"""
        return self.cache.get(self.cache.key(name, work, version), self.cache.alias_key(name))

    async def prepare(self, umo: str, name: str, work: str = "", version: str = "",
                      source_urls: Sequence[str] = ()) -> dict:
        name, work, version = name.strip(), (work or "").strip(), (version or "").strip()
        if not name:
            return {"status": "unavailable", "reason": "缺少人物名称",
                    "instruction": "不要编造外观；有可靠文字依据时可继续。"}

        cached = self.lookup(name, work, version)
        if cached:
            logger.info(f"qiniu-image: 外观缓存命中｜name={name[:40]}")
            return {"status": "confirmed", "cached": True, **cached}

        urls = [url.strip() for url in source_urls or []
                if isinstance(url, str) and url.strip().startswith(("http://", "https://"))]
        urls = list(dict.fromkeys(urls))[:MAX_SOURCES]
        if not urls:
            return {"status": "unavailable", "reason": "未提供来源网址",
                    "instruction": "先用你的联网搜索找到该角色的资料页或图片地址，"
                                   "把网址填入 source_urls 后重试；仍然找不到时不要编造外观。"}

        candidates, failures = await self._candidates(urls)
        if not candidates:
            return {"status": "unavailable", "reason": "未取得可核对的参考图", "failures": failures,
                    "instruction": "不要编造外观；仅在已有可靠文字依据时继续。"}

        rows = [{"source_url": item["source_url"], "label": item["label"], "page_text": item["page_text"]}
                for item in candidates]
        record = await self._extract(umo, name, work, version, candidates, rows)
        if not record:
            return {"status": "unavailable", "reason": "参考图未通过核对", "failures": failures,
                    "instruction": "不要使用未核对的候选图；仅在已有可靠文字依据时继续。"}

        payload = {
            "canonical_name": record["canonical_name"],
            "version": record.get("version", "") or version,
            "identity_basis": record["identity_basis"],
            "appearance": appearance.normalize(record["appearance"]),
            "source_url": candidates[record["selected_index"] - 1]["source_url"],
            "image_url": candidates[record["selected_index"] - 1]["image_url"],
        }
        self.cache.put(self.cache.key(name, work, version), payload,
                       alias_name=payload["canonical_name"])
        logger.info(
            f"qiniu-image: 外观已提取｜name={name[:40]} canonical={payload['canonical_name'][:40]} "
            f"candidates={len(candidates)} fields={len(appearance.render(payload['appearance']))}"
        )
        return {"status": "confirmed", "cached": False, **payload, "failures": failures}

    async def _candidates(self, urls: Sequence[str]) -> Tuple[List[dict], List[str]]:
        """下载来源页并从中挑出候选图，最多 MAX_CANDIDATES 张。"""
        failures: List[str] = []
        direct: List[dict] = []
        queued: List[dict] = []
        seen: set = set()
        deadline = time.monotonic() + FETCH_BUDGET_SECONDS

        def queue(image_url: str, label: str, page_text: str, source_url: str) -> None:
            if image_url in seen:
                return
            seen.add(image_url)
            queued.append({"image_url": image_url, "label": label, "page_text": page_text,
                           "source_url": source_url})

        for url in urls:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append("取图超出时间预算")
                break
            try:
                raw, mime, final_url = await asyncio.wait_for(
                    self.fetcher.fetch(url), timeout=min(10, remaining)
                )
            except (ValueError, OSError, aiohttp.ClientError, asyncio.TimeoutError):
                failures.append("部分来源无法读取")
                continue
            if mime.startswith("image/"):
                # 模型直接给了图片直链：刚下载的字节就是候选图，不用再取一次。
                seen.add(final_url)
                direct.append({"image_url": final_url, "label": "", "page_text": "",
                               "source_url": final_url, "bytes": raw})
                continue
            parser = PageImages(final_url)
            parser.feed(raw.decode("utf-8", errors="replace"))
            page_text = " ".join(parser.text)[:6000]
            for image in sorted(parser.images, key=lambda item: item["rank"]):
                queue(image["url"], image["label"], page_text, final_url)

        candidates: List[dict] = list(direct)
        for item in queued:
            if len(candidates) >= MAX_CANDIDATES:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append("取图超出时间预算")
                break
            try:
                data, _, final_url = await asyncio.wait_for(
                    self.fetcher.fetch(item["image_url"], image_only=True), timeout=min(10, remaining)
                )
            except (ValueError, OSError, aiohttp.ClientError, asyncio.TimeoutError):
                failures.append("部分候选图片无法读取")
                continue
            # source_url 始终保持为来源**页面**，image_url 才是像素出处；两者都会
            # 交给视觉模型，页面正文与图片必须成对，不能混。
            candidates.append({**item, "image_url": final_url or item["image_url"], "bytes": data})
        return candidates[:MAX_CANDIDATES], sorted(set(failures))

    async def _extract(self, umo: str, name: str, work: str, version: str,
                       candidates: List[dict], rows: List[dict]) -> Optional[dict]:
        """一次抽取，失败则按缺失字段定向重试一次。"""
        from .prompt_rewriter import visual_json

        seen: List[Any] = []
        problems: Sequence[str] = ()
        for attempt in (1, 2):
            seen.clear()
            result = await visual_json(
                self.context, umo,
                prompt=_extraction_prompt(name, work, version, rows, problems),
                image_urls=[image_data(item["bytes"]) for item in candidates],
                validate=lambda value, total=len(candidates): _valid_assessment(value, total),
                purpose="角色外观提取",
                system_prompt=EXTRACTION_SYSTEM,
                on_result=seen.append,
                **self.providers,
            )
            if result:
                return result
            last = seen[-1] if seen and isinstance(seen[-1], dict) else {}
            # 点名要求补的是**致命**问题。可选字段的瑕疵不该把模型再叫回来一次，
            # 它渲染时会被丢掉，重试一次只是白花钱。
            problems = appearance.fatal_problems(last.get("appearance")) or (
                [f"未按要求的格式返回（status={last.get('status', '未返回 JSON')}）"] if last
                else ["未返回可解析的 JSON"]
            )
            if attempt == 1:
                logger.info(f"qiniu-image: 外观抽取不合格，定向重试｜name={name[:40]} "
                            f"status={last.get('status', '未返回 JSON')} problems={list(problems)}")
        return None
