"""解析输入图片。"""

import re
from typing import Any, List, Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import filter as event_filter

IMG_CQ_RE = re.compile(r"\[CQ:image,(?P<kv>[^\]]+)\]")
KV_RE = re.compile(r"([^,=]+)=([^,]+)")

_ACCEPTED_PREFIXES = ("http://", "https://", "base64://")


def _segments(event: Any) -> List[Any]:
    try:
        return list(event.get_messages() or [])
    except Exception:
        return []


def _direct_reference(seg: Any) -> Optional[str]:
    """不下载、不解码，直接可用的图片引用。"""
    for value in (getattr(seg, "url", None), getattr(seg, "file", None)):
        if isinstance(value, str) and value.startswith(_ACCEPTED_PREFIXES):
            return value
    return None


async def _image_to_reference(seg: Any, client: Any) -> Optional[str]:
    """把一个 Image 组件转成上游可用的引用。"""
    direct = _direct_reference(seg)
    if direct:
        return direct

    try:
        encoded = await seg.convert_to_base64()
    except Exception as exc:
        logger.debug(f"qiniu-image: 图片组件转 base64 失败：{type(exc).__name__}")
        return None
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        client.decode_base64_image(encoded)
    except ValueError as exc:
        logger.warning(f"qiniu-image: 引用图片不可用（{exc}）")
        return None
    return f"base64://{encoded}"


def _first_image_from_cq_string(cq_text: str) -> Optional[str]:
    """从 CQ 码字符串里提取第一张图片的可用地址。"""
    match = IMG_CQ_RE.search(cq_text)
    if not match:
        return None
    values = dict(KV_RE.findall(match.group("kv")))
    url = values.get("url")
    if url:
        return url
    file_ = values.get("file")
    if isinstance(file_, str) and file_.startswith(_ACCEPTED_PREFIXES):
        return file_
    return None


async def _image_from_get_msg(context: Any, event: Any, reply_id: Any) -> Optional[str]:
    """通过 OneBot get_msg 查询被引用消息里的第一张图片。"""
    if event.get_platform_name() != "aiocqhttp":
        return None
    try:
        platform = context.get_platform(event_filter.PlatformAdapterType.AIOCQHTTP)
        if not platform:
            return None
        bot = platform.get_client()
    except Exception as exc:
        logger.debug(f"qiniu-image: 获取 aiocqhttp client 失败：{type(exc).__name__}")
        return None

    candidates = [reply_id]
    try:
        candidates.append(int(reply_id))
    except (TypeError, ValueError):
        pass

    for candidate in candidates:
        try:
            resp = await bot.api.call_action("get_msg", message_id=candidate)
        except Exception:
            continue
        if not isinstance(resp, dict):
            continue

        message = resp.get("message")
        if isinstance(message, list):
            for part in message:
                if not isinstance(part, dict) or part.get("type") != "image":
                    continue
                data = part.get("data") or {}
                url = data.get("url")
                if isinstance(url, str) and url.startswith(_ACCEPTED_PREFIXES):
                    return url
                file_ = data.get("file")
                if isinstance(file_, str) and file_.startswith(_ACCEPTED_PREFIXES):
                    return file_
        elif isinstance(message, str):
            found = _first_image_from_cq_string(message)
            if found:
                return found

        raw = resp.get("raw_message")
        if isinstance(raw, str):
            found = _first_image_from_cq_string(raw)
            if found:
                return found
    return None


async def resolve_input_images(context: Any, event: Any, client: Any) -> List[str]:
    """Stable input:1..N ordering: direct images, then images in the quoted message."""
    result: List[str] = []
    segments = _segments(event)
    reply = next((seg for seg in segments if isinstance(seg, Comp.Reply)), None)
    quoted = list(getattr(reply, "chain", None) or [])
    for seg in [*segments, *quoted]:
        if isinstance(seg, Comp.Image):
            reference = await _image_to_reference(seg, client)
            if reference is None:
                raise ValueError("输入图片无法读取，不能忽略该图片继续生成")
            if reference not in result:
                result.append(reference)
    if reply and not any(isinstance(seg, Comp.Image) for seg in quoted):
        reply_id = getattr(reply, "id", None)
        if reply_id not in (None, "", 0):
            reference = await _image_from_get_msg(context, event, reply_id)
            if reference and reference not in result:
                result.append(reference)
    return result
