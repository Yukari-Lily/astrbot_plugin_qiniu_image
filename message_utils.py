"""解析输入图片。"""

import re
from typing import Any, List, Optional, Tuple

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


def _reference(*values: Any) -> Optional[str]:
    """不同消息编码共用的直接图片引用解析。"""
    for value in values:
        if isinstance(value, str) and value.startswith(_ACCEPTED_PREFIXES):
            return value
    return None


async def _image_to_reference(seg: Any, client: Any) -> Optional[str]:
    """把一个 Image 组件转成上游可用的引用。"""
    direct = _reference(getattr(seg, "url", None), getattr(seg, "file", None))
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
    for match in IMG_CQ_RE.finditer(cq_text):
        values = dict(KV_RE.findall(match.group("kv")))
        found = _reference(values.get("url"), values.get("file"))
        if found:
            return found
    return None


def _own_platform(context: Any, event: Any) -> Any:
    """取当前这条事件所属适配器实例的平台。

    同一个 AstrBot 进程里可能同时跑多个 aiocqhttp 适配器（一个群里挂两个 bot），按类型取只会
    拿到第一个命中的那个——于是 get_msg 可能跑在另一个 bot 的连接上，查不到图，甚至取回别的
    消息里的图片当底图。按实例 id 取才是这条事件自己的连接；旧版本没有这个方法时退回按类型取。
    """
    getter = getattr(context, "get_platform_inst", None)
    platform_id = getattr(event, "get_platform_id", None)
    if getter is not None and platform_id is not None:
        platform = getter(platform_id())
        if platform is not None:
            return platform
    return context.get_platform(event_filter.PlatformAdapterType.AIOCQHTTP)


async def _image_from_get_msg(context: Any, event: Any, reply_id: Any) -> Tuple[Optional[str], bool]:
    """通过 OneBot get_msg 查询被引用消息里的第一张图片。

    第二个返回值表示"被引用消息里确实有图片，只是没拿到可用地址"——调用方据此区分
    "用户没有发图"和"用户发了图但读不出来"。查询本身失败时无从判断，按没有图处理。
    """
    if event.get_platform_name() != "aiocqhttp":
        return None, False
    try:
        platform = _own_platform(context, event)
        if not platform:
            return None, False
        bot = platform.get_client()
    except Exception as exc:
        logger.debug(f"qiniu-image: 获取 aiocqhttp client 失败：{type(exc).__name__}")
        return None, False

    candidates = [reply_id]
    try:
        numeric_id = int(reply_id)
        if numeric_id != reply_id:
            candidates.append(numeric_id)
    except (TypeError, ValueError):
        pass

    saw_image = False
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
                saw_image = True
                data = part.get("data") or {}
                if isinstance(data, dict):
                    found = _reference(data.get("url"), data.get("file"))
                    if found:
                        return found, True
        elif isinstance(message, str):
            found = _first_image_from_cq_string(message)
            if found:
                return found, True
            saw_image = saw_image or bool(IMG_CQ_RE.search(message))

        raw = resp.get("raw_message")
        if isinstance(raw, str):
            found = _first_image_from_cq_string(raw)
            if found:
                return found, True
            saw_image = saw_image or bool(IMG_CQ_RE.search(raw))
    return None, saw_image


async def resolve_input_image(context: Any, event: Any, client: Any) -> Optional[str]:
    """返回本次绘图应使用的输入图片引用，没有则返回 None。

    被引用的消息里确实有图片、却取不到可用地址时抛 ``ValueError``：用户要的是改图，
    静默降级成文生图会画出一张与用户图片无关的图。
    """
    segments = _segments(event)
    saw_image = False

    for seg in segments:
        if isinstance(seg, Comp.Image):
            saw_image = True
            reference = await _image_to_reference(seg, client)
            if reference:
                return reference

    reply = next((seg for seg in segments if isinstance(seg, Comp.Reply)), None)
    if reply is not None:
        for seg in getattr(reply, "chain", None) or []:
            if isinstance(seg, Comp.Image):
                saw_image = True
                reference = await _image_to_reference(seg, client)
                if reference:
                    return reference
        reply_id = getattr(reply, "id", None)
        if reply_id not in (None, "", 0):
            reference, reply_has_image = await _image_from_get_msg(context, event, reply_id)
            if reference:
                return reference
            saw_image = saw_image or reply_has_image
    if saw_image:
        raise ValueError("用户输入或引用的图片无法读取")
    return None
