"""改写绘图提示词。"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from astrbot.api import logger

from .style_presets import build_style_guidance

_MIN_LENGTH = 4
_MAX_LENGTH = 1500

_HISTORY_ITEM_LIMIT = 500

_REFUSAL_MARKERS = (
    "抱歉",
    "对不起",
    "无法",
    "不能",
    "作为一个",
    "作为一名",
    "我是一个",
    "很高兴",
    "请问",
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "as an ai",
)


def _plausible(text: str, user_prompt: str, has_image: bool) -> bool:
    if not text:
        return False
    if len(text) < _MIN_LENGTH or len(text) > _MAX_LENGTH:
        return False
    lowered = text.lower()
    head = lowered[:40]
    if any(marker in head for marker in _REFUSAL_MARKERS):
        return False
    if not has_image and len(text) * 2 < len(user_prompt):
        return False
    return True


def _flatten_content(content: Any) -> str:
    """历史消息的 content 可能是字符串，也可能是 ContentPart 列表。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(p.strip() for p in parts if p.strip()).strip()
    return ""


async def load_history(context: Any, umo: str, rounds: int) -> List[Dict[str, str]]:
    """取当前会话最近 rounds 轮问答，失败一律返回空列表。"""
    if rounds <= 0:
        return []
    try:
        manager = context.conversation_manager
        cid = await manager.get_curr_conversation_id(umo)
        if not cid:
            return []
        conversation = await manager.get_conversation(umo, cid)
        raw = json.loads(getattr(conversation, "history", None) or "[]")
    except Exception as exc:
        logger.debug(f"qiniu-image: 读取历史对话失败，跳过注入（{type(exc).__name__}）")
        return []
    if not isinstance(raw, list):
        return []

    messages: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("tool_calls") or item.get("tool_call_id"):
            continue
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _flatten_content(item.get("content"))
        if not text:
            continue
        messages.append({"role": role, "content": text[:_HISTORY_ITEM_LIMIT]})

    return messages[-(rounds * 2):]


async def _call_llm(
    context: Any,
    provider_id: str,
    prompt: str,
    system_prompt: str,
    contexts: Optional[List[Dict[str, str]]],
    image_urls: Optional[List[str]],
) -> Optional[str]:
    kwargs: Dict[str, Any] = {
        "chat_provider_id": provider_id,
        "prompt": prompt,
        "system_prompt": system_prompt,
    }
    if contexts:
        kwargs["contexts"] = contexts
    if image_urls:
        kwargs["image_urls"] = image_urls

    try:
        resp = await context.llm_generate(**kwargs)
    except TypeError:
        resp = await context.llm_generate(
            chat_provider_id=provider_id,
            prompt=f"{system_prompt}\n\n---\n\n{prompt}",
        )
    return getattr(resp, "completion_text", None)


async def rewrite(
    context: Any,
    umo: str,
    user_prompt: str,
    *,
    has_image: bool,
    timeout: int,
    system_prompt: str,
    image_url: Optional[str] = None,
    history_rounds: int = 0,
    style_mode: str = "disabled",
    style_strength: str = "normal",
) -> str:
    """返回改写后的提示词；任何失败都返回 user_prompt 原文。"""
    if not user_prompt or not system_prompt:
        return user_prompt

    try:
        provider_id = await context.get_current_chat_provider_id(umo=umo)
    except Exception as exc:
        logger.warning(f"qiniu-image: 获取聊天模型失败，使用原始提示词（{type(exc).__name__}）")
        return user_prompt
    if not provider_id:
        logger.debug("qiniu-image: 当前会话没有可用聊天模型，使用原始提示词")
        return user_prompt

    contexts = await load_history(context, umo, history_rounds)
    style_guidance = build_style_guidance(
        user_prompt,
        mode=style_mode,
        strength=style_strength,
        has_image=has_image,
    )
    effective_system_prompt = f"{system_prompt}\n{style_guidance}" if style_guidance else system_prompt
    task = f"{'编辑要求' if has_image else '绘图要求'}：{user_prompt}"

    attempts: List[Optional[List[str]]] = [[image_url]] if image_url else []
    attempts.append(None)

    text: Optional[str] = None
    for image_urls in attempts:
        try:
            text = await asyncio.wait_for(
                _call_llm(context, provider_id, task, effective_system_prompt, contexts, image_urls),
                timeout=max(1, timeout),
            )
            break
        except asyncio.TimeoutError:
            logger.warning(f"qiniu-image: 提示词改写超时（{timeout}s），使用原始提示词")
            return user_prompt
        except Exception as exc:
            if image_urls:
                logger.warning(
                    f"qiniu-image: 视觉改写失败，降级为纯文本改写（{type(exc).__name__}: {exc}）"
                )
                continue
            logger.warning(f"qiniu-image: 提示词改写失败，使用原始提示词（{type(exc).__name__}: {exc}）")
            return user_prompt

    text = (text or "").strip().strip('"').strip("“”").strip()
    if not _plausible(text, user_prompt, has_image):
        logger.warning(f"qiniu-image: 改写结果不可用，使用原始提示词｜结果={text[:80]!r}")
        return user_prompt

    logger.info(
        f"qiniu-image: 提示词改写｜历史={len(contexts)}条 视觉={'是' if image_url else '否'}"
        f"｜原文={user_prompt[:60]!r}｜改写={text[:120]!r}"
    )
    return text
