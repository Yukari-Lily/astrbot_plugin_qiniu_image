"""优化与安全改写共用的文字模型调用及有界回退。"""

import asyncio
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from astrbot.api import logger

LLM_TIMEOUT_SECONDS = 45
ATTEMPTS_PER_PROVIDER = 3
T = TypeVar("T")


class ModelOutputError(ValueError):
    """结果不可执行；错误说明可安全反馈给模型，便于下一次修正。"""


# 只认句首的对话式拒答：改写结果要是画面描述，句中出现"很高兴""很遗憾"这类字词不能被误伤。
# 因此判定的是"第一句是不是以拒答开头"，不是"开头十几个字里有没有这些字样"。
_REFUSAL_MARKERS = (
    "抱歉",
    "对不起",
    "很抱歉",
    "很遗憾",
    "作为一个",
    "作为一名",
    "我无法",
    "我不能",
    "无法提供",
    "不能提供",
    "我是一个ai",
    "我是一个人工智能",
    "我是一个语言模型",
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "as an ai",
)
_SENTENCE_END_RE = re.compile(r"[。！？!?；;\n]")
_OPENING_PUNCTUATION = "「『“\"'（( 　"

def is_prompt_text(text: str) -> bool:
    """判断结果是"一段提示词"还是"模型在回答我们"。

    改写是否保住原意、写得够不够好交给模型自己与逐级重试判断，插件只挡掉句首拒答这类
    不能送进图片模型的结果。
    """
    if not text:
        return False
    opening = _SENTENCE_END_RE.split(text, maxsplit=1)[0].lstrip(_OPENING_PUNCTUATION).lower()
    return not any(opening.startswith(marker) for marker in _REFUSAL_MARKERS)


async def _call_llm(context: Any, provider_id: str, prompt: str, system_prompt: str) -> str:
    generate = context.llm_generate
    try:
        parameters = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        supports_system = True
    else:
        parameter = parameters.get("system_prompt")
        supports_system = (
            parameter is not None
            and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY)
        ) or any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())

    kwargs: Dict[str, Any] = {"chat_provider_id": provider_id, "prompt": prompt}
    if supports_system:
        kwargs["system_prompt"] = system_prompt
    else:
        kwargs["prompt"] = f"{system_prompt}\n\n---\n\n{prompt}"
    response = await generate(**kwargs)
    text = getattr(response, "completion_text", None)
    return text.strip() if isinstance(text, str) else ""


async def _resolve_provider_ids(
    context: Any,
    umo: str,
    provider_id: str,
    fallback_provider_ids: Sequence[str],
) -> List[str]:
    if not provider_id:
        try:
            provider_id = await context.get_current_chat_provider_id(umo=umo)
        except Exception as exc:
            logger.warning(f"qiniu-image: 获取当前聊天模型失败（{type(exc).__name__}）")
    candidates = (provider_id, *fallback_provider_ids)
    return list(dict.fromkeys(item.strip() for item in candidates
                             if isinstance(item, str) and item.strip()))


async def call_with_fallback(
    context: Any,
    umo: str,
    *,
    prompt: str,
    system_prompt: str,
    parse: Callable[[str], Optional[T]],
    purpose: str,
    provider_id: str = "",
    fallback_provider_ids: Sequence[str] = (),
    timeout_seconds: int = LLM_TIMEOUT_SECONDS,
) -> Optional[Tuple[T, str]]:
    """每轮按配置顺序尝试 Provider，失败先用备用模型，再进入下一轮。"""
    provider_ids = await _resolve_provider_ids(context, umo, provider_id, fallback_provider_ids)
    if not provider_ids:
        logger.warning(f"qiniu-image: {purpose}没有可用模型｜umo={umo}")
        return None
    feedback = ""
    for attempt in range(1, ATTEMPTS_PER_PROVIDER + 1):
        for current_provider in provider_ids:
            candidate = ""
            try:
                candidate = await asyncio.wait_for(
                    _call_llm(context, current_provider, prompt,
                              system_prompt + (f"\n\n输出校验修正要求：{feedback}" if feedback else "")),
                    timeout=timeout_seconds,
                )
                parsed = parse(candidate)
                if parsed is not None:
                    return parsed, current_provider
                raise ModelOutputError("结果为空或不是有效提示词，请按要求返回可执行结果。")
            except ModelOutputError as exc:
                feedback = str(exc)
                reason = f"输出校验失败：{feedback}"
                logger.debug(f"qiniu-image invalid model output | umo={umo} provider={current_provider} output={candidate[:1200]!r}")
            except asyncio.TimeoutError:
                reason = f"调用超时（{timeout_seconds}s）"
            except Exception as exc:
                reason = f"调用或解析失败（{type(exc).__name__}）"
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                if status is not None:
                    reason += f" HTTP {status}"
            logger.warning(
                f"qiniu-image: {purpose}{reason}｜umo={umo} provider={current_provider} "
                f"attempt={attempt}/{ATTEMPTS_PER_PROVIDER}"
            )
    return None
