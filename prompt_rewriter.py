"""改写绘图提示词。"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from astrbot.api import logger

from .style_presets import (
    QUALITY_GUIDANCE,
    SAFE_REFRAME_GUIDANCE,
    build_style_guidance,
    clean_style_metadata,
    find_explicit_presets,
)

_MIN_LENGTH = 4
_MAX_LENGTH = 1500

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

_SAFETY_REFRAME_STAGES = (
    "第一级（同义替换）：不改动画面内容，仅把明确露骨的词语替换为含蓄、克制的同义或近义表达；原服装、场景、姿势、构图与氛围全部保持不变。优先用二次元插画语境措辞包装（如动画角色插画、角色立绘、轻小说封面、动画剧照、角色设计图、时尚杂志内页插画），避免私密场景、暴露服装与挑逗姿势叠加组合，至少要替换掉其中一项。",
    "第二级（隐晦转述）：仍不改动画面内容，用模糊、含蓄、间接的语言整体重述提示词，避免直接点出敏感概念；把语义方向改写成二次元常见语境（角色插画、服装设计展示、青春日常、校园社团、都市街头插画等），画面保持原服装、场景与构图。",
    "第三级（轻微画面收敛）：保持含蓄措辞，同时开始最小画面调整：增加衣物覆盖，去除薄纱、透视、走光与身体敏感部位强调；仍尽量保留原服装类别、构图与氛围。",
    "第四级（中度收敛）：将可能被视为情趣或内衣的服装改为不透明的完整睡衣、家居服或时装，改用自然姿态和非色情的浪漫氛围，降低床铺、身体曲线和亲密暗示的视觉权重。",
    "第五级（最大安全）：转为适合全年龄展示的普通角色插画或时尚编辑肖像，穿完整日常服装，采用自然表情、中性动作和非私密场景，仅保留主体身份、核心配色、画风及安全的叙事元素。",
)

_OPTIMIZER_BOUNDARY_GUIDANCE = """
职责边界：主聊天模型拥有完整会话、Bot 人设和工具能力，负责全部创作决策并形成完整绘图或编辑方案；当前模型只接收这一份方案，并将它转换为图片模型容易执行的表达。将方案视为完整、权威的内容来源。除插件另外提供的全局质量规则和内置风格路由外，不得自行新增、替换或重新选择主体、人物数量、身份设定、剧情、服装、动作、表情、物品、场景、背景、构图、视角、光照、配色、媒介、画风、文字或特效。只可调整语序、消除歧义、校正客观事实、删除重复或内部冲突描述，并补充不改变既定画面语义的通用执行措辞。
""".strip()

def _plausible(
    text: str,
    user_prompt: str,
    has_image: bool,
    *,
    allow_shorter: bool = False,
) -> bool:
    if not text:
        return False
    if len(text) < _MIN_LENGTH or len(text) > _MAX_LENGTH:
        return False
    lowered = text.lower()
    head = lowered[:40]
    if any(marker in head for marker in _REFUSAL_MARKERS):
        return False
    if not allow_shorter and not has_image and len(text) * 2 < len(user_prompt):
        return False
    return True


async def _call_llm(
    context: Any,
    provider_id: str,
    prompt: str,
    system_prompt: str,
    image_urls: Optional[List[str]],
) -> Optional[str]:
    kwargs: Dict[str, Any] = {
        "chat_provider_id": provider_id,
        "prompt": prompt,
        "system_prompt": system_prompt,
    }
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


async def _resolve_provider_ids(
    context: Any,
    umo: str,
    configured_provider_id: str,
    fallback_provider_ids: Sequence[str],
) -> List[str]:
    provider_ids: List[str] = []
    if configured_provider_id:
        provider_ids.append(configured_provider_id)
    else:
        try:
            current = await context.get_current_chat_provider_id(umo=umo)
            if current:
                provider_ids.append(current)
        except Exception as exc:
            logger.warning(f"qiniu-image: 获取当前聊天模型失败（{type(exc).__name__}）")

    for provider_id in fallback_provider_ids:
        provider_id = str(provider_id or "").strip()
        if provider_id and provider_id not in provider_ids:
            provider_ids.append(provider_id)
    return provider_ids


async def _try_providers(
    context: Any,
    provider_ids: Sequence[str],
    *,
    prompt: str,
    system_prompt: str,
    image_url: Optional[str],
    timeout: int,
    attempts_per_provider: int,
    plausible: Callable[[str], bool],
    purpose: str,
) -> Optional[Tuple[str, str, bool]]:
    """依次重试主 Provider 和备用 Provider。"""
    attempts = max(1, attempts_per_provider)
    for provider_id in provider_ids:
        for attempt in range(1, attempts + 1):
            image_variants: List[Optional[List[str]]] = (
                [[image_url], None] if image_url else [None]
            )
            for image_urls in image_variants:
                try:
                    candidate = await asyncio.wait_for(
                        _call_llm(
                            context,
                            provider_id,
                            prompt,
                            system_prompt,
                            image_urls,
                        ),
                        timeout=max(1, timeout),
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"qiniu-image: {purpose}超时｜provider={provider_id} "
                        f"attempt={attempt}/{attempts}｜视觉={'是' if image_urls else '否'}"
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        f"qiniu-image: {purpose}失败｜provider={provider_id} "
                        f"attempt={attempt}/{attempts}｜视觉={'是' if image_urls else '否'}"
                        f"｜{type(exc).__name__}: {exc}"
                    )
                    continue

                text = (candidate or "").strip().strip('"').strip("“”").strip()
                if plausible(text):
                    return text, provider_id, bool(image_urls)
                logger.warning(
                    f"qiniu-image: {purpose}结果不可用｜provider={provider_id} "
                    f"attempt={attempt}/{attempts}｜结果={text[:80]!r}"
                )
    return None


async def rewrite(
    context: Any,
    umo: str,
    user_prompt: str,
    *,
    has_image: bool,
    timeout: int,
    system_prompt: str,
    image_url: Optional[str] = None,
    style_mode: str = "disabled",
    style_strength: str = "normal",
    quality_guidance: str = QUALITY_GUIDANCE,
    provider_id: str = "",
    fallback_provider_ids: Sequence[str] = (),
    attempts_per_provider: int = 2,
) -> Optional[str]:
    """返回改写后的提示词；所有 Provider 均失败时返回 None。"""
    if not user_prompt or not system_prompt:
        return None

    provider_ids = await _resolve_provider_ids(
        context,
        umo,
        provider_id,
        fallback_provider_ids,
    )
    if not provider_ids:
        logger.warning("qiniu-image: 没有可用的提示词改写 Provider")
        return None

    style_guidance = build_style_guidance(
        user_prompt,
        mode=style_mode,
        strength=style_strength,
        has_image=has_image,
    )
    instruction_parts = [system_prompt, _OPTIMIZER_BOUNDARY_GUIDANCE]
    if quality_guidance:
        instruction_parts.append(quality_guidance)
    if style_guidance:
        instruction_parts.append(style_guidance)
    effective_system_prompt = "\n\n".join(part for part in instruction_parts if part)

    task = f"主聊天模型完成的{'编辑' if has_image else '绘图'}方案：{user_prompt}"

    result = await _try_providers(
        context,
        provider_ids,
        prompt=task,
        system_prompt=effective_system_prompt,
        image_url=image_url,
        timeout=timeout,
        attempts_per_provider=attempts_per_provider,
        plausible=lambda candidate: _plausible(
            clean_style_metadata(candidate)[0],
            user_prompt,
            has_image,
        ),
        purpose="提示词改写",
    )
    if not result:
        logger.error(
            "qiniu-image: 所有提示词改写 Provider 均失败，不向图片模型发送未经优化的提示词"
        )
        return None
    raw_text, used_provider_id, used_vision = result

    mentioned_presets = find_explicit_presets(raw_text)
    text, marked_presets = clean_style_metadata(raw_text)
    explicit_presets = find_explicit_presets(user_prompt)
    selected = marked_presets or explicit_presets or mentioned_presets
    style_name = "、".join(preset.name for preset in selected) or "未识别"
    logger.info(
        f"qiniu-image: 提示词改写｜provider={used_provider_id} "
        f"历史=由主聊天模型处理 视觉={'是' if used_vision else '否'}"
        f"｜内置风格={style_name}"
        f"｜原文={user_prompt[:60]!r}｜改写={text[:120]!r}"
    )
    return text


async def rewrite_for_safety(
    context: Any,
    umo: str,
    prompt: str,
    *,
    has_image: bool,
    timeout: int,
    provider_id: str = "",
    fallback_provider_ids: Sequence[str] = (),
    attempts_per_provider: int = 2,
    safety_attempt: int = 1,
    safety_attempts_total: int = 5,
    quality_guidance: str = QUALITY_GUIDANCE,
) -> Optional[str]:
    """审核拒绝后生成合规替代提示词；所有 Provider 均失败时返回 None。"""
    provider_ids = await _resolve_provider_ids(
        context,
        umo,
        provider_id,
        fallback_provider_ids,
    )
    if not provider_ids:
        return None

    total = max(1, safety_attempts_total)
    current = min(max(1, safety_attempt), total)
    stage_index = min(
        len(_SAFETY_REFRAME_STAGES) - 1,
        ((current * len(_SAFETY_REFRAME_STAGES) + total - 1) // total) - 1,
    )
    stage_rule = _SAFETY_REFRAME_STAGES[stage_index]
    system_prompt = (
        "你是图像提示词安全转译器。将被图像平台拒绝的提示词改写为可安全生成的替代版本。\n"
        + SAFE_REFRAME_GUIDANCE
        + f"\n这是第 {current}/{total} 次安全调整，采用递进安全策略：{stage_rule}"
        + (f"\n{quality_guidance}" if quality_guidance else "")
        + "\n保持原提示词中仍然安全的画风、角色身份、色彩与构图；只输出一段最终提示词。"
    )
    task = (
        f"被审核拒绝的提示词：{prompt}\n"
        "请给出尽量接近原意、但明确非色情、衣着完整且适合全年龄展示的版本。"
    )
    result = await _try_providers(
        context,
        provider_ids,
        prompt=task,
        system_prompt=system_prompt,
        image_url=None,
        timeout=timeout,
        attempts_per_provider=attempts_per_provider,
        plausible=lambda candidate: (
            _plausible(
                clean_style_metadata(candidate)[0],
                prompt,
                has_image,
                allow_shorter=True,
            )
            and clean_style_metadata(candidate)[0].casefold()
            != prompt.strip().casefold()
        ),
        purpose="安全转译",
    )
    if not result:
        return None
    raw_text, used_provider_id, _ = result
    text, _ = clean_style_metadata(raw_text)

    logger.info(
        f"qiniu-image: 审核拒绝后已生成安全替代提示词｜provider={used_provider_id} "
        f"安全级别={stage_index + 1}/{len(_SAFETY_REFRAME_STAGES)} "
        f"attempt={current}/{total}｜改写={text[:120]!r}"
    )
    return text
