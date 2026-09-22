"""图片平台审核拒绝后改写执行稿，保留安全的视觉设计。"""

from typing import Any, Optional, Sequence

from astrbot.api import logger

from .model_call import call_with_fallback, is_prompt_text
from .style_presets import (
    QUALITY_SECTION,
    SAFE_REFRAME_GUIDANCE,
    SAFE_REFRAME_TIGHTENING,
    plan_body,
)

_SAFETY_REFRAME_STAGES = (
    "第一级（同义替换）：不改动画面内容，仅把明确露骨的词语替换为含蓄、克制的同义或近义表达；原服装、场景、姿势、构图与氛围全部保持不变，也不新增原文没有的内衣、打底、端庄或全年龄描述。优先用二次元插画语境措辞包装（如动画角色插画、角色立绘、轻小说封面、动画剧照、角色设计图、时尚杂志内页插画）。",
    "第二级（隐晦转述）：仍不改动画面内容，用模糊、含蓄、间接的语言整体重述提示词，避免直接点出敏感概念；把语义方向改写成二次元常见语境（角色插画、服装设计展示、青春日常、校园社团、都市街头插画等），画面保持原服装、场景与构图，也不新增原文没有的遮挡或端庄化描述。",
    "第三级（轻微画面收敛）：保持含蓄措辞，同时开始最小画面调整：避免私密场景、暴露服装与挑逗姿势叠加组合，至少替换掉其中一项；增加衣物覆盖，去除薄纱、透视、走光与身体敏感部位强调；仍尽量保留原服装类别、构图与氛围。",
    "第四级（中度收敛）：将可能被视为情趣或内衣的服装改为不透明的完整睡衣、家居服或时装，改用自然姿态和非色情的浪漫氛围，降低床铺、身体曲线和亲密暗示的视觉权重。",
    "第五级（最大安全）：转为适合全年龄展示的普通角色插画或时尚编辑肖像，穿完整日常服装，采用自然表情、中性动作和非私密场景，仅保留主体身份、核心配色、画风及安全的叙事元素。",
)
SAFETY_REWRITE_LEVELS = len(_SAFETY_REFRAME_STAGES)

# 任务描述与级别对齐：第一、二级只要求措辞替换，禁止先做画面收敛。
_STAGE_TASKS = (
    "只做同义替换：替换露骨用词为含蓄表达，画面内容、服装、场景、姿势与构图保持原样，"
    "不要新增原文没有的完整内搭、端庄或全年龄描述。",
    "只做隐晦转述：用含蓄语言重述同一画面，服装、场景、姿势与构图保持原样，"
    "不要新增原文没有的遮挡或端庄化描述。",
    "在含蓄措辞基础上做轻微画面收敛：优先弱化风险元素叠加，适当增加衣物覆盖，"
    "仍尽量保留原服装类别、构图与氛围。",
    "做中度画面收敛：改为不透明完整服装与非色情氛围，保留主体身份与核心视觉设计。",
    "做最大安全改写：全年龄完整日常服装与非私密场景，仅保留主体身份、核心配色与画风。",
)


def _parse_rewrite(text: str) -> Optional[str]:
    text = plan_body(text.strip().strip('"').strip("“”").strip())
    return text if is_prompt_text(text) else None


async def rewrite_for_safety(
    context: Any,
    umo: str,
    prompt: str,
    *,
    provider_id: str = "",
    fallback_provider_ids: Sequence[str] = (),
    safety_attempt: int = 1,
) -> Optional[str]:
    """审核拒绝后生成合规替代提示词；所有 Provider 均失败时返回 None。"""
    total = SAFETY_REWRITE_LEVELS
    current = min(max(1, safety_attempt), total)
    # 安全级别与重试次数一一对应，第 n 次就用第 n 级策略。
    stage_rule = _SAFETY_REFRAME_STAGES[current - 1]
    tightening = f"\n{SAFE_REFRAME_TIGHTENING}" if current >= 3 else ""
    system_prompt = (
        "你是图像提示词安全转译器。将被图像平台拒绝的提示词改写为可安全生成的替代版本。\n"
        + SAFE_REFRAME_GUIDANCE
        + tightening
        + f"\n这是第 {current}/{total} 次安全调整，采用递进安全策略：{stage_rule}"
        + "\n" + QUALITY_SECTION
        + "\n角色身份在任何一级都必须原样保留：不得替换、删减、泛化或改写角色名、作品名、外观设定与人数。"
        "\n在不与安全要求冲突的前提下，保持原文的动作、色彩与构图；只输出一段最终提示词。"
        "原稿已经融合的画风、媒介、光影和材质应保留，不重新选风格或转抄模板与质量规则。"
    )
    task = (
        f"被审核拒绝的提示词：{prompt}\n"
        f"请按当前级别要求改写：{_STAGE_TASKS[current - 1]}"
    )
    result = await call_with_fallback(
        context, umo, prompt=task, system_prompt=system_prompt,
        parse=_parse_rewrite, purpose="安全转译",
        provider_id=provider_id, fallback_provider_ids=fallback_provider_ids,
    )
    if not result:
        return None
    text, used_provider_id = result

    logger.info(
        f"qiniu-image: 审核拒绝后已生成安全替代提示词｜provider={used_provider_id} "
        f"安全级别={current}/{total}｜改写={text[:120]!r}"
    )
    return text
