"""Lossless function-call prompting: the model selects existing style fragments only."""

import json
import re
from typing import Optional

from .prompt_rewriter import _resolve_provider_ids, _try_providers, LLM_TIMEOUT_SECONDS
from .model_json import parse_model_json
from .style_presets import QUALITY_GUIDANCE, STYLE_PRESETS, find_explicit_presets


def _parts(text):
    return [part.strip() for part in re.split(r"(?<=[。；;])|\n", text) if part.strip()]


QUALITY_PARTS = _parts(QUALITY_GUIDANCE.split("\n", 1)[1])
STYLE_PARTS = {preset.id: _parts(preset.prompt) for preset in STYLE_PRESETS}
STYLE_HEADER = "画面风格（以不改变上述主体特征、动作、构图和明确要求为前提）："
QUALITY_HEADER = "全局兜底（仅在不与上述内容及画风冲突时应用）："

INTEGRATION_RULES = """
你只负责原样整合绘图方案与已有风格，不负责创作、搜索或校正主体特征。
完整方案由绘图规划模型提供、主聊天模型审核，插件会逐字保留，你不能改写或增删它。
请从提供的风格片段、全局质量片段中选择与方案相容的片段编号（从 0 开始）。
优先级：用户明确要求及主模型确认的特征/动作/构图 > 内置风格 > 全局兜底。
省略任何冲突的低优先级片段；禁止用风格里的示例人物、动作、场景替换方案，
禁止添加未要求的物件、文字、背景。主体名称原样保留，不改名、不补外观。
自动模式下，未明确指定其他具体画风时，必须选一个相容的内置风格，优先高偏好项。
anime style、高清等通用词不算指定画风；用户点名的内置风格优先，但识别否定语义。
选择内置风格时至少保留一个相容片段；只有明确指定其他画风、保留原图画风、
没有相容片段或配置禁用时才可选 none，并填写 none_reason。
none_reason 必须为 external_style（用户指定外部画风）、preserve_image（局部改图）、
incompatible（无相容片段）、disabled（配置不启用）或 existing_style（沿用已有完整风格）。
如果方案已包含上次完整的“画面风格”段，应选 none 保留它，不重复追加；
如果已包含完整“全局兜底”段，quality_parts 必须为空，沿用原文。
有用户图片时，以图中主体外观为准，不根据文字知识覆盖图片。
用户仅说参考图片、参考人物或参考构图，属于 reference，应结合内置风格；
局部修改、保留原画风属于 edit，不自动叠加风格，全局规则只作用于修改区域。
image_mode 已明确为 edit/reference 时必须遵守；auto 时根据方案判断。
全局兜底只选择相容片段，至少一个；已经确定具体媒介时不要再加入其他媒介。
subtle 只选少量核心片段，normal 选相容核心片段，strong 尽量充分但不突破方案。
只输出 JSON：
{"image_mode":"none|edit|reference", "style_id":"风格 id 或 none",
 "style_parts":[0], "quality_parts":[0], "none_reason":"选择 none 时的原因代码，否则空字符串"}
不要输出新的提示词、特征、解释或自创片段。
""".strip()


def _indices(value, parts, *, required):
    return (
        isinstance(value, list)
        and (bool(value) or not required)
        and all(type(i) is int and 0 <= i < len(parts) for i in value)
        and len(value) == len(set(value))
    )


def _assemble(selection, prompt, *, has_image, image_mode, allowed):
    if not isinstance(selection, dict):
        return None
    mode = selection.get("image_mode")
    if mode not in (("edit", "reference") if has_image else ("none",)):
        return None
    if has_image and image_mode != "auto" and mode != image_mode:
        return None
    style_id = selection.get("style_id")
    if not isinstance(style_id, str) or style_id not in (*allowed, "none"):
        return None
    if style_id == "none":
        reasons = {"external_style", "incompatible"}
        if has_image and mode == "edit":
            reasons.add("preserve_image")
        if not allowed:
            reasons.add("disabled")
        if STYLE_HEADER in prompt:
            reasons.add("existing_style")
        if not isinstance(selection.get("none_reason"), str) or selection["none_reason"] not in reasons:
            return None
    elif has_image and mode == "edit" and not find_explicit_presets(prompt):
        return None
    style_parts = STYLE_PARTS.get(style_id, [])
    if not _indices(selection.get("style_parts"), style_parts, required=style_id != "none"):
        return None
    has_style = STYLE_HEADER in prompt
    has_quality = QUALITY_HEADER in prompt
    if has_style and style_id != "none":
        return None
    if not _indices(selection.get("quality_parts"), QUALITY_PARTS, required=not has_quality):
        return None
    if has_quality and selection["quality_parts"]:
        return None
    result = [prompt]
    if has_image and "以用户图片为主体外观依据。" not in prompt:
        result.append("以用户图片为主体外观依据。" + (
            "仅执行方案要求的修改，其余部分保持原图；以下风格与质量要求只作用于修改区域。"
            if mode == "edit" else "将用户图片作为参考，按方案与以下相容画风重新绘制。"
        ))
    if style_id != "none":
        result.append(STYLE_HEADER + "\n" +
                      "\n".join(style_parts[i] for i in sorted(selection["style_parts"])))
    if not has_quality:
        result.append(QUALITY_HEADER + "\n" +
                      "\n".join(QUALITY_PARTS[i] for i in sorted(selection["quality_parts"])))
    text = "\n\n".join(result)
    return text if len(text) <= 32000 else None


async def integrate(context, umo, prompt, *, has_image, image_mode="auto",
                    style_mode="auto", style_strength="normal", provider_id="",
                    fallback_provider_ids=()) -> Optional[str]:
    """Return the exact source plan plus compatible, existing style text."""
    if not prompt or len(prompt) > 32000 or image_mode not in ("auto", "edit", "reference"):
        return None
    providers = await _resolve_provider_ids(context, umo, provider_id, fallback_provider_ids)
    explicit = find_explicit_presets(prompt)
    presets = (STYLE_PRESETS if style_mode == "auto" else
               explicit if style_mode == "explicit_only" else ())
    catalog = {p.id: {"name": p.name, "suitable_for": p.suitable_for,
                      "avoid_when": p.avoid_when, "preference": p.auto_preference,
                      "parts": STYLE_PARTS[p.id]} for p in presets}
    # In auto mode an explicit request is still resolved semantically, including negation.
    task = json.dumps({"plan": prompt, "has_image": has_image, "image_mode": image_mode,
                       "style_mode": style_mode, "style_strength": style_strength,
                       "styles": catalog, "quality_parts": QUALITY_PARTS}, ensure_ascii=False)

    def assemble(raw):
        try:
            selection = parse_model_json(raw)
        except (ValueError, TypeError):
            return None
        return _assemble(selection, prompt, has_image=has_image,
                         image_mode=image_mode, allowed=catalog)

    result = await _try_providers(
        context, providers, prompt=task, system_prompt=INTEGRATION_RULES,
        timeout=LLM_TIMEOUT_SECONDS, attempts_per_provider=2,
        plausible=lambda raw: assemble(raw) is not None, purpose="提示词整合",
    )
    return assemble(result[0]) if result else None
