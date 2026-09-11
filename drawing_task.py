"""Drawing contracts shared by tools, prompt compilation and generation records."""

import copy
import json
import re
from typing import Any


REFERENCE_FEATURE_GUIDANCE = (
    "参考图使用边界（严格）：人物参考图只用于核对身份和稳定、可见的外观特征，"
    "包括脸部、发型、发色、眼睛、肤色、发饰、服装结构、服装颜色和标志物。"
    "不得从参考图继承或猜测动作、姿势、手势、表情、镜头、视角、构图、布局、背景、场景、"
    "光照、色调、材质、文字、特效或画风。最终动作、镜头、构图、场景和画风只服从本次任务的文字方案；"
    "用户文字明确要求的换装、动作或画风优先于参考图。"
)


def _string(description: str = "") -> dict:
    return {"type": "string", "maxLength": 8000, "description": description}


def _strings(description: str = "") -> dict:
    return {"type": "array", "items": _string(), "maxItems": 30, "description": description}


CHARACTER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "name"],
    "properties": {
        "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$"},
        "name": _string("准确人物名称；编辑时沿用原人物 id"),
        "work": _string("作品或人物所属身份"),
        "version": _string("具体形象版本"),
        "position": _string("从观看者视角描述位置"),
        "features": _strings("有依据的标志性外观，不混入本轮换装要求"),
        "evidence": _string("分别说明身份指代与外观依据，可沿用用户确认、会话或目标作品已确认资料；陌生或冲突信息需搜索核实，不能把搜索排名当作依据"),
        "identity_status": {"type": "string", "enum": ["confirmed", "original", "uncertain"]},
        "reference_id": _string("prepare_character_reference 返回的标识；空字符串表示清除旧参考"),
    },
}
IMAGE_ROLE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["source", "role"],
    "properties": {
        "source": _string("input:1 等本条/引用图片序号，或 prepare_character_reference 返回的标识"),
        "role": {"type": "string", "enum": ["character", "style", "edit"],
                 "description": "character 只取人物稳定外观；style 仅为兼容保留，不读取画风；edit 为编辑原图"},
        "character_id": _string("人物参考必须绑定人物 id"),
    },
}
TASK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["operation"],
    "properties": {
        "operation": {"type": "string", "enum": ["create", "edit", "redraw"]},
        "base_generation_id": _string("继承指定作品；latest 仅限本会话当前用户；新画留空"),
        "user_requirements": _strings("用户本轮明确要求，优先级最高"),
        "changes": _strings("本轮改动；原作事实不覆盖用户明确的换装等要求"),
        "preserve": _strings("必须保留的内容"),
        "creative_choices": _strings("主聊天模型自行选择的创作细节，不能当成用户硬性要求"),
        "characters": {"type": "array", "items": CHARACTER_SCHEMA, "maxItems": 20,
                       "description": "新画提供完整人物；edit 只提交待更新条目，按 id 合并保留其余人物"},
        "character_count": {"type": "integer", "minimum": 0, "maximum": 20},
        "image_roles": {"type": "array", "items": IMAGE_ROLE_SCHEMA, "maxItems": 25},
    },
}
DRAW_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["prompt"],
    "properties": {"prompt": _string("结合人设与完整会话形成的绘图方案"), "task": TASK_SCHEMA},
}


def validate(value: Any, schema: dict, path: str = "task") -> None:
    """Validate the small schema subset used here without an extra dependency."""
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "integer": type(value) is int}
    if kind and not valid.get(kind, False):
        raise ValueError(f"{path} 类型无效，应为 {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} 取值无效")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path} 缺少 {key}")
        props = schema.get("properties", {})
        for key, item in value.items():
            if key not in props:
                raise ValueError(f"{path} 包含未知字段 {key}")
            validate(item, props[key], f"{path}.{key}")
    elif kind == "array":
        if len(value) > schema.get("maxItems", 100):
            raise ValueError(f"{path} 条目过多")
        for item in value:
            validate(item, schema["items"], path)
    elif kind == "string":
        if len(value) > schema.get("maxLength", 8000):
            raise ValueError(f"{path} 太长")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(f"{path} 格式无效")
    elif kind == "integer":
        if not schema.get("minimum", 0) <= value <= schema.get("maximum", 20):
            raise ValueError(f"{path} 超出范围")


def normalize_task(task: dict, base: dict | None = None) -> dict:
    validate(task, TASK_SCHEMA)
    result = copy.deepcopy(task)
    incoming = result.get("characters", [])
    ids = [c["id"] for c in incoming]
    if len(ids) != len(set(ids)):
        raise ValueError("人物 id 重复")
    if base and base.get("characters") and task["operation"] == "edit":
        chars = {c["id"]: copy.deepcopy(c) for c in base.get("characters", [])}
        for char in incoming:
            if char["id"] not in chars:
                raise ValueError("局部编辑不能新增人物；改变人物名单请使用 redraw")
            old = chars[char["id"]]
            if char["name"] != old["name"]:
                raise ValueError("局部编辑不能替换人物身份；请使用 redraw")
            old.update(char)
        result["characters"] = list(chars.values())
    elif base and "characters" not in task:
        result["characters"] = copy.deepcopy(base.get("characters", []))
    else:
        result.setdefault("characters", [])
    chars = result["characters"]
    if "character_count" in result and result["character_count"] != len(chars):
        raise ValueError("人物数量与人物列表不一致")
    result["character_count"] = len(chars)
    for char in chars:
        if not char["name"].strip():
            raise ValueError("人物名称不能为空")
        if char.get("identity_status") not in ("confirmed", "original"):
            raise ValueError("人物身份尚未确定，本次不生成")
        if not char.get("features") or any(not feature.strip() for feature in char["features"]) or (char["identity_status"] == "confirmed" and not char.get("evidence", "").strip()):
            raise ValueError("人物缺少可靠外观或身份依据，本次不生成")
        if len(chars) > 1 and not char.get("position", "").strip():
            raise ValueError("多人图必须逐人指定位置")
    role_sources, role_characters = set(), set()
    for role in result.get("image_roles", []):
        if not role["source"].strip() or role["source"] in role_sources:
            raise ValueError("图片来源为空或重复绑定")
        role_sources.add(role["source"])
        if role["role"] == "character" and role.get("character_id") not in {c["id"] for c in chars}:
            raise ValueError("人物参考未绑定有效人物 id")
        if role["role"] == "character":
            if role["character_id"] in role_characters:
                raise ValueError("每个人物只能绑定一张人物参考图")
            role_characters.add(role["character_id"])
        if role["role"] != "character" and role.get("character_id"):
            raise ValueError("非人物参考不能绑定人物 id")
        if task["operation"] != "edit" and role["role"] == "edit":
            raise ValueError("只有 edit 操作可以绑定编辑原图")
    return result


def task_context(task: dict, base: dict | None, bindings: list[dict]) -> str:
    payload = {"本次任务": task, "输入图片顺序及用途": bindings,
               "参考图使用约束": REFERENCE_FEATURE_GUIDANCE}
    if base:
        payload["目标作品"] = {"实际提示词": base["prompt"], "任务": base["task"]}
    return json.dumps(payload, ensure_ascii=False)


def parse_compilation(text: str, characters: list[dict]) -> dict | None:
    try:
        value = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        if not isinstance(value, dict) or not isinstance(value.get("scene"), str) or not value["scene"].strip():
            return None
        rows = value.get("characters")
        if not isinstance(rows, list) or len(rows) != len(characters):
            return None
        wanted = [c["id"] for c in characters]
        if [r.get("id") for r in rows if isinstance(r, dict)] != wanted:
            return None
        if any(not isinstance(r.get("description"), str) or not r["description"].strip() for r in rows):
            return None
        return value
    except (ValueError, TypeError, AttributeError):
        return None


def render_compilation(value: dict, characters: list[dict]) -> str:
    parts = [value["scene"].strip()]
    if characters:
        parts.append(f"画面共有 {len(characters)} 位主体。以下人物标签仅用于对应，不画成文字。")
    for char, row in zip(characters, value["characters"]):
        identity = " / ".join(str(char.get(k, "")) for k in ("name", "work", "version") if char.get(k))
        parts.append(f"人物 {char['id']}（{identity}；{char.get('position', '')}）：{row['description'].strip()}")
    return "\n\n".join(parts)


def looks_like_group(prompt: str) -> bool:
    return bool(re.search(r"(?:[2-9]|1\d)\s*(?:anime\s*)?(?:girls?|boys?|people|characters)|[二三四五六七八九十两][人位]|合照|同框|group portrait", prompt, re.I))
