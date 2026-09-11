"""绘图契约：人物名单 schema、校验、以及最终提示词的确定性拼装。

这里只保留"画一张新图"需要的东西。编辑继承（base_generation_id）、图片用途
绑定（image_roles）、身份状态枚举都在重构中删除了，因为插件改为**按名字**
自动从外观缓存取记录，模型不需要传递任何句柄。
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from . import appearance


def _string(max_length: int, description: str = "") -> dict:
    return {"type": "string", "maxLength": max_length, "description": description}


CHARACTER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name"],
    "properties": {
        "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$",
               "description": "人物标识，多人时用于逐人对应；可省略"},
        "name": _string(120, "准确人物名称"),
        "work": _string(120, "所属作品或身份"),
        "version": _string(80, "具体形象版本"),
        "position": _string(120, "从观看者视角描述位置；多人时必填"),
    },
}

DRAW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompt"],
    "properties": {
        "prompt": _string(8000, "结合人设与完整会话形成的绘图方案：主体、动作、表情、场景、构图、光照、画风、文字"),
        "characters": {
            "type": "array",
            "items": CHARACTER_SCHEMA,
            "maxItems": 20,
            "description": "画面中的人物名单。不确定角色外观时先用 prepare_character_reference 取得准确名称，再填这里",
        },
    },
}


def validate(value: Any, schema: dict, path: str = "draw") -> None:
    """校验本文件用到的那一小撮 schema 子集，不引入额外依赖。"""
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


def normalize_characters(raw: Any) -> List[Dict[str, str]]:
    """整理人物名单：补 id、查重、多人必须逐人给位置。"""
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("人物名单必须是列表")
    characters: List[Dict[str, str]] = []
    seen: set = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError("人物条目必须是对象")
        validate(item, CHARACTER_SCHEMA, f"characters[{index}]")
        char = {key: str(item.get(key, "") or "").strip()
                for key in ("id", "name", "work", "version", "position")}
        if not char["name"]:
            raise ValueError("人物名称不能为空")
        if not char["id"]:
            char["id"] = f"p{index + 1}"
        if char["id"] in seen:
            raise ValueError("人物 id 重复")
        seen.add(char["id"])
        characters.append(char)
    if len(characters) > 1:
        for char in characters:
            if not char["position"]:
                raise ValueError("多人图必须逐人指定位置")
    return characters


def parse_scene(text: str, characters: Sequence[Dict[str, str]]) -> Optional[dict]:
    """解析优化模型的输出：``{scene, characters:[{id, description}]}``。

    人物顺序与数量必须严格对应，否则返回 None 让调用方重试——错位会把
    甲的动作安到乙身上。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("scene"), str) or not value["scene"].strip():
        return None
    rows = value.get("characters")
    if not characters:
        return {"scene": value["scene"]}
    if not isinstance(rows, list) or len(rows) != len(characters):
        return None
    if [row.get("id") for row in rows if isinstance(row, dict)] != [c["id"] for c in characters]:
        return None
    if any(not isinstance(row.get("description"), str) or not row["description"].strip() for row in rows):
        return None
    return value


def render_prompt(
    value: dict,
    characters: Sequence[Dict[str, str]],
    appearances: Dict[str, Any],
    *,
    level: str = appearance.FULL,
) -> str:
    """确定性地拼出最终提示词——**外观块由这里逐字写入，不经过任何模型**。

    外观段紧贴在其所属人物的标题行下方，多人场景下属性不会串味。
    """
    parts = [value["scene"].strip()]
    if characters:
        parts.append(f"画面共有 {len(characters)} 位主体。以下人物标签仅用于对应，不画成文字。")
    # 优化模型没能给出逐人 JSON 时退化为"只有场景"，人物标题与外观块照旧写入。
    rows = value.get("characters") or [{} for _ in characters]
    for char, row in zip(characters, rows):
        identity = " / ".join(char.get(key, "") for key in ("name", "work", "version") if char.get(key))
        line = f"人物 {char['id']}（{identity}；{char.get('position', '')}）"
        description = str(row.get("description", "") or "").strip()
        if description:
            line += f"：{description}"
        block = appearance.render_block(appearances.get(char["id"]), level)
        parts.append(f"{line}\n{block}" if block else line)
    return "\n\n".join(parts)


def assemble(
    scene_value: dict,
    characters: Sequence[Dict[str, str]],
    appearances: Dict[str, Any],
    *,
    level: str = appearance.FULL,
) -> str:
    """`render_prompt` 的别名，语义上强调"锁存"这一步。"""
    return render_prompt(scene_value, characters, appearances, level=level)
