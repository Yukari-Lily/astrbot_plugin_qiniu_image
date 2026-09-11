"""角色外观记录：定键 schema、覆盖度校验、确定性渲染。

本模块是"参考图 → 外观文字 → 最终提示词"这条链路的锁。它只做纯计算，
不发起任何网络或模型调用，因此可以完整离线单测。

设计要点：
- 外观用**定键对象**而不是自由字符串列表，缺失字段可以直接变成校验失败，
  而不是像 `features: [str]` 那样只要非空就能蒙混过关。
- 渲染顺序按**身份承载能力**排序。提示词越长注意力越被稀释，顺序保证
  先被侵蚀的是最不要紧的属性。
- 渲染函数是确定性的：同样的记录永远产出同样的文本，因此"外观块是最终
  提示词的字面子串"是一个可测的不变量。
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _str(max_length: int, description: str = "") -> dict:
    return {"type": "string", "maxLength": max_length, "description": description}


def _list(max_items: int, max_length: int, description: str = "") -> dict:
    return {
        "type": "array",
        "items": {"type": "string", "maxLength": max_length},
        "maxItems": max_items,
        "description": description,
    }


APPEARANCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "silhouette": _str(120, "整体剪影：体型、身高比例与发量轮廓的印象"),
        "hair": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "color": _str(40, "发色"),
                "length": _str(20, "发长"),
                "style": _str(60, "发型，如双马尾、单马尾、姬发式、波波头、编发"),
                "front": _str(80, "前发结构：刘海形状、鬓发、呆毛"),
            },
        },
        "eyes": {
            "type": "object", "additionalProperties": False,
            "properties": {"color": _str(40, "瞳色"), "shape": _str(60, "眼型与瞳孔样式")},
        },
        "skin": _str(40, "肤色"),
        "build": _str(60, "身材比例"),
        "outfit": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "pieces": _list(6, 40, "自上而下的服装构成"),
                "cut": _str(80, "服装剪裁与版型"),
                "trim": _str(60, "滚边、袖口、腰带等配色与装饰"),
            },
        },
        "headwear": _str(80, "头饰、帽子等头部穿戴；没有则留空"),
        "accessory": _str(80, "配饰；左右不对称时必须写明哪一侧"),
        "palette": _list(5, 30, "整体配色词"),
        "marks": _str(80, "徽记、纹样、标志性印记"),
        "asymmetry": _str(80, "不对称特征及其所在侧，如左眼下方泪痣"),
        "signature": _str(80, "独一无二、最能认出该角色的标志物"),
        "version": _str(60, "该记录描述的具体官方形象版本"),
    },
}

#: 必须非空才算抽取成功的字段。缺一个就说明视觉模型没看全，而不是"这个角色没有"。
REQUIRED_PATHS: Tuple[str, ...] = (
    "silhouette",
    "hair.color",
    "hair.length",
    "hair.style",
    "hair.front",
    "eyes.color",
    "eyes.shape",
    "outfit.pieces",
    "outfit.cut",
    "skin",
    "build",
)

#: 渲染顺序 = 身份承载能力由强到弱。先被提示词长度稀释侵蚀的应该是末尾。
RENDER_ORDER: Tuple[Tuple[str, str], ...] = (
    ("silhouette", "剪影"),
    ("hair.style", "发型"),
    ("hair.length", "发长"),
    ("hair.color", "发色"),
    ("hair.front", "前发"),
    ("eyes.color", "瞳色"),
    ("eyes.shape", "眼型"),
    ("signature", "标志物"),
    ("asymmetry", "不对称"),
    ("headwear", "头饰"),
    ("accessory", "配饰"),
    ("outfit.pieces", "服装"),
    ("outfit.cut", "剪裁"),
    ("outfit.trim", "滚边"),
    ("palette", "配色"),
    ("skin", "肤色"),
    ("build", "体型"),
    ("marks", "印记"),
)

#: 这些字段的合法取值可以只有一个字（发色"黑"、发长"长"、瞳色"红"、肤色"白"、
#: 配色"金"）。别的一字答案仍然算偷懒——"高"不是剪影，"黑"不是发型。
SHORT_VALUE_PATHS = frozenset({"hair.color", "hair.length", "eyes.color", "skin", "palette"})

#: 身份内核：安全回退时不重新注入服装类字段，否则会把安全链路合法软化掉的
#: 暴露服装又原样塞回去，导致反复被拒。
CORE_PATHS = frozenset({
    "silhouette", "hair.style", "hair.length", "hair.color", "hair.front",
    "eyes.color", "eyes.shape", "signature", "asymmetry", "headwear",
    "accessory", "skin", "build", "marks",
})

FULL = "full"
IDENTITY_CORE = "core"

_PLACEHOLDER = re.compile(
    r"^(?:unknown|n/?a|none|null|not\s+(?:visible|applicable|available|clear|sure)"
    r"|不清楚|看不清|看不出来|未知|无法判断|无法识别|无法确定|未提及|不详|待定|待补充|待确认"
    r"|不确定|疑似|可能|也许|大概|无|没有|无资料|无相关信息|略|-+|—+|\.+|/+|\?+|？+)$",
    re.IGNORECASE,
)

#: 判断"模板灌水"：同一个值铺满多个字段，说明模型没真的看图。
_TEMPLATE_REPEAT_LIMIT = 4


def _get(record: Dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _values(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    """按渲染顺序取出可用的字段，返回 (标签, 值) 列表。

    有瑕疵的单个值在这里被丢掉——可选字段上写了个占位词，只该少渲染这一项，
    不该让整份记录失效。
    """
    rows: List[Tuple[str, str]] = []
    for path, label in RENDER_ORDER:
        raw = _get(record, path)
        if isinstance(raw, list):
            items = [text for text in (_clean(item) for item in raw)
                     if text and _value_problem(path, text) is None]
            text = "、".join(items)
        else:
            text = _clean(raw)
            if text and _value_problem(path, text) is not None:
                text = ""
        if text:
            rows.append((label, text))
    return rows


def _clean_by_schema(value: Any, schema: Dict[str, Any]) -> Any:
    """按 schema 递归清理。空容器一律返回 None，交由调用方丢弃。"""
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return None
        inner = {key: cleaned for key, sub in schema.get("properties", {}).items()
                 if (cleaned := _clean_by_schema(value.get(key), sub)) not in (None, "", [], {})}
        return inner or None
    if kind == "array":
        if not isinstance(value, list):
            return None
        items = [text for text in (_clean(item) for item in value) if text]
        return items or None
    return _clean(value) or None


def normalize(record: Any) -> Dict[str, Any]:
    """清理外观记录：去空白、丢空容器、剥掉 schema 之外的键。

    不抛异常——校验交给 `coverage_problems`，这里只负责把记录整理成
    渲染和缓存都能直接消费的形状。按 schema 递归，所以嵌套的数组
    （`outfit.pieces`、`palette`）不会被当成字符串抹掉。
    """
    if not isinstance(record, dict):
        return {}
    cleaned = _clean_by_schema(record, APPEARANCE_SCHEMA)
    return cleaned if isinstance(cleaned, dict) else {}


def _value_problem(path: str, text: str) -> Optional[str]:
    """单个取值的瑕疵；None 表示这个值可以直接用。"""
    if _PLACEHOLDER.match(text):
        return f"{path} 是占位内容：{text[:20]}"
    # 短值字段的合法取值本来就可以只有一个字（发色"黑"、发长"长"、肤色"白"、
    # 配色"金"），拿长度卡它们会把正常记录判成不合格。其余字段是描述性的，
    # 一个字说明模型没看图。ASCII 单字符（"a"、"#"）在任何字段都是噪声。
    if len(text) < 2 and (text.isascii() or path not in SHORT_VALUE_PATHS):
        return f"{path} 过短：{text}"
    if "?" in text or "？" in text:
        return f"{path} 含有未确定的表述：{text[:20]}"
    return None


def _problems(record: Any) -> List[Tuple[bool, str]]:
    """返回 (是否致命, 说明)。

    分档是因为两种问题的正确处置不同：**致命** = 模型没真的看这张图
    （必需字段缺失、还是偷懒的值、或把同一个值铺满多个字段），整份记录不可用；
    **不致命** = 可选字段上写了个占位词或给得太短，丢掉那一项就是了，不该让
    整个角色退回纯文字出图——那恰恰是这套记录要修的毛病。
    """
    if not isinstance(record, dict):
        return [(True, "外观记录缺失")]

    problems: List[Tuple[bool, str]] = []
    seen: Dict[str, int] = {}
    for path, _label in RENDER_ORDER:
        required = path in REQUIRED_PATHS
        raw = _get(record, path)
        values: Iterable[Any] = raw if isinstance(raw, list) else [raw]
        found = False
        for value in values:
            text = _clean(value)
            if not text:
                continue
            found = True
            issue = _value_problem(path, text)
            if issue:
                problems.append((required, issue))
                continue
            seen[text] = seen.get(text, 0) + 1
        if required and not found:
            problems.append((True, f"{path} 缺失"))

    # 同一个值铺满多个字段 = 没看图，与它出现在必需还是可选字段无关。
    for text, count in seen.items():
        if count >= _TEMPLATE_REPEAT_LIMIT:
            problems.append((True, f"同一描述重复出现在 {count} 个字段：{text[:20]}"))
    return problems


def fatal_problems(record: Any) -> List[str]:
    """致命问题：模型没真的看图。非空即不可用于出图。"""
    return [message for fatal, message in _problems(record) if fatal]


def coverage_problems(record: Any) -> List[str]:
    """全部问题，含只影响单个可选字段的瑕疵；供日志与诊断。

    这是把"模型偷懒"变成**确定性失败**的地方：以前 `features` 只要非空
    就能过校验，模型答一句"银发、蓝眼"照样出图。
    """
    return [message for _fatal, message in _problems(record)]


def is_usable(record: Any) -> bool:
    """记录是否达到可出图的最低标准。可选字段的瑕疵不在此列。"""
    return not fatal_problems(record)


def render(record: Any, level: str = FULL) -> str:
    """把外观记录渲染成逐字保留的标签段落。

    返回值是**确定性**的：同样的记录永远得到同样的字符串，因此调用方
    可以断言它是最终提示词的字面子串。空字段一律省略，不填空泛默认值。
    """
    if not isinstance(record, dict):
        return ""
    rows = [(label, text) for label, text in _values(record)
            if level == FULL or _path_of(label) in CORE_PATHS]
    if not rows:
        return ""
    return "；".join(f"{label}：{text}" for label, text in rows)


_LABEL_TO_PATH = {label: path for path, label in RENDER_ORDER}


def _path_of(label: str) -> str:
    return _LABEL_TO_PATH.get(label, "")


def render_block(record: Any, level: str = FULL) -> str:
    """带引导语的完整外观段，供最终提示词使用。

    有致命问题的记录一律渲染为空——"绝不写出未经核对的外观锚点"这条不变量
    必须落在这里，而不是靠每一个调用方自觉检查。可选字段的瑕疵只是不渲染
    那一项，见 `_values`。
    """
    if fatal_problems(record):
        return ""
    body = render(record, level)
    if not body:
        return ""
    return f"外观锚点（稳定特征，与文字要求冲突时以此为准）：{body}"
