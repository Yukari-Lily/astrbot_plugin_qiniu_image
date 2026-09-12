"""Parse model JSON, accepting a single Markdown JSON fence without weakening validation."""

import json
import re


def parse_model_json(text):
    if not isinstance(text, str):
        raise ValueError("模型没有返回 JSON 文本")
    text = text.strip().lstrip("\ufeff").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)
