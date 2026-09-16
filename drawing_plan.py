"""绘图请求、优化方案和成功记录；均为不可变快照。"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ImagePlan:
    prompt: str
    style: str = ""
    style_exception: str = ""
    # 聊天优化结果已融合风格与质量；关键词原文仍需追加固定质量段。
    integrated: bool = False
    people_count: Optional[int] = None


@dataclass(frozen=True)
class ImageRecord:
    plan: ImagePlan
    has_image: bool
    keep_layout: bool
    updated_at: float
    submitted_prompt: str = ""


@dataclass(frozen=True)
class DrawingRequest:
    prompt: str
    user_message: str = ""
    subject_info: str = ""
    optimize: bool = True
    keep_layout: bool = True
    previous: Optional[ImageRecord] = None
