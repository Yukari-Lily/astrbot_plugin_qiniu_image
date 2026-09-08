"""七牛 AI 绘图 / 改图插件。"""

import asyncio
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .message_utils import resolve_input_image
from .prompt_rewriter import rewrite
from .qiniu_api import (
    QiniuApiError,
    QiniuAuthError,
    QiniuImageClient,
    QiniuImageDownloadError,
    QiniuInputError,
    QiniuNotConfiguredError,
    QiniuRateLimitError,
    QiniuRequestUncertainError,
    QiniuResponseError,
    QiniuSafetyError,
    QiniuTransientApiError,
)
from .style_presets import STYLE_MODES, STYLE_STRENGTHS

REWRITE_SCOPES = ("always", "image_only", "never")
TOOL_MODES = ("background", "sync")


@register(
    "astrbot_plugin_qiniu_image",
    "Yukari Lily",
    "七牛 AI 绘图 / 改图",
    "1.2.1",
    "https://github.com/YukariLily/astrbot_plugin_qiniu_image",
)
class QiniuImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.client = QiniuImageClient(config)

        raw_triggers = config.get("triggers") or []
        triggers = {item.strip() for item in raw_triggers if isinstance(item, str) and item.strip()}
        self.triggers: Tuple[str, ...] = tuple(sorted(triggers, key=len, reverse=True))

        scope = str(config.get("rewrite_scope", "always") or "always").strip().lower()
        if scope not in REWRITE_SCOPES:
            raise ValueError(f"qiniu_image 配置项 rewrite_scope 必须是 {'/'.join(REWRITE_SCOPES)} 之一")
        self.rewrite_scope = scope
        self.rewrite_vision = bool(config.get("rewrite_vision", True))
        self.rewrite_history_rounds = max(0, int(config.get("rewrite_history_rounds", 4) or 0))
        self.rewrite_timeout = int(config.get("rewrite_timeout", 20) or 20)
        self.rewrite_prompt_t2i = config.get("rewrite_system_prompt_t2i") or ""
        self.rewrite_prompt_i2i = config.get("rewrite_system_prompt_i2i") or ""

        style_mode = str(config.get("style_mode", "auto") or "auto").strip().lower()
        if style_mode not in STYLE_MODES:
            raise ValueError(f"qiniu_image 配置项 style_mode 必须是 {'/'.join(STYLE_MODES)} 之一")
        self.style_mode = style_mode
        style_strength = str(config.get("style_strength", "normal") or "normal").strip().lower()
        if style_strength not in STYLE_STRENGTHS:
            raise ValueError(
                f"qiniu_image 配置项 style_strength 必须是 {'/'.join(STYLE_STRENGTHS)} 之一"
            )
        self.style_strength = style_strength

        self.dedup_ttl = int(config.get("dedup_ttl", 20) or 20)
        self._recent_msg: Dict[str, float] = {}

        mode = str(config.get("tool_mode", "background") or "background").strip().lower()
        if mode not in TOOL_MODES:
            raise ValueError(f"qiniu_image 配置项 tool_mode 必须是 {'/'.join(TOOL_MODES)} 之一")
        self.tool_mode = mode
        self.background_notice = (config.get("background_notice") or "").strip()
        self._tasks: Set[asyncio.Task] = set()

        if not self.client.configured:
            logger.warning("qiniu-image: 未配置 api_key，插件已加载但无法出图")

    async def terminate(self):
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.client.close()

    def _match_trigger(self, event: AstrMessageEvent) -> Optional[str]:
        """命中返回触发词，未命中返回 None。未命中时绝不能有任何副作用。"""
        if not self.triggers:
            return None
        text = (event.message_str or "").strip()
        if not text:
            return None
        return next((trigger for trigger in self.triggers if text.startswith(trigger)), None)

    def _dedup_hit(self, event: AstrMessageEvent) -> bool:
        """协议端重连重投时防止同一条消息出两次图。"""
        mid = getattr(getattr(event, "message_obj", None), "message_id", None)
        if mid is None:
            return False
        mid = str(mid)

        now = time.monotonic()
        for key, timestamp in list(self._recent_msg.items()):
            if now - timestamp > self.dedup_ttl:
                self._recent_msg.pop(key, None)

        if mid in self._recent_msg:
            return True
        self._recent_msg[mid] = now
        return False

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message(self, event: AstrMessageEvent):
        trigger = self._match_trigger(event)
        if trigger is None:
            return

        event.stop_event()

        if self._dedup_hit(event):
            return
        if not self.client.configured:
            yield event.plain_result("生成失败喵（未配置 api_key）")
            return

        user_prompt = (event.message_str or "").strip()[len(trigger):].strip()
        image_b64, error_text = await self._draw(event, user_prompt)
        if image_b64:
            yield event.chain_result([Comp.Image.fromBase64(image_b64)])
        else:
            yield event.plain_result(error_text or "生成失败喵")

    @filter.llm_tool(name="draw_image")
    async def draw_image(self, event: AstrMessageEvent, prompt: str):
        """生成或修改图片。输入图会自动用于改图，完成后自动发送，请勿重复调用。

        Args:
            prompt(string): 具体的绘图或编辑描述。
        """
        if not self.client.configured:
            yield event.plain_result("生成失败喵（未配置 api_key）")
            return

        prompt = (prompt or "").strip()

        if self.tool_mode == "background":
            task = asyncio.create_task(self._draw_and_push(event, prompt))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            if self.background_notice:
                yield event.plain_result(self.background_notice)
            return

        image_b64, error_text = await self._draw(event, prompt)
        if image_b64:
            yield event.chain_result([Comp.Image.fromBase64(image_b64)])
        else:
            yield event.plain_result(error_text or "生成失败喵")

    async def _draw_and_push(self, event: AstrMessageEvent, prompt: str) -> None:
        """后台出图并推送。"""
        try:
            image_b64, error_text = await self._draw(event, prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                f"qiniu-image background task failed | {self._ctx(event)} "
                f"error_type={type(exc).__name__}\n" + "".join(traceback.format_tb(exc.__traceback__))
            )
            image_b64, error_text = None, "生成失败喵"

        chain = (
            MessageChain().base64_image(image_b64)
            if image_b64
            else MessageChain().message(error_text or "生成失败喵")
        )
        try:
            await self.context.send_message(event.unified_msg_origin, chain)
        except Exception as exc:
            logger.error(f"qiniu-image: 推送结果失败（{type(exc).__name__}: {exc}）")

    def _should_rewrite(self, has_image: bool) -> bool:
        if self.rewrite_scope == "never":
            return False
        if self.rewrite_scope == "image_only":
            return has_image
        return True

    def _vision_url(self, image_ref: Optional[str]) -> Optional[str]:
        """转换输入图片引用。"""
        if not image_ref or not self.rewrite_vision:
            return None
        try:
            return self.client.as_image_reference(image_ref)
        except QiniuInputError:
            return None

    async def _draw(
        self,
        event: AstrMessageEvent,
        user_prompt: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        image_ref = await resolve_input_image(self.context, event, self.client)

        prompt = user_prompt
        if user_prompt and self._should_rewrite(bool(image_ref)):
            prompt = await rewrite(
                self.context,
                event.unified_msg_origin,
                user_prompt,
                has_image=bool(image_ref),
                timeout=self.rewrite_timeout,
                system_prompt=self.rewrite_prompt_i2i if image_ref else self.rewrite_prompt_t2i,
                image_url=self._vision_url(image_ref),
                history_rounds=self.rewrite_history_rounds,
                style_mode=self.style_mode,
                style_strength=self.style_strength,
            )

        return await self._generate(event, prompt, image_ref)

    async def _generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        image_ref: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """返回图片或错误提示。"""
        try:
            if image_ref:
                images: List[str] = await self.client.image_to_image(image_ref, prompt)
            else:
                images = await self.client.text_to_image(prompt)
            if images and images[0]:
                return images[0], None
            logger.error(f"qiniu-image: 上游未返回图片 | model={self.client.model}")
            return None, "生成失败喵（上游没有返回图片）"

        except QiniuNotConfiguredError:
            return None, "生成失败喵（未配置 api_key）"
        except QiniuSafetyError as exc:
            logger.warning(
                f"qiniu-image rejected by safety | {self._ctx(event)} "
                f"model={self.client.model} status={exc.status} code={exc.code}"
            )
            return None, "生成失败喵，太色了喵"
        except QiniuAuthError as exc:
            logger.error(
                f"qiniu-image authentication failed | status={exc.status} "
                f"code={exc.code} model={self.client.model}"
            )
            return None, "生成失败喵（API Key 无效或没有模型权限）"
        except QiniuRateLimitError as exc:
            logger.warning(
                f"qiniu-image rate limited | status={exc.status} code={exc.code} model={self.client.model}"
            )
            return None, "生成服务繁忙喵，请稍后再试"
        except QiniuTransientApiError as exc:
            logger.warning(
                f"qiniu-image upstream unavailable | status={exc.status} "
                f"code={exc.code} model={self.client.model}"
            )
            return None, "生成服务暂时不可用喵，请稍后再试"
        except QiniuApiError as exc:
            logger.warning(
                f"qiniu-image request rejected | status={exc.status} code={exc.code} "
                f"detail={exc.detail} model={self.client.model}"
            )
            if exc.status == 400:
                return None, "生成失败喵（请求参数或模型配置无效）"
            return None, "生成失败喵（上游拒绝了请求）"
        except QiniuRequestUncertainError as exc:
            logger.error(f"qiniu-image request failed after retries | model={self.client.model} reason={exc}")
            return None, "生成失败喵（多次重试仍未成功）"
        except QiniuResponseError as exc:
            logger.error(f"qiniu-image invalid response after retries | model={self.client.model} reason={exc}")
            return None, "生成失败喵（上游响应异常，多次重试仍未成功）"
        except QiniuImageDownloadError as exc:
            logger.error(f"qiniu-image download failed | model={self.client.model} reason={exc}")
            return None, "生成失败喵（图片下载失败）"
        except QiniuInputError as exc:
            logger.warning(f"qiniu-image invalid input | model={self.client.model} reason={exc}")
            return None, "生成失败喵（输入图片无效、格式不支持或文件过大）"
        except Exception as exc:
            logger.error(
                f"qiniu-image failed | {self._ctx(event)} model={self.client.model} "
                f"error_type={type(exc).__name__}\n" + "".join(traceback.format_tb(exc.__traceback__))
            )
            return None, "生成失败喵"

    @staticmethod
    def _ctx(event: AstrMessageEvent) -> str:
        mid = getattr(getattr(event, "message_obj", None), "message_id", None)
        return f"mid={mid} gid={event.get_group_id()} uid={event.get_sender_id()}"
