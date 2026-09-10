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
from .prompt_rewriter import SAFETY_REWRITE_LEVELS, rewrite, rewrite_for_safety
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
from .style_presets import STYLE_MODES, STYLE_STRENGTHS, style_catalog_text

DEDUP_TTL_SECONDS = 20

@register(
    "astrbot_plugin_qiniu_image",
    "Yukari Lily",
    "七牛 AI 绘图 / 改图",
    "1.3.0",
    "https://github.com/Yukari-Lily/astrbot_plugin_qiniu_image",
)
class QiniuImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.client = QiniuImageClient(config)

        raw_triggers = config.get("triggers") or []
        triggers = {item.strip() for item in raw_triggers if isinstance(item, str) and item.strip()}
        self.triggers: Tuple[str, ...] = tuple(sorted(triggers, key=len, reverse=True))

        raw_rewrite_providers = config.get("rewrite_provider_ids") or []
        if not isinstance(raw_rewrite_providers, (list, tuple)):
            raw_rewrite_providers = []
        rewrite_provider_ids = tuple(dict.fromkeys(
            provider_id.strip()
            for provider_id in raw_rewrite_providers
            if isinstance(provider_id, str) and provider_id.strip()
        ))
        self.rewrite_provider_id = rewrite_provider_ids[0] if rewrite_provider_ids else ""
        self.rewrite_fallback_provider_ids = rewrite_provider_ids[1:]

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

        self._recent_msg: Dict[str, float] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._last_image_prompts: Dict[str, Dict[str, object]] = {}

        if not self.client.configured:
            logger.warning("qiniu-image: 未配置 api_key，插件已加载但无法出图")

    async def terminate(self):
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.client.close()

    def _match_trigger(self, event: AstrMessageEvent) -> Optional[str]:
        """命中返回触发词；未命中不产生副作用。"""
        if not self.triggers:
            return None
        text = (event.message_str or "").strip()
        if not text:
            return None
        return next((trigger for trigger in self.triggers if text.startswith(trigger)), None)

    def _dedup_hit(self, event: AstrMessageEvent) -> bool:
        """协议重投时避免同一消息重复出图。"""
        mid = getattr(getattr(event, "message_obj", None), "message_id", None)
        if mid is None:
            return False
        mid = str(mid)
        now = time.monotonic()
        for key, timestamp in list(self._recent_msg.items()):
            if now - timestamp > DEDUP_TTL_SECONDS:
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

        prompt = (event.message_str or "").strip()[len(trigger):].strip()
        image_b64, error_text = await self._draw(event, prompt)
        if image_b64:
            yield event.chain_result([Comp.Image.fromBase64(image_b64)])
        else:
            yield event.plain_result(error_text or "生成失败喵")

    @filter.llm_tool(name="draw_image")
    async def draw_image(self, event: AstrMessageEvent, prompt: str):
        """生成或修改图片。输入图会自动用于改图，完成后自动发送，请勿重复调用。

        调用前先结合你的系统人设、完整群聊上下文和用户当前意图形成一份完整绘图方案。
        例如用户让“你”画自拍时，要把你的人格、外观设定和自拍语境具体写入 prompt，
        不能只传“自拍”。用户把创作选择交给你时，可合理决定服装、动作、场景、构图、
        光照、配色和画风。除非你明确留给插件自动选择内置风格，否则不要把创作决定留给
        提示词优化模型；它只负责整理表达、校正事实和执行插件明确配置的规则。

        对你已知的常见作品、角色和昵称直接补全准确的官方作品名、角色名和关键外观。
        只有现有知识不足以可靠还原或无法唯一识别主体时才搜索；搜索与绘图必须串行，
        先阅读搜索结果，再调用本工具，禁止与搜索工具并行调用。

        若用户要求修改、延续或重画插件上一张作品，而当前上下文没有完整执行提示词，
        先调用 get_last_image_prompt 取得图片模型实际收到的版本，再基于它编写本次完整方案。
        用户本轮明确要求始终优先，不要把旧提示词中已被用户推翻的内容带回来。

        Args:
            prompt(string): 结合用户要求、Bot 人设、完整会话和必要考据写成的完整绘图或编辑方案。
        """
        if not self.client.configured:
            yield event.plain_result("生成失败喵（未配置 api_key）")
            return

        prompt = (prompt or "").strip()
        task = asyncio.create_task(self._draw_and_push(event, prompt))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return

    @filter.llm_tool(name="list_image_styles")
    async def list_image_styles(self, event: AstrMessageEvent):
        """查询本插件实际可用的内置绘图风格。当用户询问你会哪些画风、支持哪些风格、推荐什么画风或要求列出风格时，调用此工具获取最新目录。"""
        yield event.plain_result(
            "本插件的内置绘图风格如下。回答用户时使用中文名称，不要编造目录外的内置风格。\n"
            f"当前模式：{self.style_mode}；强度：{self.style_strength}。\n\n"
            f"{style_catalog_text(concise=True)}"
        )

    @filter.llm_tool(name="get_last_image_prompt")
    async def get_last_image_prompt(self, event: AstrMessageEvent):
        """取得当前会话中插件最近一次实际送给图片模型的提示词。用户要求修改、延续、重画或追问上一张生成图，而上下文没有完整执行稿时调用；不要用于无关的新绘图。"""
        record = self._last_image_prompts.get(str(event.unified_msg_origin))
        if not record:
            return "当前会话还没有可读取的实际出图提示词。请根据现有对话理解用户需求。"
        mode = "改图" if record.get("has_image") else "文生图"
        return (
            "以下是插件最近一次真正提交给图片模型的执行记录。修改时以用户本轮要求覆盖旧内容，"
            "其余需要延续的主体身份、外观、场景和画风可从实际提示词继承。\n"
            f"模式：{mode}\n"
            f"实际提示词：{record['prompt']}"
        )

    async def _draw_and_push(
        self,
        event: AstrMessageEvent,
        prompt: str,
    ) -> None:
        """后台出图并推送。"""
        try:
            image_b64, error_text = await self._draw(
                event,
                prompt,
            )
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

    async def _draw(
        self,
        event: AstrMessageEvent,
        user_prompt: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            return None, "生成失败喵（请输入具体的绘图或编辑描述）"

        image_ref = await resolve_input_image(self.context, event, self.client)

        prompt = user_prompt
        rewritten_prompt = await rewrite(
            self.context,
            event.unified_msg_origin,
            user_prompt,
            has_image=bool(image_ref),
            style_mode=self.style_mode,
            style_strength=self.style_strength,
            provider_id=self.rewrite_provider_id,
            fallback_provider_ids=self.rewrite_fallback_provider_ids,
        )
        if not rewritten_prompt:
            return None, "生成失败喵（所有提示词优化模型均不可用）"
        prompt = rewritten_prompt

        try:
            result = await self._generate(
                event,
                prompt,
                image_ref,
            )
        except QiniuSafetyError as exc:
            logger.warning(
                f"qiniu-image rejected by safety, starting fallback | {self._ctx(event)} "
                f"model={self.client.model} status={exc.status} code={exc.code}"
            )
        else:
            if result[0]:
                self._remember_image_prompt(
                    event,
                    prompt,
                    has_image=bool(image_ref),
                )
            return result

        for safety_attempt in range(1, SAFETY_REWRITE_LEVELS + 1):
            safe_prompt = await rewrite_for_safety(
                self.context,
                event.unified_msg_origin,
                prompt,
                provider_id=self.rewrite_provider_id,
                fallback_provider_ids=self.rewrite_fallback_provider_ids,
                safety_attempt=safety_attempt,
            )
            if not safe_prompt:
                logger.warning(
                    f"qiniu-image: safety rewrite produced no usable prompt, advancing stage | "
                    f"{self._ctx(event)} attempt={safety_attempt}/{SAFETY_REWRITE_LEVELS}"
                )
                continue

            try:
                result = await self._generate(
                    event,
                    safe_prompt,
                    image_ref,
                )
            except QiniuSafetyError as exc:
                logger.warning(
                    f"qiniu-image safety fallback rejected, advancing stage | {self._ctx(event)} "
                    f"model={self.client.model} status={exc.status} code={exc.code} "
                    f"attempt={safety_attempt}/{SAFETY_REWRITE_LEVELS}"
                )
                prompt = safe_prompt
                continue

            if result[0]:
                self._remember_image_prompt(
                    event,
                    safe_prompt,
                    has_image=bool(image_ref),
                )
            return result

        return None, "生成失败喵（所有安全级别均未能生成可用图片）"

    def _remember_image_prompt(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        has_image: bool,
    ) -> None:
        """记录当前会话最后一次实际提交的提示词，供后续修改透明继承。"""
        key = str(event.unified_msg_origin)
        self._last_image_prompts[key] = {
            "prompt": prompt,
            "has_image": has_image,
            "updated_at": time.monotonic(),
        }
        if len(self._last_image_prompts) > 100:
            oldest = min(
                self._last_image_prompts,
                key=lambda item: float(self._last_image_prompts[item]["updated_at"]),
            )
            self._last_image_prompts.pop(oldest, None)

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
        except QiniuSafetyError:
            raise
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
