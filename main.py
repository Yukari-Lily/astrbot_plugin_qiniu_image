"""七牛 AI 绘图 / 改图插件。"""

import asyncio
import json
import re
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .message_utils import resolve_input_image
from .prompt_integrator import integrate
from .prompt_rewriter import SAFETY_REWRITE_LEVELS, rewrite, rewrite_for_safety
from .drawing_planner import DrawingPlanner, SEARCH_TOOLS
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
_SILENT_DRAWING = "qiniu_image_silent_drawing"
PLANNING_TOOL_TIMEOUT_SECONDS = 55

_DRAWING_RULES = """
绘图时先调用 prepare_drawing：交代用户意图、必要人设/指代、硬性要求及允许自由设计的部分。
独立规划模型负责搜索、特征核对和动作构图。检查返回的摘要，不满意则带 base_plan_id 提修改意见；
确认符合委托后调用 draw_image(plan_id)，不要并行准备与出图，也不用额外询问用户批准。
修改上一张作品可用 use_last=true，让规划模型直接读取完整缓存；必要时用 get_drawing_plan(full=true)
或 get_last_image_prompt(full=true) 查看全文和依据。默认只返回摘要，accepted 后等自动发图，不重复调用。
整个绘图工具链静默执行：调用工具前后都不要发“稍等、正在规划、方案已确认、重试、后台加速”等过程消息，
不要复述内部摘要或错误堆栈。完整提示词准备好后插件会统一发一句画面说明并发送图片，你无需再补回复。
失败时只简短说明失败原因，不要承诺已经在出图或“马上就来”。
""".strip()

@register(
    "astrbot_plugin_qiniu_image",
    "Yukari Lily",
    "七牛 AI 绘图 / 改图",
    "2.2.1",
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
        self.planner = DrawingPlanner(context, str(config.get("planning_provider_id") or "").strip())
        self._search_tools = {}
        self._preparing = set()

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

    @filter.on_llm_request()
    async def on_llm_request(self, event, req):
        if _DRAWING_RULES not in (req.system_prompt or ""):
            req.system_prompt = (req.system_prompt or "") + "\n\n" + _DRAWING_RULES
        # Only inherit web tools actually enabled for this request/persona.
        tools = getattr(req, "func_tool", None)
        self._search_tools[self._prompt_key(event)] = tuple(
            tool for tool in (getattr(tools, "tools", None) or [])
            if tool.name in SEARCH_TOOLS and getattr(tool, "active", True)
        )
        if len(self._search_tools) > 100:
            self._search_tools.pop(next(iter(self._search_tools)))

    def _prompt_key(self, event):
        return json.dumps([str(event.unified_msg_origin), str(event.get_sender_id())])

    @filter.on_decorating_result()
    async def suppress_drawing_chatter(self, event):
        """Only suppress this drawing turn's LLM chatter; plugin deliveries bypass this hook."""
        if not event.get_extra(_SILENT_DRAWING, False):
            return
        result = event.get_result()
        if result and result.is_llm_result():
            result.chain.clear()

    @filter.llm_tool(name="prepare_drawing")
    async def prepare_drawing(self, event: AstrMessageEvent, brief: str,
                              base_plan_id: str = "", use_last: bool = False,
                              image_mode: str = "auto"):
        """静默委托独立模型规划绘图，返回内部审核用的编号和摘要，不向用户播报进度。

        Args:
            brief(string): 用户意图、已解析的指代/必要人设、硬性要求和创作自由度；修改时写具体意见。
            base_plan_id(string): 修改待审方案时填原方案编号，否则留空。
            use_last(boolean): 修改上一张作品时为 true，规划模型直接读取完整提示词，无需复述。
            image_mode(string): 有图时 reference=参考创作、edit=局部修改、auto=按委托判断。
        """
        if not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 16000:
            return "请提供不超过16000字符的绘图委托。"
        if image_mode not in ("auto", "edit", "reference") or (base_plan_id and use_last):
            return "图片用途应为 auto/edit/reference；base_plan_id 与 use_last 不能同时使用。"
        owner = self._prompt_key(event)
        if owner in self._preparing:
            event.set_extra(_SILENT_DRAWING, True)
            return '{"status":"pending","instruction":"已有方案正在准备，请等待。"}'
        self._preparing.add(owner)
        try:
            previous_prompt = ""
            base = None
            if base_plan_id:
                base = self.planner.get(owner, base_plan_id)
                if not base:
                    return "方案不存在、已过期或不属于当前用户，请重新准备。"
                if base["status"] in ("integrating", "generating"):
                    return "该方案正在出图，请等待完成再修改。"
                previous_prompt = base.get("final_prompt") or base["prompt"]
            elif use_last:
                last = self._last_image_prompts.get(owner)
                if not last:
                    return "当前用户没有可沿用的提示词，请提交新的绘图委托。"
                previous_prompt = str(last["prompt"])
            image_ref = await resolve_input_image(self.context, event, self.client)
            if base and not image_ref:
                image_ref = base["image_ref"]
                if image_mode == "auto" and image_ref:
                    image_mode = base["image_mode"]
            event.set_extra(_SILENT_DRAWING, True)
            record = await asyncio.wait_for(self.planner.prepare(
                event, owner, brief.strip(), image_ref=image_ref,
                vision_ref=self.client.as_image_reference(image_ref) if image_ref else None,
                image_mode=image_mode, previous_prompt=previous_prompt,
                search_tools=self._search_tools.get(owner, ()),
            ), timeout=PLANNING_TOOL_TIMEOUT_SECONDS)
            if base and base["status"] == "ready":
                base["status"] = "superseded"
            result = self.planner.describe(record)
            result["instruction"] = "静默检查摘要，不要向用户复述或播报进度；需要修改则重新准备，符合委托后直接调用 draw_image(plan_id)。"
            return json.dumps(result, ensure_ascii=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"qiniu-image: 规划失败（{type(exc).__name__}）")
            event.set_extra(_SILENT_DRAWING, False)
            return "规划失败：" + (str(exc) if isinstance(exc, ValueError) else "规划模型或工具不可用/超时，请重试。")
        finally:
            self._preparing.discard(owner)

    @filter.llm_tool(name="get_drawing_plan")
    async def get_drawing_plan(self, event: AstrMessageEvent, plan_id: str, full: bool = False):
        """读取方案，默认只返回摘要；需要核查完整方案或搜索依据时才取全文。

        Args:
            plan_id(string): prepare_drawing 返回的方案编号。
            full(boolean): true 返回完整方案与搜索核对依据，默认 false。
        """
        record = self.planner.get(self._prompt_key(event), plan_id)
        if not record:
            return "方案不存在、已过期或不属于当前用户。"
        return json.dumps(self.planner.describe(record, full), ensure_ascii=False)

    @filter.llm_tool(name="draw_image")
    async def draw_image(self, event: AstrMessageEvent, plan_id: str):
        """静默确认并提交方案，立即返回；后台整合提示词和出图。不要另发确认、等待或重试消息。

        Args:
            plan_id(string): 已检查并符合用户委托的 prepare_drawing 方案编号。
        """
        if not self.client.configured:
            event.set_extra(_SILENT_DRAWING, False)
            return "生成失败喵（未配置 api_key）"
        owner = self._prompt_key(event)
        record = self.planner.get(owner, plan_id)
        if not record:
            return "请先调用 prepare_drawing 并检查返回的摘要，再提交有效方案编号。"
        event.set_extra(_SILENT_DRAWING, True)
        if record["status"] != "ready":
            return json.dumps({"status": record["status"], "plan_id": plan_id,
                               "instruction": "不要重复出图；如需修改或重画，请准备新方案。"}, ensure_ascii=False)
        record["status"] = "integrating"
        task = asyncio.create_task(self._integrate_and_push(event, record))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return json.dumps({"status": "accepted", "plan_id": plan_id, "phase": "integrating",
                           "instruction": "静默等待，勿再回复用户或重复调用。插件会在完整提示词准备好后只发一句画面说明，再自动发图。"}, ensure_ascii=False)

    async def _integrate_and_push(self, event, record):
        """Keep provider fallbacks outside AstrBot's tool timeout, with exactly one caption."""
        try:
            final_prompt = await integrate(
                self.context, event.unified_msg_origin, record["prompt"], has_image=bool(record["image_ref"]),
                image_mode=record["image_mode"] if record["image_ref"] else "auto",
                style_mode=self.style_mode, style_strength=self.style_strength,
                provider_id=self.rewrite_provider_id, fallback_provider_ids=self.rewrite_fallback_provider_ids,
            )
            if not final_prompt:
                record["status"] = "failed"
                await self._push_text(event, "生成失败喵（提示词整合暂时不可用）")
                return
            record["final_prompt"] = final_prompt
            record["status"] = "generating"
            self._remember_image_prompt(event, final_prompt, has_image=bool(record["image_ref"]), plan=record)
            await self._push_text(event, self._drawing_caption(record))
            await self._draw_and_push(event, final_prompt, image_ref=record["image_ref"], plan=record)
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception as exc:
            record["status"] = "failed"
            logger.error(f"qiniu-image: 后台整合失败（{type(exc).__name__}）")
            await self._push_text(event, "生成失败喵，请稍后重试")

    @staticmethod
    def _drawing_caption(record):
        caption = record.get("caption")
        if not isinstance(caption, str) or not caption.strip() or len(caption.strip()) > 60:
            caption = re.split(r"[。！？\n]", record["summary"], maxsplit=1)[0]
        caption = " ".join(caption.split()).strip("。！？ ")
        return caption[:59].rstrip("，、； ") + "。"

    async def _push_text(self, event, text):
        try:
            await self.context.send_message(event.unified_msg_origin, MessageChain().message(text))
        except Exception as exc:
            logger.warning(f"qiniu-image: 发送绘图消息失败（{type(exc).__name__}）")

    @filter.llm_tool(name="list_image_styles")
    async def list_image_styles(self, event: AstrMessageEvent):
        """查询本插件实际可用的内置绘图风格。当用户询问你会哪些画风、支持哪些风格、推荐什么画风或要求列出风格时，调用此工具获取最新目录。"""
        yield event.plain_result(
            "本插件的内置绘图风格如下。回答用户时使用中文名称，不要编造目录外的内置风格。\n"
            f"当前模式：{self.style_mode}；强度：{self.style_strength}。\n\n"
            f"{style_catalog_text(concise=True)}"
        )

    @filter.llm_tool(name="get_last_image_prompt")
    async def get_last_image_prompt(self, event: AstrMessageEvent, full: bool = False):
        """读取当前用户最近绘图记录，默认摘要；普通修改用 prepare_drawing(use_last=true) 即可。

        Args:
            full(boolean): 只有需要检查完整执行提示词时设为 true。
        """
        record = self._last_image_prompts.get(self._prompt_key(event))
        if not record:
            return "当前用户还没有可读取的出图提示词。"
        result = {"mode": "图生图" if record["has_image"] else "文生图",
                  "plan_id": record.get("plan_id", ""), "summary": record.get("summary", "关键词直出；完整提示词已保存"),
                  "instruction": "修改时调用 prepare_drawing(use_last=true)，无需复述完整提示词；此记录不代表已出图成功。"}
        if full:
            result["prompt"] = record["prompt"]
        if record.get("adjusted_after_review"):
            result["notice"] = "已使用审核后的安全替代提示词，原摘要仅供参考，可按需读取实际全文。"
        return json.dumps(result, ensure_ascii=False)

    async def _draw_and_push(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        image_ref: Optional[str] = None,
        plan=None,
    ) -> None:
        """后台出图并推送。"""
        try:
            image_b64, error_text = await self._draw_prepared(event, prompt, image_ref, plan=plan)
        except asyncio.CancelledError:
            if plan is not None:
                plan["status"] = "cancelled"
            raise
        except Exception as exc:
            logger.error(
                f"qiniu-image background task failed | {self._ctx(event)} "
                f"error_type={type(exc).__name__}\n" + "".join(traceback.format_tb(exc.__traceback__))
            )
            image_b64, error_text = None, "生成失败喵"

        if plan is not None:
            plan["status"] = "completed" if image_b64 else "failed"
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

        return await self._draw_prepared(event, prompt, image_ref)

    async def _draw_prepared(self, event, prompt, image_ref, plan=None):

        try:
            result = await self._generate(
                event,
                prompt,
                image_ref,
                plan=plan,
            )
        except QiniuSafetyError as exc:
            logger.warning(
                f"qiniu-image rejected by safety, starting fallback | {self._ctx(event)} "
                f"model={self.client.model} status={exc.status} code={exc.code}"
            )
        else:
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
                    plan=plan,
                )
            except QiniuSafetyError as exc:
                logger.warning(
                    f"qiniu-image safety fallback rejected, advancing stage | {self._ctx(event)} "
                    f"model={self.client.model} status={exc.status} code={exc.code} "
                    f"attempt={safety_attempt}/{SAFETY_REWRITE_LEVELS}"
                )
                prompt = safe_prompt
                continue

            return result

        return None, "生成失败喵（所有安全级别均未能生成可用图片）"

    def _remember_image_prompt(
        self,
        event: AstrMessageEvent,
        prompt: str,
        *,
        has_image: bool,
        plan=None,
    ) -> None:
        """记录当前会话最后一次实际提交的提示词，供后续修改透明继承。"""
        if plan is not None:
            if plan.get("final_prompt") and plan["final_prompt"] != prompt:
                plan["adjusted_after_review"] = True
            plan["final_prompt"] = prompt
        key = self._prompt_key(event)
        self._last_image_prompts[key] = {
            "prompt": prompt,
            "has_image": has_image,
            "updated_at": time.monotonic(),
            "plan_id": plan["plan_id"] if plan else "",
            "summary": plan["summary"] if plan else "关键词直出；完整提示词已保存",
            "adjusted_after_review": bool(plan and plan.get("adjusted_after_review")),
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
        plan=None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """返回图片或错误提示。"""
        try:
            self._remember_image_prompt(event, prompt, has_image=bool(image_ref), plan=plan)
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
