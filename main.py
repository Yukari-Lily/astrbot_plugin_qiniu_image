"""七牛 AI 绘图 / 改图插件。"""

import asyncio
import json
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .message_utils import resolve_input_image
from .prompt_integrator import integrate, integrate_fallback
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
_DRAWING_SUBMISSIONS = "qiniu_image_submissions"
PREPARATION_TIMEOUT_SECONDS = 45
INTEGRATION_TIMEOUT_SECONDS = 20

_DRAWING_RULES = """
绘图只需调用 prepare_drawing，即刻受理并在后台规划、搜索和出图，无需再调用 draw_image。
brief 只写用户明确的主体、动作、构图、文字、必要人设/指代和硬性要求，不把你凭记忆补充的角色外观写进去。
对已有角色仅交代名称、作品及用户明确设定，外观知识由后台按需核实；原创主体完整保留用户设定。
research_mode=skip 仅用于原创角色/场景或不依赖外部知识的请求；已有 IP 角色需要查外观时用 auto。
style_request 只写用户明确要求的画风原话，未要求留空，恢复自动选风格填 auto；内容修改留空沿用旧风格。
caption 用你当前人格自然地说一句画谁，60字以内；插件在提示词准备好后代发，不要声称已经画好。
有图时 reference=参考创作，edit=局部修改，auto=按委托判断；明确局部改图无需联网核对。
修改上一张作品用 use_last=true；修改指定任务用 base_plan_id，但正在生成图片的任务不能修改。
每个新增请求分别调用 prepare_drawing，accepted 仅受理这一份。不要轮询、重复提交或等待审核摘要。
只有用户明确取消时调用 cancel_drawing，允许取消尚未开始图片生成的后台任务。
需要查看依据时调用 get_drawing_plan(full=true) 或 get_last_image_prompt(full=true)，默认只返回状态与摘要。
整个绘图过程静默，不播报搜索、规划、确认、重试等过程消息。插件会发送 caption 和图片。
所有请求受理完毕后最终只输出内部标记 QINIU_DRAWING_DONE，插件会隐藏它；不要返回空内容。
受理失败时只简短说明原因，不承诺已经开始出图。
""".strip()

@register(
    "astrbot_plugin_qiniu_image",
    "Yukari Lily",
    "七牛 AI 绘图 / 改图",
    "2.3.0",
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
        self._plan_tasks = {}
        self._image_sources = {}

        if not self.client.configured:
            logger.warning("qiniu-image: 未配置 api_key，插件已加载但无法出图")

    async def terminate(self):
        for plan_id in list(self._plan_tasks):
            record = self.planner.plans.get(plan_id)
            if record and record["status"] in ("planning", "integrating", "generating"):
                self._cancel_plan(record)
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
                              image_mode: str = "auto", style_request: str = "",
                              caption: str = "", research_mode: str = "auto"):
        """立即受理绘图，返回任务编号；后台限时规划并自动发图，无需再调用 draw_image。

        Args:
            brief(string): 用户明确的主体、动作、构图、硬性要求和必要指代；不要添加记忆中的角色外观。
            base_plan_id(string): 修改指定任务时填原编号，正在生成图片时不能修改。
            use_last(boolean): 修改上一张作品时为 true，自动沿用完整提示词。
            image_mode(string): reference=参考创作，edit=局部修改，auto=按委托判断。
            style_request(string): 用户明确要求的画风原话，未要求留空，恢复自动填 auto。
            caption(string): 按当前聊天人格说一句画谁，60字以内，由插件代发。
            research_mode(string): auto=按需核实；skip=原创或无需外部知识的简单请求，不联网。
        """
        if not self.client.configured:
            return "生成失败喵（未配置 api_key）"
        if not isinstance(brief, str) or not 1 <= len(brief.strip()) <= 16000:
            return "请提供不超过16000字符的绘图委托。"
        if not isinstance(style_request, str) or len(style_request) > 1000:
            return "画风要求仅填写用户明确说出的原话，1000字以内。"
        if not isinstance(caption, str) or len(caption.strip()) > 60:
            return "caption 请用当前人格简单说画谁，60字以内。"
        if image_mode not in ("auto", "edit", "reference") or (base_plan_id and use_last):
            return "图片用途应为 auto/edit/reference；base_plan_id 与 use_last 不能同时使用。"
        if research_mode not in ("auto", "skip"):
            return "research_mode 应为 auto 或 skip。"
        owner = self._prompt_key(event)
        request_key = json.dumps([brief.strip(), base_plan_id, use_last, image_mode, style_request.strip(), research_mode], ensure_ascii=False)
        submitted = event.get_extra(_DRAWING_SUBMISSIONS, {})
        if request_key in submitted:
            return submitted[request_key]
        previous_prompt, previous_style_id, base = "", "none", None
        if base_plan_id:
            base = self.planner.get(owner, base_plan_id)
            if not base:
                return "方案不存在、已过期或不属于当前用户。"
            if base["status"] == "generating":
                return "该方案正在出图，请等待完成再修改。"
            previous_prompt = base.get("final_prompt") or base["prompt"]
            previous_style_id = base["style_id"]
        elif use_last:
            last = self._last_image_prompts.get(owner)
            if not last:
                return "当前用户没有可沿用的提示词，请提交新的绘图委托。"
            previous_prompt, previous_style_id = str(last["prompt"]), str(last.get("style_id", "none"))
        options = dict(previous_prompt=previous_prompt, previous_style_id=previous_style_id,
                       image_ref=base["image_ref"] if base else None,
                       image_mode=(base["image_mode"] if base["image_ref"] else base.get("requested_image_mode", "auto"))
                       if base and image_mode == "auto" else image_mode,
                       style_mode=self.style_mode, style_request=style_request.strip())
        try:
            record = self.planner.snapshot(owner, brief.strip(), state={}, **options)
        except ValueError as exc:
            return "受理失败：" + str(exc)
        record.update(status="planning", caption=caption.strip() or "按你的要求画一张。",
                      deadline=time.monotonic() + PREPARATION_TIMEOUT_SECONDS, research_mode=research_mode,
                      requested_image_mode=options["image_mode"])
        image_sources = (event,)
        if base and not base["image_ref"]:
            image_sources += tuple(source for source in self._image_sources.get(base["plan_id"], ()) if source is not event)
        self._image_sources[record["plan_id"]] = image_sources
        # Reserve the id and immutable request inputs before any provider or image I/O.
        if base and base["status"] in ("ready", "planning", "integrating"):
            self._cancel_plan(base, status="superseded")
        search_tools = tuple(self._search_tools.get(owner, ())) if research_mode == "auto" else ()
        task = asyncio.create_task(self._plan_and_push(event, record, options, search_tools, image_sources))
        self._track_task(record, task)
        event.set_extra(_SILENT_DRAWING, True)
        accepted = json.dumps({"status": "accepted", "plan_id": record["plan_id"], "phase": "planning",
                               "instruction": "本请求已受理，后台会自动发图；不要重复准备、提交或轮询。继续处理其他请求，全部受理后只输出 QINIU_DRAWING_DONE。"}, ensure_ascii=False)
        submitted = dict(submitted)
        submitted[request_key] = accepted
        event.set_extra(_DRAWING_SUBMISSIONS, submitted)
        logger.info(f"qiniu-image accepted | umo={event.unified_msg_origin} plan_id={record['plan_id']} phase=planning")
        return accepted

    def _track_task(self, record, task):
        self._tasks.add(task)
        self._plan_tasks[record["plan_id"]] = task

        def done(completed):
            self._tasks.discard(completed)
            self._plan_tasks.pop(record["plan_id"], None)
            self._image_sources.pop(record["plan_id"], None)

        task.add_done_callback(done)

    def _cancel_plan(self, record, status="cancelled"):
        record["status"] = status
        task = self._plan_tasks.get(record["plan_id"])
        if task:
            task.cancel()

    @staticmethod
    def _remaining(record):
        return max(0, record.get("deadline", time.monotonic() + INTEGRATION_TIMEOUT_SECONDS) - time.monotonic())

    async def _plan_and_push(self, event, record, options, search_tools, image_sources):
        state = {}
        try:
            image_ref = None
            for source in image_sources:
                image_ref = await asyncio.wait_for(
                    resolve_input_image(self.context, source, self.client), timeout=min(10, self._remaining(record)))
                segments = source.get_messages() or []
                supplied_image = any(isinstance(seg, Comp.Image) or
                                     (isinstance(seg, Comp.Reply) and any(isinstance(part, Comp.Image)
                                      for part in (getattr(seg, "chain", None) or []))) for seg in segments)
                if supplied_image and not image_ref:
                    raise ValueError("用户输入图片无法读取")
                if image_ref:
                    break
            options = dict(options, image_ref=image_ref or options["image_ref"])
            if options["image_mode"] in ("edit", "reference") and not options["image_ref"]:
                raise ValueError("没有可用的输入图片")
            self.planner.snapshot(record["owner"], record["brief"], state=state, record=record, **options)
            direct_edit = bool(options["image_ref"]) and options["image_mode"] == "edit" and not options["style_request"]
            record["fast_path"] = direct_edit or record["research_mode"] == "skip"
            if not direct_edit:
                try:
                    await asyncio.wait_for(self.planner.prepare(
                        event, record["owner"], record["brief"], fallback_state=state, target_record=record,
                        vision_ref=self.client.as_image_reference(options["image_ref"]) if options["image_ref"] else None,
                        search_tools=search_tools, no_research=not search_tools, **options,
                    ), timeout=self._remaining(record))
                except Exception as exc:
                    # The snapshot is already usable; never start another model after the deadline.
                    self.planner.snapshot(record["owner"], record["brief"], state=state, record=record, **options)
                    record.update(degraded=True, degraded_reason="规划已停止，使用最新方案快照；未确认外观不作确定事实。")
                    logger.warning(f"qiniu-image snapshot used | plan_id={record['plan_id']} revision={record['snapshot_revision']} cause={type(exc).__name__}")
            if record["status"] in ("cancelled", "superseded"):
                return
            record["status"] = "integrating"
            await self._integrate_and_push(event, record)
        except asyncio.CancelledError:
            if record["status"] != "superseded":
                record["status"] = "cancelled"
            raise
        except Exception as exc:
            record["status"] = "failed"
            logger.warning(f"qiniu-image preparation failed | plan_id={record['plan_id']} cause={type(exc).__name__}")
            await self._push_text(event, "生成失败喵（输入图片或绘图方案不可用）")

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

    @filter.llm_tool(name="cancel_drawing")
    async def cancel_drawing(self, event: AstrMessageEvent, plan_id: str):
        """仅在用户明确取消或替换请求时，取消尚未开始生成图片的后台任务；新增绘图不表示取消。

        Args:
            plan_id(string): 用户明确不再需要的任务编号。
        """
        record = self.planner.get(self._prompt_key(event), plan_id)
        if not record or record["status"] not in ("ready", "planning", "integrating"):
            return "只能取消当前用户尚未开始图片生成的任务。"
        self._cancel_plan(record)
        event.set_extra(_SILENT_DRAWING, True)
        return json.dumps({"status": "cancelled", "plan_id": plan_id,
                           "instruction": "继续处理其他请求；全部处理后仅输出 QINIU_DRAWING_DONE。"}, ensure_ascii=False)

    @filter.llm_tool(name="draw_image")
    async def draw_image(self, event: AstrMessageEvent, plan_id: str, caption: str = ""):
        """兼容旧待审方案的提交入口。prepare_drawing 已自动受理的任务无需调用，重复调用只返回状态。

        Args:
            plan_id(string): 已检查并符合用户委托的 prepare_drawing 方案编号。
            caption(string): 用你当前聊天人格写的简短口语，60字以内，说画谁即可；由插件在完整提示词就绪时代发，不要另行回复。
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
                               "instruction": "不要重复出图；继续处理其他请求，全部处理后仅输出 QINIU_DRAWING_DONE。"}, ensure_ascii=False)
        if not isinstance(caption, str) or not caption.strip() or len(caption.strip()) > 60:
            return json.dumps({"status": "ready", "plan_id": plan_id,
                               "instruction": "请补上 caption 后再调用：按你当前人格自然地说一句画谁，60字以内，不复述方案，不声称已经画好。"}, ensure_ascii=False)
        record["caption"] = caption.strip()
        record["status"] = "integrating"
        logger.info(f"qiniu-image accepted | umo={event.unified_msg_origin} plan_id={plan_id}")
        task = asyncio.create_task(self._integrate_and_push(event, record))
        self._track_task(record, task)
        return json.dumps({"status": "accepted", "plan_id": plan_id, "phase": "integrating",
                           "instruction": "本方案已受理，不重复调用。继续处理尚未完成的其他绘图请求；全部处理后仅输出 QINIU_DRAWING_DONE 作为内部收尾，不能返回空内容。插件会代发 caption 和图片。"}, ensure_ascii=False)

    async def _integrate_and_push(self, event, record):
        """Keep provider fallbacks outside AstrBot's tool timeout, with exactly one caption."""
        try:
            selection = {}
            final_prompt = None
            if not record.get("degraded") and not record.get("fast_path") and self._remaining(record) > 0:
                try:
                    final_prompt = await asyncio.wait_for(integrate(
                        self.context, event.unified_msg_origin, record["prompt"], has_image=bool(record["image_ref"]),
                        image_mode=record["image_mode"] if record["image_ref"] else "auto",
                        style_mode=self.style_mode, style_strength=self.style_strength,
                        provider_id=self.rewrite_provider_id, fallback_provider_ids=self.rewrite_fallback_provider_ids,
                        selection_out=selection,
                        planned_style_id=record["style_id"],
                    ), timeout=min(INTEGRATION_TIMEOUT_SECONDS, self._remaining(record)))
                except Exception as exc:
                    logger.warning(f"qiniu-image integration fallback | plan_id={record['plan_id']} cause={type(exc).__name__}")
            if not final_prompt:
                final_prompt = integrate_fallback(
                    record["prompt"], has_image=bool(record["image_ref"]), image_mode=record["image_mode"],
                    planned_style_id=record["style_id"], selection_out=selection,
                )
            if not final_prompt:
                record["status"] = "failed"
                await self._push_text(event, "生成失败喵（提示词整合暂时不可用）")
                return
            record["final_prompt"] = final_prompt
            record["style_selection"] = selection
            logger.info(f"qiniu-image prompt ready | umo={event.unified_msg_origin} plan_id={record['plan_id']} selection={json.dumps(selection, ensure_ascii=False)}")
            record["status"] = "generating"
            self._remember_image_prompt(event, final_prompt, has_image=bool(record["image_ref"]), plan=record)
            await self._push_text(event, self._drawing_caption(record))
            await self._draw_and_push(event, final_prompt, image_ref=record["image_ref"], plan=record)
        except asyncio.CancelledError:
            if record["status"] != "superseded":
                record["status"] = "cancelled"
            raise
        except Exception as exc:
            record["status"] = "failed"
            logger.error(f"qiniu-image: 后台整合失败（{type(exc).__name__}）")
            await self._push_text(event, "生成失败喵，请稍后重试")

    @staticmethod
    def _drawing_caption(record):
        return record["caption"]

    async def _push_text(self, event, text):
        try:
            await self.context.send_message(event.unified_msg_origin, MessageChain().message(text))
        except Exception as exc:
            logger.warning(f"qiniu-image: 发送绘图消息失败（{type(exc).__name__}）")

    @filter.llm_tool(name="list_image_styles")
    async def list_image_styles(self, event: AstrMessageEvent):
        """查询本插件实际可用的内置绘图风格。当用户询问你会哪些画风、支持哪些风格、推荐什么画风或要求列出风格时，调用此工具获取最新目录。列出可用风格时只列中文标题，每行一个，不添加说明、配置或推荐语，不编造目录外的内置风格。"""
        yield event.plain_result(style_catalog_text(concise=True))

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
            result["style_selection"] = record.get("style_selection", {})
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

        chain = (
            MessageChain().base64_image(image_b64)
            if image_b64
            else MessageChain().message(error_text or "生成失败喵")
        )
        try:
            delivered = await self.context.send_message(event.unified_msg_origin, chain)
            if delivered is False:
                raise RuntimeError("消息平台未接受发送")
            if plan is not None:
                plan["status"] = "completed" if image_b64 else "failed"
                logger.info(f"qiniu-image delivered | umo={event.unified_msg_origin} plan_id={plan['plan_id']} status={plan['status']}")
        except Exception as exc:
            if plan is not None:
                plan["status"] = "delivery_failed"
            logger.error(f"qiniu-image: 推送结果失败 | umo={event.unified_msg_origin} plan_id={plan['plan_id'] if plan else ''}（{type(exc).__name__}: {exc}）")

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
            "style_selection": plan.get("style_selection", {}) if plan else {},
            "style_id": plan["style_id"] if plan else "none",
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
            if plan is not None:
                logger.info("qiniu-image generation request | " + json.dumps({
                    "umo": str(event.unified_msg_origin), "plan_id": plan["plan_id"],
                    "adjusted_after_review": bool(plan.get("adjusted_after_review")),
                    "prompt": prompt}, ensure_ascii=False))
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
