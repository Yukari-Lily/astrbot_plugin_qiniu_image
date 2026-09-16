"""七牛 AI 绘图 / 改图插件。"""

import asyncio
import json
import time
import traceback
from dataclasses import replace
from typing import Any, Coroutine, Dict, List, Optional, Set, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .drawing_plan import DrawingRequest, ImagePlan, ImageRecord
from .message_utils import resolve_input_image
from .prompt_optimizer import optimize_prompt
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
from .safety_rewriter import SAFETY_REWRITE_LEVELS, rewrite_for_safety
from .style_presets import (
    STYLE_STRENGTHS,
    compose_prompt,
    plan_body,
    style_catalog_text,
)

DEDUP_TTL_SECONDS = 20
IMAGE_RESOLVE_TIMEOUT_SECONDS = 10
_NOT_CONFIGURED = "生成失败喵（未配置 api_key）"


@register(
    "astrbot_plugin_qiniu_image",
    "Yukari Lily",
    "七牛 AI 绘图 / 改图",
    "2.0.0",
    "https://github.com/Yukari-Lily/astrbot_plugin_qiniu_image",
)
class QiniuImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.client = QiniuImageClient(config)

        raw_triggers = config.get("triggers") or []
        if not isinstance(raw_triggers, (list, tuple)):
            # 配置被手改成裸字符串时不能逐字符当触发词，否则任何消息都会触发。
            logger.warning("qiniu-image: 配置项 triggers 必须是列表，已忽略")
            raw_triggers = []
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

        self.enable_styles = bool(config.get("enable_styles", True))
        style_strength = str(config.get("style_strength", "normal") or "normal").strip().lower()
        if style_strength not in STYLE_STRENGTHS:
            raise ValueError(
                f"qiniu_image 配置项 style_strength 必须是 {'/'.join(STYLE_STRENGTHS)} 之一"
            )
        self.style_strength = style_strength

        self._recent_msg: Dict[str, float] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._last_image_prompts: Dict[Tuple[str, str], ImageRecord] = {}

        if not self.client.configured:
            logger.warning("qiniu-image: 未配置 api_key，插件已加载但无法出图")

    async def terminate(self):
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.close()

    def _start_task(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _match_trigger(self, event: AstrMessageEvent) -> Optional[str]:
        """命中返回触发词；未命中不产生副作用。"""
        if not self.triggers:
            return None
        text = (event.message_str or "").strip()
        if not text:
            return None
        return next((trigger for trigger in self.triggers if text.startswith(trigger)), None)

    @staticmethod
    def _message_key(event: AstrMessageEvent) -> Optional[str]:
        """去重键必须带上来源，同群多台 Bot 会收到同一个 message_id。"""
        mid = getattr(getattr(event, "message_obj", None), "message_id", None)
        if mid is None:
            return None
        return f"{event.unified_msg_origin}|{mid}"

    def _dedup_hit(self, event: AstrMessageEvent) -> bool:
        """协议重投时避免同一消息重复出图。"""
        mid = self._message_key(event)
        if mid is None:
            return False
        now = time.monotonic()
        for key, timestamp in list(self._recent_msg.items()):
            if now - timestamp > DEDUP_TTL_SECONDS:
                self._recent_msg.pop(key, None)
        if mid in self._recent_msg:
            return True
        self._recent_msg[mid] = now
        return False

    def _prompt_key(self, event: AstrMessageEvent) -> Tuple[str, str]:
        """出图记录按会话与用户隔离，同群不同用户不会互相继承提示词。"""
        return str(event.unified_msg_origin), str(event.get_sender_id())

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message(self, event: AstrMessageEvent):
        trigger = self._match_trigger(event)
        if trigger is None:
            return

        event.stop_event()
        if self._dedup_hit(event):
            return
        if not self.client.configured:
            yield event.plain_result(_NOT_CONFIGURED)
            return

        prompt = (event.message_str or "").strip()[len(trigger):].strip()
        request = DrawingRequest(prompt=prompt, optimize=False)
        task = self._start_task(self._run_request(event, request))
        image_b64, error_text = await task
        if image_b64:
            yield event.chain_result([Comp.Image.fromBase64(image_b64)])
        else:
            yield event.plain_result(error_text or "生成失败喵")

    @filter.llm_tool(name="draw_image")
    async def draw_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        subject_info: str = "",
        use_last_image: bool = False,
        keep_layout: bool = True,
    ):
        """结合用户要求、已有会话和 Bot 人设，形成完整绘图或编辑方案后出图。

        你负责创作决策，prompt 是可执行的完整方案，不是只转发角色名或一句意图。
        用户留出创作空间时，可合理决定服装、动作、表情、道具、场景、镜头、构图、光照、
        配色和画风，组织一致的姿态、透视与遮挡关系；用户明确限制始终优先。
        已有完整方案就忠实保留。不要为了填满参数强加复杂姿势；局部编辑不扩大改动范围。
        优化器只负责整理、校正事实、消除冲突和融合适用风格，不替你重新设计画面。
        未确定具体画风时可留给插件自动匹配；明确的内置风格名称或外部画风应写入方案。
        净色动画壁纸仅限至少两名人物，且为低优先候选；不要给单人默认套净色或为此添加人物。

        常见角色与昵称直接补全可靠的官方作品名、角色名及关键外观；知识不足以可靠还原或
        无法唯一识别时，先搜索并阅读结果，再调用本工具，搜索与绘图不得并行。
        Bot 自拍结合已有人设外貌与自拍语境形成方案，不编造身份。subject_info 只提供必要的
        事实或指代补充，通常可留空；不传完整人格规则或整份群聊，不使用旧参数 context。

        修改、延续上一张时 use_last_image=true，插件自动附上上一份成功执行稿。
        此时 prompt 写清本轮修改及要保留的要求，不必转抄底稿；未涉及的设计默认继承。
        新绘图 use_last_image=false。精确改图需要用户发送或引用图片，记录不包含图片像素。
        图片自动附加，实际外观以原图为准；keep_layout=true 用于局部编辑，false 用于参考创作。

        同一请求只调用一次。受理后用一条简短消息说明已受理，不再持续发送进度；图片由插件发送。

        Args:
            prompt(string): 结合用户要求、上下文与必要考据形成的完整方案；续画时为本轮修改要求。
            subject_info(string): 必要的已有事实或 Bot 外貌补充，通常留空；不要使用旧参数 context。
            use_last_image(bool): 修改、延续或重画当前用户上一张作品时为 True。
            keep_layout(bool): 当前输入图用于局部编辑时为 True，用户要求参考创作时为 False。
        """
        if not self.client.configured:
            return _NOT_CONFIGURED
        prompt = (prompt or "").strip()
        if not prompt:
            return "请提供绘图或编辑意图。"
        previous = self._last_image_prompts.get(self._prompt_key(event)) if use_last_image else None
        if use_last_image and previous is None:
            return "当前用户没有可沿用的成功出图记录，请重新描述需要的画面或引用图片。"
        # 记录不可变，在任何等待前取得快照，不受后续成功出图更新缓存的影响。
        request = DrawingRequest(
            prompt=prompt,
            user_message=event.message_str or "",
            subject_info=(subject_info or "").strip(),
            previous=previous,
            keep_layout=bool(keep_layout),
        )
        self._start_task(self._draw_and_push(event, request))
        logger.info(f"qiniu-image accepted | {self._ctx(event)} previous={previous is not None}")
        return (
            "已受理，出图完成后插件会自动发送图片，请勿重复调用。现在只用一条消息复述用户想画什么，"
            "不要补充尚未决定的画面细节。本轮到此为止，不要再发进度、确认或第二条消息。"
        )

    @filter.llm_tool(name="get_last_image_prompt")
    async def get_last_image_prompt(self, event: AstrMessageEvent):
        """用户查询上一张成功生成图的提示词时调用。续画直接用 draw_image 的 use_last_image=True，无需读取或转抄底稿。"""
        record = self._last_image_prompts.get(self._prompt_key(event))
        if record is None:
            return "当前用户还没有可读取的出图提示词。"
        return json.dumps({
            "mode": "图生图" if record.has_image else "文生图",
            "prompt": record.submitted_prompt,
            "style": record.plan.style,
            "style_exception": record.plan.style_exception,
            "integrated": record.plan.integrated,
            "people_count": record.plan.people_count,
            "keep_layout": record.keep_layout,
            "instruction": "这是成功出图时实际提交的完整提示词，不含图片像素。"
                           "续画只传本轮要求并设置 use_last_image=true，插件自动提供底稿。",
        }, ensure_ascii=False)

    @filter.llm_tool(name="list_image_styles")
    async def list_image_styles(self, event: AstrMessageEvent):
        """查询实际内置风格目录，供介绍或创作时参考。回答使用中文名称，不编造目录外的内置风格；出图无需先查询目录。"""
        yield event.plain_result(style_catalog_text())

    async def _load_input_image(self, event: AstrMessageEvent) -> Optional[str]:
        try:
            return await asyncio.wait_for(
                resolve_input_image(self.context, event, self.client),
                timeout=IMAGE_RESOLVE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise ValueError("原图读取超时") from None

    async def _draw_and_push(self, event: AstrMessageEvent, request: DrawingRequest) -> None:
        image_b64, error_text = await self._run_request(event, request)
        chain = (
            MessageChain().base64_image(image_b64)
            if image_b64 else MessageChain().message(error_text or "生成失败喵")
        )
        try:
            if await self.context.send_message(event.unified_msg_origin, chain) is False:
                raise RuntimeError("消息平台未接受发送")
            logger.info(f"qiniu-image delivered | {self._ctx(event)} success={bool(image_b64)}")
        except Exception as exc:
            logger.error(f"qiniu-image: 推送结果失败 | {self._ctx(event)}（{type(exc).__name__}）")

    async def _run_request(
        self, event: AstrMessageEvent, request: DrawingRequest,
    ) -> Tuple[Optional[str], Optional[str]]:
        """两种入口共享准备与生成；只在聊天入口进行正常优化。"""
        try:
            if not request.prompt.strip():
                return None, "生成失败喵（请输入具体的绘图或编辑描述）"
            image_ref = await self._load_input_image(event)
            if request.optimize:
                plan = await optimize_prompt(
                    self.context, event.unified_msg_origin, request,
                    has_image=bool(image_ref), enable_styles=self.enable_styles,
                    style_strength=self.style_strength,
                    provider_id=self.rewrite_provider_id,
                    fallback_provider_ids=self.rewrite_fallback_provider_ids,
                )
                if plan is None:
                    return None, "生成失败喵（所有提示词优化模型均未返回可用方案）"
            else:
                plan = ImagePlan(prompt=plan_body(request.prompt), style_exception="keyword")
            return await self._generate_plan(event, plan, image_ref, request.keep_layout)
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            logger.warning(f"qiniu-image: 方案或图片不可用（{exc}）| {self._ctx(event)}")
            return None, "生成失败喵（" + str(exc) + "）"
        except Exception as exc:
            logger.error(
                f"qiniu-image request failed | {self._ctx(event)} error_type={type(exc).__name__}\n"
                + "".join(traceback.format_tb(exc.__traceback__))
            )
            return None, "生成失败喵"

    async def _generate_plan(
        self, event: AstrMessageEvent, plan: ImagePlan,
        image_ref: Optional[str], keep_layout: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        """首次生成与审核重试共用拼装、提交及成功记录。"""
        original_prompt = compose_prompt(
            plan.prompt, has_image=bool(image_ref), integrated=plan.integrated,
            keep_layout=keep_layout,
        )
        for safety_attempt in range(SAFETY_REWRITE_LEVELS + 1):
            candidate = plan
            if safety_attempt:
                rewritten = await rewrite_for_safety(
                    self.context, event.unified_msg_origin,
                    original_prompt if plan.integrated else plan.prompt,
                    provider_id=self.rewrite_provider_id,
                    fallback_provider_ids=self.rewrite_fallback_provider_ids,
                    safety_attempt=safety_attempt,
                )
                if rewritten is None:
                    continue
                candidate = replace(plan, prompt=rewritten)
            final_prompt = compose_prompt(
                candidate.prompt, has_image=bool(image_ref), integrated=candidate.integrated,
                keep_layout=keep_layout,
            )
            logger.info(
                f"qiniu-image submit | {self._ctx(event)} style={candidate.style or '无'} "
                f"integrated={candidate.integrated} exception={candidate.style_exception or '无'} "
                f"people_count={candidate.people_count} "
                f"strength={self.style_strength} model={self.client.model} "
                f"quality={self.client.image_config['quality']} size={self.client.image_config['size']} "
                f"has_image={bool(image_ref)} safety_attempt={safety_attempt}"
            )
            logger.debug(
                f"qiniu-image submit | {self._ctx(event)} safety_attempt={safety_attempt} "
                f"style={candidate.style or '无'} prompt={final_prompt!r}"
            )
            try:
                result = await self._generate(event, final_prompt, image_ref)
            except QiniuSafetyError as exc:
                logger.warning(
                    f"qiniu-image safety rejected | {self._ctx(event)} status={exc.status} "
                    f"attempt={safety_attempt}/{SAFETY_REWRITE_LEVELS}"
                )
                continue
            if result[0]:
                self._remember_image_prompt(event, candidate, bool(image_ref), keep_layout, final_prompt)
            return result
        return None, "生成失败喵（所有安全级别均未能生成可用图片）"

    def _remember_image_prompt(
        self, event: AstrMessageEvent, plan: ImagePlan, has_image: bool, keep_layout: bool,
        submitted_prompt: str = "",
    ) -> None:
        self._last_image_prompts[self._prompt_key(event)] = ImageRecord(
            plan=plan, has_image=has_image, keep_layout=keep_layout, updated_at=time.monotonic(),
            submitted_prompt=submitted_prompt or compose_prompt(
                plan.prompt, has_image=has_image, integrated=plan.integrated, keep_layout=keep_layout,
            ),
        )
        if len(self._last_image_prompts) > 100:
            oldest = min(self._last_image_prompts, key=lambda key: self._last_image_prompts[key].updated_at)
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
            return None, _NOT_CONFIGURED
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
        return f"umo={event.unified_msg_origin} mid={mid} gid={event.get_group_id()} uid={event.get_sender_id()}"
