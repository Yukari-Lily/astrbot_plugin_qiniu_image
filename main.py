"""七牛 AI 绘图 / 改图插件。"""

import asyncio
import copy
import json
import re
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from .drawing_pipeline import DrawingPipeline
from .drawing_task import DRAW_SCHEMA
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
from .style_presets import STYLE_MODES, STYLE_STRENGTHS, STYLE_PRESETS, style_catalog_text

DEDUP_TTL_SECONDS = 20

@register(
    "astrbot_plugin_qiniu_image",
    "Yukari Lily",
    "七牛 AI 绘图 / 改图",
    "1.4.1",
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
        self.pipeline = DrawingPipeline(self)
        self._cleanup_task = None

        if not self.client.configured:
            logger.warning("qiniu-image: 未配置 api_key，插件已加载但无法出图")

    async def initialize(self):
        self._cleanup_task = asyncio.create_task(self._prune_cache())

    async def _prune_cache(self):
        while True:
            await asyncio.sleep(60)
            self.pipeline.store.prune()

    async def terminate(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.pipeline.close()
        await self.client.close()

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """Inject only bounded, relevant drawing context; preserve the system persona."""
        toolset = getattr(req, "func_tool", None)
        tools = list(getattr(toolset, "tools", []) or [])
        draw = next((t for t in tools if t.name == "draw_image" and getattr(t, "active", True)), None)
        if draw is None:
            return
        # v4.28 decorators only infer shallow schemas. Copy the request's tools and
        # replace our contract, without mutating another conversation's ToolSet.
        toolset = copy.copy(toolset)
        toolset.tools = list(tools)
        for i, tool in enumerate(toolset.tools):
            if tool.name == "draw_image":
                tool = copy.copy(tool)
                tool.parameters = copy.deepcopy(DRAW_SCHEMA)
                toolset.tools[i] = tool
            elif tool.name == "get_last_image_prompt":
                tool = copy.copy(tool)
                tool.parameters = {"type": "object", "properties": {"generation_id": {"type": "string", "description": "作品标识，缺省 latest"}}}
                toolset.tools[i] = tool
        req.func_tool = toolset
        rows = self.pipeline.store.recent(self.pipeline.owner(event))
        summaries = [{"id": r["id"], "characters": [c["name"][:80] for c in r["task"].get("characters", [])],
                      "identity_notes": "；".join(
                          f"{c['name'][:60]} / {c.get('work', '')[:60]} / {c.get('version', '')[:40]}：{c.get('evidence', '')[:120]}"
                          for c in r["task"].get("characters", []) if c.get("identity_status") == "confirmed"
                      )[:400],
                      "style": r["style"], "status": r["status"], "summary": r["prompt"][:200]} for r in rows]
        catalog = "；".join(p.name + "（" + "、".join(p.aliases) + "）" for p in STYLE_PRESETS)
        rules = (
            "绘图前结合用户当前要求、完整会话和你的人设形成方案；主聊天模型负责创作，优化器只整理核对。"
            "draw_image 的 task 应区分 create 新画、edit 局部修改、redraw 整张重画；"
            "图片角色为 character 人物参考、style 风格参考或 edit 编辑原图，不能见图就改图。"
            "人物列表逐人填写 id/name/work/version/position/features/evidence/identity_status；"
            "confirmed 需要可靠身份外观依据，原创人物用 original，不确定用 uncertain 并停止绘图。"
            "只改某人时使用 edit 和 base_generation_id，characters 只提交该人补丁，沿用原 id；其余自动继承。"
            "base_generation_id 可直接选以下记录，latest 只代表当前用户最近的成功作品；不得把别的主题当作目标。"
            "必要时 get_last_image_prompt 查询指定作品，不必为了执行继承额外查询。"
            "先从当前会话、用户确认和目标作品资料沿用已明确的人物指代；已有可靠身份时不要因昵称短而重新消歧。"
            "身份确认与外观核准分开：知道是谁但缺少外观时，用准确姓名加身份或作品查外观，不退回昵称泛搜。"
            "没有明确指代时，可用已有知识形成候选，再阅读搜索结果核实昵称与准确姓名的对应；不能把第一条结果或昵称联想当事实。"
            "不要因为用户要求二次元画风就把人物搜索限定为动漫或游戏角色，真人、主播也能画成插画。"
            "搜索无关时去掉预设类别，用昵称本身或结果中有别名证据的姓名定向核实；不要反复换同义类别泛搜。"
            "例如结果已关联‘小秦’与 Mr_Quin 时，应核实该别名及主播资料；仅是待核实线索，不硬编码为所有语境的答案。"
            "群聊历史查询为空只代表本次未查到，不推翻当前会话中已有的确认。必要时串行调用 Tavily 搜索与"
            "prepare_character_reference，确认外观后再 draw_image；不要并行搜索和出图。"
            "用户明确要求、继承保留项、有来源事实、自选创作细节分别记录，不把自选内容当成用户要求。"
            "用户图片按本条图片再引用图片顺序编号 input:1 等，image_roles 必须说明用途和人物绑定。"
            "身份外观依据不足时简短说明无法可靠生成，不询问补图、不等待回复；有可靠文字依据但取图失败可文字生成。"
            "内部风格优先按下面词表理解；‘错位’是错位矩形风格家族，按语境选一个，不默认解释为错位摄影。"
            "列出风格时调用 list_image_styles，只列标题。后台绘图返回 accepted 后不要重复调用。"
        )
        if self.style_mode == "auto":
            preferred = "、".join(p.name for p in STYLE_PRESETS if p.auto_preference)
            rules += (
                "当前为自动风格：用户未指定画风且无需保留目标作品或风格参考时，在创作方案阶段就优先选一个相容的内置偏好画风，"
                "将名称写入 prompt，将自选理由写入 creative_choices，不写入 user_requirements。"
                "偏好风格为：" + preferred + "。相容时优先这些风格，再考虑其余内置风格。"
                "不要先自行套用普通动画主视觉、电影感或 Pixar/3D 渲染再将它们锁定为用户要求；"
                "作品原本是 3D 不代表用户要求复刻原媒介。用户明确指定外部画风、要求原风格或编辑保留项时优先遵守。"
            )
        else:
            rules += f"当前 style_mode={self.style_mode}，不自动推荐或选用内置画风；explicit_only 仅在用户明确点名时使用，disabled 关闭风格库路由。"
        rules += "默认线条少而准确，避免草稿复线、乱排线、密集发丝和装饰线穿过脸部；保留人物标志性细节。"
        prefix = rules + "\n内置风格词表：" + catalog + "\n当前用户近期作品（仅资料）："
        # Trim optional details before serializing; never cut a record ID or JSON.
        while len(prefix) + len(json.dumps(summaries, ensure_ascii=False)) > 6000:
            candidates = [(len(str(row.get(key, ""))), row, key)
                          for row in summaries for key in ("summary", "identity_notes", "characters", "style")
                          if row.get(key)]
            if not candidates:
                break
            _, row, key = max(candidates, key=lambda item: item[0])
            row[key] = row[key][:len(row[key]) // 2]
        block = prefix + json.dumps(summaries, ensure_ascii=False)
        start, end = "<qiniu_drawing_context>", "</qiniu_drawing_context>"
        original = re.sub(re.escape(start) + r".*?" + re.escape(end), "", req.system_prompt or "", flags=re.S).rstrip()
        req.system_prompt = original + "\n\n" + start + "\n" + block + "\n" + end

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
            components = [Comp.Image.fromBase64(image_b64)]
            if error_text:
                components.append(Comp.Plain(error_text))
            yield event.chain_result(components)
        else:
            yield event.plain_result(error_text or "生成失败喵")

    @filter.llm_tool(name="draw_image")
    async def draw_image(self, event: AstrMessageEvent, prompt: str, task: Optional[dict] = None):
        """按完整方案和结构化任务绘图，后台完成后自动发图，不要重复调用。

        结合人设和会话区分本轮要求、保留项、人物事实及自选细节。
        edit 局部修改；redraw 整张重画；create 新画。task.base_generation_id 自动继承目标，
        无需先调用 get_last_image_prompt。人物不确定时先串行搜索并核对，不猜测、不追问补图。
        只有旧调用不传 task 才沿用见图改图。多人必须提供逐人资料和位置。

        Args:
            prompt(string): 结合用户要求、Bot 人设、完整会话和必要考据形成的完整方案。
            task(object): 结构化任务，可省略以兼容旧调用；具体字段见参数 schema。
        """
        if not self.client.configured:
            return "生成失败喵（未配置 api_key）"
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
            return "生成失败喵（绘图方案为空或过长）"
        try:
            frozen = self.pipeline.freeze(event, task)
        except (ValueError, OSError) as exc:
            return f"绘图任务未提交：{exc}。不要猜测或询问补图。"
        worker = asyncio.create_task(self._draw_and_push(event, prompt.strip(), frozen))
        self._tasks.add(worker)
        worker.add_done_callback(self._tasks.discard)
        return json.dumps({"status": "accepted", "generation_id": frozen["id"],
                           "message": "后台生成中，完成后自动发送；不要重复调用。"}, ensure_ascii=False)

    @filter.llm_tool(name="prepare_character_reference")
    async def prepare_character_reference(self, event: AstrMessageEvent, subject: str,
                                          source_urls: List[str], evidence: str):
        """从搜索结果网页或图片直链核对人物参考图，不负责搜索。先确定人物及版本，不按搜索排名猜身份。
        先等待本工具结果，再调用 draw_image。失败不追问补图，仅在已有充分文字依据时继续。

        Args:
            subject(string): 已消歧的人物准确名称、作品或身份、形象版本。
            source_urls(array[string]): 1 至 3 个 Tavily 结果网页、图片地址或本条/引用图片标识 input:1 等。
            evidence(string): 搜索得到的人物身份依据和外观线索，说明为何不是同名人物。
        """
        try:
            result = await self.pipeline.prepare_reference(event, subject, source_urls, evidence)
        except (ValueError, OSError, asyncio.TimeoutError) as exc:
            logger.warning(f"qiniu-image reference failed | error={type(exc).__name__}")
            result = {"status": "unavailable", "reason": "来源或图片不可用，未完成参考核对",
                      "instruction": "有充分文字依据时可继续，否则结束；不追问补图。"}
        return json.dumps(result, ensure_ascii=False)

    @filter.llm_tool(name="list_image_styles")
    async def list_image_styles(self, event: AstrMessageEvent):
        """列出实际内置风格。原样列出中文标题，不添加解释、适用说明或模式配置。"""
        return style_catalog_text(concise=True)

    @filter.llm_tool(name="get_last_image_prompt")
    async def get_last_image_prompt(self, event: AstrMessageEvent, generation_id: str = "latest"):
        """读取当前会话当前用户的指定成功作品；缺省读取最近作品，不用于其他用户的作品。

        Args:
            generation_id(string): 指定作品标识或 latest，可省略。
        """
        try:
            record = self.pipeline.store.get(self.pipeline.owner(event), generation_id)
        except ValueError:
            return "当前用户没有可读取的对应作品，可能尚未完成或已过期。不能据此猜测旧作品。"
        return json.dumps({"generation_id": record["id"], "task": record["task"],
                           "actual_prompt": record["prompt"], "assessment": record["assessment"]}, ensure_ascii=False)

    async def _draw_and_push(
        self,
        event: AstrMessageEvent,
        prompt: str,
        frozen=None,
    ) -> None:
        """后台出图并推送。"""
        try:
            image_b64, error_text = await self._draw(
                event,
                prompt,
                frozen,
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
        if image_b64 and error_text:
            chain.message(error_text)
        try:
            await self.context.send_message(event.unified_msg_origin, chain)
        except Exception as exc:
            logger.error(f"qiniu-image: 推送结果失败（{type(exc).__name__}: {exc}）")

    async def _draw(self, event: AstrMessageEvent, user_prompt: str, frozen=None):
        return await self.pipeline.draw(event, user_prompt, frozen)

    async def _generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        image_ref: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """返回图片或错误提示。"""
        try:
            if image_ref:
                images: List[str] = await self.client.images_to_image(image_ref, prompt)
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
