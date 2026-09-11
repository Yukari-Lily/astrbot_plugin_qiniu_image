"""七牛 AI 绘图 / 改图插件。

出图链路只有一条，且外观不经过任何模型转述：

    draw_image(prompt, characters)
        ↓ 插件按名字从外观缓存取记录（模型不传句柄）
        ↓ 优化模型只写场景/动作/表情/构图
        ↓ draw_task.assemble 把外观块逐字写进最终提示词
        ↓ 出图（被拒则五级安全回退，回退后外观仍然在位）

冷门角色的外观由 prepare_character_reference 取得，参考图**只用来产出文字**，
永远不进绘图接口；只有用户自己发的图片才会作为图片输入送出去。
"""

import asyncio
import copy
import json
import re
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register

from . import appearance
from .character_ref import CharacterReference, SafeFetcher
from .draw_task import DRAW_SCHEMA, normalize_characters
from .message_utils import resolve_input_images
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
from .style_presets import STYLE_MODES, STYLE_STRENGTHS, STYLE_PRESETS, style_catalog_text

DEDUP_TTL_SECONDS = 20
RECENT_PROMPT_LIMIT = 5

#: 注入到系统提示词的作图约定。上限约 400 字：每轮对话都要付这份开销，
#: 而"怎么消歧"属于工具描述该讲的事，不在这里重复。
_DRAWING_RULES = (
    "绘图前先结合用户要求、完整会话和你的人设形成方案；你负责创作，优化器只整理核对。"
    "但写进 prompt 的方案必须自足：图片模型看不到这段对话，人名、作品、外观版本、"
    "场景、动作和服饰都要写全，不能出现“刚才那张”“上一个”“她”这类指代。"
    "不确定某个角色的外观时（冷门角色、原创人物、具体形象版本），先联网搜索它的资料页，"
    "把网址交给 prepare_character_reference 取得准确外观，再调用 draw_image；必须串行，"
    "不要并行，也不要为此追问用户。"
    "用户本轮没有发图时，不得向图片模型传任何图片。"
    "用户发了图时按 input:1 起编号；只有用户明确要求改动的地方才改，其余保持原样。"
    "列出风格时调用 list_image_styles，只列标题。后台绘图返回 accepted 后不要重复调用。"
    "身份或外观依据不足时简短说明无法可靠生成，不询问补图。"
)

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

        # 搜索沿用 AstrBot 已有的联网能力：主聊天模型搜到来源网页后把网址交给
        # 插件。插件自己不需要搜索密钥，只负责下载、核对与抽取。
        self.fetcher = SafeFetcher()
        self.references = CharacterReference(
            context,
            {"provider_id": self.rewrite_provider_id,
             "fallback_provider_ids": self.rewrite_fallback_provider_ids},
            fetcher=self.fetcher,
        )

        self._recent_msg: Dict[str, float] = {}
        self._tasks: Set[asyncio.Task] = set()
        self._last_prompts: "OrderedDict[str, List[dict]]" = OrderedDict()

        if not self.client.configured:
            logger.warning("qiniu-image: 未配置 api_key，插件已加载但无法出图")

    async def terminate(self):
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.fetcher.close()
        await self.client.close()

    # ------------------------------------------------------------------
    # 系统提示词注入
    # ------------------------------------------------------------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """注入有限的作图约定与风格词表；不改写人设本身。"""
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
        req.func_tool = toolset

        catalog = "；".join(p.name + "（" + "、".join(p.aliases) + "）" for p in STYLE_PRESETS)
        block = f"{_DRAWING_RULES}\n内置风格词表：{catalog}"
        if self.style_mode == "auto":
            preferred = "、".join(p.name for p in STYLE_PRESETS if p.auto_preference)
            block += f"\n自动风格：未指定画风时优先选一个相容的内置偏好画风，偏好为：{preferred}。"
        recent = self._last_prompts.get(self._owner(event)) or []
        if recent:
            rows = "；".join(f"{row['id']}（{row['style']}）" for row in reversed(recent))
            block += f"\n近期作品：{rows}。需要原文时调用 get_last_image_prompt。"

        start, end = "<qiniu_drawing_context>", "</qiniu_drawing_context>"
        original = re.sub(re.escape(start) + r".*?" + re.escape(end), "", req.system_prompt or "", flags=re.S).rstrip()
        req.system_prompt = original + "\n\n" + start + "\n" + block + "\n" + end

    # ------------------------------------------------------------------
    # 触发词直出（不经过聊天模型）
    # ------------------------------------------------------------------

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
        image_b64, error_text = await self._draw(event, prompt, (), {})
        if image_b64:
            components = [Comp.Image.fromBase64(image_b64)]
            if error_text:
                components.append(Comp.Plain(error_text))
            yield event.chain_result(components)
        else:
            yield event.plain_result(error_text or "生成失败喵")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @filter.llm_tool(name="draw_image")
    async def draw_image(self, event: AstrMessageEvent, prompt: str,
                         characters: Optional[list] = None):
        """按完整方案绘图，后台完成后自动发图，不要重复调用。

        你负责创作：结合用户本轮要求、Bot 人设、完整会话与必要考据写出完整方案。
        不确定某个角色长什么样时，先调用 prepare_character_reference，再调用本工具；
        人物的发型、发色、瞳色、服装等外观由插件自动注入，不要在 prompt 里重复描述，
        也不要自己编造，写了会被丢弃。多人必须逐人给出位置并逐人描述。

        Args:
            prompt(string): 结合用户要求、Bot 人设、完整会话和必要考据形成的完整方案，含动作、表情、场景、构图、光照与画风。必须自足——图片模型看不到这段对话，人名、场景、动作要写全，不要出现“刚才那张”“她”这类指代。
            characters(array[object]): 画面中的具名人物。每项含 name（准确名称）、work（所属作品）、version（形象版本）、position（从观看者视角看的位置，多人必填）。纯风景或不含具名角色时可省略。
        """
        if not self.client.configured:
            return "生成失败喵（未配置 api_key）"
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32000:
            return "生成失败喵（绘图方案为空或过长）"
        try:
            people = normalize_characters(characters)
        except ValueError as exc:
            return f"绘图任务未提交：{exc}。不要猜测或询问补图。"

        # 外观在提交时同步取出：prepare_character_reference 刚写进缓存，这里必中；
        # 缓存在后台任务开始前就定型，也不受后续请求影响。
        # 缓存项是含来源信息的整包，这里只取 appearance 本身；只注入通过覆盖度
        # 校验的记录，宁可不用也不能把半真半假的外观锚点写进提示词。
        appearances: Dict[str, Any] = {}
        for char in people:
            entry = self.references.lookup(char["name"], char["work"], char["version"])
            record = entry.get("appearance") if isinstance(entry, dict) else None
            if appearance.is_usable(record):
                appearances[char["id"]] = record
        unknown = [char["name"] for char in people if char["id"] not in appearances]
        if unknown:
            logger.info(f"qiniu-image: 以下人物没有可用外观记录，按纯文字生成｜{'、'.join(unknown)}")

        worker = asyncio.create_task(self._draw_and_push(event, prompt.strip(), people, appearances))
        self._tasks.add(worker)
        worker.add_done_callback(self._tasks.discard)
        return json.dumps({"status": "accepted", "message": "后台生成中，完成后自动发送；不要重复调用。"},
                          ensure_ascii=False)

    @filter.llm_tool(name="prepare_character_reference")
    async def prepare_character_reference(self, event: AstrMessageEvent, name: str,
                                          source_urls: Optional[list] = None,
                                          work: str = "", version: str = ""):
        """查询角色的准确外观：你负责搜索，插件负责下载、核对与提取。

        当你不确定某个角色长什么样，或用户提到的是冷门角色、原创人物、某个具体形象
        版本时：先用你的联网搜索找到该角色的资料页或图片地址，再把搜索结果网址填入
        source_urls 调用本工具。下载、选图、核对与外观提取全部由插件完成。
        结果会把外观写入插件缓存，随后 draw_image 会自动使用，你不需要复述外观。
        本工具不返回网址。查不到时会明确告知，此时不要编造外观。

        Args:
            name(string): 角色的准确名称，例如「小秦」。
            source_urls(array[string]): 1 至 4 个搜索结果网页或图片直链；不要填你无法访问的地址。
            work(string): 所属作品、系列或身份，用于区分同名角色，可省略。
            version(string): 具体形象版本，例如某代立绘、某种服装，可省略。
        """
        urls = [url for url in (source_urls or []) if isinstance(url, str)]
        try:
            result = await self.references.prepare(
                event.unified_msg_origin, name, work, version, urls
            )
        except (ValueError, OSError, asyncio.TimeoutError) as exc:
            logger.warning(f"qiniu-image: 角色外观查询失败（{type(exc).__name__}: {exc}）")
            result = {"status": "unavailable", "reason": "来源或图片不可用，未完成外观核对",
                      "instruction": "不要编造外观；有可靠文字依据时可继续，否则简短说明无法可靠生成。"}
        except Exception as exc:
            logger.error(
                f"qiniu-image: 角色外观查询异常｜error_type={type(exc).__name__}\n"
                + "".join(traceback.format_tb(exc.__traceback__))
            )
            result = {"status": "unavailable", "reason": "外观查询异常",
                      "instruction": "不要编造外观；有可靠文字依据时可继续，否则简短说明无法可靠生成。"}
        return json.dumps(result, ensure_ascii=False)

    @filter.llm_tool(name="list_image_styles")
    async def list_image_styles(self, event: AstrMessageEvent):
        """列出实际内置风格。原样列出中文标题，不添加解释、适用说明或模式配置。"""
        return style_catalog_text(concise=True)

    @filter.llm_tool(name="get_last_image_prompt")
    async def get_last_image_prompt(self, event: AstrMessageEvent, generation_id: str = "latest"):
        """读取当前会话当前用户的近期作品原文；缺省读取最近一次，不用于其他用户的作品。

        Args:
            generation_id(string): 指定作品标识或 latest，可省略。
        """
        rows = self._last_prompts.get(self._owner(event)) or []
        if not rows:
            return "当前用户没有可读取的近期作品。不能据此猜测旧作品。"
        if generation_id in (None, "", "latest"):
            record = rows[-1]
        else:
            record = next((row for row in rows if row["id"] == generation_id), None)
            if record is None:
                return "当前用户没有可读取的对应作品。不能据此猜测旧作品。"
        return json.dumps({"generation_id": record["id"], "prompt": record["prompt"],
                           "style": record["style"], "characters": record["characters"]},
                          ensure_ascii=False)

    # ------------------------------------------------------------------
    # 出图
    # ------------------------------------------------------------------

    @staticmethod
    def _owner(event: AstrMessageEvent) -> str:
        return f"{event.unified_msg_origin}:{event.get_sender_id()}"

    def _remember(self, event: AstrMessageEvent, prompt: str, style: str,
                  characters: Sequence[Dict[str, str]]) -> None:
        owner = self._owner(event)
        rows = self._last_prompts.pop(owner, [])
        rows.append({
            "id": f"g{int(time.time() * 1000) % 10 ** 9}",
            "prompt": prompt[:4000],
            "style": style or "未使用内置风格",
            "characters": [char["name"] for char in characters],
        })
        self._last_prompts[owner] = rows[-RECENT_PROMPT_LIMIT:]
        while len(self._last_prompts) > 200:
            self._last_prompts.popitem(last=False)

    async def _draw_and_push(self, event: AstrMessageEvent, prompt: str,
                             characters: Sequence[Dict[str, str]],
                             appearances: Dict[str, Any]) -> None:
        """后台出图并推送。"""
        try:
            image_b64, error_text = await self._draw(event, prompt, characters, appearances)
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

    async def _draw(self, event: AstrMessageEvent,
                    user_prompt: str,
                    characters: Sequence[Dict[str, str]],
                    appearances: Dict[str, Any]):
        """一次完整出图：解析输入图 → 优化场景 → 拼装外观 → 出图（含安全回退）。"""
        try:
            image_refs = await resolve_input_images(self.context, event, self.client)
        except ValueError as exc:
            return None, f"生成失败喵（{exc}）"

        metadata: Dict[str, Any] = {}
        text = await rewrite(
            self.context,
            event.unified_msg_origin,
            user_prompt,
            has_image=bool(image_refs),
            characters=characters,
            appearances=appearances,
            style_mode=self.style_mode,
            style_strength=self.style_strength,
            provider_id=self.rewrite_provider_id,
            fallback_provider_ids=self.rewrite_fallback_provider_ids,
            image_urls=image_refs,
            result_metadata=metadata,
        )
        if not text:
            return None, "生成失败喵（提示词优化失败）"

        for attempt in range(1, SAFETY_REWRITE_LEVELS + 1):
            try:
                image_b64, error_text = await self._generate(event, text, image_refs)
            except QiniuSafetyError:
                if attempt == SAFETY_REWRITE_LEVELS:
                    return None, "生成失败喵（内容审核未通过）"
                logger.warning(f"qiniu-image: 被审核拒绝，进入安全回退 {attempt}/{SAFETY_REWRITE_LEVELS}")
                replacement = await rewrite_for_safety(
                    self.context,
                    event.unified_msg_origin,
                    text,
                    provider_id=self.rewrite_provider_id,
                    fallback_provider_ids=self.rewrite_fallback_provider_ids,
                    safety_attempt=attempt + 1,
                    characters=characters,
                    appearances=appearances,
                )
                if not replacement:
                    return None, "生成失败喵（内容审核未通过）"
                text = replacement
                continue
            if image_b64:
                self._remember(event, text, str(metadata.get("style", "")), characters)
            return image_b64, error_text
        return None, "生成失败喵"

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
