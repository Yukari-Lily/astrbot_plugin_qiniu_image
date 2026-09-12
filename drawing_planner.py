"""Isolated drawing planning with read-only research tools and reviewable drafts."""

import asyncio
import copy
import json
import re
import time
import uuid

from astrbot.api import logger
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor

from .subject_reference import SubjectReferences
from .model_json import parse_model_json
from .style_presets import STYLE_PRESETS, find_explicit_presets
from .prompt_integrator import STYLE_HEADER, QUALITY_HEADER, style_preference

SEARCH_TOOLS = frozenset((
    "web_search_baidu", "web_search_tavily", "tavily_extract_web_page",
    "web_search_bocha", "web_search_brave", "web_search_firecrawl",
    "firecrawl_extract_web_page", "web_search_exa", "exa_get_contents", "web_search_anysearch",
))
PLAN_TTL = 1800
PLAN_LIMIT = 100
PLAN_BYTES = 128 * 1024 * 1024
ACTIVE_STATUSES = ("planning", "integrating", "generating")

PLANNING_RULES = """
你是独立绘图规划模型。委托中包含主聊天模型解析的意图、必要人设/指代、硬性要求及创作自由度。
只在授权的自由度内设计主体、动作、服装、场景与构图；不得推翻委托的明确要求。
用户原话中的明确要求才是硬约束；主聊天模型自行补充的角色常识也需要核实，不能因写着“核心人设”就拒绝纠错。
你看不到完整群聊，不猜测未提供的会话事实。方案必须自足。
文生图对每个主体分别判断：①是否理解名字、昵称、指代和用户绘画意图；②是否有把握描述特征。
任一没把握，使用提供的搜索工具先消歧再搜特征；搜索前保留你自己的特征认识，不能用搜索结果冒充。
搜索时 drawing_subject 填主体名，drawing_known_features 填搜索前的认识（包括主聊天模型补充的认识；无认识填空字符串）。
搜索后逐个核对配饰、服装、颜色和道具；例如记忆说红色吉他、搜索说蓝色吉他就是冲突，不能声称一致。
不要假设图片模型认识某主体。把可靠认识写出；原创角色以委托设定为准，不搜索不存在的官方形象。
文字搜索特征与原认识相符，直接用，不取图；没有先验认识时以明确可靠的搜索资料补足，不编造。
只有两者冲突，调用 compare_subject_reference 下载图片，至少两方一致才采用共同特征。
retry 后必须重新搜索并给出新来源再核对一次，最多两轮；fallback 时省略该主体不可靠特征，
仅保留名称和创作方案，让图片模型发挥。多人分别处理，不串用特征。工具必须串行调用。
confirmed 只用返回的共同特征，不能重新带入被否定特征；不得声称未实际执行的搜索或核对。
无搜索工具或搜索不可用时说明限制，省略不可靠的特征；不能编造来源或伪称已核实。
有用户图片时绝不搜索、不做三方核对，以图为准。reference=参考形象并结合内置风格，
edit=局部改动其余保持原图，auto 时按委托判断。不得凭常识覆盖图片，无法看图就报告失败。
修改方案或上一张作品时，提供的 previous_prompt 是完整基稿；只改委托指定部分，保留其他内容。
换风格时删除旧“画面风格”段；其他修改沿用已有风格及兜底段各一份。不要让修改变成重新随机创作。
画风先于自由创作确定：先根据主体、用户明确要求和 styles 目录选择一个相容风格，再设计动作、背景、光影。
从所有相容风格中按题材选择，优先高偏好项；不要一律选目录第一项净色壁纸，不要随机硬套不相容风格。
preference 是按独立主体数量区分的权重：单人/单主体优先错位矩形、诗意窗口等，净色动画壁纸低优先；多人/多主体时净色动画壁纸高优先。
衣服、随身配饰和普通背景不单独算主体；用户明确点名的风格及硬性要求优先于权重，单人也可明确指定净色壁纸。
style_request 是用户明确说出的画风要求；为空时，brief 中主聊天模型添加的“3D/厚涂/电影光影/极致细节”等不是用户指定画风，不得以此排除内置风格。
角色原作采用3D/厚涂不代表本次要沿用原作媒介，保留角色外观即可。auto 下必须从 styles 选一个方向。
选好风格后，方案只写可靠主体特征、动作构图与必要场景；按该风格控制细节、光影和留白，不额外堆叠画质套话或另一种渲染媒介。
styles 提供的风格正文由后续整合器原样追加，不要抄进新方案；你只规划，不出图，不发送消息。
只有用户明确指定外部画风、局部改图或 styles 为空时可用 style_id=none；不能因自己设计的场景不适合就跳过风格，应调整自由设计部分。
preserve_previous_style=true 时沿用 previous_style_id 和原基稿画风，不另选；否则换风格时删除旧风格/兜底段，重新按所选风格规划。
最终只输出 JSON：{"prompt":"完整绘图方案","summary":"用于查询的摘要，写出各主体、采用的特征、
动作构图及画风，指出放弃的特征/不确定性，最多1200字符","image_mode":"none|reference|edit","style_id":"所选内置风格id或none"}。
凡执行过搜索，最终另给 subject_assessments 数组，每个已搜索主体一项：
{"subject":"搜索时的主体名","relation":"consistent|conflict|unknown|unavailable","search_features":"搜到的特征","source_urls":["本次搜索网址"]}。
consistent 表示文字一致，conflict 表示有任何冲突（即使已通过图片纠正也仍填 conflict），unknown 表示搜索前无认识。
搜索返回 Error/无结果时可换查询；仍无可用资料则填 unavailable、空 search_features 和 source_urls，省略不可靠特征并在摘要说明，绝不称“已核实/官方一致”。
conflict 必须先完成 compare_subject_reference 两轮以内的核对；不要用文字选择代替取图。无认识才能填 unknown。
若提供 save_subject_assessment，每个主体文字核实后立即调用它保存结果，不要攒到最后才写。
图片核对成功会自动保存快照。资料足够就立即返回最终方案，不为了补细枝末节继续搜索；最终 subject_assessments 可省略已保存主体。
剩余时间由 preparation_seconds_left 给出，必须给后续整合留余量，不需要耗尽时间。未确认外观省略，保留名称及用户要求。
summary 必须忠实反映完整方案，不能掩盖与委托不符的决定。用户明确要求和原样文字必须保留。
""".strip()


def _text_result(result):
    if isinstance(result, str):
        return result
    return "\n".join(getattr(part, "text", "") for part in getattr(result, "content", []))


class ResearchTool(FunctionTool):
    """Delegate to the exact tool available to this request, recording real evidence."""

    def __init__(self, original, research):
        parameters = copy.deepcopy(original.parameters)
        parameters.setdefault("properties", {}).update({
            "drawing_subject": {"type": "string", "description": "本次搜索的单个绘图主体名"},
            "drawing_known_features": {"type": "string", "description": "搜索前你及委托中的角色认识；没有则为空，不能填搜索后的结论"},
        })
        parameters["required"] = list(parameters.get("required", [])) + ["drawing_subject", "drawing_known_features"]
        super().__init__(name=original.name, description=original.description, parameters=parameters)
        self.original, self.research = original, research

    async def call(self, context, drawing_subject, drawing_known_features, **kwargs):
        # Serialize even if a provider emits parallel tool calls.
        async with self.research["lock"]:
            if not isinstance(drawing_subject, str) or not drawing_subject.strip() or not isinstance(drawing_known_features, str):
                return "请先提供单个主体名和搜索前认识。"
            name = drawing_subject.strip().casefold()
            self.research["known"].setdefault(name, drawing_known_features.strip())
            self.research.get("assessments", {}).pop(name, None)
            update = self.research.get("on_update")
            if update:
                update()
            outputs = []
            async for result in FunctionToolExecutor.execute(self.original, context, **kwargs):
                outputs.append(_text_result(result))
            text = "\n".join(outputs)[:24000]
            self.research["calls"].append({"tool": self.name, "arguments": kwargs, "result": text,
                                          "subject": drawing_subject.strip(), "model_features": self.research["known"][name],
                                          "failed": not text.strip() or bool(re.match(r"(?i)^\s*(error:|错误[:：])", text))})
            self.research["urls"].update(re.findall(r'https?://[^\s<>"\\]+', text))
            if update:
                update()
            return text + "\n绘图核对提醒：对照搜索前认识；任何颜色/服装/道具冲突均需 compare_subject_reference，最终填写 subject_assessments。"


class DrawingPlanner:
    def __init__(self, context, provider_id=""):
        self.context, self.provider_id = context, provider_id
        self.plans = {}

    def prune(self):
        now = time.monotonic()
        for key, record in list(self.plans.items()):
            if record["status"] not in ACTIVE_STATUSES and now - record["created"] >= PLAN_TTL:
                self.plans.pop(key, None)

    def get(self, owner, plan_id):
        self.prune()
        record = self.plans.get(plan_id)
        return record if record and record["owner"] == owner else None

    @staticmethod
    def describe(record, full=False):
        result = {key: record[key] for key in ("plan_id", "status", "summary", "image_mode")}
        result["style_id"] = record["style_id"]
        if "snapshot_revision" in record:
            result["snapshot_revision"] = record["snapshot_revision"]
        if record.get("degraded"):
            result["degraded"] = True
            result["limitation"] = record["degraded_reason"]
        result["checks"] = [{"subject": row["subject"], "status": row["status"], "round": row["round"]}
                            for row in record["checks"]]
        result["search_count"] = len(record["research"])
        result["search_failures"] = sum(bool(row.get("failed")) for row in record["research"])
        result["subject_assessments"] = [{"subject": row["subject"], "relation": row["relation"]}
                                         for row in record.get("subject_assessments", [])]
        if record.get("adjusted_after_review"):
            result["notice"] = "上游审核拒绝后使用了安全替代提示词；原摘要仅代表审核前方案，全文以实际执行版本为准。"
        if not record["has_search"] and record["image_mode"] == "none" and not record.get("degraded"):
            result["limitation"] = "本次没有可用的联网搜索工具；规划模型仅能采用已有可靠信息。"
        if full:
            result.update(prompt=record.get("final_prompt") or record["prompt"],
                          brief=record["brief"], research=record["research"], checks=record["checks"])
            result["style_selection"] = record.get("style_selection", {})
            result["subject_assessments"] = record.get("subject_assessments", [])
        return result

    async def prepare(self, event, owner, brief, *, image_ref=None, vision_ref=None, image_mode="auto",
                      previous_prompt="", search_tools=(), style_mode="auto", style_request="",
                      previous_style_id="none", fallback_state=None, target_record=None, no_research=False):
        provider = self.provider_id or await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        if not provider:
            raise ValueError("没有可用的绘图规划模型")
        checker = SubjectReferences(self.context, provider_id=provider)
        research = {"calls": [], "urls": set(), "lock": asyncio.Lock(), "known": {}, "assessments": {}}
        snapshot_state = fallback_state if fallback_state is not None else {}
        snapshot_state.update(provider=provider, research=research, checker=checker)
        if target_record is not None:
            def update_snapshot():
                if target_record["status"] == "planning":
                    self.snapshot(owner, brief, state=snapshot_state, record=target_record, image_ref=image_ref,
                                  image_mode=image_mode, previous_prompt=previous_prompt, style_mode=style_mode,
                                  style_request=style_request, previous_style_id=previous_style_id)
            research["on_update"] = update_snapshot
        checked_at = {}
        unresolved = set()
        tools = ToolSet()
        if not image_ref and not no_research:
            for tool in search_tools:
                if tool.name in SEARCH_TOOLS and getattr(tool, "active", True):
                    tools.add_tool(ResearchTool(tool, research))

            async def compare(_event, subject, model_features, search_features, source_urls):
                async with research["lock"]:
                    name = subject.strip().casefold()
                    cached = checker.rounds.get(("plan", name))
                    if cached and cached.get("status") in ("confirmed", "fallback"):
                        return json.dumps(cached["result"], ensure_ascii=False)
                    unresolved.add(name)
                    research["assessments"].pop(name, None)
                    if target_record is not None:
                        update_snapshot()
                    if not research["calls"] or len(research["calls"]) <= checked_at.get(name, -1):
                        return json.dumps({"status": "research_required", "instruction": "先重新搜索主体特征和图片来源。"})
                    own_urls = self._subject_urls(research, name)
                    if not isinstance(source_urls, list) or any(not isinstance(url, str) or url not in own_urls for url in source_urls):
                        return json.dumps({"status": "invalid_sources", "instruction": "仅使用本次真实搜索结果中的完整网址。"})
                    if name not in research["known"]:
                        return json.dumps({"status": "research_required", "instruction": "请先用相同主体名搜索并记录原认识。"})
                    model_features = research["known"][name]
                    result = await checker.compare("plan", event.unified_msg_origin, subject,
                                                   model_features, search_features, source_urls)
                    checked_at[name] = len(research["calls"])
                    if result["status"] in ("confirmed", "fallback"):
                        unresolved.discard(name)
                    if target_record is not None:
                        update_snapshot()
                    return json.dumps(result, ensure_ascii=False)

            tools.add_tool(FunctionTool(
                name="compare_subject_reference", description="仅在搜索特征与原有认识冲突时下载图片核对；retry 后重新搜索再调用，最多两轮。",
                parameters={"type": "object", "properties": {
                    "subject": {"type": "string"}, "model_features": {"type": "string"},
                    "search_features": {"type": "string"},
                    "source_urls": {"type": "array", "items": {"type": "string"}}},
                    "required": ["subject", "model_features", "search_features", "source_urls"]},
                handler=compare,
            ))
        if target_record is not None and search_tools and not image_ref and not no_research:
            async def save_assessment(_event, subject, relation, search_features, source_urls):
                async with research["lock"]:
                    if target_record["status"] != "planning":
                        return '{"status":"closed"}'
                    row = dict(subject=subject, relation=relation, search_features=search_features, source_urls=source_urls)
                    previous = None
                    name = None
                    try:
                        name = self._validate_assessment(row, research, checker)
                        if name in unresolved:
                            name = None
                            raise ValueError("该主体的冲突尚未核对完成")
                        previous = research["assessments"].get(name)
                        research["assessments"][name] = row
                        update_snapshot()
                    except ValueError as exc:
                        if name is not None:
                            if previous is None:
                                research["assessments"].pop(name, None)
                            else:
                                research["assessments"][name] = previous
                        return json.dumps({"status": "invalid", "reason": str(exc)}, ensure_ascii=False)
                    return json.dumps({"status": "saved", "revision": target_record["snapshot_revision"],
                                       "instruction": "此主体已保存；资料足够就返回最终方案，不要重复搜索。"}, ensure_ascii=False)

            tools.add_tool(FunctionTool(
                name="save_subject_assessment", description="每核实一个主体立即保存文字判断到可出图快照；只允许真实来源，冲突需先图片核对。",
                parameters={"type": "object", "properties": {
                    "subject": {"type": "string"}, "relation": {"type": "string", "enum": ["consistent", "unknown", "conflict", "unavailable"]},
                    "search_features": {"type": "string"}, "source_urls": {"type": "array", "items": {"type": "string"}}},
                    "required": ["subject", "relation", "search_features", "source_urls"]}, handler=save_assessment))
        preserve_style = bool(previous_prompt) and not style_request
        requested = find_explicit_presets(style_request)
        presets = STYLE_PRESETS if style_mode == "auto" else requested if style_mode == "explicit_only" else ()
        if preserve_style:
            presets = tuple(p for p in STYLE_PRESETS if p.id == previous_style_id)
        catalog = {p.id: {"name": p.name, "suitable_for": p.suitable_for,
                          "avoid_when": p.avoid_when, "preference": style_preference(p),
                          "prompt": p.prompt} for p in presets}
        task = json.dumps({"brief": brief, "previous_prompt": previous_prompt,
                           "styles": catalog, "style_mode": style_mode, "style_request": style_request,
                           "preserve_previous_style": preserve_style, "previous_style_id": previous_style_id,
                           "has_image": bool(image_ref), "image_mode": image_mode,
                           "has_search": bool(search_tools) and not image_ref and not no_research,
                           "preparation_seconds_left": max(0, target_record["deadline"] - time.monotonic()) if target_record else 105}, ensure_ascii=False)
        kwargs = dict(chat_provider_id=provider, prompt=task, system_prompt=PLANNING_RULES, contexts=[])
        if image_ref:
            kwargs["image_urls"] = [vision_ref or image_ref]
            response = await asyncio.wait_for(self.context.llm_generate(**kwargs), timeout=90)
        elif no_research:
            response = await self.context.llm_generate(**kwargs)
        else:
            response = await asyncio.wait_for(self.context.tool_loop_agent(
                event=event, tools=tools, max_steps=20, tool_call_timeout=80, **kwargs), timeout=105)
        if checker.pending("plan") or unresolved:
            raise ValueError("规划模型未完成第二轮核对，请重新准备方案")
        result = parse_model_json(response.completion_text)
        if not isinstance(result, dict):
            raise ValueError("规划模型未返回有效方案")
        if research["calls"]:
            assessments = result.get("subject_assessments", [] if research["assessments"] else None)
            if not isinstance(assessments, list):
                raise ValueError("搜索后缺少逐主体的文字一致性判断，请补充核对")
            merged, reviewed = dict(research["assessments"]), set()
            for row in assessments:
                name = self._validate_assessment(row, research, checker)
                if name in reviewed:
                    raise ValueError("主体核对记录重复")
                merged[name] = row
                reviewed.add(name)
            if set(merged) != set(research["known"]):
                raise ValueError("有搜索主体未完成一致性判断")
            result["subject_assessments"] = list(merged.values())
        else:
            result["subject_assessments"] = []
        for field, limit in (("prompt", 30000), ("summary", 1200)):
            if not isinstance(result.get(field), str) or not 1 <= len(result[field].strip()) <= limit:
                raise ValueError("规划模型未返回完整方案和简短摘要")
        mode = result.get("image_mode")
        if mode not in (("reference", "edit") if image_ref else ("none",)):
            raise ValueError("规划模型返回了错误的图片模式")
        if image_ref and image_mode != "auto" and mode != image_mode:
            raise ValueError("规划模型改变了指定的图片用途")
        style_id = result.get("style_id")
        if not isinstance(style_id, str) or style_id not in (*catalog, "none"):
            raise ValueError("规划模型未选择有效的内置风格，请先选风格再设计画面")
        if preserve_style:
            if style_id != previous_style_id:
                raise ValueError("本次未要求换画风，不能改变原有风格")
        elif style_id == "none" and catalog and mode != "edit" and not (style_request and style_request != "auto"):
            raise ValueError("auto 必须选用内置风格，不能用自行补充的画风跳过")
        elif (not previous_prompt or style_request) and (STYLE_HEADER in result["prompt"] or QUALITY_HEADER in result["prompt"]):
            raise ValueError("新方案不得自行填写风格/兜底段，应由整合器原样追加")
        if mode == "edit" and style_id != "none" and not style_request and not preserve_style:
            raise ValueError("局部修改不得擅自改变原图画风")
        checks = [state["result"] for state in checker.rounds.values() if "result" in state]
        for check in checks:
            if check["status"] == "confirmed" and check["features"] not in result["prompt"]:
                result["prompt"] += f"\n\n{check['subject']}的已核对特征（覆盖前文与之冲突的外观描述）：{check['features']}"
        for assessment in result.get("subject_assessments", []):
            if assessment["relation"] in ("consistent", "unknown") and assessment["search_features"] not in result["prompt"]:
                result["prompt"] += f"\n\n{assessment['subject']}的已核实特征（用户明确改设要求优先）：{assessment['search_features']}"
        return self._store(owner, brief, result, image_ref=image_ref, style_request=style_request,
                           research=research["calls"], checks=checks, has_search=bool(search_tools), record=target_record)

    @staticmethod
    def _subject_urls(research, name):
        return set().union(*(set(re.findall(r'https?://[^\s<>"\\]+', call["result"]))
                             for call in research["calls"] if call["subject"].strip().casefold() == name and not call["failed"]))

    @staticmethod
    def _validate_assessment(row, research, checker):
        if not isinstance(row, dict) or not isinstance(row.get("subject"), str):
            raise ValueError("主体核对记录无效")
        name = row["subject"].strip().casefold()
        relation, urls = row.get("relation"), row.get("source_urls")
        calls = [call for call in research["calls"] if call["subject"].strip().casefold() == name]
        own_urls = DrawingPlanner._subject_urls(research, name)
        if (name not in research["known"] or relation not in ("consistent", "conflict", "unknown", "unavailable")
                or not isinstance(row.get("search_features"), str) or len(row["search_features"]) > 8000 or not isinstance(urls, list)
                or any(not isinstance(url, str) or url not in own_urls for url in urls)):
            raise ValueError("主体核对记录或搜索来源无效")
        if (not calls or all(call["failed"] for call in calls)) and relation != "unavailable":
            raise ValueError("搜索没有可用结果，不能声称已核实主体特征")
        if relation == "unavailable" and (row["search_features"] or urls):
            raise ValueError("搜索不可用时不能编造搜索特征或来源")
        if relation == "unknown" and research["known"][name]:
            raise ValueError("已有搜索前认识，不能用 unknown 跳过一致性判断")
        if relation in ("consistent", "unknown") and (not row["search_features"].strip() or not urls):
            raise ValueError("核实特征必须提供文字和真实来源")
        check = checker.rounds.get(("plan", name), {})
        if (relation == "conflict" and check.get("status") not in ("confirmed", "fallback")) or check.get("status") == "retry":
            raise ValueError("文字特征冲突却未完成图片核对，请先核对再准备方案")
        return name

    def snapshot(self, owner, brief, *, state, image_ref=None, image_mode="auto", previous_prompt="",
                 style_mode="auto", style_request="", previous_style_id="none", record=None):
        """Rebuild from user requirements and committed evidence only, without a model call."""
        research = state.get("research", {"calls": [], "known": {}, "assessments": {}})
        checker = state.get("checker")
        checks = [row["result"] for row in checker.rounds.values() if "result" in row] if checker else []
        assessments = dict(research.get("assessments", {}))
        trusted = {name: row["search_features"] for name, row in assessments.items()
                   if row["relation"] in ("consistent", "unknown")}
        for check in checks:
            name = check["subject"].strip().casefold()
            if check["status"] == "confirmed":
                trusted[name] = check["features"]
                assessments[name] = {"subject": check["subject"], "relation": "confirmed"}
            else:
                trusted.pop(name, None)
        unknown = set(research["known"]) - set(trusted)
        for name in unknown:
            assessments[name] = {"subject": name, "relation": "unverified"}
        preserve_style = bool(previous_prompt) and not style_request
        mode = (image_mode if image_mode != "auto" else "edit") if image_ref else "none"
        explicit = tuple(p for p in find_explicit_presets(style_request)
                         if style_request.strip() in (p.name, p.id, *p.aliases))
        style_id = previous_style_id if preserve_style else "none"
        if not preserve_style and style_mode != "disabled":
            if explicit:
                style_id = explicit[0].id
            elif style_mode == "auto" and mode != "edit" and (not style_request or style_request == "auto"):
                style_id = "clean_anime_wallpaper"
        base = previous_prompt
        if style_request:
            base = re.split(f"{re.escape(STYLE_HEADER)}|{re.escape(QUALITY_HEADER)}", base, maxsplit=1)[0].rstrip()
        prompt = (base + "\n\n本次修改要求（仅修改指定部分，其余沿用基稿）：\n" if base else "绘图委托：\n") + brief
        if style_request and style_request != "auto":
            prompt += "\n用户明确指定的画风：" + style_request
        if image_ref:
            prompt += "\n以用户图片为准，仅按委托修改或参考创作；不要凭文字知识覆盖图中外观。"
        else:
            prompt += "\n角色名称及用户明确设定优先；未核实的外观不作硬约束，不补写猜测的服装、发色或道具。"
        for name, features in trusted.items():
            prompt += f"\n{name}的已核实特征（覆盖冲突的外观描述，用户明确改设要求优先）：{features}"
        if unknown:
            prompt += "\n以下主体外观尚未核实，保留名称和用户意图，不指定存疑外观：" + "、".join(sorted(unknown))
        result = dict(prompt=prompt, summary=("委托：" + brief[:400] + "；已保存特征：" + ("、".join(trusted) or "暂无")
                      + "；未确认：" + ("、".join(sorted(unknown)) or "无新增记录"))[:1200],
                      style_id=style_id, image_mode=mode, subject_assessments=list(assessments.values()),
                      snapshot_revision=(record.get("snapshot_revision", 0) if record else 0) + 1)
        saved = self._store(owner, brief, result, image_ref=image_ref, style_request=style_request,
                            research=list(research["calls"]), checks=checks, has_search=bool(research["calls"]), record=record)
        if record is not None:
            logger.debug(f"qiniu-image snapshot | plan_id={saved['plan_id']} revision={saved['snapshot_revision']} subjects={len(trusted)}")
        return saved

    def _store(self, owner, brief, result, *, image_ref, style_request, research, checks, has_search, record=None):
        if len(result["prompt"]) > 32000:
            raise ValueError("完整绘图提示词过长，请缩短委托")
        self.prune()
        size = (len(image_ref or "") + 4 * (len(brief) + len(result["prompt"]) +
                len(result["summary"]) + sum(len(row["result"]) for row in research)))
        if size > PLAN_BYTES:
            raise ValueError("方案及图片超过缓存大小限制")
        old_size = record["size"] if record else 0
        while len(self.plans) - bool(record) >= PLAN_LIMIT or sum(item["size"] for item in self.plans.values()) - old_size + size > PLAN_BYTES:
            oldest = next((key for key, item in self.plans.items()
                           if item is not record and item["status"] not in ACTIVE_STATUSES), None)
            if oldest is None:
                raise ValueError("绘图任务繁忙，请稍后重试")
            self.plans.pop(oldest)
        plan_id = record["plan_id"] if record else uuid.uuid4().hex[:16]
        updated = dict(result, plan_id=plan_id, owner=owner, brief=brief, image_ref=image_ref,
                      style_request=style_request,
                      research=research, checks=checks, has_search=has_search,
                      created=record["created"] if record else time.monotonic(), size=size,
                      status=record["status"] if record else "ready")
        if record is None:
            record = updated
        else:
            record.update(updated)
        self.plans[plan_id] = record
        return record
