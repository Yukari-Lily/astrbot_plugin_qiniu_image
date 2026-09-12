"""Isolated drawing planning with read-only research tools and reviewable drafts."""

import asyncio
import copy
import json
import re
import time
import uuid

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor

from .subject_reference import SubjectReferences
from .model_json import parse_model_json

SEARCH_TOOLS = frozenset((
    "web_search_baidu", "web_search_tavily", "tavily_extract_web_page",
    "web_search_bocha", "web_search_brave", "web_search_firecrawl",
    "firecrawl_extract_web_page", "web_search_exa", "exa_get_contents", "web_search_anysearch",
))
PLAN_TTL = 1800
PLAN_LIMIT = 100
PLAN_BYTES = 128 * 1024 * 1024

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
未指定具体画风时留给内置风格，不擅自锁定画风。你只规划，不出图，不发送消息。
最终只输出 JSON：{"prompt":"完整绘图方案","summary":"供主模型审核的摘要，写出各主体、采用的特征、
动作构图及画风，指出放弃的特征/不确定性，最多1200字符","image_mode":"none|reference|edit"}。
凡执行过搜索，最终另给 subject_assessments 数组，每个已搜索主体一项：
{"subject":"搜索时的主体名","relation":"consistent|conflict|unknown|unavailable","search_features":"搜到的特征","source_urls":["本次搜索网址"]}。
consistent 表示文字一致，conflict 表示有任何冲突（即使已通过图片纠正也仍填 conflict），unknown 表示搜索前无认识。
搜索返回 Error/无结果时可换查询；仍无可用资料则填 unavailable、空 search_features 和 source_urls，省略不可靠特征并在摘要说明，绝不称“已核实/官方一致”。
conflict 必须先完成 compare_subject_reference 两轮以内的核对；不要用文字选择代替取图。无认识才能填 unknown。
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
            outputs = []
            async for result in FunctionToolExecutor.execute(self.original, context, **kwargs):
                outputs.append(_text_result(result))
            text = "\n".join(outputs)[:24000]
            self.research["calls"].append({"tool": self.name, "arguments": kwargs, "result": text,
                                          "subject": drawing_subject.strip(), "model_features": self.research["known"][name],
                                          "failed": not text.strip() or bool(re.match(r"(?i)^\s*(error:|错误[:：])", text))})
            self.research["urls"].update(re.findall(r'https?://[^\s<>"\\]+', text))
            return text + "\n绘图核对提醒：对照搜索前认识；任何颜色/服装/道具冲突均需 compare_subject_reference，最终填写 subject_assessments。"


class DrawingPlanner:
    def __init__(self, context, provider_id=""):
        self.context, self.provider_id = context, provider_id
        self.plans = {}

    def prune(self):
        now = time.monotonic()
        for key, record in list(self.plans.items()):
            if record["status"] not in ("integrating", "generating") and now - record["created"] >= PLAN_TTL:
                self.plans.pop(key, None)

    def get(self, owner, plan_id):
        self.prune()
        record = self.plans.get(plan_id)
        return record if record and record["owner"] == owner else None

    @staticmethod
    def describe(record, full=False):
        result = {key: record[key] for key in ("plan_id", "status", "summary", "image_mode")}
        result["checks"] = [{"subject": row["subject"], "status": row["status"], "round": row["round"]}
                            for row in record["checks"]]
        result["search_count"] = len(record["research"])
        result["search_failures"] = sum(bool(row.get("failed")) for row in record["research"])
        result["subject_assessments"] = [{"subject": row["subject"], "relation": row["relation"]}
                                         for row in record.get("subject_assessments", [])]
        if record.get("adjusted_after_review"):
            result["notice"] = "上游审核拒绝后使用了安全替代提示词；原摘要仅代表审核前方案，全文以实际执行版本为准。"
        if not record["has_search"] and record["image_mode"] == "none":
            result["limitation"] = "本次没有可用的联网搜索工具；规划模型仅能采用已有可靠信息。"
        if full:
            result.update(prompt=record.get("final_prompt") or record["prompt"],
                          brief=record["brief"], research=record["research"], checks=record["checks"])
            result["style_selection"] = record.get("style_selection", {})
            result["subject_assessments"] = record.get("subject_assessments", [])
        return result

    async def prepare(self, event, owner, brief, *, image_ref=None, vision_ref=None, image_mode="auto",
                      previous_prompt="", search_tools=()):
        provider = self.provider_id or await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
        if not provider:
            raise ValueError("没有可用的绘图规划模型")
        checker = SubjectReferences(self.context, provider_id=provider)
        research = {"calls": [], "urls": set(), "lock": asyncio.Lock(), "known": {}}
        checked_at = {}
        unresolved = set()
        tools = ToolSet()
        if not image_ref:
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
                    if not research["calls"] or len(research["calls"]) <= checked_at.get(name, -1):
                        return json.dumps({"status": "research_required", "instruction": "先重新搜索主体特征和图片来源。"})
                    if not isinstance(source_urls, list) or any(url not in research["urls"] for url in source_urls):
                        return json.dumps({"status": "invalid_sources", "instruction": "仅使用本次真实搜索结果中的完整网址。"})
                    if name not in research["known"]:
                        return json.dumps({"status": "research_required", "instruction": "请先用相同主体名搜索并记录原认识。"})
                    model_features = research["known"][name]
                    result = await checker.compare("plan", event.unified_msg_origin, subject,
                                                   model_features, search_features, source_urls)
                    checked_at[name] = len(research["calls"])
                    if result["status"] in ("confirmed", "fallback"):
                        unresolved.discard(name)
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
        task = json.dumps({"brief": brief, "previous_prompt": previous_prompt,
                           "has_image": bool(image_ref), "image_mode": image_mode,
                           "has_search": bool(search_tools) and not image_ref}, ensure_ascii=False)
        kwargs = dict(chat_provider_id=provider, prompt=task, system_prompt=PLANNING_RULES, contexts=[])
        if image_ref:
            kwargs["image_urls"] = [vision_ref or image_ref]
            response = await asyncio.wait_for(self.context.llm_generate(**kwargs), timeout=90)
        else:
            response = await asyncio.wait_for(self.context.tool_loop_agent(
                event=event, tools=tools, max_steps=20, tool_call_timeout=80, **kwargs), timeout=105)
        if checker.pending("plan") or unresolved:
            raise ValueError("规划模型未完成第二轮核对，请重新准备方案")
        result = parse_model_json(response.completion_text)
        if not isinstance(result, dict):
            raise ValueError("规划模型未返回有效方案")
        if research["calls"]:
            assessments = result.get("subject_assessments")
            if not isinstance(assessments, list):
                raise ValueError("搜索后缺少逐主体的文字一致性判断，请补充核对")
            reviewed = set()
            for row in assessments:
                if not isinstance(row, dict) or not isinstance(row.get("subject"), str):
                    raise ValueError("主体核对记录无效")
                name = row["subject"].strip().casefold()
                relation = row.get("relation")
                urls = row.get("source_urls")
                if (name not in research["known"] or name in reviewed or
                        relation not in ("consistent", "conflict", "unknown", "unavailable") or
                        not isinstance(row.get("search_features"), str) or
                        not isinstance(urls, list) or any(not isinstance(url, str) or url not in research["urls"] for url in urls)):
                    raise ValueError("主体核对记录或搜索来源无效")
                calls = [call for call in research["calls"] if call["subject"].casefold() == name]
                if all(call["failed"] for call in calls) and relation != "unavailable":
                    raise ValueError("搜索没有可用结果，不能声称已核实主体特征")
                if relation == "unavailable" and (row["search_features"] or urls):
                    raise ValueError("搜索不可用时不能编造搜索特征或来源")
                if relation == "unknown" and research["known"][name]:
                    raise ValueError("已有搜索前认识，不能用 unknown 跳过一致性判断")
                if relation == "conflict":
                    state = checker.rounds.get(("plan", name), {})
                    if state.get("status") not in ("confirmed", "fallback"):
                        raise ValueError("文字特征冲突却未完成图片核对，请先核对再准备方案")
                reviewed.add(name)
            if reviewed != set(research["known"]):
                raise ValueError("有搜索主体未完成一致性判断")
        for field, limit in (("prompt", 30000), ("summary", 1200)):
            if not isinstance(result.get(field), str) or not 1 <= len(result[field].strip()) <= limit:
                raise ValueError("规划模型未返回完整方案和简短摘要")
        mode = result.get("image_mode")
        if mode not in (("reference", "edit") if image_ref else ("none",)):
            raise ValueError("规划模型返回了错误的图片模式")
        if image_ref and image_mode != "auto" and mode != image_mode:
            raise ValueError("规划模型改变了指定的图片用途")
        checks = [state["result"] for state in checker.rounds.values() if "result" in state]
        for check in checks:
            if check["status"] == "confirmed" and check["features"] not in result["prompt"]:
                raise ValueError("规划方案遗漏了已核对的特征")
        self.prune()
        size = (len(image_ref or "") + 4 * (len(brief) + len(result["prompt"]) +
                len(result["summary"]) + sum(len(row["result"]) for row in research["calls"])))
        if size > PLAN_BYTES:
            raise ValueError("方案及图片超过缓存大小限制")
        while len(self.plans) >= PLAN_LIMIT or sum(item["size"] for item in self.plans.values()) + size > PLAN_BYTES:
            oldest = next((key for key, item in self.plans.items()
                           if item["status"] not in ("integrating", "generating")), None)
            if oldest is None:
                raise ValueError("绘图任务繁忙，请稍后重试")
            self.plans.pop(oldest)
        plan_id = uuid.uuid4().hex[:16]
        record = dict(result, plan_id=plan_id, owner=owner, brief=brief, image_ref=image_ref,
                      research=research["calls"], checks=checks, has_search=bool(search_tools),
                      created=time.monotonic(), size=size, status="ready")
        self.plans[plan_id] = record
        return record
