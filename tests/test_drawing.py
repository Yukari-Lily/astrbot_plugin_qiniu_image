"""Offline behavior tests: providers, image service and web downloads are mocked."""

import asyncio
import importlib
import json
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("drawing_plugin")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package


class Image:
    def __init__(self, url="", file=""):
        self.url, self.file = url, file

    @classmethod
    def fromBase64(cls, data):
        return cls(file="base64://" + data)


class Reply:
    pass


class Chain:
    def base64_image(self, data):
        self.image = data
        return self

    def message(self, text):
        self.text = text
        return self


class Star:
    def __init__(self, context):
        self.context = context


def decorator(*args, **kwargs):
    return lambda fn: fn


for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star", "astrbot.api.message_components"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["astrbot.api"].logger = logging.getLogger("drawing_tests")
sys.modules["astrbot.api"].AstrBotConfig = dict
sys.modules["astrbot.api.event"].AstrMessageEvent = object
sys.modules["astrbot.api.event"].MessageChain = Chain
sys.modules["astrbot.api.event"].filter = types.SimpleNamespace(
    llm_tool=decorator, on_llm_request=decorator, on_decorating_result=decorator, event_message_type=decorator,
    EventMessageType=types.SimpleNamespace(ALL="all"),
    PlatformAdapterType=types.SimpleNamespace(AIOCQHTTP="aiocqhttp"))
sys.modules["astrbot.api.star"].Star = Star
sys.modules["astrbot.api.star"].Context = object
sys.modules["astrbot.api.star"].register = decorator
sys.modules["astrbot.api.message_components"].Image = Image
sys.modules["astrbot.api.message_components"].Reply = Reply


class FunctionTool:
    def __init__(self, name, description, parameters, handler=None):
        self.name, self.description, self.parameters = name, description, parameters
        self.handler, self.active = handler, True


class ToolSet:
    def __init__(self, tools=()):
        self.tools = list(tools)

    def add_tool(self, tool):
        self.tools.append(tool)


class Executor:
    @staticmethod
    async def execute(tool, context, **kwargs):
        yield await tool.call(context, **kwargs)


for name in ("astrbot.core", "astrbot.core.agent", "astrbot.core.agent.tool", "astrbot.core.astr_agent_tool_exec"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["astrbot.core.agent.tool"].FunctionTool = FunctionTool
sys.modules["astrbot.core.agent.tool"].ToolSet = ToolSet
sys.modules["astrbot.core.astr_agent_tool_exec"].FunctionToolExecutor = Executor

main = importlib.import_module("drawing_plugin.main")
planning = importlib.import_module("drawing_plugin.drawing_planner")
integrator = importlib.import_module("drawing_plugin.prompt_integrator")
references = importlib.import_module("drawing_plugin.subject_reference")


def response(value):
    if isinstance(value, dict) and "prompt" in value and "summary" in value:
        value = dict(value)
        value.setdefault("style_id", "none" if value.get("image_mode") == "edit" else "clean_anime_wallpaper")
    return types.SimpleNamespace(completion_text=json.dumps(value, ensure_ascii=False))


def selection(mode="none", style="clean_anime_wallpaper", quality=None):
    return {"image_mode": mode, "style_id": style,
            "style_parts": [1] if style != "none" else [],
            "quality_parts": [0] if quality is None else quality,
            "none_reason": ("preserve_image" if mode == "edit" else "external_style") if style == "none" else ""}


def context(value=None):
    return types.SimpleNamespace(
        get_current_chat_provider_id=AsyncMock(return_value="main-model"),
        llm_generate=AsyncMock(return_value=response(value or selection())),
        tool_loop_agent=AsyncMock(return_value=response({"prompt": "银发红瞳的甲坐在海边", "summary": "甲，银发红瞳，海边坐姿；自动画风", "image_mode": "none"})),
        send_message=AsyncMock())


class Event:
    def __init__(self, *, user="alice", mid="1", images=(), text="画一只猫"):
        self.unified_msg_origin = "group"
        self.user, self.images, self.message_str = user, images, text
        self.message_obj = types.SimpleNamespace(message_id=mid)
        self.stopped = False
        self.extras = {}
        self.result = None

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def get_result(self):
        return self.result

    def get_sender_id(self):
        return self.user

    def get_group_id(self):
        return self.unified_msg_origin

    def get_messages(self):
        return self.images

    def get_platform_name(self):
        return "test"

    def stop_event(self):
        self.stopped = True

    def plain_result(self, text):
        return text

    def chain_result(self, parts):
        return parts


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_logged_hollow_wallpaper_selection_is_rejected_then_core_is_kept(self):
        ctx = context()
        hollow = selection()
        hollow["style_parts"] = [0, 2]  # Observed in the real log: purpose and ratio only.
        ctx.llm_generate.side_effect = [response(hollow), response(selection())]
        result = await integrator.integrate(ctx, "g", "喜多背着吉他招手", has_image=False,
                                            planned_style_id="clean_anime_wallpaper")
        self.assertIn(integrator.STYLE_PARTS["clean_anime_wallpaper"][1], result)
        self.assertEqual(ctx.llm_generate.await_count, 2)

    async def test_integrator_cannot_drop_or_replace_reviewed_style(self):
        for replacement in ("none", "clean_anime_wallpaper"):
            ctx = context(selection(style=replacement))
            self.assertIsNone(await integrator.integrate(ctx, "g", "奶龙开心地挥手", has_image=False,
                                                        planned_style_id="window_overlay_poetic"))
        chosen = selection(style="window_overlay_poetic")
        chosen["style_parts"] = [0, 1]
        ctx = context(chosen)
        prompt = "奶龙开心地挥手"
        result = await integrator.integrate(ctx, "g", prompt, has_image=False,
                                            planned_style_id="window_overlay_poetic")
        self.assertTrue(result.startswith(prompt + "\n\n"))
        self.assertIn(integrator.STYLE_PARTS["window_overlay_poetic"][0], result)
        sent = json.loads(ctx.llm_generate.call_args.kwargs["prompt"])
        self.assertEqual(list(sent["styles"]), ["window_overlay_poetic"])

    async def test_reviewed_style_can_be_reused_without_duplicate_sections(self):
        original = await integrator.integrate(context(), "g", "三个人合影", has_image=False,
                                              planned_style_id="clean_anime_wallpaper")
        choice = selection(style="none", quality=[])
        choice["none_reason"] = "existing_style"
        changed = original.replace("三个人合影", "三个人坐着合影")
        result = await integrator.integrate(context(choice), "g", changed, has_image=False,
                                            planned_style_id="clean_anime_wallpaper")
        self.assertEqual(result, changed)

    async def test_reviewed_explicit_edit_style_does_not_need_name_in_image_prompt(self):
        result = await integrator.integrate(context(selection("edit")), "g", "只重画帽子",
                                            has_image=True, image_mode="edit",
                                            planned_style_id="clean_anime_wallpaper")
        self.assertIn("只作用于修改区域", result)
        self.assertIn(integrator.STYLE_PARTS["clean_anime_wallpaper"][1], result)

    async def test_selection_audit_matches_exact_assembled_fragments(self):
        audit = {}
        text = await integrator.integrate(context(), "g", "千束招手", has_image=False, selection_out=audit)
        self.assertEqual(audit, selection())
        self.assertEqual(text, "千束招手\n\n" + integrator.STYLE_HEADER + "\n" +
                         integrator.STYLE_PARTS[audit["style_id"]][1] + "\n\n" +
                         integrator.QUALITY_HEADER + "\n" + integrator.QUALITY_PARTS[0])

    async def test_gemini_fenced_json_is_accepted_on_first_attempt(self):
        for fence in ("json", "JSON", ""):
            ctx = context()
            ctx.llm_generate.return_value = types.SimpleNamespace(
                completion_text="```" + fence + "\n" + json.dumps(selection()) + "\n```")
            text = await integrator.integrate(ctx, "group", "千束在街边微笑", has_image=False)
            self.assertTrue(text.startswith("千束在街边微笑"))
            ctx.llm_generate.assert_awaited_once()

    async def test_fences_do_not_bypass_content_validation(self):
        ctx = context()
        bad = selection()
        bad["style_parts"] = [999]
        ctx.llm_generate.return_value = types.SimpleNamespace(completion_text="```json\n" + json.dumps(bad) + "\n```")
        self.assertIsNone(await integrator.integrate(ctx, "g", "甲", has_image=False))

    async def test_subjects_actions_composition_and_text_are_preserved_exactly(self):
        plan = '左边甲：银发红瞳，右边乙：黑发金瞳；两人背靠背。标题逐字写“净色动画壁纸”。'
        ctx = context()
        text = await integrator.integrate(ctx, "group", plan, has_image=False)
        self.assertTrue(text.startswith(plan + "\n\n"))
        self.assertIn(integrator.STYLE_PARTS["clean_anime_wallpaper"][1], text)
        self.assertIn(integrator.QUALITY_PARTS[0], text)
        self.assertNotIn("image_urls", ctx.llm_generate.call_args.kwargs)

    async def test_invented_fragments_and_out_of_range_indices_fail_closed(self):
        for bad in (["新加一个人物"], [999], [-1], [True], [0, 0]):
            choice = selection()
            choice["style_parts"] = bad
            with self.subTest(bad=bad):
                self.assertIsNone(await integrator.integrate(context(choice), "g", "原方案", has_image=False))

    async def test_fallback_provider_used_without_changing_plan(self):
        ctx = context()
        ctx.llm_generate.side_effect = [RuntimeError(), RuntimeError(), response(selection())]
        result = await integrator.integrate(ctx, "g", "原方案", has_image=False,
                                           provider_id="first", fallback_provider_ids=["backup"])
        self.assertTrue(result.startswith("原方案"))
        self.assertEqual([call.kwargs["chat_provider_id"] for call in ctx.llm_generate.call_args_list],
                         ["first", "first", "backup"])

    async def test_disabled_and_explicit_only_reject_unrequested_styles(self):
        for mode in ("disabled", "explicit_only"):
            self.assertIsNone(await integrator.integrate(context(), "g", "一只猫", has_image=False, style_mode=mode))
            result = await integrator.integrate(context(selection(style="none")), "g", "一只猫",
                                                has_image=False, style_mode=mode)
            self.assertIsNotNone(result)

    async def test_edit_scope_and_reference_style(self):
        edit = await integrator.integrate(context(selection("edit", "none")), "g", "只改帽子",
                                          has_image=True, image_mode="edit")
        self.assertIn("其余部分保持原图", edit)
        self.assertNotIn(integrator.STYLE_HEADER, edit)
        ref = await integrator.integrate(context(selection("reference")), "g", "参考图片画人物",
                                         has_image=True, image_mode="reference")
        self.assertIn(integrator.STYLE_HEADER, ref)
        self.assertIsNone(await integrator.integrate(context(selection("edit")), "g", "参考图片",
                                                     has_image=True, image_mode="reference"))

    async def test_followup_keeps_full_style_without_duplication(self):
        original = await integrator.integrate(context(), "g", "银发少女站在海边", has_image=False)
        changed = original.replace("站在海边", "坐在海边")
        updated = await integrator.integrate(context(selection(style="none", quality=[])), "g",
                                             changed, has_image=False)
        self.assertEqual(updated, changed)

    async def test_length_limit_never_truncates_source(self):
        self.assertIsNone(await integrator.integrate(context(), "g", "字" * 32000, has_image=False))

    async def test_local_fallback_preserves_edit_scope_and_existing_style(self):
        edit = integrator.integrate_fallback("只把帽子改蓝", has_image=True, image_mode="edit", planned_style_id="none")
        self.assertIn("其余部分保持原图", edit)
        self.assertNotIn(integrator.STYLE_HEADER, edit)
        existing = "原图方案\n" + integrator.STYLE_HEADER + "\n原画风\n" + integrator.QUALITY_HEADER + "\n原质量要求"
        self.assertEqual(integrator.integrate_fallback(existing, has_image=False, image_mode="none",
                                                     planned_style_id="window_overlay_poetic"), existing)


def evidence(digest="first"):
    return [{"source_url": "https://example.com/page", "image_url": "https://example.com/image.png",
             "label": "角色", "digest": digest, "data": "data:image/png;base64,REAL_DOWNLOAD"}]


def consensus(pair=None):
    return {"image_matches_subject": True, "image_features": "银发红瞳",
            "matched_pair": pair if pair is not None else ["image", "model"], "features": "银发红瞳"}


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_and_model_agree(self):
        ctx = context(consensus())
        checker = references.SubjectReferences(ctx)
        with patch.object(references, "download_images", AsyncMock(return_value=evidence())):
            result = await checker.compare("request", "g", "甲", "银发红瞳", "黑发金瞳", ["https://example.com"])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["features"], "银发红瞳")
        self.assertEqual(ctx.llm_generate.call_args.kwargs["chat_provider_id"], "main-model")
        self.assertEqual(ctx.llm_generate.call_args.kwargs["image_urls"], [evidence()[0]["data"]])

    async def test_each_valid_pair_and_single_source_rejection(self):
        for pair in (["image", "model"], ["image", "search"], ["model", "search"]):
            self.assertEqual(references.SubjectReferences._consensus(consensus(pair), "甲", "乙"), "银发红瞳")
        for pair in ([], ["image"], ["image", "image"], ["image", "invented"], [{}, "model"]):
            self.assertEqual(references.SubjectReferences._consensus(consensus(pair), "甲", "乙"), "")
        self.assertEqual(references.SubjectReferences._consensus(consensus(), "", "乙"), "")
        wrong = consensus()
        wrong["image_matches_subject"] = False
        self.assertEqual(references.SubjectReferences._consensus(wrong, "甲", "乙"), "")

    async def test_retry_requires_new_download_and_preserves_original_knowledge(self):
        ctx = context(consensus())
        ctx.llm_generate.side_effect = [response(consensus([])), response(consensus(["image", "search"]))]
        checker = references.SubjectReferences(ctx)
        with patch.object(references, "download_images", AsyncMock(side_effect=[evidence(), evidence("second")])) as fetch:
            first = await checker.compare("r", "g", "甲", "原认识", "第一次搜索", ["https://example.com/1"])
            self.assertEqual(first["status"], "retry")
            self.assertTrue(checker.pending("r"))
            second = await checker.compare("r", "g", "甲", "不应覆盖原认识", "第二次搜索", ["https://example.com/2"])
        self.assertEqual(second["status"], "confirmed")
        self.assertEqual(fetch.await_count, 2)
        sent = json.loads(ctx.llm_generate.call_args.kwargs["prompt"])
        self.assertEqual(sent["model_features"], "原认识")
        self.assertEqual(sent["search_features"], "第二次搜索")

    async def test_two_failed_rounds_fall_back_and_third_does_no_work(self):
        ctx = context(consensus([]))
        checker = references.SubjectReferences(ctx)
        with patch.object(references, "download_images", AsyncMock(return_value=evidence())) as fetch:
            statuses = [(await checker.compare("r", "g", "甲", "原认识", "搜索", ["https://example.com"]))["status"]
                        for _ in range(3)]
        self.assertEqual(statuses, ["retry", "fallback", "fallback"])
        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(ctx.llm_generate.await_count, 1)  # Identical pixels aren't a second opinion.
        self.assertFalse(checker.pending("r"))

    async def test_subjects_and_requests_are_independent(self):
        checker = references.SubjectReferences(context())
        with patch.object(references, "download_images", AsyncMock(return_value=[])):
            for request, subject in (("r1", "甲"), ("r1", "乙"), ("r2", "甲")):
                result = await checker.compare(request, "g", subject, "特征", "搜索", [])
                self.assertEqual(result["round"], 1)

    async def test_vision_unavailable_still_reaches_fallback(self):
        ctx = context()
        ctx.llm_generate.side_effect = TypeError("no vision support")
        checker = references.SubjectReferences(ctx)
        with patch.object(references, "download_images", AsyncMock(side_effect=[evidence(), evidence("new")])):
            first = await checker.compare("r", "g", "甲", "原认识", "搜索", [])
            second = await checker.compare("r", "g", "甲", "原认识", "搜索", [])
        self.assertEqual((first["status"], second["status"]), ("retry", "fallback"))
        self.assertEqual(second["features"], "")

    async def test_parallel_draw_sees_comparison_in_progress(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def download(urls):
            started.set()
            await release.wait()
            return []

        checker = references.SubjectReferences(context())
        with patch.object(references, "download_images", download):
            task = asyncio.create_task(checker.compare("r", "g", "甲", "原认识", "搜索", []))
            await started.wait()
            self.assertTrue(checker.pending("r"))
            release.set()
            await task


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ctx = context()
        self.plugin = main.QiniuImagePlugin(self.ctx, {"api_key": "test-key", "planning_provider_id": "planner",
                                                       "rewrite_provider_ids": ["integrator"], "triggers": ["画图"]})
        self.plugin.client.text_to_image = AsyncMock(return_value=["generated"])
        self.plugin.client.image_to_image = AsyncMock(return_value=["edited"])
        self.web = FunctionTool("web_search_tavily", "search", {})
        self.plugin._search_tools[self.plugin._prompt_key(Event())] = (self.web,)

    async def asyncTearDown(self):
        await self.plugin.terminate()

    async def finish(self):
        await asyncio.wait_for(asyncio.gather(*list(self.plugin._tasks), return_exceptions=True), timeout=1)

    async def accept(self, event=None, brief="画甲，动作自由设计", **kwargs):
        return json.loads(await self.plugin.prepare_drawing(event or Event(), brief, **kwargs))

    async def test_acceptance_returns_before_any_io_and_auto_delivers(self):
        event = Event()
        accepted = await self.accept(event, caption="千束喵～")
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["phase"], "planning")
        self.ctx.tool_loop_agent.assert_not_awaited()
        self.ctx.llm_generate.assert_not_awaited()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertIn("画甲", record["prompt"])
        await self.finish()
        self.assertEqual(record["status"], "completed")
        self.assertEqual(self.ctx.tool_loop_agent.call_args.kwargs["chat_provider_id"], "planner")
        self.assertEqual(self.ctx.tool_loop_agent.call_args.kwargs["contexts"], [])
        self.assertEqual(self.ctx.send_message.call_args_list[0].args[1].text, "千束喵～")
        self.assertEqual(self.ctx.send_message.call_args_list[1].args[1].image, "generated")
        self.plugin.client.text_to_image.assert_awaited_once_with(record["final_prompt"])
        self.assertNotIn("prompt", json.loads(await self.plugin.get_last_image_prompt(event)))
        self.assertIn("prompt", json.loads(await self.plugin.get_drawing_plan(event, record["plan_id"], full=True)))

    async def test_slow_image_resolution_does_not_block_tool_and_is_cancellable(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def slow(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        event = Event()
        with patch.object(main, "resolve_input_image", slow):
            accepted = await asyncio.wait_for(self.accept(event), timeout=.1)
            await started.wait()
            result = json.loads(await self.plugin.cancel_drawing(event, accepted["plan_id"]))
            await self.finish()
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(cancelled.is_set())
        self.plugin.client.text_to_image.assert_not_awaited()
        self.ctx.send_message.assert_not_awaited()

    async def test_total_deadline_uses_snapshot_without_wrapup_model(self):
        cancelled = asyncio.Event()

        async def slow(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.ctx.tool_loop_agent.side_effect = slow
        with patch.object(main, "PREPARATION_TIMEOUT_SECONDS", .03):
            accepted = await self.accept(brief="画一只蓝色猫")
            await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertTrue(cancelled.is_set())
        self.assertTrue(record["degraded"])
        self.assertEqual(record["status"], "completed")
        self.assertIn("蓝色猫", record["final_prompt"])
        self.ctx.llm_generate.assert_not_awaited()

    async def test_integration_only_gets_remaining_total_budget(self):
        cancelled = asyncio.Event()

        async def slow(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def plan(**kwargs):
            await asyncio.sleep(.02)
            return response({"prompt": "甲在海边招手", "summary": "甲", "image_mode": "none"})

        self.ctx.tool_loop_agent.side_effect = plan
        with patch.object(main, "PREPARATION_TIMEOUT_SECONDS", .05), patch.object(main, "integrate", slow):
            accepted = await self.accept()
            await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertTrue(cancelled.is_set())
        self.assertTrue(record["style_selection"]["fallback"])
        self.assertEqual(record["status"], "completed")

    async def test_completed_subjects_are_visible_before_deadline_and_survive_it(self):
        first_saved, second_started, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.web.call = AsyncMock(return_value='{"url":"https://example.com/a","features":"银发红瞳"}')

        async def run(**kwargs):
            search, compare, save = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="银发红瞳", query="甲")
            saved = json.loads(await save.handler(Event(), "甲", "consistent", "银发红瞳", ["https://example.com/a"]))
            self.assertEqual(saved["status"], "saved")
            first_saved.set()
            self.web.call.side_effect = slow_search
            await search.call(None, drawing_subject="乙", drawing_known_features="猜测黑发金瞳", query="乙")

        async def slow_search(*args, **kwargs):
            second_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.ctx.tool_loop_agent.side_effect = run
        with patch.object(main, "PREPARATION_TIMEOUT_SECONDS", .15):
            accepted = await self.accept(brief="画甲和乙在海边合影")
            await first_saved.wait()
            await second_started.wait()
            snapshot = json.loads(await self.plugin.get_drawing_plan(Event(), accepted["plan_id"], full=True))
            self.assertEqual(snapshot["status"], "planning")
            self.assertIn("银发红瞳", snapshot["prompt"])
            self.assertNotIn("猜测黑发金瞳", snapshot["prompt"])
            await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertTrue(cancelled.is_set())
        self.assertIn("银发红瞳", record["final_prompt"])
        self.assertNotIn("黑发金瞳", record["final_prompt"])
        self.assertEqual(record["subject_assessments"][-1], {"subject": "乙", "relation": "unverified"})
        self.ctx.llm_generate.assert_not_awaited()

    async def test_conflict_check_updates_snapshot_even_if_next_model_step_hangs(self):
        self.web.call = AsyncMock(return_value='{"url":"https://example.com/a"}')
        self.ctx.llm_generate.return_value = response(consensus())

        async def run(**kwargs):
            search, compare, save = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="银发红瞳", query="甲")
            await compare.handler(Event(), "甲", "银发红瞳", "黑发金瞳", ["https://example.com/a"])
            await asyncio.Event().wait()

        self.ctx.tool_loop_agent.side_effect = run
        with patch.object(main, "PREPARATION_TIMEOUT_SECONDS", .05), patch.object(references, "download_images", AsyncMock(return_value=evidence())):
            accepted = await self.accept()
            await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertIn("银发红瞳", record["final_prompt"])
        self.assertEqual(record["checks"][0]["status"], "confirmed")
        self.ctx.llm_generate.assert_awaited_once()  # Only the image comparison, never a wrapup.

    async def test_simple_original_request_uses_one_model_and_no_search(self):
        self.ctx.llm_generate.return_value = response({"prompt": "蓝色原创猫在窗边", "summary": "原创猫", "image_mode": "none"})
        accepted = await self.accept(brief="原创蓝色猫", research_mode="skip")
        await self.finish()
        self.ctx.tool_loop_agent.assert_not_awaited()
        self.ctx.llm_generate.assert_awaited_once()
        self.assertNotIn("tools", self.ctx.llm_generate.call_args.kwargs)
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "completed")

    async def test_saved_subject_can_finish_early_without_repeating_assessment(self):
        self.web.call = AsyncMock(return_value='{"url":"https://example.com/a","features":"银发红瞳"}')

        async def run(**kwargs):
            search, compare, save = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="银发红瞳", query="甲")
            await save.handler(Event(), "甲", "consistent", "银发红瞳", ["https://example.com/a"])
            return response({"prompt": "甲在海边招手", "summary": "甲在海边", "image_mode": "none"})

        self.ctx.tool_loop_agent.side_effect = run
        accepted = await self.accept()
        await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertEqual(record["status"], "completed")
        self.assertFalse(record.get("degraded"))
        self.assertIn("银发红瞳", record["final_prompt"])
        self.assertEqual(record["subject_assessments"][0]["relation"], "consistent")

    async def test_checkpoint_rejects_invented_or_other_subject_sources(self):
        self.web.call = AsyncMock(side_effect=['{"url":"https://example.com/a"}', '{"url":"https://example.com/b"}'])

        async def run(**kwargs):
            search, compare, save = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="", query="甲")
            await search.call(None, drawing_subject="乙", drawing_known_features="", query="乙")
            for urls in (["https://example.com/invented"], ["https://example.com/b"]):
                result = json.loads(await save.handler(Event(), "甲", "unknown", "假的金瞳", urls))
                self.assertEqual(result["status"], "invalid")
            result = json.loads(await compare.handler(Event(), "甲", "", "假的金瞳", ["https://example.com/b"]))
            self.assertEqual(result["status"], "invalid_sources")
            return response({"prompt": "invalid"})

        self.ctx.tool_loop_agent.side_effect = run
        accepted = await self.accept(brief="画甲和乙")
        await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertNotIn("假的金瞳", record["final_prompt"])
        self.assertTrue(all(row["relation"] == "unverified" for row in record["subject_assessments"]))

    async def test_background_lifetime_survives_returning_tool_task(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(**kwargs):
            started.set()
            await release.wait()
            return response({"prompt": "甲在海边", "summary": "甲", "image_mode": "none"})

        self.ctx.tool_loop_agent.side_effect = slow
        caller = asyncio.create_task(self.accept())
        accepted = await caller
        await started.wait()
        self.assertTrue(caller.done())
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "planning")
        release.set()
        await self.finish()
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "completed")

    async def test_explicit_image_edit_needs_no_planning_or_integration_model(self):
        event = Event(images=[Image(url="https://example.com/original.png")])
        accepted = await self.accept(event, brief="只把帽子改蓝", image_mode="edit")
        await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.ctx.tool_loop_agent.assert_not_awaited()
        self.ctx.llm_generate.assert_not_awaited()
        self.assertEqual(record["style_id"], "none")
        self.assertIn("其余部分保持原图", record["final_prompt"])
        self.plugin.client.image_to_image.assert_awaited_once_with("https://example.com/original.png", record["final_prompt"])

    async def test_reference_image_is_normalized_and_never_uses_web_tools(self):
        encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
        self.ctx.llm_generate.side_effect = [response({"prompt": "以图中人物为依据，海边坐姿", "summary": "参考原图", "image_mode": "reference"}), response(selection("reference"))]
        accepted = await self.accept(Event(images=[Image(file="base64://" + encoded)]), image_mode="reference")
        await self.finish()
        self.ctx.tool_loop_agent.assert_not_awaited()
        self.assertEqual(self.ctx.llm_generate.call_args_list[0].kwargs["image_urls"], ["data:image/png;base64," + encoded])
        self.assertNotIn("image_urls", self.ctx.llm_generate.call_args_list[1].kwargs)
        self.assertEqual(self.plugin.client.image_to_image.call_args.args[0], "base64://" + encoded)
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "completed")

    async def test_invalid_plan_uses_seed_without_another_model(self):
        self.ctx.tool_loop_agent.return_value = response({"prompt": "invalid"})
        accepted = await self.accept(brief="画一只蓝色猫")
        await self.finish()
        self.ctx.llm_generate.assert_not_awaited()
        self.assertIn("蓝色猫", self.plugin.client.text_to_image.call_args.args[0])
        self.assertTrue(self.plugin.planner.plans[accepted["plan_id"]]["degraded"])

    async def test_cancel_before_background_starts_and_shutdown_never_draw(self):
        event = Event()
        accepted = await self.accept(event)
        await self.plugin.cancel_drawing(event, accepted["plan_id"])
        await self.finish()
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "cancelled")
        accepted = await self.accept(Event(mid="2"))
        await self.plugin.terminate()
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "cancelled")
        self.plugin.client.text_to_image.assert_not_awaited()
        self.ctx.send_message.assert_not_awaited()

    async def test_cancel_during_integration_and_reject_cancel_during_generation(self):
        started = asyncio.Event()

        async def slow(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()

        event = Event()
        with patch.object(main, "integrate", slow):
            accepted = await self.accept(event)
            await started.wait()
            await self.plugin.cancel_drawing(event, accepted["plan_id"])
            await self.finish()
        self.plugin.client.text_to_image.assert_not_awaited()
        self.ctx.send_message.assert_not_awaited()
        release = asyncio.Event()
        started.clear()

        async def generate(*args):
            started.set()
            await release.wait()
            return ["generated"]

        self.plugin.client.text_to_image.side_effect = generate
        accepted = await self.accept(Event(mid="2"))
        await started.wait()
        result = await self.plugin.cancel_drawing(event, accepted["plan_id"])
        self.assertIn("只能取消", result)
        release.set()
        await self.finish()

    async def test_duplicate_tool_calls_are_single_use_but_new_requests_are_independent(self):
        event = Event()
        first = await self.accept(event)
        duplicate = await self.accept(event, caption="另一句")
        self.assertEqual(first["plan_id"], duplicate["plan_id"])
        second = await self.accept(event, brief="再画乙")
        self.assertNotEqual(first["plan_id"], second["plan_id"])
        await self.finish()
        await self.plugin.draw_image(event, first["plan_id"], caption="重复")
        self.assertEqual(self.plugin.client.text_to_image.await_count, 2)
        third = await self.accept(Event(mid="2"))
        self.assertNotEqual(third["plan_id"], first["plan_id"])
        await self.finish()
        self.assertEqual(self.plugin.client.text_to_image.await_count, 3)

    async def test_other_user_cannot_read_cancel_revise_or_execute(self):
        accepted = await self.accept()
        bob = Event(user="bob")
        self.assertIn("不属于", await self.plugin.get_drawing_plan(bob, accepted["plan_id"], full=True))
        self.assertIn("不属于", await self.plugin.prepare_drawing(bob, "修改", base_plan_id=accepted["plan_id"]))
        self.assertIn("只能取消", await self.plugin.cancel_drawing(bob, accepted["plan_id"]))
        self.assertIn("有效方案编号", await self.plugin.draw_image(bob, accepted["plan_id"], caption="test"))
        await self.finish()
        self.plugin.client.text_to_image.assert_awaited_once()

    async def test_revision_supersedes_background_task_and_preserves_snapshot(self):
        event = Event()
        first = await self.accept(event, brief="甲在海边")
        second = await self.accept(event, brief="只把帽子改蓝", base_plan_id=first["plan_id"])
        await self.finish()
        self.assertEqual(self.plugin.planner.plans[first["plan_id"]]["status"], "superseded")
        request = json.loads(self.ctx.tool_loop_agent.call_args.kwargs["prompt"])
        self.assertIn("甲在海边", request["previous_prompt"])
        self.assertEqual(request["brief"], "只把帽子改蓝")
        self.assertEqual(self.plugin.planner.plans[second["plan_id"]]["status"], "completed")
        self.plugin.client.text_to_image.assert_awaited_once()

    async def test_revision_keeps_original_image_even_before_resolution_finishes(self):
        original = Event(images=[Image(url="https://example.com/original.png")])
        started, release = asyncio.Event(), asyncio.Event()

        async def resolve(_context, event, _client):
            if event is original:
                started.set()
                await release.wait()
                return "https://example.com/original.png"
            return None

        with patch.object(main, "resolve_input_image", resolve):
            first = await self.accept(original, brief="只把帽子改蓝", image_mode="edit")
            await started.wait()
            second = await self.accept(Event(mid="2"), brief="帽子改为红色", base_plan_id=first["plan_id"])
            release.set()
            await self.finish()
        self.assertEqual(self.plugin.planner.plans[first["plan_id"]]["status"], "superseded")
        self.assertEqual(self.plugin.planner.plans[second["plan_id"]]["status"], "completed")
        self.assertEqual(self.plugin.client.image_to_image.call_args.args[0], "https://example.com/original.png")
        self.plugin.client.image_to_image.assert_awaited_once()
        self.assertFalse(self.plugin._image_sources)

    async def test_active_snapshot_cannot_expire_or_be_evicted(self):
        accepted = await self.accept()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        record["created"] -= planning.PLAN_TTL + 1
        self.assertIs(self.plugin.planner.get(record["owner"], record["plan_id"]), record)
        with patch.object(planning, "PLAN_LIMIT", 1):
            result = await self.plugin.prepare_drawing(Event(mid="2"), "另一个任务")
        self.assertIn("繁忙", result)
        await self.finish()

    async def test_followup_uses_full_prompt_and_keeps_selected_style(self):
        self.ctx.tool_loop_agent.return_value = response({"prompt": "喜多挥手", "summary": "喜多", "image_mode": "none", "style_id": "window_overlay_poetic"})
        self.ctx.llm_generate.return_value = response(selection(style="window_overlay_poetic"))
        await self.accept()
        await self.finish()
        original = self.plugin.client.text_to_image.call_args.args[0]
        self.assertIn(integrator.STYLE_PARTS["window_overlay_poetic"][0], original)
        self.ctx.tool_loop_agent.side_effect = asyncio.TimeoutError()
        second = await self.accept(Event(mid="2"), brief="只改成坐姿", use_last=True)
        await self.finish()
        task = json.loads(self.ctx.tool_loop_agent.call_args.kwargs["prompt"])
        self.assertEqual(task["previous_prompt"], original)
        record = self.plugin.planner.plans[second["plan_id"]]
        self.assertTrue(record["final_prompt"].startswith(original))
        self.assertEqual(record["final_prompt"].count(integrator.STYLE_HEADER), 1)
        self.assertEqual(record["style_id"], "window_overlay_poetic")

    async def test_search_tool_scope_is_frozen_at_acceptance(self):
        disabled = FunctionTool("web_search_bocha", "search", {})
        disabled.active = False
        req = types.SimpleNamespace(system_prompt="人设", func_tool=ToolSet([self.web, FunctionTool("shell", "command", {}), disabled]))
        event = Event()
        await self.plugin.on_llm_request(event, req)
        await self.plugin.on_llm_request(event, req)
        self.assertEqual(req.system_prompt.count(main._DRAWING_RULES), 1)
        await self.accept(event)
        self.plugin._search_tools[self.plugin._prompt_key(event)] = ()
        await self.finish()
        names = [tool.name for tool in self.ctx.tool_loop_agent.call_args.kwargs["tools"].tools]
        self.assertEqual(names, ["web_search_tavily", "compare_subject_reference", "save_subject_assessment"])

    async def test_failed_integration_preserves_plan_and_delivers(self):
        with patch.object(main, "integrate", AsyncMock(return_value=None)):
            accepted = await self.accept()
            await self.finish()
        record = self.plugin.planner.plans[accepted["plan_id"]]
        self.assertTrue(record["final_prompt"].startswith(record["prompt"]))
        self.assertTrue(record["style_selection"]["fallback"])
        self.assertEqual(record["status"], "completed")

    async def test_delivery_failure_is_not_completed(self):
        for failure in (False, RuntimeError("platform unavailable")):
            self.ctx.send_message.side_effect = [True, failure]
            accepted = await self.accept(Event())
            await self.finish()
            self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "delivery_failed")

    async def test_safety_retry_updates_actual_prompt(self):
        self.plugin.client.text_to_image.side_effect = [main.QiniuSafetyError(400, "safety"), ["generated"]]
        with patch.object(main, "rewrite_for_safety", AsyncMock(return_value="安全的完整替代提示词")):
            accepted = await self.accept()
            await self.finish()
        self.assertIn("安全的完整替代提示词", await self.plugin.get_drawing_plan(Event(), accepted["plan_id"], full=True))
        self.assertIn("notice", json.loads(await self.plugin.get_last_image_prompt(Event())))

    async def test_chatter_suppression_is_only_for_its_event(self):
        event = Event()
        await self.accept(event)
        event.result = types.SimpleNamespace(chain=["QINIU_DRAWING_DONE"], is_llm_result=lambda: True)
        await self.plugin.suppress_drawing_chatter(event)
        self.assertEqual(event.result.chain, [])
        other = Event()
        other.result = types.SimpleNamespace(chain=["正常聊天"], is_llm_result=lambda: True)
        await self.plugin.suppress_drawing_chatter(other)
        self.assertEqual(other.result.chain, ["正常聊天"])
        await self.finish()

    async def test_bad_input_and_missing_image_never_generate(self):
        for kwargs in ({"research_mode": "bad"}, {"caption": "长" * 61}, {"image_mode": "bad"}):
            self.assertNotIn("accepted", await self.plugin.prepare_drawing(Event(), "画猫", **kwargs))
        accepted = await self.accept(image_mode="edit")
        await self.finish()
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "failed")
        self.plugin.client.text_to_image.assert_not_awaited()
        self.ctx.send_message.assert_awaited_once()

    async def test_unreadable_input_image_cannot_fall_back_to_text_generation(self):
        with patch.object(main, "resolve_input_image", AsyncMock(return_value=None)):
            accepted = await self.accept(Event(images=[Image(file="invalid-image")]))
            await self.finish()
        self.assertEqual(self.plugin.planner.plans[accepted["plan_id"]]["status"], "failed")
        self.plugin.client.text_to_image.assert_not_awaited()
        self.plugin.client.image_to_image.assert_not_awaited()

    async def test_keyword_trigger_keeps_baseline_path(self):
        event = Event(text="画图 一只猫")
        with patch.object(main, "rewrite", AsyncMock(return_value="原关键词优化结果")) as rewrite:
            results = [row async for row in self.plugin.on_message(event)]
            duplicate = [row async for row in self.plugin.on_message(event)]
        self.assertTrue(event.stopped)
        self.assertEqual((len(results), duplicate), (1, []))
        rewrite.assert_awaited_once()
        self.ctx.tool_loop_agent.assert_not_awaited()
        self.plugin.client.text_to_image.assert_awaited_once_with("原关键词优化结果")


class PlanningResearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_keeps_completed_evidence_and_cancels_pending_search(self):
        ctx = context(consensus())
        planner = planning.DrawingPlanner(ctx, "planner")
        source = FunctionTool("web_search_tavily", "search", {})
        cancelled = asyncio.Event()

        async def search(_context, query):
            if query == "甲":
                return '{"url":"https://example.com/real"}'
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        source.call = search

        async def run(**kwargs):
            search_tool, compare = kwargs["tools"].tools
            await search_tool.call(None, drawing_subject="甲", drawing_known_features="银发红瞳", query="甲")
            await compare.handler(Event(), "甲", "银发红瞳", "黑发金瞳", ["https://example.com/real"])
            await search_tool.call(None, drawing_subject="乙", drawing_known_features="不确定", query="乙")

        ctx.tool_loop_agent.side_effect = run
        state = {}
        with patch.object(references, "download_images", AsyncMock(return_value=evidence())):
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(planner.prepare(Event(), "owner", "画甲和乙", search_tools=[source], fallback_state=state), timeout=.05)
        self.assertTrue(cancelled.is_set())
        ctx.llm_generate.side_effect = RuntimeError("wrapup unavailable")
        record = planner.snapshot("owner", "画甲和乙", state=state)
        self.assertEqual(len(record["research"]), 1)
        self.assertEqual(record["checks"][0]["status"], "confirmed")
        self.assertIn("银发红瞳", record["prompt"])
        self.assertEqual(record["subject_assessments"], [{"subject": "甲", "relation": "confirmed"}, {"subject": "乙", "relation": "unverified"}])
        self.assertEqual(planner.describe(record)["search_count"], 1)
        ctx.tool_loop_agent.assert_awaited_once()

    async def test_fallback_preserves_base_and_respects_style_modes(self):
        ctx = context()
        ctx.llm_generate.side_effect = RuntimeError("offline")
        planner = planning.DrawingPlanner(ctx)
        base = "甲坐在海边\n" + integrator.STYLE_HEADER + "\n原画风\n" + integrator.QUALITY_HEADER + "\n原质量"
        record = planner.snapshot("owner", "只把帽子改蓝", state={},
                                                previous_prompt=base, previous_style_id="window_overlay_poetic")
        self.assertTrue(record["prompt"].startswith(base))
        self.assertEqual(record["style_id"], "window_overlay_poetic")
        changed = planner.snapshot("owner", "只把帽子改蓝", state={},
                                                previous_prompt=base, style_request="水彩")
        self.assertNotIn("原画风", changed["prompt"])
        self.assertIn("甲坐在海边", changed["prompt"])
        self.assertIn("水彩", changed["prompt"])
        for style_mode in ("disabled", "explicit_only"):
            result = planner.snapshot("owner", "画猫", state={}, style_mode=style_mode)
            self.assertEqual(result["style_id"], "none")

    async def test_logged_paraphrase_of_confirmed_features_no_longer_restarts_planning(self):
        ctx = context(consensus())
        source = FunctionTool("web_search_tavily", "search", {})
        source.call = AsyncMock(return_value='{"url":"https://example.com/real"}')

        async def run(**kwargs):
            search, compare = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="银发红瞳", query="甲")
            await compare.handler(Event(), "甲", "银发红瞳", "黑发金瞳", ["https://example.com/real"])
            return response({"prompt": "甲，银色头发、红色眼睛，在海边招手", "summary": "甲在海边招手", "image_mode": "none",
                             "subject_assessments": [{"subject": "甲", "relation": "conflict", "search_features": "黑发金瞳", "source_urls": ["https://example.com/real"]}]})

        ctx.tool_loop_agent.side_effect = run
        with patch.object(references, "download_images", AsyncMock(return_value=evidence())):
            record = await planning.DrawingPlanner(ctx).prepare(Event(), "owner", "画甲", search_tools=[source])
        self.assertEqual(record["status"], "ready")
        self.assertIn("银发红瞳", record["prompt"])
        ctx.tool_loop_agent.assert_awaited_once()

    async def test_auto_cannot_skip_style_because_brief_invents_3d_or_heavy_paint(self):
        for brief in ("画奶龙，3D CGI动画风格", "画秦彻，日系厚涂", "画全家福，剧场版插画"):
            ctx = context()
            ctx.tool_loop_agent.return_value = response({"prompt": brief, "summary": brief, "image_mode": "none", "style_id": "none"})
            planner = planning.DrawingPlanner(ctx)
            with self.assertRaisesRegex(ValueError, "auto 必须"):
                await planner.prepare(Event(), "owner", brief)
            self.assertFalse(planner.plans)

    async def test_style_is_planned_from_catalog_with_subject_count_preferences(self):
        ctx = context()
        ctx.tool_loop_agent.return_value = response({"prompt": "奶龙挥手，留出几何窗口空间", "summary": "奶龙；诗意窗口", "image_mode": "none", "style_id": "window_overlay_poetic"})
        planner = planning.DrawingPlanner(ctx)
        record = await planner.prepare(Event(), "owner", "画奶龙，3D动画或插画")
        task = json.loads(ctx.tool_loop_agent.call_args.kwargs["prompt"])
        self.assertEqual(len(task["styles"]), 13)
        self.assertEqual(task["style_request"], "")
        clean = task["styles"]["clean_anime_wallpaper"]["preference"]
        for name in ("glitch_rectangles", "window_overlay_poetic"):
            other = task["styles"][name]["preference"]
            self.assertLess(clean["single_subject"], other["single_subject"])
            self.assertGreater(clean["multiple_subjects"], other["multiple_subjects"])
        self.assertEqual(planner.describe(record)["style_id"], "window_overlay_poetic")
        ctx.tool_loop_agent.assert_awaited_once()  # Style selection adds no separate model request.

    async def test_explicit_external_style_disabled_and_image_edit_allow_no_preset(self):
        cases = [({"style_request": "用写实3D渲染"}, "none"), ({"style_mode": "disabled"}, "none"),
                 ({"style_mode": "explicit_only"}, "none"), ({"image_ref": "https://example.com/user.png", "image_mode": "edit"}, "edit")]
        for kwargs, mode in cases:
            ctx = context()
            result = response({"prompt": "按要求画奶龙", "summary": "奶龙", "image_mode": mode, "style_id": "none"})
            ctx.tool_loop_agent.return_value = result
            ctx.llm_generate.return_value = result
            record = await planning.DrawingPlanner(ctx).prepare(Event(), "owner", "画奶龙", **kwargs)
            self.assertEqual(record["style_id"], "none")

    async def test_image_reference_requires_preset_and_previous_style_is_preserved(self):
        ctx = context()
        ctx.llm_generate.return_value = response({"prompt": "参考图中人物", "summary": "参考原图", "image_mode": "reference", "style_id": "none"})
        with self.assertRaisesRegex(ValueError, "auto 必须"):
            await planning.DrawingPlanner(ctx).prepare(Event(), "owner", "参考图", image_ref="https://example.com/user.png")
        ctx.tool_loop_agent.return_value = response({"prompt": "只改动作", "summary": "新动作", "image_mode": "none", "style_id": "clean_anime_wallpaper"})
        with self.assertRaises(ValueError):
            await planning.DrawingPlanner(ctx).prepare(Event(), "owner", "只改动作", previous_prompt="原完整提示词",
                                                      previous_style_id="window_overlay_poetic")

    async def test_text_conflict_cannot_skip_image_comparison(self):
        for relation in ("conflict", "unknown", None, "consistent"):
            ctx = context()
            planner = planning.DrawingPlanner(ctx)
            source = FunctionTool("web_search_tavily", "search", {"properties": {"query": {"type": "string"}}, "required": ["query"]})
            source.call = AsyncMock(return_value='{"url":"https://example.com/kita","features":"Pelham Blue"}')

            async def run(**kwargs):
                search = kwargs["tools"].tools[0]
                await search.call(None, drawing_subject="喜多", drawing_known_features="Pelham Blue" if relation == "consistent" else "红色吉他", query="喜多 吉他")
                result = {"prompt": "喜多背吉他", "summary": "喜多", "image_mode": "none"}
                if relation is not None:
                    result["subject_assessments"] = [{"subject": "喜多", "relation": relation,
                                                       "search_features": "Pelham Blue", "source_urls": ["https://example.com/kita"]}]
                return response(result)

            ctx.tool_loop_agent.side_effect = run
            if relation == "consistent":
                record = await planner.prepare(Event(), "owner", "画喜多", search_tools=[source])
                self.assertEqual(record["research"][0]["model_features"], "Pelham Blue")
            else:
                with self.assertRaises(ValueError):
                    await planner.prepare(Event(), "owner", "画喜多", search_tools=[source])
                self.assertFalse(planner.plans)
            self.assertNotIn("drawing_subject", source.parameters["properties"])

    async def test_empty_search_cannot_claim_consistent_official_features(self):
        for relation in ("consistent", "unknown", "unavailable"):
            ctx = context()
            planner = planning.DrawingPlanner(ctx)
            source = FunctionTool("web_search_tavily", "search", {})
            source.call = AsyncMock(return_value="Error: Tavily web searcher does not return any results.")

            async def run(**kwargs):
                await kwargs["tools"].tools[0].call(None, drawing_subject="喜多", drawing_known_features="红色吉他", query="喜多吉他")
                return response({"prompt": "喜多，省略不确定的吉他特征", "summary": "搜索无结果，未核实吉他", "image_mode": "none",
                                 "subject_assessments": [{"subject": "喜多", "relation": relation, "search_features": "", "source_urls": []}]})

            ctx.tool_loop_agent.side_effect = run
            if relation == "unavailable":
                record = await planner.prepare(Event(), "owner", "画喜多", search_tools=[source])
                self.assertEqual(planner.describe(record)["search_failures"], 1)
                self.assertEqual(planner.describe(record)["subject_assessments"][0]["relation"], "unavailable")
            else:
                with self.assertRaisesRegex(ValueError, "没有可用结果"):
                    await planner.prepare(Event(), "owner", "画喜多", search_tools=[source])

    async def test_real_search_then_two_failed_comparisons_returns_reviewable_fallback(self):
        ctx = context(consensus([]))
        planner = planning.DrawingPlanner(ctx, "planner")
        source = FunctionTool("web_search_tavily", "search", {})
        source.call = AsyncMock(side_effect=[json.dumps({"url": "https://example.com/1"}),
                                            json.dumps({"url": "https://example.com/2"})])

        async def run(**kwargs):
            search, compare = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="原认识", query="甲 外观")
            first = json.loads(await compare.handler(Event(), "甲", "原认识", "搜索结果", ["https://example.com/1"]))
            self.assertEqual(first["status"], "retry")
            early = json.loads(await compare.handler(Event(), "甲", "原认识", "搜索结果", ["https://example.com/1"]))
            self.assertEqual(early["status"], "research_required")
            await search.call(None, drawing_subject="甲", drawing_known_features="不能覆盖", query="甲 官方立绘 新来源")
            second = json.loads(await compare.handler(Event(), "甲", "不能覆盖", "新搜索结果", ["https://example.com/2"]))
            self.assertEqual(second["status"], "fallback")
            return response({"prompt": "甲坐在海边，不指定外观", "summary": "甲坐在海边；两轮核对无共识，放弃不可靠外观", "image_mode": "none",
                             "subject_assessments": [{"subject": "甲", "relation": "conflict", "search_features": "新搜索结果", "source_urls": ["https://example.com/2"]}]})

        ctx.tool_loop_agent.side_effect = run
        with patch.object(references, "download_images", AsyncMock(side_effect=[evidence(), evidence("new")])):
            record = await planner.prepare(Event(), "owner", "画甲", search_tools=[source])
        self.assertEqual(record["checks"][0]["status"], "fallback")
        self.assertEqual(len(record["research"]), 2)
        self.assertEqual(record["research"][1]["model_features"], "原认识")
        self.assertEqual(source.call.call_args.kwargs, {"query": "甲 官方立绘 新来源"})
        self.assertTrue(all(call.kwargs["chat_provider_id"] == "planner" for call in ctx.llm_generate.call_args_list))
        self.assertNotIn("research", planner.describe(record))
        self.assertIn("research", planner.describe(record, full=True))

    async def test_unfinished_second_round_and_invented_sources_cannot_pass(self):
        ctx = context()
        planner = planning.DrawingPlanner(ctx, "planner")
        source = FunctionTool("web_search_tavily", "search", {})
        source.call = AsyncMock(return_value='{"url":"https://example.com/real"}')

        async def run(**kwargs):
            search, compare = kwargs["tools"].tools
            await search.call(None, drawing_subject="甲", drawing_known_features="原认识", query="甲")
            result = json.loads(await compare.handler(Event(), "甲", "原认识", "搜索", ["https://example.com/invented"]))
            self.assertEqual(result["status"], "invalid_sources")
            await compare.handler(Event(), "甲", "原认识", "搜索", ["https://example.com/real"])
            return response({"prompt": "未核对方案", "summary": "试图提前完成", "image_mode": "none"})

        ctx.tool_loop_agent.side_effect = run
        with patch.object(references, "download_images", AsyncMock(return_value=[])):
            with self.assertRaisesRegex(ValueError, "第二轮"):
                await planner.prepare(Event(), "owner", "画甲", search_tools=[source])
        self.assertEqual(planner.plans, {})

    async def test_invalid_summary_does_not_create_a_draft(self):
        ctx = context()
        ctx.tool_loop_agent.return_value = response({"prompt": "完整方案", "summary": "长" * 1201, "image_mode": "none"})
        planner = planning.DrawingPlanner(ctx)
        with self.assertRaises(ValueError):
            await planner.prepare(Event(), "owner", "画一只猫")
        self.assertEqual(planner.plans, {})


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_nonpublic_addresses_and_dns_answers_are_rejected(self):
        for url in ("http://127.0.0.1/x", "http://10.0.0.1", "http://[::1]", "file:///tmp/x", "http://u:p@example.com"):
            with self.assertRaises(ValueError):
                references.public_url(url)
        resolver = references.PublicResolver()
        try:
            with patch.object(resolver.resolver, "resolve", AsyncMock(return_value=[{"host": "127.0.0.1"}])):
                with self.assertRaises(OSError):
                    await resolver.resolve("example.com")
        finally:
            await resolver.close()

    async def test_page_image_links_resolve_against_source(self):
        parser = references.PageImages("https://example.com/wiki/person")
        parser.feed('<img data-src="../portrait.png" alt="人物立绘"><meta property="og:image" content="/cover.png">')
        self.assertEqual(parser.images[0], ("https://example.com/portrait.png", "人物立绘"))


if __name__ == "__main__":
    unittest.main()
