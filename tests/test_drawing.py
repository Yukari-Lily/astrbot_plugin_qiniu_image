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

    async def asyncTearDown(self):
        await self.plugin.terminate()

    async def finish(self):
        await asyncio.gather(*list(self.plugin._tasks))

    async def prepare(self, event=None, **kwargs):
        return json.loads(await self.plugin.prepare_drawing(event or Event(), "画甲，银发红瞳；动作自由设计", **kwargs))

    async def test_preparation_is_isolated_and_cannot_generate_until_main_approves(self):
        event = Event()
        draft = await self.prepare(event)
        self.assertEqual(draft["status"], "ready")
        self.assertNotIn("prompt", draft)
        self.assertEqual(self.ctx.tool_loop_agent.call_args.kwargs["chat_provider_id"], "planner")
        self.assertEqual(self.ctx.tool_loop_agent.call_args.kwargs["contexts"], [])
        self.assertNotIn("draw_image", [t.name for t in self.ctx.tool_loop_agent.call_args.kwargs["tools"].tools])
        self.plugin.client.text_to_image.assert_not_awaited()
        self.ctx.llm_generate.assert_not_awaited()
        rejected = await self.plugin.draw_image(event, "直接画一个主体", caption="test caption~")
        self.assertIn("prepare_drawing", rejected)
        accepted = json.loads(await self.plugin.draw_image(event, draft["plan_id"], caption="test caption~"))
        self.assertEqual(accepted["status"], "accepted")
        self.assertNotIn("prompt", accepted)
        await self.finish()
        self.assertEqual(self.ctx.llm_generate.call_args.kwargs["chat_provider_id"], "integrator")
        full = json.loads(await self.plugin.get_last_image_prompt(event, full=True))
        self.plugin.client.text_to_image.assert_awaited_once_with(full["prompt"])
        self.assertNotIn("prompt", json.loads(await self.plugin.get_last_image_prompt(event)))

    async def test_single_subject_style_choice_reaches_image_and_followup(self):
        event = Event()
        self.ctx.tool_loop_agent.return_value = response({"prompt": "喜多挥手，几何窗口留白", "summary": "喜多；诗意窗口", "image_mode": "none", "style_id": "window_overlay_poetic"})
        choice = selection(style="window_overlay_poetic")
        choice["style_parts"] = [0, 1]
        self.ctx.llm_generate.return_value = response(choice)
        draft = await self.prepare(event)
        self.assertEqual(draft["style_id"], "window_overlay_poetic")
        await self.plugin.draw_image(event, draft["plan_id"], caption="喜多喵～")
        await self.finish()
        self.assertIn(integrator.STYLE_PARTS["window_overlay_poetic"][0], self.plugin.client.text_to_image.call_args.args[0])
        self.assertEqual(json.loads(self.ctx.llm_generate.call_args.kwargs["prompt"])["planned_style_id"], "window_overlay_poetic")
        await self.plugin.prepare_drawing(event, "只改成坐姿", use_last=True)
        task = json.loads(self.ctx.tool_loop_agent.call_args.kwargs["prompt"])
        self.assertTrue(task["preserve_previous_style"])
        self.assertEqual(task["previous_style_id"], "window_overlay_poetic")
        self.assertEqual(task["style_mode"], "auto")

    async def test_explicit_single_person_wallpaper_remains_available(self):
        draft = json.loads(await self.plugin.prepare_drawing(Event(), "画喜多", style_request="净色动画壁纸"))
        self.assertEqual(draft["style_id"], "clean_anime_wallpaper")
        self.assertEqual(json.loads(self.ctx.tool_loop_agent.call_args.kwargs["prompt"])["style_request"], "净色动画壁纸")

    async def test_followup_cannot_skip_unsubmitted_plan_and_bots_are_independent(self):
        event = Event()
        first = await self.prepare(event)
        second = json.loads(await self.plugin.prepare_drawing(event, "再画喜多"))
        self.assertEqual(second["status"], "pending_review")
        self.assertEqual(second["plans"][0]["plan_id"], first["plan_id"])
        self.ctx.tool_loop_agent.assert_awaited_once()
        self.plugin.client.text_to_image.assert_not_awaited()
        other_bot = Event()
        other_bot.unified_msg_origin = "second-bot:group"
        other = await self.prepare(other_bot)
        self.assertEqual(other["status"], "ready")
        await self.plugin.draw_image(event, first["plan_id"], caption="千束喵～")
        second = json.loads(await self.plugin.prepare_drawing(event, "再画喜多"))
        self.assertEqual(second["status"], "ready")
        await self.plugin.draw_image(event, second["plan_id"], caption="再画个喜多喵")
        await self.finish()
        self.assertEqual(self.plugin.client.text_to_image.await_count, 2)

    async def test_cancellation_releases_pending_plan_without_drawing(self):
        event = Event()
        draft = await self.prepare(event)
        self.assertIn("只能取消", await self.plugin.cancel_drawing(Event(user="bob"), draft["plan_id"]))
        result = json.loads(await self.plugin.cancel_drawing(event, draft["plan_id"]))
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual((await self.prepare(event))["status"], "ready")
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_caption_comes_from_main_persona_and_is_sent_only_after_prompt_ready(self):
        event = Event()
        draft = await self.prepare(event)
        rejected = json.loads(await self.plugin.draw_image(event, draft["plan_id"]))
        self.assertEqual(rejected["status"], "ready")
        self.assertFalse(self.plugin._tasks)
        caption = "千束喵～元气满满的那种！"
        accepted = json.loads(await self.plugin.draw_image(event, draft["plan_id"], caption=caption))
        self.assertIn("QINIU_DRAWING_DONE", accepted["instruction"])
        await self.finish()
        self.assertEqual(self.ctx.send_message.call_args_list[0].args[1].text, caption)
        self.ctx.llm_generate.assert_awaited_once()  # Only integration; no extra caption model.
        full = json.loads(await self.plugin.get_drawing_plan(event, draft["plan_id"], full=True))
        self.assertEqual(full["style_selection"]["style_id"], "clean_anime_wallpaper")
        self.assertEqual(full["prompt"], self.plugin.client.text_to_image.call_args.args[0])

    async def test_delivery_failure_is_not_marked_completed(self):
        for failure in (False, RuntimeError("platform unavailable")):
            event = Event()
            draft = await self.prepare(event)
            self.ctx.send_message.side_effect = [True, failure]
            await self.plugin.draw_image(event, draft["plan_id"], caption="千束喵～")
            await self.finish()
            self.assertEqual(self.plugin.planner.plans[draft["plan_id"]]["status"], "delivery_failed")

    async def test_revision_changes_id_and_supersedes_unapproved_draft(self):
        event = Event()
        first = await self.prepare(event)
        second = await self.prepare(event, base_plan_id=first["plan_id"])
        self.assertNotEqual(first["plan_id"], second["plan_id"])
        previous = json.loads(self.ctx.tool_loop_agent.call_args.kwargs["prompt"])["previous_prompt"]
        self.assertIn("银发红瞳", previous)
        blocked = json.loads(await self.plugin.draw_image(event, first["plan_id"], caption="test caption~"))
        self.assertEqual(blocked["status"], "superseded")
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_other_user_cannot_read_revise_or_execute_draft(self):
        draft = await self.prepare()
        bob = Event(user="bob")
        self.assertIn("不属于", await self.plugin.get_drawing_plan(bob, draft["plan_id"], full=True))
        self.assertIn("不属于", await self.plugin.prepare_drawing(bob, "修改", base_plan_id=draft["plan_id"]))
        self.assertIn("有效方案编号", await self.plugin.draw_image(bob, draft["plan_id"], caption="test caption~"))
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_plan_is_single_use_even_after_completion(self):
        draft = await self.prepare()
        await self.plugin.draw_image(Event(), draft["plan_id"], caption="test caption~")
        await self.finish()
        again = json.loads(await self.plugin.draw_image(Event(), draft["plan_id"], caption="test caption~"))
        self.assertEqual(again["status"], "completed")
        self.plugin.client.text_to_image.assert_awaited_once()

    async def test_expired_plan_is_rejected(self):
        draft = await self.prepare()
        record = self.plugin.planner.plans[draft["plan_id"]]
        record["created"] -= planning.PLAN_TTL + 1
        self.assertIn("有效方案编号", await self.plugin.draw_image(Event(), draft["plan_id"], caption="test caption~"))

    async def test_followup_loads_complete_prompt_inside_planner(self):
        draft = await self.prepare()
        await self.plugin.draw_image(Event(), draft["plan_id"], caption="test caption~")
        await self.finish()
        full = json.loads(await self.plugin.get_last_image_prompt(Event(), full=True))["prompt"]
        await self.plugin.prepare_drawing(Event(mid="2"), "只把坐姿改为站姿", use_last=True)
        request = json.loads(self.ctx.tool_loop_agent.call_args.kwargs["prompt"])
        self.assertEqual(request["previous_prompt"], full)
        self.assertEqual(request["brief"], "只把坐姿改为站姿")

    async def test_image_planning_has_no_tools_and_only_user_image(self):
        self.ctx.llm_generate.side_effect = [response({"prompt": "以图中人物为依据，海边坐姿", "summary": "参考用户人物，海边坐姿", "image_mode": "reference"}),
                                            response(selection("reference"))]
        event = Event(images=[Image(url="https://example.com/user.png")])
        draft = await self.prepare(event, image_mode="reference")
        self.ctx.tool_loop_agent.assert_not_awaited()
        vision = self.ctx.llm_generate.call_args.kwargs
        self.assertEqual(vision["chat_provider_id"], "planner")
        self.assertEqual(vision["image_urls"], ["https://example.com/user.png"])
        self.assertNotIn("tools", vision)
        # Executing from a later turn uses the reviewed image, not a different new attachment.
        await self.plugin.draw_image(Event(images=[Image(url="https://example.com/different.png")]), draft["plan_id"], caption="test caption~")
        await self.finish()
        self.assertEqual(self.plugin.client.image_to_image.call_args.args[0], "https://example.com/user.png")
        self.assertNotIn("image_urls", self.ctx.llm_generate.call_args.kwargs)

    async def test_search_tool_scope_is_filtered_from_current_request(self):
        web = FunctionTool("web_search_tavily", "search", {})
        shell = FunctionTool("shell", "command", {})
        disabled = FunctionTool("web_search_bocha", "search", {})
        disabled.active = False
        req = types.SimpleNamespace(system_prompt="人设", func_tool=ToolSet([web, shell, disabled]))
        await self.plugin.on_llm_request(Event(), req)
        await self.plugin.on_llm_request(Event(), req)
        self.assertEqual(req.system_prompt.count(main._DRAWING_RULES), 1)
        await self.prepare()
        names = [tool.name for tool in self.ctx.tool_loop_agent.call_args.kwargs["tools"].tools]
        self.assertEqual(names, ["web_search_tavily", "compare_subject_reference"])
        self.assertNotIn("①", main._DRAWING_RULES)

    async def test_failed_integration_sends_one_failure_without_claiming_generation(self):
        draft = await self.prepare()
        with patch.object(main, "integrate", AsyncMock(return_value=None)):
            await self.plugin.draw_image(Event(), draft["plan_id"], caption="test caption~")
            await self.finish()
        self.assertEqual(self.plugin.planner.plans[draft["plan_id"]]["status"], "failed")
        self.plugin.client.text_to_image.assert_not_awaited()
        self.ctx.send_message.assert_awaited_once()
        self.assertIn("生成失败", self.ctx.send_message.call_args.args[1].text)
        self.assertNotIn(self.plugin._prompt_key(Event()), self.plugin._last_image_prompts)

    async def test_slow_integration_returns_immediately_and_cannot_be_started_twice(self):
        event = Event()
        draft = await self.prepare(event)
        record = self.plugin.planner.plans[draft["plan_id"]]
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(*args, **kwargs):
            started.set()
            await release.wait()
            return "完整提示词：千束在街边微笑招手"

        with patch.object(main, "integrate", slow):
            accepted = json.loads(await asyncio.wait_for(self.plugin.draw_image(event, draft["plan_id"], caption="test caption~"), timeout=.1))
            await started.wait()
            self.assertEqual(accepted["phase"], "integrating")
            self.ctx.send_message.assert_not_awaited()
            self.plugin.client.text_to_image.assert_not_awaited()
            self.assertNotIn(self.plugin._prompt_key(event), self.plugin._last_image_prompts)
            duplicate = json.loads(await self.plugin.draw_image(event, draft["plan_id"], caption="test caption~"))
            self.assertEqual(duplicate["status"], "integrating")
            self.assertEqual(len(self.plugin._tasks), 1)
            release.set()
            await self.finish()
        deliveries = self.ctx.send_message.call_args_list
        self.assertEqual(len(deliveries), 2)
        self.assertEqual(deliveries[0].args[1].text, record["caption"])
        self.assertEqual(deliveries[1].args[1].image, "generated")
        self.plugin.client.text_to_image.assert_awaited_once_with(record["final_prompt"])

    async def test_drawing_chatter_is_suppressed_only_for_its_event(self):
        event = Event()
        await self.prepare(event)
        event.result = types.SimpleNamespace(chain=["QINIU_DRAWING_DONE"], is_llm_result=lambda: True)
        await self.plugin.suppress_drawing_chatter(event)
        self.assertEqual(event.result.chain, [])
        # The second bot and ordinary messages are independent, even for the same group/user.
        other = Event()
        other.result = types.SimpleNamespace(chain=["正常聊天"], is_llm_result=lambda: True)
        await self.plugin.suppress_drawing_chatter(other)
        self.assertEqual(other.result.chain, ["正常聊天"])
        event.result = types.SimpleNamespace(chain=["插件图片"], is_llm_result=lambda: False)
        await self.plugin.suppress_drawing_chatter(event)
        self.assertEqual(event.result.chain, ["插件图片"])

    async def test_safety_retry_updates_plan_and_last_full_prompt(self):
        self.plugin.client.text_to_image.side_effect = [main.QiniuSafetyError(400, "safety"), ["generated"]]
        draft = await self.prepare()
        with patch.object(main, "rewrite_for_safety", AsyncMock(return_value="安全的完整替代提示词")):
            await self.plugin.draw_image(Event(), draft["plan_id"], caption="test caption~")
            await self.finish()
        self.assertIn("安全的完整替代提示词", await self.plugin.get_last_image_prompt(Event(), full=True))
        self.assertIn("安全的完整替代提示词", await self.plugin.get_drawing_plan(Event(), draft["plan_id"], full=True))
        self.assertIn("notice", json.loads(await self.plugin.get_last_image_prompt(Event())))

    async def test_planning_timeout_releases_guard_and_never_submits_image(self):
        async def slow(**kwargs):
            await asyncio.Event().wait()

        self.ctx.tool_loop_agent.side_effect = slow
        event = Event()
        with patch.object(main, "PLANNING_TOOL_TIMEOUT_SECONDS", .01):
            result = await asyncio.wait_for(self.plugin.prepare_drawing(event, "画猫"), timeout=.2)
        self.assertIn("超时", result)
        self.assertFalse(event.get_extra(main._SILENT_DRAWING))
        self.assertFalse(self.plugin._preparing)
        self.assertFalse(self.plugin.planner.plans)
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_base64_image_is_normalized_for_planning_and_kept_for_generation(self):
        encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
        self.ctx.llm_generate.return_value = response({"prompt": "参考图中主体", "summary": "参考原图", "image_mode": "reference"})
        event = Event(images=[Image(file="base64://" + encoded)])
        draft = await self.prepare(event)
        self.assertEqual(self.ctx.llm_generate.call_args.kwargs["image_urls"], ["data:image/png;base64," + encoded])
        self.assertEqual(self.plugin.planner.plans[draft["plan_id"]]["image_ref"], "base64://" + encoded)

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
