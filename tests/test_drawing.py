"""提示词流程与图片处理的离线测试；模型、图片服务和平台均为 mock。"""

import asyncio
import importlib
import json
import logging
import sys
import types
import unittest
from pathlib import Path
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


for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star",
             "astrbot.api.message_components"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["astrbot.api"].logger = logging.getLogger("drawing_tests")
sys.modules["astrbot.api"].AstrBotConfig = dict
sys.modules["astrbot.api.event"].AstrMessageEvent = object
sys.modules["astrbot.api.event"].MessageChain = Chain
sys.modules["astrbot.api.event"].filter = types.SimpleNamespace(
    llm_tool=decorator, event_message_type=decorator,
    EventMessageType=types.SimpleNamespace(ALL="all"),
    PlatformAdapterType=types.SimpleNamespace(AIOCQHTTP="aiocqhttp"),
)
sys.modules["astrbot.api.star"].Star = Star
sys.modules["astrbot.api.star"].Context = object
sys.modules["astrbot.api.star"].register = decorator
sys.modules["astrbot.api.message_components"].Image = Image
sys.modules["astrbot.api.message_components"].Reply = Reply

main = importlib.import_module("drawing_plugin.main")
plans = importlib.import_module("drawing_plugin.drawing_plan")
optimizer = importlib.import_module("drawing_plugin.prompt_optimizer")
qiniu = importlib.import_module("drawing_plugin.qiniu_api")
styles = importlib.import_module("drawing_plugin.style_presets")
utils = importlib.import_module("drawing_plugin.message_utils")

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"


def completion(value):
    return types.SimpleNamespace(completion_text=value)


def optimizer_json(prompt="一只猫", style="klein_order"):
    return completion(json.dumps({"prompt": prompt, "style": style}, ensure_ascii=False))


class Event:
    def __init__(self, *, user="alice", mid="1", images=(), text="画图 一只猫"):
        self.unified_msg_origin = "group"
        self.user, self.images, self.message_str = user, images, text
        self.message_obj = types.SimpleNamespace(message_id=mid)
        self.stopped = False

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


class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ctx = types.SimpleNamespace(
            get_current_chat_provider_id=AsyncMock(return_value="optimizer"),
            send_message=AsyncMock(),
        )
        self.ctx.llm_generate = AsyncMock(return_value=optimizer_json())
        self.plugin = main.QiniuImagePlugin(
            self.ctx,
            {"api_key": "test-key", "rewrite_provider_ids": ["optimizer"],
             "triggers": ["画图"]},
        )
        self.plugin.client.text_to_image = AsyncMock(return_value=["generated"])
        self.plugin.client.image_to_image = AsyncMock(return_value=["edited"])

    async def asyncTearDown(self):
        await self.plugin.terminate()

    async def drain(self):
        await asyncio.sleep(0.05)

    async def test_chat_preserves_full_plan_and_submits_integrated_prompt(self):
        draft = "蓝色短发、金色眼睛的 Bot 在窗边自拍，半身构图，柔和侧光。"
        final = "蓝色短发、金色眼睛的 Bot 在窗边自拍，半身构图，蓝白配色，柔和侧光与清晰轮廓。"
        self.ctx.llm_generate.return_value = optimizer_json(final)
        self.plugin.style_strength = "subtle"
        reply = await self.plugin.draw_image(
            Event(text="看你的自拍"), draft, "蓝色短发、金色眼睛", False, True,
        )
        self.assertIn("已受理", reply)
        await self.drain()
        self.ctx.llm_generate.assert_awaited_once()
        request = json.loads(self.ctx.llm_generate.await_args.kwargs["prompt"])
        self.assertEqual(request["plan"], draft)
        self.assertEqual(request["user_message"], "看你的自拍")
        self.assertEqual(request["subject_info"], "蓝色短发、金色眼睛")
        self.assertEqual(self.plugin.client.text_to_image.await_count, 1)
        self.assertEqual(self.plugin.client.text_to_image.await_args.args[0], final)
        instruction = self.ctx.llm_generate.await_args.kwargs["system_prompt"]
        self.assertIn(styles.QUALITY_GUIDANCE, instruction)
        self.assertIn(optimizer.STRENGTH_RULES["subtle"], instruction)
        stored = json.loads(await self.plugin.get_last_image_prompt(Event()))
        self.assertEqual(stored["prompt"], final)
        self.assertTrue(stored["integrated"])

    async def test_optimizer_failure_does_not_send_short_intent_to_image_model(self):
        self.ctx.llm_generate = AsyncMock(return_value=completion("不是 JSON"))
        result = await self.plugin._run_request(Event(), plans.DrawingRequest(prompt="一只猫"))
        self.assertIsNone(result[0])
        self.assertIn("优化模型", result[1])
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_keyword_trigger_uses_original_text_without_optimizer(self):
        event = Event(text="画图 一只猫", images=())
        rows = [row async for row in self.plugin.on_message(event)]
        self.assertEqual(rows[0][0].file, "base64://generated")
        self.ctx.llm_generate.assert_not_awaited()
        submitted = self.plugin.client.text_to_image.await_args.args[0]
        self.assertEqual(submitted, "一只猫\n\n" + styles.QUALITY_SECTION)
        stored = json.loads(await self.plugin.get_last_image_prompt(event))
        self.assertEqual(stored["prompt"], submitted)
        self.assertFalse(stored["integrated"])

    async def test_last_plan_is_snapshotted_for_revision(self):
        event = Event(user="alice")
        self.plugin._remember_image_prompt(
            event, plans.ImagePlan("上一张完整画面", "klein_order"), False, True,
        )
        self.ctx.llm_generate.reset_mock()
        await self.plugin.draw_image(event, "把帽子改蓝", "", True, True)
        await self.drain()
        request = json.loads(self.ctx.llm_generate.await_args.kwargs["prompt"])
        self.assertEqual(request["previous"]["prompt"], "上一张完整画面")
        self.assertEqual(request["previous"]["style"], "klein_order")
        self.assertEqual(request["plan"], "把帽子改蓝")

    async def test_disabled_styles_rejects_optimizer_style(self):
        self.plugin.enable_styles = False
        result = await optimizer.optimize_prompt(
            self.ctx, "group", plans.DrawingRequest(prompt="一只猫"), has_image=False,
            enable_styles=False, provider_id="optimizer",
        )
        self.assertIsNone(result)

    async def test_single_person_clean_style_is_retried_before_image_submission(self):
        self.ctx.llm_generate.side_effect = [
            completion(json.dumps({"prompt": "千束的净色壁纸", "style": "clean_anime_wallpaper", "people_count": 1})),
            optimizer_json("千束的蓝白错位窗口海报，半身构图。", "window_overlay_poetic"),
        ]
        result = await self.plugin._run_request(Event(), plans.DrawingRequest(prompt="画千束的壁纸"))
        self.assertEqual(result[0], "generated")
        self.assertEqual(self.ctx.llm_generate.await_count, 2)
        self.plugin.client.text_to_image.assert_awaited_once_with("千束的蓝白错位窗口海报，半身构图。")
        stored = json.loads(await self.plugin.get_last_image_prompt(Event()))
        self.assertEqual(stored["style"], "window_overlay_poetic")

    async def test_repeated_invalid_clean_style_never_reaches_image_api(self):
        self.ctx.llm_generate.return_value = optimizer_json("千束", "clean_anime_wallpaper")
        result = await self.plugin._run_request(Event(), plans.DrawingRequest(prompt="千束"))
        self.assertIsNone(result[0])
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_style_catalog_has_one_source_for_query_and_optimizer(self):
        query = styles.style_catalog_text()
        optimizer_catalog = styles.style_catalog_text(include_prompts=True)
        self.assertIn("klein_order", query)
        self.assertIn("klein_order", optimizer_catalog)
        self.assertIn("风格参考", optimizer_catalog)

    async def test_chat_edit_and_reference_keep_image_authority_without_template(self):
        for keep_layout, line in ((True, styles.IMAGE_AUTHORITY_LINE),
                                  (False, styles.IMAGE_REFERENCE_LINE)):
            with self.subTest(keep_layout=keep_layout):
                self.ctx.llm_generate.return_value = completion(json.dumps({
                    "prompt": "将帽子改为蓝色，其余保持。", "style": "",
                    "style_exception": "preserve_image" if keep_layout else "preserve_plan",
                }))
                with patch.object(main, "resolve_input_image", AsyncMock(return_value="https://example.com/a.png")):
                    result = await self.plugin._run_request(Event(), plans.DrawingRequest(
                        prompt="把帽子改蓝", keep_layout=keep_layout,
                    ))
                self.assertEqual(result[0], "edited")
                submitted = self.plugin.client.image_to_image.await_args.args[1]
                self.assertEqual(submitted, "将帽子改为蓝色，其余保持。\n\n" + line)
                self.assertEqual(json.loads(await self.plugin.get_last_image_prompt(Event()))["prompt"], submitted)
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_safety_retry_keeps_integrated_mode_and_records_actual_success(self):
        final = "锦木千束，红黑拼贴海报，半身侧向构图，清晰的面部与服装明暗。"
        safe = "锦木千束，完整制服，红黑拼贴海报，半身侧向构图，清晰明暗。"
        self.ctx.llm_generate.return_value = optimizer_json(final, "cream_red_circuit")
        self.plugin.client.text_to_image.side_effect = [qiniu.QiniuSafetyError(400, "safety"), ["safe"]]
        with patch.object(main, "rewrite_for_safety", AsyncMock(return_value=safe)) as rewrite:
            result = await self.plugin._run_request(Event(), plans.DrawingRequest(prompt=final))
        self.assertEqual(result[0], "safe")
        self.assertEqual(rewrite.await_args.args[2], final)
        self.assertEqual([call.args[0] for call in self.plugin.client.text_to_image.await_args_list], [final, safe])
        record = self.plugin._last_image_prompts[self.plugin._prompt_key(Event())]
        self.assertTrue(record.plan.integrated)
        self.assertEqual(record.submitted_prompt, safe)
        self.assertEqual(record.plan.style, "cream_red_circuit")
        self.plugin.client.text_to_image.side_effect = None
        self.plugin.client.text_to_image.return_value = ["revised"]
        await self.plugin._run_request(Event(), plans.DrawingRequest(prompt="把背景改蓝", previous=record))
        previous = json.loads(self.ctx.llm_generate.await_args.kwargs["prompt"])["previous"]
        self.assertEqual(previous["prompt"], safe)
        self.assertEqual(previous["style"], "cream_red_circuit")

    async def test_keyword_safety_retry_keeps_existing_quality_append(self):
        self.plugin.client.text_to_image.side_effect = [qiniu.QiniuSafetyError(400, "safety"), ["safe"]]
        with patch.object(main, "rewrite_for_safety", AsyncMock(return_value="安全的角色插画")) as rewrite:
            result = await self.plugin._run_request(Event(), plans.DrawingRequest(prompt="角色插画", optimize=False))
        self.assertEqual(result[0], "safe")
        self.ctx.llm_generate.assert_not_awaited()
        self.assertEqual(rewrite.await_args.args[2], "角色插画")
        self.assertEqual(self.plugin.client.text_to_image.await_args.args[0], "安全的角色插画\n\n" + styles.QUALITY_SECTION)

    async def test_failed_generation_does_not_replace_record_or_trigger_safety(self):
        event = Event()
        self.plugin._remember_image_prompt(event, plans.ImagePlan("原稿", integrated=True), False, True)
        before = self.plugin._last_image_prompts[self.plugin._prompt_key(event)]
        self.plugin.client.text_to_image.side_effect = qiniu.QiniuAuthError(401, "unauthorized")
        with patch.object(main, "rewrite_for_safety", AsyncMock()) as rewrite:
            result = await self.plugin._run_request(event, plans.DrawingRequest(prompt="新稿"))
        self.assertIsNone(result[0])
        rewrite.assert_not_awaited()
        self.assertIs(self.plugin._last_image_prompts[self.plugin._prompt_key(event)], before)

    async def test_revision_snapshot_survives_later_record_change_and_user_isolation(self):
        event = Event(user="alice")
        self.plugin._remember_image_prompt(event, plans.ImagePlan("旧执行稿", integrated=True), False, True)
        with patch.object(self.plugin, "_load_input_image", AsyncMock(return_value=None)):
            await self.plugin.draw_image(event, "把帽子改蓝", use_last_image=True)
            self.plugin._remember_image_prompt(event, plans.ImagePlan("新执行稿", integrated=True), False, True)
            await self.drain()
        request = json.loads(self.ctx.llm_generate.await_args.kwargs["prompt"])
        self.assertEqual(request["previous"]["prompt"], "旧执行稿")
        count = self.ctx.llm_generate.await_count
        reply = await self.plugin.draw_image(Event(user="bob"), "把帽子改蓝", use_last_image=True)
        self.assertIn("没有可沿用", reply)
        self.assertEqual(self.ctx.llm_generate.await_count, count)

    async def test_input_image_is_sent_to_edit_api(self):
        event = Event(images=[Image(url="https://example.com/original.png")])
        with patch.object(main, "resolve_input_image", AsyncMock(return_value="https://example.com/original.png")):
            result = await self.plugin._run_request(
                event, plans.DrawingRequest(prompt="把帽子改蓝", optimize=False),
            )
        self.assertEqual(result[0], "edited")
        self.assertEqual(self.plugin.client.image_to_image.await_args.args[0], "https://example.com/original.png")
        self.plugin.client.text_to_image.assert_not_awaited()

    async def test_response_selection_skips_invalid_item_and_keeps_selected_source(self):
        client = qiniu.QiniuImageClient({"api_key": "k", "model": "openai/gpt-image-2"})
        data = {"data": [{"url": "invalid"}, {"url": "https://example.com/a.png"}]}
        source = client._select_response_image(data)
        self.assertEqual(source.url, "https://example.com/a.png")
        client._download_bytes = AsyncMock(return_value=b"raw")
        result = await client._images_from_source(types.SimpleNamespace(), source)
        self.assertEqual(result, ["cmF3"])

    async def test_response_base64_is_normalized_once(self):
        client = qiniu.QiniuImageClient({"api_key": "k", "model": "openai/gpt-image-2"})
        source = client._select_response_image({"data": [{"b64_json": "data:image/png;base64," + PNG_B64}]})
        self.assertEqual(source.b64_json, PNG_B64)
        self.assertEqual(await client._images_from_source(types.SimpleNamespace(), source), [PNG_B64])

    async def test_duplicate_message_id_is_scoped_by_origin(self):
        first = Event(mid="77")
        second = Event(mid="77")
        second.unified_msg_origin = "other-group"
        self.assertFalse(self.plugin._dedup_hit(first))
        self.assertFalse(self.plugin._dedup_hit(second))
        self.assertTrue(self.plugin._dedup_hit(first))


class PromptTests(unittest.TestCase):
    def parse(self, data, *, has_image=False, request=None, enable_styles=True):
        return optimizer._parse_plan(
            json.dumps(data, ensure_ascii=False), request or plans.DrawingRequest(prompt="完整画面方案"),
            has_image=has_image, enable_styles=enable_styles,
        )

    def test_empty_style_accepts_existing_design_without_verbatim_evidence(self):
        for reason in ("external", "user_opt_out", "preserve_plan", "incompatible"):
            with self.subTest(reason=reason):
                plan = self.parse({"prompt": "水彩人物肖像，柔和侧光与纸张肌理。", "style": "", "style_exception": reason})
                self.assertTrue(plan.integrated)
                self.assertEqual(plan.style_exception, reason)
                self.assertEqual(styles.compose_prompt(plan.prompt, has_image=False, integrated=plan.integrated), plan.prompt)

    def test_clean_style_requires_multiple_people_and_input_evidence(self):
        request = plans.DrawingRequest(prompt="千束和泷奈一起合影，留出天空背景。")
        base = {"prompt": "千束和泷奈的清爽合影。", "style": "clean_anime_wallpaper",
                "people_evidence": "千束和泷奈一起合影"}
        for count in (None, 0, 1, True, "2", 2.5):
            with self.subTest(count=count), self.assertRaises(optimizer.ModelOutputError):
                self.parse({**base, "people_count": count}, request=request)
        for evidence in (None, "", "只有输出里出现的两个人", 2):
            with self.subTest(evidence=evidence), self.assertRaises(optimizer.ModelOutputError):
                self.parse({**base, "people_count": 2, "people_evidence": evidence}, request=request)
        plan = self.parse({**base, "people_count": 2}, request=request)
        self.assertEqual(plan.people_count, 2)
        self.assertEqual(plan.style, "clean_anime_wallpaper")
        self.assertTrue(plan.integrated)

    def test_clean_style_revision_uses_current_count_and_previous_evidence(self):
        previous = plans.ImageRecord(
            plans.ImagePlan("千束和泷奈合影", "clean_anime_wallpaper", integrated=True, people_count=2),
            False, True, 0,
        )
        request = plans.DrawingRequest(prompt="把天空改蓝", previous=previous)
        data = {"prompt": "千束和泷奈合影，蓝色天空。", "style": "clean_anime_wallpaper",
                "people_count": 2, "people_evidence": "千束和泷奈合影"}
        self.assertEqual(self.parse(data, request=request).people_count, 2)
        with self.assertRaises(optimizer.ModelOutputError):
            self.parse({**data, "prompt": "千束独照", "people_count": 1},
                       request=plans.DrawingRequest(prompt="只留下千束", previous=previous))

    def test_clean_style_is_last_and_not_default_priority(self):
        clean = next(preset for preset in styles.STYLE_PRESETS if preset.id == "clean_anime_wallpaper")
        self.assertLess(clean.auto_preference, 0)
        for include_prompts in (False, True):
            catalog = styles.style_catalog_text(include_prompts=include_prompts)
            entries = [line for line in catalog.splitlines() if line.startswith("- ")]
            self.assertIn("clean_anime_wallpaper", entries[-1])
            self.assertIn("低优先", entries[-1])
            self.assertNotIn("默认优先", entries[-1])

    def test_invalid_style_or_invalid_preservation_state_is_rejected(self):
        for data in (
            {"prompt": "人物", "style": "invented"},
            {"prompt": "人物", "style": ""},
            {"prompt": "人物", "style": "", "style_exception": "preserve_image"},
            {"prompt": "人物", "style": "", "style_exception": "preserve_previous"},
            {"prompt": "人物", "style": "klein_order", "style_exception": "external"},
            {"prompt": "", "style": "klein_order"},
        ):
            with self.subTest(data=data), self.assertRaises(optimizer.ModelOutputError):
                self.parse(data)
        with self.assertRaises(optimizer.ModelOutputError):
            self.parse({"prompt": "人物", "style": "", "style_exception": "preserve_image"},
                       has_image=True, request=plans.DrawingRequest(prompt="重画", keep_layout=False))

    def test_disabled_and_unstyled_revision_are_integrated(self):
        disabled = self.parse({"prompt": "人物肖像，保留侧光。", "style": ""}, enable_styles=False)
        self.assertTrue(disabled.integrated)
        self.assertEqual(disabled.style_exception, "disabled")
        previous = plans.ImageRecord(disabled, False, True, 0)
        revision = self.parse(
            {"prompt": "人物肖像，帽子改蓝，保留侧光。", "style": "", "style_exception": "preserve_previous"},
            request=plans.DrawingRequest(prompt="把帽子改蓝", previous=previous),
        )
        self.assertTrue(revision.integrated)

    def test_composition_is_idempotent_and_keeps_requested_style_text(self):
        body = "在海报上写出“米白红色电路图”，人物直立。"
        for integrated in (True, False):
            for keep_layout in (True, False):
                with self.subTest(integrated=integrated, keep_layout=keep_layout):
                    kwargs = dict(has_image=True, integrated=integrated, keep_layout=keep_layout)
                    prompt = styles.compose_prompt(body, **kwargs)
                    self.assertEqual(styles.compose_prompt(prompt, **kwargs), prompt)
                    self.assertTrue(prompt.startswith(body))
                    self.assertEqual(prompt.count(styles.QUALITY_HEAD), 0 if integrated else 1)
        legacy = body + "\n\n" + styles.STYLE_LEADS["normal"] + "\n旧模板"
        self.assertEqual(styles.compose_prompt(legacy, has_image=False, integrated=True), body)


class SafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_refusal_is_rejected_but_normal_prose_is_kept(self):
        safety = importlib.import_module("drawing_plugin.safety_rewriter")
        self.assertFalse(safety.is_prompt_text("抱歉，我无法改写"))
        self.assertTrue(safety.is_prompt_text("少女很高兴地跑到海边"))


if __name__ == "__main__":
    unittest.main()
