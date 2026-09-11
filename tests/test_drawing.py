"""Offline behavioral tests; AstrBot events/providers and network calls are mocked."""

import asyncio
import base64
import copy
import importlib
import json
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "qiniu_test_plugin"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


class Image:
    def __init__(self, url="", file=""):
        self.url, self.file = url, file

    @classmethod
    def fromBase64(cls, value):
        return cls(file="base64://" + value)


class Reply:
    def __init__(self, chain=None, id=None):
        self.chain, self.id = chain, id


class MessageChain:
    def __init__(self):
        self.parts = []

    def base64_image(self, data):
        self.parts.append(("image", data))
        return self

    def message(self, data):
        self.parts.append(("text", data))
        return self


def decorator(*args, **kwargs):
    return lambda fn: fn


class Star:
    def __init__(self, context):
        self.context = context


for name in ("astrbot", "astrbot.api", "astrbot.api.event", "astrbot.api.star", "astrbot.api.message_components"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["astrbot.api"].logger = logging.getLogger("test")
sys.modules["astrbot.api"].AstrBotConfig = dict
sys.modules["astrbot.api.event"].AstrMessageEvent = object
sys.modules["astrbot.api.event"].MessageChain = MessageChain
sys.modules["astrbot.api.event"].filter = types.SimpleNamespace(
    llm_tool=decorator, on_llm_request=decorator, event_message_type=decorator,
    EventMessageType=types.SimpleNamespace(ALL="all"),
    PlatformAdapterType=types.SimpleNamespace(AIOCQHTTP="aiocqhttp"),
)
sys.modules["astrbot.api.star"].Star = Star
sys.modules["astrbot.api.star"].Context = object
sys.modules["astrbot.api.star"].register = decorator
sys.modules["astrbot.api.message_components"].Image = Image
sys.modules["astrbot.api.message_components"].Reply = Reply
sys.modules["astrbot.api.message_components"].Plain = str

def module(name):
    return importlib.import_module(f"{PACKAGE}.{name}")


tasks = module("drawing_task")
store_module = module("generation_store")
references = module("reference_images")
rewriter = module("prompt_rewriter")
api = module("qiniu_api")
pipeline_module = module("drawing_pipeline")
main = module("main")
styles = module("style_presets")
messages = module("message_utils")

# Real 1x1 PNG, not a URL or a model-generated substitute.
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC")
B64 = base64.b64encode(PNG).decode()


class Event:
    def __init__(self, user="alice", group="group", segments=()):
        self.unified_msg_origin, self.user = group, user
        self.segments = list(segments)
        self.message_str = "画图"
        self.message_obj = types.SimpleNamespace(message_id="message-1")

    def get_sender_id(self):
        return self.user

    def get_group_id(self):
        return self.unified_msg_origin

    def get_messages(self):
        return self.segments

    def get_platform_name(self):
        return "test"


def character(id="kanon", name="中川花音", position="观看者左侧"):
    return {"id": id, "name": name, "work": "已知作品", "version": "校服",
            "position": position, "features": ["粉色短发", "黄色蝴蝶结"],
            "identity_status": "confirmed", "evidence": "作品角色资料与用户确认"}


def group_task():
    return {"operation": "create", "characters": [character(str(i), f"角色{i}", f"从左第{i + 1}位") for i in range(5)],
            "character_count": 5, "user_requirements": ["五人同框"]}


def stored_task(store, event, task=None, *, assets=()):
    return store.put((event.unified_msg_origin, event.user), PNG,
                     {"task": task or group_task(), "prompt": "五人同框已执行稿", "style": "净色动画壁纸",
                      "status": "success", "assessment": {"status": "ok", "issues": []}}, assets=assets)


class TaskTests(unittest.TestCase):
    def test_partial_edit_preserves_other_four(self):
        base = group_task()
        task = tasks.normalize_task({"operation": "edit", "characters": [{"id": "2", "name": "角色2", "features": ["修正发饰"]}]}, base)
        self.assertEqual(len(task["characters"]), 5)
        for i in (0, 1, 3, 4):
            self.assertEqual(task["characters"][i], base["characters"][i])
        self.assertEqual(base["characters"][2]["features"], ["粉色短发", "黄色蝴蝶结"])

    def test_count_duplicate_unknown_and_uncertain_rejected(self):
        invalid = [dict(group_task(), character_count=4),
                   {"operation": "create", "characters": [character(), character()]},
                   {"operation": "create", "surprise": 1},
                   {"operation": "create", "characters": [dict(character(), identity_status="uncertain")]},
                   {"operation": "create", "characters": [dict(character(), evidence="")]},
                   {"operation": "edit", "characters": [{"id": "new", "name": "新人"}]}]
        for task in invalid:
            with self.subTest(task=task), self.assertRaises(ValueError):
                tasks.normalize_task(task, group_task() if task["operation"] == "edit" else None)

    def test_structured_output_cannot_omit_or_reorder_characters(self):
        chars = group_task()["characters"]
        value = {"scene": "合照", "characters": [{"id": c["id"], "description": "人物描写"} for c in chars]}
        self.assertIsNotNone(tasks.parse_compilation(json.dumps(value), chars))
        value["characters"].reverse()
        self.assertIsNone(tasks.parse_compilation(json.dumps(value), chars))
        value["characters"].pop()
        self.assertIsNone(tasks.parse_compilation(json.dumps(value), chars))

    def test_render_has_separate_identified_characters(self):
        chars = group_task()["characters"]
        value = {"scene": "场景", "characters": [{"id": c["id"], "description": "外观"} for c in chars]}
        text = tasks.render_compilation(value, chars)
        self.assertEqual(text.count("）：外观"), 5)
        self.assertIn("人物 4（角色4", text)
        self.assertIn("\n\n", styles.clean_style_metadata(text)[0])

    def test_catalog_only_titles(self):
        self.assertEqual(styles.style_catalog_text(concise=True).splitlines(), ["- " + p.name for p in styles.STYLE_PRESETS])
        self.assertEqual(len(styles.STYLE_PRESETS), 13)


class StoreTests(unittest.TestCase):
    def test_owner_ttl_and_pinned_snapshot(self):
        now = [0]
        store = store_module.GenerationStore(clock=lambda: now[0])
        self.addCleanup(store.close)
        row = stored_task(store, Event())
        frozen = store.snapshot(("group", "alice"), "latest")
        with self.assertRaises(ValueError):
            store.get(("group", "bob"), row["id"])
        now[0] = 1800
        with self.assertRaises(ValueError):
            store.get(("group", "alice"), row["id"])
        self.assertFalse(Path(row["image_path"]).exists())
        self.assertEqual(frozen["image_bytes"], PNG)

    def test_attachments_count_toward_size_and_cleanup(self):
        store = store_module.GenerationStore(max_bytes=len(PNG) * 2)
        self.addCleanup(store.close)
        first = stored_task(store, Event(), assets=[{"binding": {"source": "ref_x", "role": "style"}, "bytes": PNG}])
        self.assertEqual(first["size"], 2 * len(PNG))
        stored_task(store, Event())
        self.assertFalse(Path(first["image_path"]).exists())
        self.assertFalse(Path(first["assets"][0]["image_path"]).exists())

    def test_record_limit(self):
        store = store_module.GenerationStore(max_records=2)
        self.addCleanup(store.close)
        first = stored_task(store, Event())
        stored_task(store, Event())
        stored_task(store, Event())
        self.assertEqual(len(store.records), 2)
        self.assertNotIn(first["id"], store.records)


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = types.SimpleNamespace(send_message=AsyncMock(), llm_generate=AsyncMock(),
                                             get_current_chat_provider_id=AsyncMock(return_value="vision"))
        self.plugin = main.QiniuImagePlugin(self.context, {"api_key": "test", "rewrite_provider_ids": ["vision", "fallback"]})
        self.event = Event()
        self.plugin._generate = AsyncMock(return_value=(B64, None))

    async def asyncTearDown(self):
        await self.plugin.terminate()

    def reference(self, char):
        return self.plugin.pipeline.store.put(("group", "alice"), PNG,
            {"subject": char["name"], "assessment": {"canonical_name": char["name"], "features": char["features"],
             "summary": "参考依据"}}, kind="reference")

    async def test_freeze_latest_before_other_generation_finishes(self):
        first = stored_task(self.plugin.pipeline.store, self.event)
        frozen = self.plugin.pipeline.freeze(self.event, {"operation": "edit", "base_generation_id": "latest"})
        stored_task(self.plugin.pipeline.store, self.event, {"operation": "create", "characters": []})
        self.assertEqual(frozen["base"]["id"], first["id"])
        with patch.object(pipeline_module, "rewrite", AsyncMock(return_value="只改花音，其余保留")), patch.object(pipeline_module, "visual_json", AsyncMock(return_value={"status": "ok", "issues": []})):
            output, _ = await self.plugin.pipeline.draw(self.event, "只改花音", frozen)
        self.assertEqual(output, B64)
        self.assertEqual(self.plugin._generate.await_args.args[2], [])

    async def test_five_references_reach_optimizer_but_not_image_api(self):
        task = group_task()
        for char in task["characters"]:
            char["reference_id"] = self.reference(char)["id"]
        frozen = self.plugin.pipeline.freeze(self.event, task)
        compile_mock = AsyncMock(return_value="五个人各自保留外观")
        check_mock = AsyncMock(return_value={"status": "issues", "issues": ["中间人物发饰不匹配"]})
        with patch.object(pipeline_module, "rewrite", compile_mock), patch.object(pipeline_module, "visual_json", check_mock):
            output, note = await self.plugin.pipeline.draw(self.event, "五人同框", frozen)
        self.assertEqual(output, B64)
        self.assertEqual(len(compile_mock.await_args.kwargs["image_urls"]), 5)
        self.assertEqual(self.plugin._generate.await_args.args[2], [])
        self.assertEqual(len(check_mock.await_args.kwargs["image_urls"]), 6)
        self.assertIn("中间人物发饰", note)
        self.plugin._generate.assert_awaited_once()
        self.assertEqual(self.plugin.pipeline.store.get(("group", "alice"), frozen["id"])["assessment"]["status"], "issues")

    async def test_style_reference_is_hidden_from_optimizer_and_image_api_without_input(self):
        style_ref = self.reference(character())["id"]
        task = {"operation": "create", "characters": [character()],
                "image_roles": [{"source": style_ref, "role": "style"}]}
        frozen = self.plugin.pipeline.freeze(self.event, task)
        compile_mock = AsyncMock(return_value="只使用文字方案的新构图")
        with patch.object(pipeline_module, "rewrite", compile_mock):
            output, _ = await self.plugin.pipeline.draw(self.event, "新构图", frozen)
        self.assertEqual(output, B64)
        self.assertEqual(compile_mock.await_args.kwargs["image_urls"], [])
        self.assertEqual(self.plugin._generate.await_args.args[2], [])

    async def test_explicit_character_reference_is_not_edit(self):
        event = Event(segments=[Image(file="base64://" + B64)])
        task = {"operation": "create", "characters": [character()],
                "image_roles": [{"source": "input:1", "role": "character", "character_id": "kanon"}]}
        with patch.object(pipeline_module, "rewrite", AsyncMock(return_value="新构图")) as mock:
            output, _ = await self.plugin.pipeline.draw(event, "参考人物画新图", self.plugin.pipeline.freeze(event, task))
        self.assertEqual(output, B64)
        self.assertFalse(mock.await_args.kwargs["has_image"])
        self.assertEqual(self.plugin._generate.await_args.args[2], ["base64://" + B64])
        row = self.plugin.pipeline.store.get(("group", "alice"))
        self.assertTrue(row["task"]["characters"][0]["reference_id"].startswith("ref_"))
        self.assertFalse(row["assets"][0]["binding"]["source"].startswith("input:"))

    async def test_new_input_replaces_saved_character_reference(self):
        char = character()
        reference = self.reference(char)
        char["reference_id"] = reference["id"]
        base = stored_task(self.plugin.pipeline.store, self.event,
                           {"operation": "create", "characters": [char]},
                           assets=[{"binding": {"source": reference["id"], "role": "character", "character_id": "kanon"}, "bytes": PNG}])
        event = Event(segments=[Image(file="base64://" + B64)])
        task = {"operation": "edit", "base_generation_id": base["id"],
                "image_roles": [{"source": "input:1", "role": "character", "character_id": "kanon"}]}
        frozen = self.plugin.pipeline.freeze(event, task)
        assets, edit = await self.plugin.pipeline.resolve_images(event, frozen)
        self.assertTrue(edit)
        self.assertEqual(len(assets), 2)
        self.assertEqual(assets[1]["binding"]["source"], "input:1")

    async def test_character_reference_cannot_be_relabelled_as_style(self):
        char = character()
        char["reference_id"] = self.reference(char)["id"]
        task = {"operation": "create", "characters": [char],
                "image_roles": [{"source": char["reference_id"], "role": "style"}]}
        with self.assertRaises(ValueError):
            self.plugin.pipeline.freeze(self.event, task)

    async def test_two_characters_cannot_silently_share_reference(self):
        char = character()
        char["reference_id"] = self.reference(char)["id"]
        other = {**char, "id": "other", "position": "右侧"}
        frozen = self.plugin.pipeline.freeze(self.event, {"operation": "create", "characters": [char, other]})
        with self.assertRaises(ValueError):
            await self.plugin.pipeline.resolve_images(self.event, frozen)

    async def test_two_explicit_references_for_same_character_rejected(self):
        char = character()
        char["reference_id"] = self.reference(char)["id"]
        task = {"operation": "create", "characters": [char],
                "image_roles": [{"source": "input:1", "role": "character", "character_id": char["id"]}]}
        with self.assertRaises(ValueError):
            self.plugin.pipeline.freeze(self.event, task)

    async def test_explicit_task_ignores_unbound_unreadable_image(self):
        event = Event(segments=[Image(file="unreadable")])
        frozen = self.plugin.pipeline.freeze(event, {"operation": "create"})
        with patch.object(pipeline_module, "resolve_input_images", AsyncMock(side_effect=ValueError)) as resolver:
            assets, edit = await self.plugin.pipeline.resolve_images(event, frozen)
        self.assertEqual(assets, [])
        self.assertFalse(edit)
        resolver.assert_not_awaited()

    async def test_selected_style_is_saved_in_generation(self):
        async def compile_prompt(*args, **kwargs):
            kwargs["result_metadata"]["style"] = "净色动画壁纸"
            return "自动选用画风的执行稿"
        with patch.object(pipeline_module, "rewrite", compile_prompt):
            await self.plugin.pipeline.draw(self.event, "画图", self.plugin.pipeline.freeze(self.event, {"operation": "create"}))
        self.assertEqual(self.plugin.pipeline.store.get(("group", "alice"))["style"], "净色动画壁纸")

    async def test_user_input_can_use_reference_verification(self):
        event = Event(segments=[Image(file="base64://" + B64)])
        self.context.llm_generate.return_value = types.SimpleNamespace(completion_text=json.dumps({
            "status": "confirmed", "selected_index": 1, "canonical_name": "中川花音",
            "features": ["粉色短发"], "summary": "用户提供作品人物和参考图"}))
        result = await self.plugin.pipeline.prepare_reference(event, "中川花音", ["input:1"], "用户说明作品人物")
        self.assertEqual(result["status"], "confirmed")
        row = self.plugin.pipeline.store.get(("group", "alice"), result["reference_id"], kind="reference")
        self.assertEqual(row["source_url"], "input:1")
        self.assertEqual(len(self.context.llm_generate.await_args.kwargs["image_urls"]), 1)

    async def test_explicit_missing_and_cross_owner_base_rejected(self):
        base = stored_task(self.plugin.pipeline.store, self.event)
        with self.assertRaises(ValueError):
            self.plugin.pipeline.freeze(Event(user="bob"), {"operation": "edit", "base_generation_id": base["id"]})
        with self.assertRaises(ValueError):
            self.plugin.pipeline.freeze(self.event, {"operation": "edit", "base_generation_id": "gen_missing"})

    async def test_edit_requires_original(self):
        output, _ = await self.plugin.pipeline.draw(self.event, "改图", self.plugin.pipeline.freeze(self.event, {"operation": "edit"}))
        self.assertIsNone(output)
        self.plugin._generate.assert_not_awaited()

    async def test_qq_404_does_not_silently_generate(self):
        event = Event(segments=[Image(url="https://gchat.qpic.cn/missing")])
        self.plugin.pipeline.fetcher.fetch = AsyncMock(side_effect=ValueError("HTTP 404"))
        output, _ = await self.plugin.pipeline.draw(event, "编辑")
        self.assertIsNone(output)
        self.plugin._generate.assert_not_awaited()

    async def test_failed_assessment_still_delivers_one_image(self):
        frozen = self.plugin.pipeline.freeze(self.event, group_task())
        with patch.object(pipeline_module, "rewrite", AsyncMock(return_value="五人")), patch.object(pipeline_module, "visual_json", AsyncMock(return_value=None)):
            output, note = await self.plugin.pipeline.draw(self.event, "五人", frozen)
        self.assertEqual(output, B64)
        self.assertIn("未完成", note)
        self.plugin._generate.assert_awaited_once()

    async def test_hook_preserves_persona_scopes_records_and_deduplicates(self):
        stored_task(self.plugin.pipeline.store, Event(user="bob"))
        row = stored_task(self.plugin.pipeline.store, self.event)
        original_tool = types.SimpleNamespace(name="draw_image", parameters={}, active=True)
        req = types.SimpleNamespace(system_prompt="原人设", func_tool=types.SimpleNamespace(tools=[original_tool]))
        await self.plugin.on_llm_request(self.event, req)
        await self.plugin.on_llm_request(self.event, req)
        self.assertTrue(req.system_prompt.startswith("原人设"))
        self.assertEqual(req.system_prompt.count("<qiniu_drawing_context>"), 1)
        self.assertIn(row["id"], req.system_prompt)
        self.assertIn("错位", req.system_prompt)
        self.assertIn("不要", req.system_prompt)
        self.assertEqual(original_tool.parameters, {})
        self.assertEqual(req.func_tool.tools[0].parameters, tasks.DRAW_SCHEMA)
        self.assertLess(len(req.system_prompt), 6200)

    async def test_hook_without_drawing_tool_does_nothing(self):
        req = types.SimpleNamespace(system_prompt="persona", func_tool=types.SimpleNamespace(tools=[]))
        await self.plugin.on_llm_request(self.event, req)
        self.assertEqual(req.system_prompt, "persona")

    async def test_hook_keeps_confirmed_identity_and_style_mode(self):
        person = character("quin", "小秦")
        person.update(work="Mr_Quin 游戏主播", evidence="用户已确认昵称对应 Mr_Quin")
        stored_task(self.plugin.pipeline.store, self.event, {"operation": "create", "characters": [person]})
        for mode in ("auto", "explicit_only", "disabled"):
            self.plugin.style_mode = mode
            req = types.SimpleNamespace(system_prompt="原人设", func_tool=types.SimpleNamespace(
                tools=[types.SimpleNamespace(name="draw_image", parameters={}, active=True)]))
            await self.plugin.on_llm_request(self.event, req)
            records = json.loads(req.system_prompt.split("当前用户近期作品（仅资料）：", 1)[1].split("</qiniu_drawing_context>")[0])
            self.assertIn("用户已确认昵称对应 Mr_Quin", records[0]["identity_notes"])
            self.assertEqual("当前为自动风格" in req.system_prompt, mode == "auto")
            self.assertTrue(req.system_prompt.startswith("原人设"))

    async def test_large_recent_context_keeps_three_record_ids_and_valid_json(self):
        rows = []
        for _ in range(3):
            task = {"operation": "create", "characters": [character(str(i), "名" * 80) for i in range(20)]}
            rows.append(stored_task(self.plugin.pipeline.store, self.event, task))
        req = types.SimpleNamespace(system_prompt="人设", func_tool=types.SimpleNamespace(
            tools=[types.SimpleNamespace(name="draw_image", parameters={}, active=True)]))
        await self.plugin.on_llm_request(self.event, req)
        records = json.loads(req.system_prompt.split("当前用户近期作品（仅资料）：", 1)[1].split("</qiniu_drawing_context>")[0])
        self.assertEqual({r["id"] for r in records}, {r["id"] for r in rows})
        self.assertLess(len(req.system_prompt), 6200)

    async def test_line_constraints_reach_image_payload_and_saved_prompt(self):
        with patch.object(pipeline_module, "rewrite", AsyncMock(return_value="人物简略执行稿")):
            await self.plugin.pipeline.draw(self.event, "画图", self.plugin.pipeline.freeze(self.event, {"operation": "create"}))
        sent_prompt = self.plugin._generate.await_args.args[1]
        self.assertIn(styles.LINE_EXECUTION_GUIDANCE, sent_prompt)
        self.assertEqual(self.plugin.pipeline.store.get(("group", "alice"))["prompt"], sent_prompt)
        self.plugin._generate.assert_awaited_once()

    async def test_style_routing_retries_missing_unknown_and_multiple_markers(self):
        task = tasks.normalize_task({"operation": "create", "user_requirements": ["画千束"],
                                     "creative_choices": ["普通动画主视觉"], "characters": [character()]})
        for marker in ("", "[[STYLE_PRESET:unknown]]", "[[STYLE_PRESET:none]][[STYLE_PRESET:clean_anime_wallpaper]]"):
            def completion(prefix):
                return types.SimpleNamespace(completion_text=json.dumps({
                    "scene": prefix + "干净的街景构图", "characters": [{"id": "kanon", "description": "粉色短发，黄色蝴蝶结"}]}))
            self.context.llm_generate.reset_mock()
            self.context.llm_generate.side_effect = [completion(marker), completion("[[STYLE_PRESET:clean_anime_wallpaper]]")]
            metadata = {}
            result = await rewriter.rewrite(self.context, "group", "普通动画主视觉", has_image=False,
                                            drawing_task=task, provider_id="vision", result_metadata=metadata)
            self.assertEqual(self.context.llm_generate.await_count, 2)
            self.assertEqual(metadata["style"], "净色动画壁纸")
            self.assertNotIn("STYLE_PRESET", result)
            payload = self.context.llm_generate.await_args.kwargs
            self.assertIn(json.dumps(task, ensure_ascii=False), payload["prompt"])
            self.assertIn("风格来源规则", payload["system_prompt"])

    async def test_style_modes_do_not_route_model_choice_as_user_request(self):
        task = tasks.normalize_task({"operation": "create", "user_requirements": ["画图"],
                                     "creative_choices": ["净色动画壁纸"]})
        response = types.SimpleNamespace(completion_text=json.dumps({"scene": "普通人物插画", "characters": []}))
        self.context.llm_generate.return_value = response
        for mode in ("explicit_only", "disabled"):
            metadata = {}
            result = await rewriter.rewrite(self.context, "group", "净色动画壁纸", has_image=False,
                                            drawing_task=task, style_mode=mode, provider_id="vision", result_metadata=metadata)
            self.assertIsNotNone(result)
            self.assertNotEqual(metadata["style"], "净色动画壁纸")
            self.assertNotIn("以下是内置风格库", self.context.llm_generate.await_args.kwargs["system_prompt"])
        task["user_requirements"] = ["用净色动画壁纸画图"]
        guidance = styles.build_style_guidance("画图", mode="explicit_only", strength="normal", has_image=False, drawing_task=task)
        self.assertIn("clean_anime_wallpaper", guidance)
        self.assertNotIn("id: glitch_rectangles", guidance)

    async def test_explicit_none_does_not_log_rejected_style_as_selected(self):
        task = tasks.normalize_task({"operation": "create", "user_requirements": ["不要净色动画壁纸，画水彩"]})
        self.context.llm_generate.return_value = types.SimpleNamespace(completion_text=json.dumps({
            "scene": "[[STYLE_PRESET:none]] 水彩风景画", "characters": []}))
        metadata = {}
        result = await rewriter.rewrite(self.context, "group", "不要净色动画壁纸，画水彩", has_image=False,
                                        drawing_task=task, provider_id="vision", result_metadata=metadata)
        self.assertIn("水彩", result)
        self.assertNotEqual(metadata["style"], "净色动画壁纸")

    async def test_background_returns_id_and_sends_note(self):
        with patch.object(self.plugin.pipeline, "draw", AsyncMock(return_value=(B64, "检查说明"))):
            result = json.loads(await self.plugin.draw_image(self.event, "画图", {"operation": "create"}))
            self.assertEqual(result["status"], "accepted")
            await asyncio.gather(*self.plugin._tasks)
        chain = self.context.send_message.await_args.args[1]
        self.assertEqual(chain.parts, [("image", B64), ("text", "检查说明")])

    async def test_visual_typeerror_fallback_keeps_images(self):
        self.context.llm_generate.side_effect = [TypeError("images unsupported"), types.SimpleNamespace(completion_text='{"status":"ok"}')]
        result = await rewriter.visual_json(self.context, "group", prompt="check", image_urls=[references.image_data(PNG)],
            validate=lambda r: r.get("status") == "ok", purpose="test", provider_id="bad", fallback_provider_ids=["good"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.assertTrue(all(call.kwargs.get("image_urls") for call in self.context.llm_generate.await_args_list))

    async def test_structured_rewriter_real_roundtrip_and_retry(self):
        task = tasks.normalize_task(group_task())
        valid = {"scene": "[[STYLE_PRESET:clean_anime_wallpaper]] 清爽五人合照", "characters": [{"id": c["id"], "description": "粉色头发与独立配饰"} for c in task["characters"]]}
        invalid = copy.deepcopy(valid)
        invalid["characters"].pop()
        self.context.llm_generate.side_effect = [types.SimpleNamespace(completion_text=json.dumps(invalid)), types.SimpleNamespace(completion_text=json.dumps(valid))]
        result = await rewriter.rewrite(self.context, "group", "五人合照", has_image=False, drawing_task=task,
                                       evidence_context=tasks.task_context(task, None, []), provider_id="vision")
        self.assertIn("人物 4", result)
        self.assertNotIn("STYLE_PRESET", result)
        self.assertIn("\n\n", result)
        self.assertEqual(self.context.llm_generate.await_count, 2)

    async def test_api_keeps_multi_image_order(self):
        client = api.QiniuImageClient({"api_key": "test"})
        client._get_session = AsyncMock(return_value=object())
        client._post_json_with_retry = AsyncMock(return_value={"data": [{"b64_json": B64}]})
        result = await client.images_to_image(["base64://" + B64, "https://example.org/two.png"], "五人")
        payload = client._post_json_with_retry.await_args.args[2]
        self.assertEqual(len(payload["images"]), 2)
        self.assertTrue(payload["images"][0]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(payload["images"][1]["image_url"], "https://example.org/two.png")
        self.assertEqual(result, [B64])

    async def test_resolver_rejects_private_dns_address(self):
        resolver = references.PublicResolver()
        resolver.delegate.resolve = AsyncMock(return_value=[{"host": "127.0.0.1"}])
        with self.assertRaises(OSError):
            await resolver.resolve("public-looking.example", 443)
        await resolver.close()

    async def test_reference_selection_uses_validated_image_not_first(self):
        self.plugin.pipeline.fetcher.candidates = AsyncMock(return_value=([
            {"bytes": PNG, "source_url": "https://example.org/one", "url": "https://example.org/logo.png", "label": "logo", "page_text": "站点标志"},
            {"bytes": PNG, "source_url": "https://example.org/two", "url": "https://example.org/person.png", "label": "角色", "page_text": "人物资料"},
        ], []))
        self.context.llm_generate.return_value = types.SimpleNamespace(completion_text=json.dumps({
            "status": "confirmed", "selected_index": 2, "canonical_name": "中川花音", "features": ["粉色短发"], "summary": "第二页角色资料"}))
        result = await self.plugin.pipeline.prepare_reference(self.event, "中川花音", ["https://example.org"], "作品角色资料")
        row = self.plugin.pipeline.store.get(("group", "alice"), result["reference_id"], kind="reference")
        self.assertEqual(row["image_url"], "https://example.org/person.png")
        self.assertEqual(len(self.context.llm_generate.await_args.kwargs["image_urls"]), 2)


class WebTests(unittest.TestCase):
    def test_public_url_blocks_local_credentials_and_ports(self):
        for url in ("file:///tmp/a", "http://127.0.0.1/a", "http://10.1.2.3/a", "http://[::ffff:127.0.0.1]/a",
                    "http://localhost/a", "http://foo.local/a", "http://user:password@example.org/a", "http://example.org:22/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                references.public_url(url)

    def test_html_relative_lazy_images_caption_and_cover(self):
        parser = references.PageImages("https://example.org/wiki/person")
        parser.feed('<meta property="og:image" content="/logo.png"><script>evil()</script><figure><img src="loading.gif" data-src="../person.png" alt="角色"><figcaption>官方立绘</figcaption></figure>')
        row = next(r for r in parser.images if "person.png" in r["url"])
        self.assertEqual(row["url"], "https://example.org/person.png")
        self.assertIn("官方立绘", row["label"])
        self.assertLess(row["rank"], parser.images[0]["rank"])
        self.assertNotIn("evil()", parser.text)


if __name__ == "__main__":
    unittest.main()
