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


appearance = module("appearance")
references = module("character_ref")
tasks = module("draw_task")
rewriter = module("prompt_rewriter")
api = module("qiniu_api")
main = module("main")
styles = module("style_presets")

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


RECORD = {
    "silhouette": "高挑纤细，发量厚重",
    "hair": {"color": "银白", "length": "及腰长发", "style": "双马尾",
             "front": "齐刘海，右侧一缕呆毛"},
    "eyes": {"color": "酒红", "shape": "上挑凤眼"},
    "skin": "冷白",
    "build": "纤细修长",
    "outfit": {"pieces": ["白色衬衫", "深蓝百褶裙"], "cut": "收腰短外套", "trim": "金色滚边"},
    "headwear": "红色发箍",
    "accessory": "左耳银色耳坠",
    "palette": ["银白", "酒红", "深蓝"],
    "marks": "锁骨处星形刺青",
    "asymmetry": "左眼下方泪痣",
    "signature": "腰间怀表",
    "version": "夏季制服",
}


def character(id="kanon", name="中川花音", position="观看者左侧"):
    return {"id": id, "name": name, "work": "已知作品", "version": "校服", "position": position}


def group_characters():
    return [character(str(i), f"角色{i}", f"从左第{i + 1}位") for i in range(5)]


class AppearanceTests(unittest.TestCase):
    def test_record_is_clean_and_complete(self):
        self.assertEqual(appearance.coverage_problems(RECORD), [])
        self.assertEqual(appearance.fatal_problems(RECORD), [])
        self.assertTrue(appearance.is_usable(RECORD))

    def test_render_order_is_identity_importance(self):
        labels = [row.split("：", 1)[0] for row in appearance.render(RECORD).split("；")]
        self.assertEqual(labels[0], "剪影")
        self.assertEqual(labels[-1], "印记")
        self.assertLess(labels.index("发色"), labels.index("服装"))
        self.assertLess(labels.index("标志物"), labels.index("肤色"))
        # 不引入录制顺序之外的字段：字段顺序恒定，与 dict 插入顺序无关。
        shuffled = {key: RECORD[key] for key in reversed(list(RECORD))}
        self.assertEqual(appearance.render(shuffled), appearance.render(RECORD))

    def test_empty_fields_are_omitted_not_defaulted(self):
        sparse = {"silhouette": "小巧", "hair": {"color": "黑"}, "eyes": {}, "skin": "白皙",
                  "build": "娇小", "outfit": {"pieces": ["连衣裙"], "cut": "宽松"}}
        text = appearance.render(sparse)
        self.assertIn("剪影：小巧", text)
        self.assertNotIn("头饰", text)
        self.assertNotIn("配色", text)
        self.assertNotIn("配饰", text)

    def test_asymmetry_keeps_its_side(self):
        self.assertIn("不对称：左眼下方泪痣", appearance.render(RECORD))

    def test_lazy_placeholder_and_duplicate_answers_are_rejected(self):
        lazy = {"silhouette": "银发", "hair": {"color": "银发", "length": "银发", "style": "银发",
                                               "front": "银发"},
                "eyes": {"color": "银发", "shape": "银发"}, "skin": "银发", "build": "银发",
                "outfit": {"pieces": ["银发"], "cut": "银发"}}
        problems = appearance.coverage_problems(lazy)
        self.assertTrue(problems)
        self.assertTrue(any("重复" in problem for problem in problems))
        self.assertIn("eyes.shape 缺失", appearance.coverage_problems({**RECORD, "eyes": {"color": "酒红"}}))
        self.assertTrue(appearance.fatal_problems({**RECORD, "hair": {**RECORD["hair"], "style": "？"}}))
        self.assertTrue(appearance.fatal_problems({**RECORD, "silhouette": "高"}))
        self.assertEqual(appearance.fatal_problems(None), ["外观记录缺失"])
        # 自造字段在校验阶段就被 additionalProperties 之外的路径挡下：normalize 直接丢弃。
        self.assertNotIn("extra_key", appearance.normalize({**RECORD, "extra_key": "自造字段"}))

    def test_single_character_values_are_legal_only_in_short_value_fields(self):
        # "黑" 是合法的发色，"金" 是合法的配色词。拿长度卡它们会让一份完好的记录
        # 被判不合格，进而整个角色退化成纯文字出图——这是最不该发生的失败。
        short = {**RECORD, "hair": {**RECORD["hair"], "color": "黑"},
                 "eyes": {**RECORD["eyes"], "color": "红"},
                 "skin": "白", "palette": ["金"]}
        self.assertEqual(appearance.coverage_problems(short), [])
        self.assertTrue(appearance.is_usable(short))
        self.assertIn("发色：黑", appearance.render(short))
        # 描述性字段仍然不能只给一个字；ASCII 单字符在任何字段都是噪声。
        self.assertTrue(appearance.fatal_problems({**RECORD, "build": "高"}))
        # 可选字段上的瑕疵只在日志里留痕，不能连累整份记录——那会让角色退回纯文字出图。
        optional = {**RECORD, "signature": "刀", "palette": ["x"], "headwear": "未知"}
        self.assertEqual(appearance.fatal_problems(optional), [])
        self.assertTrue(appearance.is_usable(optional))
        self.assertTrue(appearance.coverage_problems(optional))
        rendered = appearance.render(optional)
        self.assertNotIn("头饰", rendered)
        self.assertNotIn("配色", rendered)
        self.assertNotIn("标志物", rendered)
        self.assertIn("发型：双马尾", rendered)

    def test_identity_core_drops_clothing_but_keeps_hair(self):
        core = appearance.render(RECORD, appearance.IDENTITY_CORE)
        self.assertIn("发型：双马尾", core)
        self.assertIn("瞳色：酒红", core)
        self.assertNotIn("服装：", core)
        self.assertNotIn("滚边：", core)
        self.assertNotIn("配色：", core)


class DrawTaskTests(unittest.TestCase):
    def test_characters_get_ids_and_multi_person_needs_positions(self):
        people = tasks.normalize_characters([{"name": "甲", "position": "左"}, {"name": "乙", "position": "右"}])
        self.assertEqual([p["id"] for p in people], ["p1", "p2"])
        with self.assertRaises(ValueError):
            tasks.normalize_characters([{"name": "甲"}, {"name": "乙"}])
        with self.assertRaises(ValueError):
            tasks.normalize_characters([{"name": "甲", "id": "x"},
                                        {"name": "乙", "id": "x", "position": "右"}])
        self.assertEqual(tasks.normalize_characters(None), [])

    def test_scene_must_match_characters_exactly(self):
        people = group_characters()
        value = {"scene": "合照", "characters": [{"id": p["id"], "description": "动作"} for p in people]}
        self.assertIsNotNone(tasks.parse_scene(json.dumps(value), people))
        reversed_value = copy.deepcopy(value)
        reversed_value["characters"].reverse()
        self.assertIsNone(tasks.parse_scene(json.dumps(reversed_value), people))
        short = copy.deepcopy(value)
        short["characters"].pop()
        self.assertIsNone(tasks.parse_scene(json.dumps(short), people))

    def test_render_puts_each_appearance_below_its_own_heading(self):
        people = group_characters()
        value = {"scene": "场景", "characters": [{"id": p["id"], "description": "动作"} for p in people]}
        records = {p["id"]: dict(RECORD, signature=f"{p['name']}的标志物") for p in people}
        text = tasks.render_prompt(value, people, records)
        self.assertEqual(text.count("）：动作"), 5)
        for person in people:
            heading = f"人物 {person['id']}（{person['name']} / 已知作品 / 校服；{person['position']}）：动作"
            self.assertIn(heading, text)
            # 外观块紧贴在它所属人物的标题行下方，不会串到别人身上。
            self.assertIn(heading + "\n外观锚点（稳定特征，与文字要求冲突时以此为准）：", text)
            self.assertIn(f"标志物：{person['name']}的标志物；", text)

    def test_degraded_scene_still_carries_headings_and_appearances(self):
        people = [character()]
        text = tasks.render_prompt({"scene": "只有一段整体描述"}, people, {"kanon": RECORD})
        self.assertIn(
            "人物 kanon（中川花音 / 已知作品 / 校服；观看者左侧）\n"
            "外观锚点（稳定特征，与文字要求冲突时以此为准）：", text)
        self.assertNotIn("）：动作", text)


class CacheTests(unittest.TestCase):
    def test_dual_key_lookup_and_owner_free_keys(self):
        cache = references.AppearanceCache()
        key = cache.key("小秦", "Mr_Quin", "夏季")
        cache.put(key, {"appearance": RECORD}, alias_name="秦某")
        self.assertIsNotNone(cache.get(key))
        # 准备时叫小秦、画时叫秦某：别名键兜住这种写法差异。
        self.assertIsNotNone(cache.get(cache.alias_key("秦某")))
        self.assertIsNone(cache.get(cache.key("别人")))

    def test_ttl_and_lru_bound(self):
        now = [0.0]
        cache = references.AppearanceCache(ttl=10, max_entries=2)
        with patch.object(references.time, "monotonic", side_effect=lambda: now[0]):
            for index in range(3):
                cache.put(f"k{index}", {"appearance": RECORD})
            self.assertIsNone(cache.get("k0"))
            now[0] = 100.0
            self.assertIsNone(cache.get("k2"))


class FakeFetcher:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    async def fetch(self, url, *, image_only=False):
        self.calls.append((url, image_only))
        value = self.pages[url]
        if isinstance(value, Exception):
            raise value
        return value[0], value[1], url

    async def close(self):
        pass


def extraction(**overrides):
    payload = {"status": "confirmed", "selected_index": 1, "canonical_name": "中川花音",
               "version": "夏季制服", "identity_basis": "角色资料页与官方立绘一致",
               "appearance": copy.deepcopy(RECORD)}
    payload.update(overrides)
    return types.SimpleNamespace(completion_text=json.dumps(payload, ensure_ascii=False))


PAGE = (b'<meta property="og:image" content="/logo.png">'
        b'<figure><img data-src="person.png" alt="\xe8\xa7\x92\xe8\x89\xb2"><figcaption>\xe5\xae\x98\xe6\x96\xb9\xe7\xab\x8b\xe7\xbb\x98</figcaption></figure>',
        "text/html")


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = types.SimpleNamespace(
            llm_generate=AsyncMock(return_value=extraction()),
            get_current_chat_provider_id=AsyncMock(return_value="vision"),
        )

    def build(self, pages):
        self.fetcher = FakeFetcher(pages)
        return references.CharacterReference(self.context, {"provider_id": "vision"}, fetcher=self.fetcher)

    async def test_no_source_urls_never_calls_the_model(self):
        reference = self.build({})
        result = await reference.prepare("group", "中川花音")
        self.assertEqual(result["status"], "unavailable")
        self.context.llm_generate.assert_not_awaited()
        self.assertEqual(self.fetcher.calls, [])

    async def test_page_candidates_reach_the_model_by_index(self):
        reference = self.build({
            "https://example.org/wiki": PAGE,
            "https://example.org/logo.png": (PNG, "image/png"),
            "https://example.org/person.png": (PNG, "image/png"),
        })
        self.context.llm_generate.return_value = extraction(selected_index=1)
        result = await reference.prepare("group", "中川花音", "已知作品", "夏季制服",
                                         ["https://example.org/wiki"])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["appearance"], appearance.normalize(RECORD))
        # 来源页与像素出处分开记录，页面正文才能和它对应的图片成对送进模型。
        self.assertEqual(result["source_url"], "https://example.org/wiki")
        self.assertEqual(result["image_url"], "https://example.org/person.png")
        # 两张候选图都提交给视觉模型，选哪张由它按索引决定，插件不猜。
        self.assertEqual(len(self.context.llm_generate.await_args.kwargs["image_urls"]), 2)
        payload = self.context.llm_generate.await_args.kwargs["prompt"]
        # 模型按序号选图，图片地址不进提示词：页面正文与图注才是判断依据。
        self.assertIn("官方立绘", payload)
        self.assertIn('"index": 1', payload)
        self.assertEqual(self.fetcher.calls,
                         [("https://example.org/wiki", False),
                          ("https://example.org/person.png", True),
                          ("https://example.org/logo.png", True)])

    async def test_image_direct_link_is_used_as_candidate(self):
        reference = self.build({"https://example.org/person.png": (PNG, "image/png")})
        result = await reference.prepare("group", "中川花音", source_urls=["https://example.org/person.png"])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(self.fetcher.calls, [("https://example.org/person.png", False)])

    async def test_lazy_extraction_is_retried_once_with_named_fields(self):
        self.context.llm_generate.side_effect = [
            extraction(appearance={"silhouette": "银白", "hair": {"color": "银白"}}),
            extraction(),
        ]
        reference = self.build({"https://example.org/wiki": PAGE,
                                "https://example.org/logo.png": (PNG, "image/png"),
                                "https://example.org/person.png": (PNG, "image/png")})
        result = await reference.prepare("group", "中川花音", source_urls=["https://example.org/wiki"])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(self.context.llm_generate.await_count, 2)
        retry_prompt = self.context.llm_generate.await_args_list[1].kwargs["prompt"]
        self.assertIn("上一次抽取有这些问题", retry_prompt)
        self.assertIn("eyes.shape", retry_prompt)

    async def test_still_lazy_after_retry_returns_unavailable_and_caches_nothing(self):
        self.context.llm_generate.return_value = extraction(appearance={"silhouette": "银白"})
        reference = self.build({"https://example.org/wiki": PAGE,
                                "https://example.org/logo.png": (PNG, "image/png"),
                                "https://example.org/person.png": (PNG, "image/png")})
        result = await reference.prepare("group", "中川花音", source_urls=["https://example.org/wiki"])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.assertIsNone(reference.lookup("中川花音"))

    async def test_optional_field_flaws_are_accepted_without_a_retry(self):
        # 模型爱在可选字段上写"未知"。那只是一项不渲染，不该再把视觉模型叫回来一次，
        # 更不该让整个角色退回纯文字出图。
        record = dict(RECORD, headwear="未知", palette=["无"], signature="？")
        self.context.llm_generate.return_value = extraction(appearance=record)
        reference = self.build({"https://example.org/wiki": PAGE,
                                "https://example.org/logo.png": (PNG, "image/png"),
                                "https://example.org/person.png": (PNG, "image/png")})
        result = await reference.prepare("group", "中川花音", source_urls=["https://example.org/wiki"])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(self.context.llm_generate.await_count, 1)
        cached = reference.lookup("中川花音")
        self.assertIsNotNone(cached)
        rendered = appearance.render_block(cached["appearance"])
        self.assertIn("发型：双马尾", rendered)
        self.assertNotIn("头饰", rendered)

    async def test_required_field_flaws_still_block_and_cache_nothing(self):
        record = dict(RECORD, eyes={**RECORD["eyes"], "shape": "未知"})
        self.context.llm_generate.return_value = extraction(appearance=record)
        reference = self.build({"https://example.org/wiki": PAGE,
                                "https://example.org/logo.png": (PNG, "image/png"),
                                "https://example.org/person.png": (PNG, "image/png")})
        result = await reference.prepare("group", "中川花音", source_urls=["https://example.org/wiki"])
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.assertIsNone(reference.lookup("中川花音"))

    async def test_uncertain_verdict_is_not_cached(self):
        self.context.llm_generate.return_value = types.SimpleNamespace(
            completion_text=json.dumps({"status": "uncertain", "reason": "候选图是别的角色"}))
        reference = self.build({"https://example.org/wiki": PAGE,
                                "https://example.org/logo.png": (PNG, "image/png"),
                                "https://example.org/person.png": (PNG, "image/png")})
        result = await reference.prepare("group", "中川花音", source_urls=["https://example.org/wiki"])
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(reference.lookup("中川花音"))

    async def test_cache_hit_costs_no_fetch_and_no_vision_call(self):
        pages = {"https://example.org/wiki": PAGE,
                 "https://example.org/logo.png": (PNG, "image/png"),
                 "https://example.org/person.png": (PNG, "image/png")}
        reference = self.build(pages)
        reference.cache.put(reference.cache.key("中川花音", "已知作品", "夏季制服"),
                            {"canonical_name": "中川花音", "appearance": RECORD}, alias_name="秦某")
        self.context.llm_generate.reset_mock()
        self.fetcher.calls = []
        result = await reference.prepare("group", "中川花音", "已知作品", "夏季制服",
                                         ["https://example.org/wiki"])
        self.assertTrue(result["cached"])
        self.context.llm_generate.assert_not_awaited()
        self.assertEqual(self.fetcher.calls, [])
        # 换用准确名重查同样命中，不会重新抓取。
        self.assertIsNotNone(reference.lookup("秦某"))


class AppearanceLockTests(unittest.IsolatedAsyncioTestCase):
    """整条链路的要点：外观块是最终提示词的字面子串。"""

    async def asyncSetUp(self):
        self.context = types.SimpleNamespace(
            send_message=AsyncMock(),
            llm_generate=AsyncMock(),
            get_current_chat_provider_id=AsyncMock(return_value="vision"),
        )
        self.plugin = main.QiniuImagePlugin(
            self.context, {"api_key": "test", "rewrite_provider_ids": ["vision"]})
        self.event = Event()
        self.plugin._generate = AsyncMock(return_value=(B64, None))

    async def asyncTearDown(self):
        await self.plugin.terminate()

    def scene_source(self, people):
        """每次调用换一个场景：安全回退逐级升级，模型不可能每级都原样返回。"""
        state = {"n": 0}

        def generate(*args, **kwargs):
            state["n"] += 1
            return self.scene_reply(people, scene=f"第{state['n']}版场景，逆光")

        return generate

    @staticmethod
    def scene_reply(people, **overrides):
        payload = {"scene": "[[STYLE_PRESET:clean_anime_wallpaper]] 夏日海滩，逆光",
                   "characters": [{"id": p["id"], "description": "挥手微笑"} for p in people]}
        payload.update(overrides)
        return types.SimpleNamespace(completion_text=json.dumps(payload, ensure_ascii=False))

    async def test_optimizer_never_receives_appearance_yet_output_contains_it(self):
        people = group_characters()
        records = {p["id"]: dict(RECORD, signature=f"{p['name']}的标志物") for p in people}
        records["2"] = None
        self.context.llm_generate.return_value = self.scene_reply(people)
        metadata = {}
        text = await rewriter.rewrite(
            self.context, "group", "五人海滩合照", has_image=False,
            characters=people, appearances=records, provider_id="vision",
            result_metadata=metadata,
        )
        sent_prompt = self.context.llm_generate.await_args.kwargs["prompt"]
        self.assertNotIn("双马尾", sent_prompt)
        self.assertNotIn("外观锚点", sent_prompt)
        self.assertIn("不要写任何外观特征", self.context.llm_generate.await_args.kwargs["system_prompt"])
        for person in people:
            if records[person["id"]] is None:
                self.assertNotIn(f"标志物：{person['name']}的标志物", text)
                continue
            # 优化模型说"挥手微笑"，外观块照旧逐字出现。
            self.assertIn(appearance.render_block(records[person["id"]]), text)
        self.assertEqual(text.count("挥手微笑"), 5)

    async def test_plain_text_optimizer_output_degrades_but_keeps_the_lock(self):
        people = [character()]
        # 逐人 JSON 这条路每个 Provider 会试满 3 次才放弃，之后才走退化路径。
        self.context.llm_generate.side_effect = [
            types.SimpleNamespace(completion_text="不是 JSON"),
            types.SimpleNamespace(completion_text="还是不是 JSON"),
            types.SimpleNamespace(completion_text="仍然不是 JSON"),
            types.SimpleNamespace(completion_text="[[STYLE_PRESET:none]] 一段整体的海滩场景描述"),
        ]
        text = await rewriter.rewrite(self.context, "group", "画花音", has_image=False,
                                      characters=people, appearances={"kanon": RECORD},
                                      provider_id="vision")
        self.assertIn("一段整体的海滩场景描述", text)
        self.assertIn("人物 kanon（中川花音 / 已知作品 / 校服；观看者左侧）", text)
        self.assertIn(appearance.render_block(RECORD), text)

    async def test_appearance_survives_every_safety_level(self):
        people = [character()]
        self.context.llm_generate.side_effect = self.scene_source(people)
        rejected = await rewriter.rewrite(self.context, "group", "画花音", has_image=False,
                                          characters=people, appearances={"kanon": RECORD},
                                          provider_id="vision")
        self.assertIn(appearance.render_block(RECORD), rejected)
        for level in range(2, rewriter.SAFETY_REWRITE_LEVELS + 1):
            safety = await rewriter.rewrite_for_safety(
                self.context, "group", rejected, provider_id="vision",
                safety_attempt=level, characters=people, appearances={"kanon": RECORD})
            self.assertIn("人物 kanon（中川花音", safety)
            # 身份内核保留，服装类字段不重新注入，否则合法软化会被反复驳回。
            self.assertIn("发型：双马尾", safety)
            self.assertIn("标志物：腰间怀表", safety)
            self.assertNotIn("服装：", safety)
            self.assertNotIn("滚边：", safety)

    async def test_full_draw_sends_appearance_verbatim_with_no_input_images(self):
        people = [character()]
        self.context.llm_generate.return_value = self.scene_reply(people)
        output, error = await self.plugin._draw(self.event, "画花音", people, {"kanon": RECORD})
        self.assertEqual(output, B64)
        self.assertIsNone(error)
        prompt = self.plugin._generate.await_args.args[1]
        self.assertIn(appearance.render_block(RECORD), prompt)
        self.assertIn("挥手微笑", prompt)
        self.assertEqual(self.plugin._generate.await_args.args[2], [])
        self.assertIn("画花音", self.context.llm_generate.await_args.kwargs["prompt"])

    async def test_plain_draw_without_a_character_list_still_draws(self):
        # 没有人物名单时优化模型输出的是普通提示词文本，不是逐人 JSON。这条分支
        # 曾经被误当成 JSON 解析，导致每一次"画一张某物"都判成优化失败。
        self.context.llm_generate.return_value = types.SimpleNamespace(
            completion_text="[[STYLE_PRESET:clean_anime_wallpaper]] 一只橘猫趴在窗台，午后逆光")
        output, error = await self.plugin._draw(self.event, "画一只猫", [], {})
        self.assertIsNone(error)
        self.assertEqual(output, B64)
        prompt = self.plugin._generate.await_args.args[1]
        self.assertIn("橘猫趴在窗台", prompt)
        self.assertNotIn("STYLE_PRESET", prompt)
        self.assertEqual(self.plugin._generate.await_args.args[2], [])

    async def test_safety_ladder_retries_then_gives_up(self):
        people = [character()]
        self.context.llm_generate.side_effect = self.scene_source(people)
        self.plugin._generate = AsyncMock(side_effect=api.QiniuSafetyError(400, "blocked"))
        output, error = await self.plugin._draw(self.event, "画花音", people, {"kanon": RECORD})
        self.assertIsNone(output)
        self.assertIn("审核", error)
        self.assertEqual(self.plugin._generate.await_count, rewriter.SAFETY_REWRITE_LEVELS)

    async def test_recovers_after_first_safety_rejection(self):
        people = [character()]
        self.context.llm_generate.side_effect = self.scene_source(people)
        self.plugin._generate = AsyncMock(
            side_effect=[api.QiniuSafetyError(400, "blocked"), (B64, None)])
        output, _ = await self.plugin._draw(self.event, "画花音", people, {"kanon": RECORD})
        self.assertEqual(output, B64)
        self.assertEqual(self.plugin._generate.await_count, 2)
        # 回退后重出的是身份内核：安全链路软化掉的服装不会再被塞回去。
        retried = self.plugin._generate.await_args.args[1]
        self.assertIn(appearance.render_block(RECORD, appearance.IDENTITY_CORE), retried)
        self.assertNotIn("服装：", retried)

    async def test_input_images_reach_the_api_but_never_the_character_fetcher(self):
        event = Event(segments=[Image(file="base64://" + B64)])
        people = [character()]
        self.context.llm_generate.return_value = self.scene_reply(people)
        with patch.object(references.CharacterReference, "prepare", AsyncMock()) as prepare:
            output, _ = await self.plugin._draw(event, "改成海边", people, {"kanon": RECORD})
        self.assertEqual(output, B64)
        self.assertEqual(self.plugin._generate.await_args.args[2], ["base64://" + B64])
        prepare.assert_not_awaited()

    async def test_unusable_appearance_is_not_written_and_draw_still_proceeds(self):
        people = [character()]
        self.context.llm_generate.return_value = self.scene_reply(people)
        output, _ = await self.plugin._draw(self.event, "画花音", people, {"kanon": {"silhouette": "银发"}})
        self.assertEqual(output, B64)
        self.assertNotIn("外观锚点", self.plugin._generate.await_args.args[1])

    async def test_draw_image_reads_the_cache_at_submit_time(self):
        people = [character()]
        self.plugin.references.cache.put(
            self.plugin.references.cache.key("中川花音", "已知作品", "校服"),
            {"canonical_name": "中川花音", "appearance": appearance.normalize(RECORD)})
        worker = AsyncMock()
        with patch.object(self.plugin, "_draw_and_push", worker):
            result = json.loads(await self.plugin.draw_image(self.event, "画花音", people))
        self.assertEqual(result["status"], "accepted")
        await asyncio.gather(*self.plugin._tasks, return_exceptions=True)
        # 外观在提交时同步取出并随任务定型，后台修改缓存不会影响这一张。
        self.assertEqual(worker.await_args.args[3]["kanon"], appearance.normalize(RECORD))

    async def test_characters_are_validated_before_any_work(self):
        result = await self.plugin.draw_image(self.event, "画图", [{"name": "甲"}, {"name": "乙"}])
        self.assertIn("绘图任务未提交", result)
        self.assertEqual(self.plugin._tasks, set())


class StyleApplicationTests(unittest.IsolatedAsyncioTestCase):
    """内置风格要尽可能用上：none 不该成为省事的默认答案。"""

    async def asyncSetUp(self):
        self.context = types.SimpleNamespace(
            send_message=AsyncMock(),
            llm_generate=AsyncMock(),
            get_current_chat_provider_id=AsyncMock(return_value="vision"),
        )

    @staticmethod
    def reply(text):
        return types.SimpleNamespace(completion_text=text)

    async def test_style_rule_is_sent_only_when_the_router_is_on(self):
        self.context.llm_generate.return_value = self.reply("[[STYLE_PRESET:none]] 干净的海滩插画，大色块留白")
        await rewriter.rewrite(self.context, "group", "画一张海滩插画", has_image=False,
                               style_mode="disabled", provider_id="vision")
        self.assertNotIn("[[STYLE_PRESET:none]]。必须从中选一个",
                         self.context.llm_generate.await_args.kwargs["system_prompt"])

        self.context.llm_generate.reset_mock()
        self.context.llm_generate.return_value = self.reply("[[STYLE_PRESET:none]] 干净的海滩插画，大色块留白")
        await rewriter.rewrite(self.context, "group", "画一张海滩插画", has_image=False,
                               style_mode="auto", provider_id="vision")
        self.assertIn("不要输出 [[STYLE_PRESET:none]]",
                      self.context.llm_generate.await_args.kwargs["system_prompt"])

    async def test_a_bare_none_gets_one_targeted_retry(self):
        self.context.llm_generate.side_effect = [
            self.reply("[[STYLE_PRESET:none]] 夏日海滩，逆光，构图干净"),
            self.reply("[[STYLE_PRESET:klein_order]] 克莱因蓝大色块与秩序网格，夏日海滩，逆光"),
        ]
        metadata = {}
        text = await rewriter.rewrite(self.context, "group", "画一张海滩插画", has_image=False,
                                      style_mode="auto", provider_id="vision",
                                      result_metadata=metadata)
        self.assertIn("克莱因蓝大色块与秩序网格", text)
        self.assertNotIn("构图干净", text)
        self.assertEqual(metadata["style"], "克莱因秩序")
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.assertIn("上一次输出选择了 [[STYLE_PRESET:none]]",
                      self.context.llm_generate.await_args.kwargs["system_prompt"])
        # 标记本身不进最终提示词，风格是以视觉语言进正文的。
        self.assertNotIn("STYLE_PRESET", text)

    async def test_a_second_none_is_accepted_rather_than_blocking_the_draw(self):
        # 用户明确要别的画风时 none 是合法答案，插件不能为此拒绝出图。
        self.context.llm_generate.side_effect = [
            self.reply("[[STYLE_PRESET:none]] 厚涂油画质感的海滩，笔触明显"),
            self.reply("[[STYLE_PRESET:none]] 厚涂油画质感的海滩，笔触明显"),
        ]
        text = await rewriter.rewrite(self.context, "group", "用厚涂油画画海滩", has_image=False,
                                      style_mode="auto", provider_id="vision")
        self.assertIn("厚涂油画质感的海滩", text)
        self.assertEqual(self.context.llm_generate.await_count, 2)

    async def test_disabled_router_never_retries(self):
        self.context.llm_generate.return_value = self.reply("干净的海滩插画，大色块留白")
        text = await rewriter.rewrite(self.context, "group", "画一张海滩插画", has_image=False,
                                      style_mode="disabled", provider_id="vision")
        self.assertIn("大色块留白", text)
        self.assertEqual(self.context.llm_generate.await_count, 1)


class HookTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = types.SimpleNamespace(
            send_message=AsyncMock(), llm_generate=AsyncMock(),
            get_current_chat_provider_id=AsyncMock(return_value="vision"))
        self.plugin = main.QiniuImagePlugin(self.context, {"api_key": "test"})
        self.event = Event()

    async def asyncTearDown(self):
        await self.plugin.terminate()

    def request(self, prompt="原人设"):
        tool = types.SimpleNamespace(name="draw_image", parameters={}, active=True)
        return types.SimpleNamespace(system_prompt=prompt,
                                     func_tool=types.SimpleNamespace(tools=[tool]))

    async def test_hook_preserves_persona_swaps_schema_and_is_bounded(self):
        req = self.request()
        await self.plugin.on_llm_request(self.event, req)
        await self.plugin.on_llm_request(self.event, req)
        self.assertTrue(req.system_prompt.startswith("原人设"))
        self.assertEqual(req.system_prompt.count("<qiniu_drawing_context>"), 1)
        self.assertEqual(req.func_tool.tools[0].parameters, tasks.DRAW_SCHEMA)
        self.assertLess(len(req.system_prompt), 1900)

    async def test_hook_lists_preferred_styles_only_in_auto_mode(self):
        for mode in ("auto", "explicit_only", "disabled"):
            self.plugin.style_mode = mode
            req = self.request()
            await self.plugin.on_llm_request(self.event, req)
            self.assertIn("错位", req.system_prompt)
            self.assertEqual("自动风格：" in req.system_prompt, mode == "auto")

    async def test_recent_works_are_listed_after_a_successful_draw(self):
        self.plugin._remember(self.event, "执行稿原文", "净色动画壁纸", [character()])
        req = self.request()
        await self.plugin.on_llm_request(self.event, req)
        self.assertIn("近期作品：", req.system_prompt)
        self.assertIn("净色动画壁纸", req.system_prompt)
        record = json.loads(await self.plugin.get_last_image_prompt(self.event))
        self.assertEqual(record["prompt"], "执行稿原文")
        self.assertEqual(record["characters"], ["中川花音"])
        self.assertEqual(self.plugin._last_prompts.get("group:bob"), None)

    async def test_hook_without_drawing_tool_does_nothing(self):
        req = types.SimpleNamespace(system_prompt="persona", func_tool=types.SimpleNamespace(tools=[]))
        await self.plugin.on_llm_request(self.event, req)
        self.assertEqual(req.system_prompt, "persona")

    async def test_get_last_image_prompt_without_history(self):
        self.assertIn("没有可读取", await self.plugin.get_last_image_prompt(self.event))


class WireTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_typeerror_fallback_keeps_images(self):
        context = types.SimpleNamespace(
            get_current_chat_provider_id=AsyncMock(return_value="bad"),
            llm_generate=AsyncMock(side_effect=[
                TypeError("images unsupported"),
                types.SimpleNamespace(completion_text='{"status":"ok"}')]),
        )
        result = await rewriter.visual_json(
            context, "group", prompt="check", image_urls=[references.image_data(PNG)],
            validate=lambda r: r.get("status") == "ok", purpose="test",
            provider_id="bad", fallback_provider_ids=["good"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(context.llm_generate.await_count, 2)
        self.assertTrue(all(call.kwargs.get("image_urls") for call in context.llm_generate.await_args_list))

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

    async def test_background_push_sends_image_and_note(self):
        context = types.SimpleNamespace(
            send_message=AsyncMock(), llm_generate=AsyncMock(),
            get_current_chat_provider_id=AsyncMock(return_value="vision"))
        plugin = main.QiniuImagePlugin(context, {"api_key": "test"})
        self.addCleanup(asyncio.run, plugin.terminate())
        with patch.object(plugin, "_draw", AsyncMock(return_value=(B64, "检查说明"))):
            await plugin._draw_and_push(Event(), "画图", (), {})
        chain = context.send_message.await_args.args[1]
        self.assertEqual(chain.parts, [("image", B64), ("text", "检查说明")])


class WebTests(unittest.TestCase):
    def test_public_url_blocks_local_credentials_and_ports(self):
        for url in ("file:///tmp/a", "http://127.0.0.1/a", "http://10.1.2.3/a",
                    "http://[::ffff:127.0.0.1]/a", "http://localhost/a", "http://foo.local/a",
                    "http://user:password@example.org/a", "http://example.org:22/a"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                references.public_url(url)

    def test_public_ip_allows_only_global_unicast(self):
        for address in ("127.0.0.1", "10.1.2.3", "169.254.1.1", "::1", "224.0.0.1", "::ffff:127.0.0.1"):
            with self.subTest(address=address):
                self.assertFalse(references.public_ip(address))
        self.assertTrue(references.public_ip("93.184.216.34"))

    def test_html_relative_lazy_images_caption_and_cover(self):
        parser = references.PageImages("https://example.org/wiki/person")
        parser.feed('<meta property="og:image" content="/logo.png"><script>evil()</script>'
                    '<figure><img src="loading.gif" data-src="../person.png" alt="角色">'
                    '<figcaption>官方立绘</figcaption></figure>')
        row = next(r for r in parser.images if "person.png" in r["url"])
        self.assertEqual(row["url"], "https://example.org/person.png")
        self.assertIn("官方立绘", row["label"])
        self.assertLess(row["rank"], parser.images[0]["rank"])
        self.assertNotIn("evil()", parser.text)

    def test_image_data_prefixes_real_mime(self):
        self.assertTrue(references.image_data(PNG).startswith("data:image/png;base64,"))
        with self.assertRaises(ValueError):
            references.image_data(b"not an image")

    def test_validate_image_rejects_corrupt_bytes(self):
        references.validate_image(PNG)
        with self.assertRaises(ValueError):
            references.validate_image(b"not an image")

    def test_resolver_rejects_private_dns_address(self):
        # 必须用 asyncio.run 驱动：这个方法原先写成 async def 放在 unittest.TestCase
        # 里，unittest 只会创建一个协程然后丢掉，从来没有真正跑过。
        async def go():
            resolver = references.PublicResolver()
            resolver.delegate.resolve = AsyncMock(return_value=[{"host": "127.0.0.1"}])
            try:
                with self.assertRaises(OSError):
                    await resolver.resolve("public-looking.example", 443)
            finally:
                await resolver.close()

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
