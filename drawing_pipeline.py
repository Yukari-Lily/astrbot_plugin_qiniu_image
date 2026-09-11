"""Resolve a drawing contract once, then compile, generate and assess it."""

import asyncio
import base64
import copy
import json

import aiohttp
from astrbot.api import logger

from .drawing_task import TASK_SCHEMA, looks_like_group, normalize_task, task_context, validate
from .generation_store import GenerationStore
from .message_utils import resolve_input_images
from .prompt_rewriter import SAFETY_REWRITE_LEVELS, rewrite, rewrite_for_safety, visual_json
from .qiniu_api import MAX_INPUT_IMAGE_BYTES, MAX_OUTPUT_IMAGE_BYTES, QiniuSafetyError
from .reference_images import ReferenceFetcher, image_data, validate_image


class DrawingPipeline:
    def __init__(self, plugin):
        self.plugin = plugin
        self.store = GenerationStore()
        self.fetcher = ReferenceFetcher()

    @staticmethod
    def owner(event):
        return str(event.unified_msg_origin), str(event.get_sender_id())

    @property
    def providers(self):
        return {"provider_id": self.plugin.rewrite_provider_id,
                "fallback_provider_ids": self.plugin.rewrite_fallback_provider_ids}

    async def close(self):
        await self.fetcher.close()
        self.store.close()

    def freeze(self, event, task=None):
        """Snapshot latest and all cache references before starting a background task."""
        frozen = {"id": self.store.new_id(), "task": None, "base": None, "references": {}}
        if task is None:
            return frozen
        validate(task, TASK_SCHEMA)
        task = copy.deepcopy(task)
        base_id = task.get("base_generation_id", "")
        owner = self.owner(event)
        if base_id:
            if task["operation"] == "create":
                raise ValueError("继承作品请使用 edit 或 redraw")
            frozen["base"] = self.store.snapshot(owner, base_id)
            task["base_generation_id"] = frozen["base"]["id"]
        base = frozen["base"]
        # Cached references already copied into the base record survive reference TTL expiry.
        inherited = {a["binding"].get("source"): a for a in (base or {}).get("assets", [])}
        ref_ids = {c.get("reference_id") for c in task.get("characters", []) if c.get("reference_id")}
        ref_ids.update(r["source"] for r in task.get("image_roles", []) if not r["source"].startswith("input:"))
        for ref_id in ref_ids:
            if not ref_id.startswith("ref_"):
                raise ValueError("参考图标识无效")
            if ref_id in inherited:
                continue
            frozen["references"][ref_id] = self.store.snapshot(owner, ref_id, kind="reference")
        for char in task.get("characters", []):
            reference = frozen["references"].get(char.get("reference_id"))
            if reference:
                canonical = reference["assessment"]["canonical_name"]
                if char["name"].casefold() not in (canonical.casefold(), reference["subject"].casefold()):
                    raise ValueError("参考图人物与人物条目的名称不一致，请使用工具返回的 canonical_name")
                char.setdefault("features", reference["assessment"]["features"])
                char.setdefault("evidence", reference["assessment"]["summary"])
                char.setdefault("identity_status", "confirmed")
        frozen["task"] = normalize_task(task, (base or {}).get("task"))
        roles = frozen["task"].get("image_roles", [])
        for char in task.get("characters", []):
            ref_id = char.get("reference_id")
            if ref_id and any(r["source"] == ref_id and
                              (r["role"] != "character" or r.get("character_id") != char["id"])
                              for r in roles):
                raise ValueError("人物 reference_id 与图片用途绑定冲突")
            if ref_id and any(r["role"] == "character" and r.get("character_id") == char["id"]
                              and r["source"] != ref_id for r in roles):
                raise ValueError("同一人物提交了互相冲突的参考图")
        for role in frozen["task"].get("image_roles", []):
            if role["role"] == "character":
                char = next(c for c in frozen["task"]["characters"] if c["id"] == role["character_id"])
                char["reference_id"] = role["source"] if role["source"].startswith("ref_") else ""
        # Resolve inherited character references that were not included in this edit patch.
        for char in frozen["task"]["characters"]:
            ref_id = char.get("reference_id")
            if ref_id and ref_id not in inherited and ref_id not in frozen["references"]:
                frozen["references"][ref_id] = self.store.snapshot(owner, ref_id, kind="reference")
        return frozen

    async def prepare_reference(self, event, subject, source_urls, evidence):
        if not isinstance(subject, str) or not subject.strip() or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("必须先提供明确人物及搜索消歧依据")
        if len(subject) > 500 or len(evidence) > 8000:
            raise ValueError("人物或依据过长")
        if not isinstance(source_urls, list) or not 1 <= len(source_urls) <= 3 or any(not isinstance(s, str) for s in source_urls):
            raise ValueError("请提供 1 至 3 个来源地址或 input:N 图片标识")
        input_sources = [s for s in source_urls if s.startswith("input:")]
        web_sources = [s for s in source_urls if not s.startswith("input:")]
        candidates, failures = [], []
        if input_sources:
            inputs = await resolve_input_images(self.plugin.context, event, self.plugin.client)
            for source in dict.fromkeys(input_sources):
                try:
                    index = int(source.split(":", 1)[1])
                    if not 1 <= index <= len(inputs):
                        raise ValueError("输入图片不存在")
                    raw = await self._input_bytes(inputs[index - 1])
                    candidates.append({"bytes": raw, "url": source, "source_url": source,
                                       "label": "用户主动提供的参考图片", "page_text": ""})
                except (ValueError, OSError, aiohttp.ClientError, asyncio.TimeoutError):
                    failures.append("用户提供的参考图片无法读取")
        if web_sources:
            web_candidates, web_failures = await self.fetcher.candidates(web_sources)
            candidates.extend(web_candidates[:6 - len(candidates)])
            failures.extend(web_failures)
        if not candidates:
            return {"status": "unavailable", "reason": "未取得可核对的参考图", "failures": failures,
                    "instruction": "仅在已有可靠身份及外观文字依据时继续，否则结束本次生成；不追问补图。"}
        rows = [{"index": i + 1, **{k: c[k] for k in ("source_url", "label", "page_text")}} for i, c in enumerate(candidates)]

        def valid(result):
            if result.get("status") == "uncertain":
                return isinstance(result.get("summary"), str)
            return (result.get("status") == "confirmed" and type(result.get("selected_index")) is int
                    and 1 <= result["selected_index"] <= len(candidates)
                    and isinstance(result.get("canonical_name"), str) and bool(result["canonical_name"].strip())
                    and isinstance(result.get("features"), list) and 0 < len(result["features"]) <= 20
                    and all(isinstance(f, str) and 0 < len(f) <= 1000 for f in result["features"])
                    and isinstance(result.get("summary"), str) and 0 < len(result["summary"]) <= 4000)

        prompt = ("核对候选图与目标人物及形象版本，排除网页 logo、其他角色、无依据的封面。"
                  "来源不足或版本冲突时返回 {\"status\":\"uncertain\",\"summary\":\"原因\"}。"
                  "有明确来源且图片外观适用时，选一张，返回 {\"status\":\"confirmed\","
                  "\"selected_index\":1,\"canonical_name\":\"准确名称\",\"features\":[\"可见关键外观\"],"
                  "\"summary\":\"身份与外观依据及版本\"}。不要凭真人脸部推断身份。\n"
                  + json.dumps({"subject": subject, "evidence": evidence, "candidates": rows}, ensure_ascii=False))
        assessment = await visual_json(self.plugin.context, event.unified_msg_origin, prompt=prompt,
                                       image_urls=[image_data(c["bytes"]) for c in candidates],
                                       validate=valid, purpose="参考图核对", **self.providers)
        if not assessment or assessment["status"] != "confirmed":
            return {"status": "unavailable", "reason": "参考图未通过核对" if assessment else "视觉模型不可用",
                    "instruction": "不要使用未核对的候选图；仅有充分文字依据时继续，不追问补图。"}
        candidate = candidates[assessment["selected_index"] - 1]
        record = self.store.put(self.owner(event), candidate["bytes"],
                                {"subject": subject, "assessment": assessment,
                                 "source_url": candidate["source_url"], "image_url": candidate["url"]}, kind="reference")
        logger.info(f"qiniu-image reference prepared | reference={record['id']} candidates={len(candidates)}")
        return {"status": "confirmed", "reference_id": record["id"], "source_url": candidate["source_url"],
                **assessment, "failures": failures}

    async def _input_bytes(self, value):
        if value.startswith("base64://"):
            raw = self.plugin.client.decode_base64_image(value[len("base64://"):])
            await asyncio.to_thread(validate_image, raw)
        else:
            raw, _, _ = await self.fetcher.fetch(value, image_only=True)
        if len(raw) > MAX_INPUT_IMAGE_BYTES:
            raise ValueError("输入图片超过 40 MB")
        image_data(raw)  # validate format before either provider receives it
        return raw

    async def resolve_images(self, event, frozen):
        task, base = frozen["task"], frozen["base"]
        needs_inputs = task is None or any(r["source"].startswith("input:") for r in task.get("image_roles", []))
        inputs = await resolve_input_images(self.plugin.context, event, self.plugin.client) if needs_inputs else []
        assets = []
        if task is None:
            if inputs:
                assets.append({"binding": {"source": "input:1", "role": "edit"}, "bytes": await self._input_bytes(inputs[0])})
            return assets, bool(assets)
        edit = task["operation"] == "edit"
        if edit and base:
            assets.append({"binding": {"source": base["id"], "role": "edit"}, "bytes": base["image_bytes"]})
        explicit_roles = copy.deepcopy(task.get("image_roles", []))
        char_ids = {c["id"] for c in task["characters"]}
        for char in task["characters"]:
            ref_id = char.get("reference_id")
            if ref_id:
                existing = next((r for r in explicit_roles if r["source"] == ref_id), None)
                if existing and (existing["role"] != "character" or existing.get("character_id") != char["id"]):
                    raise ValueError("同一参考图不能静默绑定到不同人物或用途")
                if not existing:
                    explicit_roles.append({"source": ref_id, "role": "character", "character_id": char["id"]})
        inherited = {a["binding"]["source"]: a for a in (base or {}).get("assets", [])}
        # Explicit image_roles replaces inherited style references; otherwise preserve them.
        if base and "image_roles" not in task:
            for asset in base.get("assets", []):
                binding = asset["binding"]
                cid = binding.get("character_id")
                if binding["role"] == "edit":
                    continue
                if binding["role"] == "character":
                    char = next((c for c in task["characters"] if c["id"] == cid), None)
                    if not char or ("reference_id" in char and char["reference_id"] != binding["source"]):
                        continue
                if not any(r["source"] == binding["source"] for r in explicit_roles):
                    explicit_roles.append(copy.deepcopy(binding))
        seen, bound_characters = set(), set()
        for binding in explicit_roles:
            source, role = binding["source"], binding["role"]
            cid = binding.get("character_id")
            if source in seen:
                raise ValueError("参考图片重复绑定")
            seen.add(source)
            if role == "character":
                if cid not in char_ids or cid in bound_characters:
                    raise ValueError("每个人物只能绑定一张人物参考图")
                bound_characters.add(cid)
                char = next(c for c in task["characters"] if c["id"] == cid)
                if char.get("reference_id") and char["reference_id"] != source:
                    raise ValueError("人物 reference_id 与图片用途绑定冲突")
            if source in inherited:
                if role == "character" and inherited[source]["binding"].get("character_id") != cid:
                    raise ValueError("不能把已有作品的参考图绑定给另一人物")
                raw = inherited[source]["bytes"]
            elif source.startswith("input:"):
                try:
                    index = int(source.split(":", 1)[1])
                except ValueError:
                    raise ValueError("输入图片序号无效") from None
                if not 1 <= index <= len(inputs):
                    raise ValueError("指定的输入图片不存在或无法读取")
                raw = await self._input_bytes(inputs[index - 1])
            else:
                reference = frozen["references"].get(source)
                if not reference:
                    raise ValueError("参考图不存在或已过期")
                if role == "character" and char["name"].casefold() not in (reference["subject"].casefold(), reference["assessment"]["canonical_name"].casefold()):
                    raise ValueError("参考图与绑定人物身份不一致")
                raw = reference["image_bytes"]
            assets.append({"binding": binding, "bytes": raw})
        originals = [a for a in assets if a["binding"]["role"] == "edit"]
        if edit and len(originals) != 1:
            raise ValueError("编辑需要且只能指定一张原图或基础作品")
        return assets, edit

    async def assess(self, event, frozen, prompt, output, assets):
        task = frozen["task"]
        if not ((task and task["character_count"] > 1) or (task is None and looks_like_group(prompt))):
            return {"status": "not_requested", "issues": []}

        def valid(result):
            return (result.get("status") in ("ok", "issues", "uncertain")
                    and isinstance(result.get("issues"), list) and len(result["issues"]) <= 10
                    and all(isinstance(i, str) and 0 < len(i) <= 300 for i in result["issues"])
                    and (result["status"] != "issues" or bool(result["issues"]))
                    and (result["status"] != "ok" or not result["issues"]))

        instructions = ("第一张图是本次成图，其余为输入参考，顺序对应给定绑定。核对人数、人物位置、"
                        "关键外观和特征是否串到别人身上；edit 时特别对照编辑原图检查其他人物与构图是否被改坏。"
                        "以本轮用户明确要求和实际执行稿为准，换装不因不同于原参考而报错。"
                        "不把不确定的小细节断言为错误。不要求修图，不追问。"
                        "返回 {\"status\":\"ok|issues|uncertain\",\"issues\":[\"简短明确的问题\"]}。\n"
                        + json.dumps({"task": task, "actual_prompt": prompt,
                                      "references": [a["binding"] for a in assets]}, ensure_ascii=False))
        result = await visual_json(self.plugin.context, event.unified_msg_origin, prompt=instructions,
                                   image_urls=[image_data(output), *[image_data(a["bytes"]) for a in assets]],
                                   validate=valid, purpose="多人图检查", **self.providers)
        return result or {"status": "unavailable", "issues": []}

    async def draw(self, event, user_prompt, frozen=None):
        frozen = frozen or self.freeze(event)
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            return None, "生成失败喵（请输入绘图方案）"
        notices = []
        try:
            assets, edit = await self.resolve_images(event, frozen)
        except Exception as exc:
            logger.warning(f"qiniu-image input failed | generation={frozen['id']} error={type(exc).__name__}")
            # Explicit image bindings must not vanish silently, including an edit original.
            return None, "生成失败喵（指定图片无法读取或图片绑定无效）"
        task = frozen["task"]
        bindings = [{"index": i + 1, **a["binding"]} for i, a in enumerate(assets)]
        context_text = task_context(task, frozen["base"], bindings) if task is not None else ""
        visual_inputs = [image_data(a["bytes"]) for a in assets]
        rewrite_metadata = {}
        prompt = await rewrite(self.plugin.context, event.unified_msg_origin, user_prompt,
                               has_image=edit, style_mode=self.plugin.style_mode,
                               style_strength=self.plugin.style_strength, drawing_task=task,
                               evidence_context=context_text, image_urls=visual_inputs,
                               result_metadata=rewrite_metadata, **self.providers)
        if not prompt and visual_inputs and task and not edit and all(c.get("evidence") and c.get("features") for c in task["characters"]):
            # Only character references can fall back to fully evidenced text; a style reference cannot.
            if assets and all(a["binding"]["role"] == "character" for a in assets):
                assets, bindings, visual_inputs = [], [], []
                fallback_task = copy.deepcopy(task)
                for char in fallback_task["characters"]:
                    char.pop("reference_id", None)
                fallback_task["image_roles"] = []
                frozen["task"] = task = fallback_task
                context_text = task_context(task, frozen["base"], [])
                prompt = await rewrite(self.plugin.context, event.unified_msg_origin, user_prompt,
                                       has_image=False, style_mode=self.plugin.style_mode,
                                       style_strength=self.plugin.style_strength, drawing_task=task,
                                       evidence_context=context_text, result_metadata=rewrite_metadata, **self.providers)
                notices.append("本次未使用图片参考，已依据确认的文字资料生成。")
                logger.info(f"qiniu-image text fallback | generation={frozen['id']} images=0")
        if not prompt:
            return None, "生成失败喵（提示词优化模型不可用或人物结构校验失败）"
        from .style_presets import LINE_EXECUTION_GUIDANCE
        image_instructions = "\n\n" + LINE_EXECUTION_GUIDANCE
        if bindings:
            roles = {"edit": "待编辑原图", "character": "人物身份外观参考", "style": "仅画风参考"}
            labels = [f"输入图 {b['index']}：{roles[b['role']]}" + (f"，对应人物 {b['character_id']}" if b.get("character_id") else "") for b in bindings]
            image_instructions += "\n\n图片用途（编号不画入画面）：\n" + "\n".join(labels)
        image_refs = ["base64://" + base64.b64encode(a["bytes"]).decode("ascii") for a in assets]
        logger.info(f"qiniu-image drawing | generation={frozen['id']} operation={task['operation'] if task else 'legacy'} "
                    f"base={(frozen['base'] or {}).get('id', '-')} images={len(image_refs)} "
                    f"characters={[c['id'] for c in task['characters']] if task else []} "
                    f"bindings={[(b['index'], b['role'], b.get('character_id', '')) for b in bindings]}")
        result = None
        for attempt in range(SAFETY_REWRITE_LEVELS + 1):
            if attempt:
                safe = await rewrite_for_safety(self.plugin.context, event.unified_msg_origin, prompt,
                                                safety_attempt=attempt, drawing_task=task, **self.providers)
                if not safe:
                    continue
                prompt = safe
            try:
                result = await self.plugin._generate(event, prompt + image_instructions, image_refs)
                break
            except QiniuSafetyError:
                logger.warning(f"qiniu-image safety rejected | generation={frozen['id']} stage={attempt}")
        if not result:
            return None, "生成失败喵（所有安全级别均未能生成可用图片）"
        output, error = result
        if not output:
            return result
        prompt += image_instructions
        raw = self.plugin.client.decode_base64_image(output, max_bytes=MAX_OUTPUT_IMAGE_BYTES)
        try:
            assessment = await self.assess(event, frozen, prompt, raw, assets)
        except Exception as exc:
            logger.warning(f"qiniu-image assessment failed | generation={frozen['id']} error={type(exc).__name__}")
            assessment = {"status": "unavailable", "issues": []}
        if assessment["status"] == "issues":
            notices.append("画面检查发现：" + "；".join(assessment["issues"])[:500])
        elif assessment["status"] in ("unavailable", "uncertain"):
            notices.append("本次未完成可靠的多人外观核对。")
        from .style_presets import find_explicit_presets
        styles = find_explicit_presets(user_prompt)
        record_task = task or {"operation": "edit" if edit else "create", "characters": [], "character_count": 0}
        record_task = copy.deepcopy(record_task)
        stored_assets = copy.deepcopy([a for a in assets if a["binding"]["role"] != "edit"])
        for i, asset in enumerate(stored_assets):
            binding = asset["binding"]
            if binding["source"].startswith("input:"):
                binding["source"] = f"ref_{frozen['id'][4:]}_{i + 1}"
            if binding["role"] == "character":
                for char in record_task["characters"]:
                    if char["id"] == binding["character_id"]:
                        char["reference_id"] = binding["source"]
        record_task["image_roles"] = [copy.deepcopy(a["binding"]) for a in stored_assets]
        try:
            self.store.put(self.owner(event), raw,
                           {"task": record_task, "prompt": prompt, "assessment": assessment,
                            "bindings": bindings, "style": rewrite_metadata.get("style") or "、".join(p.name for p in styles) or "依执行稿",
                            "status": "success"}, record_id=frozen["id"],
                           assets=stored_assets)
        except (ValueError, OSError):
            notices.append("图片已生成，但本次作品缓存未能保存。")
        logger.info(f"qiniu-image completed | generation={frozen['id']} assessment={assessment['status']} images={len(image_refs)}")
        return output, "\n".join(notices) or None
