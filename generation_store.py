"""Bounded, owner-scoped, temporary image and generation storage."""

import copy
import tempfile
import time
import uuid
from pathlib import Path

TTL_SECONDS = 30 * 60
MAX_RECORDS = 100
MAX_CACHE_BYTES = 512 * 1024 * 1024


class GenerationStore:
    def __init__(self, *, clock=time.monotonic, max_bytes=MAX_CACHE_BYTES, max_records=MAX_RECORDS):
        self._directory = tempfile.TemporaryDirectory(prefix="qiniu-image-")
        self.root = Path(self._directory.name)
        self.records: dict[str, dict] = {}
        self.clock = clock
        self.max_bytes = max_bytes
        self.max_records = max_records

    @staticmethod
    def new_id(kind="generation") -> str:
        return ("ref_" if kind == "reference" else "gen_") + uuid.uuid4().hex[:16]

    def _drop(self, key: str) -> None:
        record = self.records.pop(key)
        Path(record["image_path"]).unlink(missing_ok=True)
        for asset in record.get("assets", []):
            Path(asset["image_path"]).unlink(missing_ok=True)

    def prune(self) -> None:
        now = self.clock()
        for key, record in list(self.records.items()):
            if now - record["created_at"] >= TTL_SECONDS:
                self._drop(key)

    def put(self, owner: tuple[str, str], image: bytes, data: dict, *, kind="generation", record_id=None, assets=()) -> dict:
        self.prune()
        size = len(image) + sum(len(a["bytes"]) for a in assets)
        if size > self.max_bytes:
            raise ValueError("图片超过缓存容量")
        while self.records and (len(self.records) >= self.max_records or sum(r["size"] for r in self.records.values()) + size > self.max_bytes):
            self._drop(min(self.records, key=lambda key: self.records[key]["created_at"]))
        key = record_id or self.new_id(kind)
        if key in self.records:
            raise ValueError("作品标识重复")
        # Only internally generated identifiers ever become filenames.
        path = self.root / (uuid.uuid4().hex + ".img")
        files, stored_assets = [path], []
        try:
            path.write_bytes(image)
            for asset in assets:
                asset_path = self.root / (uuid.uuid4().hex + ".img")
                files.append(asset_path)
                asset_path.write_bytes(asset["bytes"])
                stored_assets.append({"binding": copy.deepcopy(asset["binding"]), "image_path": str(asset_path)})
        except OSError:
            for file in files:
                file.unlink(missing_ok=True)
            raise
        record = {**copy.deepcopy(data), "id": key, "kind": kind, "owner": owner,
                  "created_at": self.clock(), "image_path": str(path), "size": size, "assets": stored_assets}
        self.records[key] = record
        return copy.deepcopy(record)

    def get(self, owner: tuple[str, str], key="latest", *, kind="generation") -> dict:
        self.prune()
        if key == "latest":
            choices = [r for r in self.records.values() if r["owner"] == owner and r["kind"] == kind]
            record = max(choices, key=lambda r: r["created_at"], default=None)
        else:
            record = self.records.get(key)
        if not record or record["owner"] != owner or record["kind"] != kind:
            raise ValueError("指定作品或参考图不存在、已过期或不属于当前会话用户")
        return copy.deepcopy(record)

    def snapshot(self, owner: tuple[str, str], key="latest", *, kind="generation") -> dict:
        record = self.get(owner, key, kind=kind)
        record["image_bytes"] = Path(record["image_path"]).read_bytes()
        for asset in record.get("assets", []):
            asset["bytes"] = Path(asset["image_path"]).read_bytes()
        return record

    def recent(self, owner: tuple[str, str]) -> list[dict]:
        self.prune()
        rows = [r for r in self.records.values() if r["owner"] == owner and r["kind"] == "generation"]
        return copy.deepcopy(sorted(rows, key=lambda r: r["created_at"], reverse=True)[:3])

    def close(self) -> None:
        self.records.clear()
        self._directory.cleanup()
