"""External durable memory, kept separate from differentiable model memory.

This is a small versioned JSON key-value example. Not a semantic vector DB.
"""
import json
from pathlib import Path
from datetime import datetime, timezone


class ExternalMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.records = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def write(self, key: str, value, *, source: str, verified: bool):
        if not key or not source:
            raise ValueError("key and source required")
        item = {"value": value, "source": source, "verified": bool(verified),
                "timestamp": datetime.now(timezone.utc).isoformat()}
        self.records.setdefault(key, []).append(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)
        return item

    def read(self, key: str, *, verified_only=True):
        versions = self.records.get(key, [])
        if verified_only:
            versions = [v for v in versions if v["verified"]]
        return versions[-1] if versions else None

    def history(self, key: str):
        return list(self.records.get(key, []))
