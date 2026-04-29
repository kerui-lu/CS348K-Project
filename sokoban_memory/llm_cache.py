from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(stable_json_dumps(data).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLMResponseCache:
    def __init__(self, cache_path: str | Path | None = None, namespace: str = "main"):
        self.cache_path = Path(cache_path) if cache_path else None
        self.namespace = namespace

    @property
    def enabled(self) -> bool:
        return self.cache_path is not None

    def make_key(self, request: dict[str, Any]) -> str:
        return stable_hash({"namespace": self.namespace, "request": request})

    def get(self, key: str) -> dict[str, Any] | None:
        if self.cache_path is None:
            return None
        path = self._path_for_key(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_path is None:
            return
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)

    def _path_for_key(self, key: str) -> Path:
        assert self.cache_path is not None
        return self.cache_path / f"{key}.json"
