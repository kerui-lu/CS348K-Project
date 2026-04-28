from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sokoban_memory.types import EpisodeResult


class RawTrajectoryMemory:
    def __init__(self) -> None:
        self.episodes: list[dict[str, Any]] = []

    def load(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            self.episodes = []
            return self.episodes
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.episodes = data if isinstance(data, list) else [data]
        return self.episodes

    def add_episode(self, episode: EpisodeResult) -> None:
        self.episodes.append(episode.to_dict())


class HeuristicMemory:
    def __init__(self) -> None:
        self.heuristics: list[str] = []

    def load(self, path: Path) -> list[str]:
        if not path.exists():
            self.heuristics = []
            return self.heuristics
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.heuristics = list(data.get("heuristics", data))
        return self.heuristics

    def save(self, heuristics: list[str], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.heuristics = heuristics
        with path.open("w", encoding="utf-8") as f:
            json.dump({"heuristics": heuristics}, f, indent=2)

