from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sokoban_memory.llm_cache import stable_hash
from sokoban_memory.types import EpisodeResult, Level

RAW_MEMORY_SCHEMA_VERSION = "raw_trajectory_memory_v2"
HEURISTIC_MEMORY_SCHEMA_VERSION = "reflection_heuristic_memory_v2"
RAW_RENDER_BANNED_WORDS = ("lesson", "heuristic", "should", "avoid", "must", "key mistake")


@dataclass(frozen=True)
class MemoryRenderConfig:
    max_memory_items: int = 3
    max_steps_per_memory: int = 6
    max_memory_chars: int = 4000

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def hash_file(path: Path) -> str:
    return stable_hash(json.loads(path.read_text(encoding="utf-8")))


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "\n[truncated to memory character budget]"
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


class RawTrajectoryMemory:
    def __init__(
        self,
        episodes: list[dict[str, Any]] | None = None,
        source_metadata: dict[str, Any] | None = None,
        memory_hash: str | None = None,
    ) -> None:
        self.episodes = episodes or []
        self.source_metadata = source_metadata or {}
        self.memory_hash = memory_hash or self.compute_hash()

    def load(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            self.episodes = []
            self.source_metadata = {}
            self.memory_hash = self.compute_hash()
            return self.episodes

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") == RAW_MEMORY_SCHEMA_VERSION:
            self.episodes = list(data.get("episodes", []))
            self.source_metadata = dict(data.get("source_metadata", {}))
        else:
            self.episodes = data if isinstance(data, list) else [data]
            self.source_metadata = {"legacy_format": True}
        self.memory_hash = hash_file(path)
        return self.episodes

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def add_episode(self, episode: EpisodeResult) -> None:
        self.episodes.append(compress_episode(episode))
        self.memory_hash = self.compute_hash()

    def render(self, config: MemoryRenderConfig) -> str:
        selected = self.episodes[: config.max_memory_items]
        if not selected:
            return "No prior trajectory records are available."

        sections = ["Prior trajectory records:"]
        for idx, episode in enumerate(selected, start=1):
            sections.append(_render_episode_summary(idx, episode, config.max_steps_per_memory))
        rendered = truncate_text("\n\n".join(sections), config.max_memory_chars)
        assert_raw_render_has_no_strategic_words(rendered)
        return rendered

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RAW_MEMORY_SCHEMA_VERSION,
            "source_metadata": self.source_metadata,
            "memory_item_count": len(self.episodes),
            "episodes": self.episodes,
        }

    def compute_hash(self) -> str:
        return stable_hash(self.to_dict())


class HeuristicMemory:
    def __init__(
        self,
        heuristics: list[str] | None = None,
        source_metadata: dict[str, Any] | None = None,
        memory_hash: str | None = None,
    ) -> None:
        self.heuristics = heuristics or []
        self.source_metadata = source_metadata or {}
        self.memory_hash = memory_hash or self.compute_hash()

    def load(self, path: Path) -> list[str]:
        if not path.exists():
            self.heuristics = []
            self.source_metadata = {}
            self.memory_hash = self.compute_hash()
            return self.heuristics

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema_version") == HEURISTIC_MEMORY_SCHEMA_VERSION:
            self.heuristics = list(data.get("heuristics", []))
            self.source_metadata = dict(data.get("source_metadata", {}))
        elif isinstance(data, dict):
            self.heuristics = list(data.get("heuristics", []))
            self.source_metadata = {"legacy_format": True}
        else:
            self.heuristics = list(data)
            self.source_metadata = {"legacy_format": True}
        self.memory_hash = hash_file(path)
        return self.heuristics

    def save(self, heuristics_or_path: list[str] | Path, path: Path | None = None) -> None:
        # Backward compatible with the v1 save(heuristics, path) shape.
        if path is None:
            output_path = Path(heuristics_or_path)
        else:
            self.heuristics = list(heuristics_or_path)  # type: ignore[arg-type]
            output_path = path
        self.memory_hash = self.compute_hash()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def render(self, config: MemoryRenderConfig) -> str:
        selected = self.heuristics[: config.max_memory_items]
        if not selected:
            return "No reflection heuristics are available."
        lines = ["Reflection heuristics distilled from previous failures:"]
        lines.extend(f"{idx}. {heuristic}" for idx, heuristic in enumerate(selected, start=1))
        return truncate_text("\n".join(lines), config.max_memory_chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HEURISTIC_MEMORY_SCHEMA_VERSION,
            "source_metadata": self.source_metadata,
            "memory_item_count": len(self.heuristics),
            "heuristics": self.heuristics,
        }

    def compute_hash(self) -> str:
        return stable_hash(self.to_dict())


def build_raw_memory_bank(
    episodes: list[EpisodeResult | dict[str, Any]],
    source_metadata: dict[str, Any] | None = None,
    max_steps_per_memory: int = 6,
) -> RawTrajectoryMemory:
    metadata = dict(source_metadata or {})
    metadata.setdefault(
        "source_train_level_ids",
        sorted({str(_episode_level_id(episode)) for episode in episodes if _episode_status(episode) != "success"}),
    )
    compressed = [
        compress_episode(episode, max_steps=max_steps_per_memory)
        for episode in episodes
        if _episode_status(episode) != "success"
    ]
    return RawTrajectoryMemory(episodes=compressed, source_metadata=metadata)


def compress_episode(episode: EpisodeResult | dict[str, Any], max_steps: int | None = None) -> dict[str, Any]:
    data = episode.to_dict() if isinstance(episode, EpisodeResult) else episode
    trajectory = list(data.get("trajectory", []))
    selected_steps = _select_steps(trajectory, max_steps)
    return {
        "level_id": data.get("level_id"),
        "status": data.get("status"),
        "step_count": data.get("step_count", len(trajectory)),
        "total_reward": data.get("total_reward"),
        "steps": [_compress_step(step) for step in selected_steps],
    }


def _episode_status(episode: EpisodeResult | dict[str, Any]) -> str:
    return episode.status if isinstance(episode, EpisodeResult) else str(episode.get("status"))


def _episode_level_id(episode: EpisodeResult | dict[str, Any]) -> str:
    return episode.level_id if isinstance(episode, EpisodeResult) else str(episode.get("level_id"))


def _select_steps(trajectory: list[dict[str, Any]], max_steps: int | None) -> list[dict[str, Any]]:
    if max_steps is None or max_steps <= 0 or len(trajectory) <= max_steps:
        return trajectory
    return trajectory[-max_steps:]


def _compress_step(step: dict[str, Any]) -> dict[str, Any]:
    info = dict(step.get("info", {}))
    return {
        "step": step.get("step"),
        "state": step.get("state"),
        "raw_action": step.get("raw_action"),
        "parsed_action": step.get("parsed_action"),
        "executed_action": step.get("executed_action"),
        "reward": step.get("reward"),
        "invalid_reason": info.get("invalid_reason"),
        "pushed_box": info.get("pushed_box"),
        "deadlocked": info.get("deadlocked"),
        "solved": info.get("solved"),
        "next_state": step.get("next_state"),
    }


def _render_episode_summary(index: int, episode: dict[str, Any], max_steps: int) -> str:
    lines = [
        f"record_index: {index}",
        f"level_id: {episode.get('level_id')}",
        f"final_status: {episode.get('status')}",
        f"step_count: {episode.get('step_count')}",
        f"total_reward: {episode.get('total_reward')}",
    ]
    for step in _select_steps(list(episode.get("steps", [])), max_steps):
        lines.extend(
            [
                f"step {step.get('step')}:",
                "state:",
                str(step.get("state")),
                f"executed_action: {step.get('executed_action')}",
                f"reward: {step.get('reward')}",
                (
                    "outcome: "
                    f"pushed_box={step.get('pushed_box')}, "
                    f"deadlocked={step.get('deadlocked')}, "
                    f"solved={step.get('solved')}"
                ),
                "next_state:",
                str(step.get("next_state")),
            ]
        )
    return "\n".join(lines)


def get_memory_source_level_ids(memory: RawTrajectoryMemory | HeuristicMemory) -> list[str]:
    metadata = getattr(memory, "source_metadata", {})
    ids = metadata.get("source_train_level_ids") or metadata.get("source_level_ids") or []
    return [str(level_id) for level_id in ids]


def validate_no_eval_memory_leak(levels: list[Level], memory: RawTrajectoryMemory | HeuristicMemory) -> None:
    eval_level_ids = {level.level_id for level in levels if level.split == "eval"}
    if not eval_level_ids:
        return
    source_level_ids = set(get_memory_source_level_ids(memory))
    if not source_level_ids:
        raise ValueError(
            "Memory file is missing source_train_level_ids metadata; refusing eval run."
        )
    overlap = sorted(eval_level_ids & source_level_ids)
    if overlap:
        raise ValueError(
            "Memory bank contains eval level IDs; refusing eval run: "
            + ", ".join(overlap)
        )


def assert_raw_render_has_no_strategic_words(rendered: str) -> None:
    lowered = rendered.lower()
    found = [word for word in RAW_RENDER_BANNED_WORDS if word in lowered]
    if found:
        raise ValueError(f"Raw trajectory render contains strategic words: {found}")
