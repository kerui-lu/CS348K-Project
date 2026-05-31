from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sokoban_memory.llm_cache import stable_hash
from sokoban_memory.types import EpisodeResult, Level
from sokoban_memory.v3_trajectory import MEMORY_RENDERER_VERSION, heuristic_scope

RAW_MEMORY_SCHEMA_VERSION = "full_path_raw_trajectory_memory_v1"
HEURISTIC_MEMORY_SCHEMA_VERSION = "reflection_heuristic_memory_v2"
RAW_RENDER_BANNED_WORDS = ("lesson", "heuristic", "should", "avoid", "must", "key mistake")


@dataclass(frozen=True)
class MemoryRenderConfig:
    max_memory_items: int = 3
    max_steps_per_memory: int = 6
    max_memory_chars: int = 4000
    # Same-level heuristic tuning (V2). These only affect the
    # `heuristic_same_level_iterative` path; cross-level conditions keep the
    # legacy caps above.
    same_level_reflection_evidence_items: int = 4
    same_level_reflection_evidence_chars: int = 16000
    same_level_heuristic_render_items: int = 6
    same_level_heuristic_render_chars: int = 8000

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
        self.episodes.append(compress_episode(episode, max_steps=None))
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

    def records_for_level(self, level_id: str) -> list[dict[str, Any]]:
        return [episode for episode in self.episodes if str(episode.get("level_id")) == level_id]

    def render_for_level(self, level_id: str, config: MemoryRenderConfig) -> str:
        selected = _select_compact_same_level_records(
            self.records_for_level(level_id),
            max_items=min(config.max_memory_items, 2),
        )
        if not selected:
            return f"No same-level trajectory records are available for {level_id}."
        sections = [
            "Same-level compact raw failure evidence:",
            f"memory_renderer_version: {MEMORY_RENDERER_VERSION}",
            f"level_id: {level_id}",
        ]
        for idx, episode in enumerate(selected, start=1):
            sections.append(_render_compact_same_level_evidence(idx, episode))
        rendered = truncate_text("\n\n".join(sections), config.max_memory_chars)
        assert_raw_render_has_no_strategic_words(rendered)
        return rendered

    def render_same_level_failure_evidence(self, level_id: str, config: MemoryRenderConfig) -> str:
        """Full same-level failure evidence for the reflection generator.

        Unlike `render_for_level` (which is shown to the *planner* and is kept
        compact), this is the *input to the reflection LLM*. It uses a larger
        item/char budget so the reflector can see precisely what failed:
        the failed push, verifier reason, and the relevant boards.
        """
        records = self.records_for_level(level_id)
        selected = _select_compact_same_level_records(
            records,
            max_items=config.same_level_reflection_evidence_items,
        )
        if not selected:
            return f"No same-level failure evidence is available for {level_id}."
        sections = [
            "Same-level failure evidence from prior attempts on this exact board:",
            f"memory_renderer_version: {MEMORY_RENDERER_VERSION}",
            f"level_id: {level_id}",
            f"attempt_records: {len(selected)}",
        ]
        for idx, episode in enumerate(selected, start=1):
            sections.append(_render_compact_same_level_evidence(idx, episode))
        rendered = truncate_text("\n\n".join(sections), config.same_level_reflection_evidence_chars)
        assert_raw_render_has_no_strategic_words(rendered)
        return rendered

    def render_verifier_summary_for_level(self, level_id: str, config: MemoryRenderConfig) -> str:
        selected = _select_compact_same_level_records(
            self.records_for_level(level_id),
            max_items=1,
        )
        if not selected:
            return f"No same-level verifier feedback is available for {level_id}."
        episode = selected[-1]
        lines = [
            "Verifier summary from the previous same-level attempt:",
            f"level_id: {level_id}",
            f"failed_push_index: {episode.get('failure_push_index')}",
            f"failure_subtype: {episode.get('failure_subtype') or episode.get('failure_reason')}",
            f"verifier_reason: {episode.get('failure_reason')}",
        ]
        return truncate_text("\n".join(lines), config.max_memory_chars)

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
        selected = [
            heuristic
            for heuristic in self.heuristics
            if classify_heuristic(heuristic)["scope"] == "global_allowed"
        ][: config.max_memory_items]
        if not selected:
            return "No reflection heuristics are available."
        lines = ["Reflection heuristics distilled from previous failures:"]
        lines.extend(f"{idx}. {heuristic}" for idx, heuristic in enumerate(selected, start=1))
        return truncate_text("\n".join(lines), config.max_memory_chars)

    def render_for_level(
        self,
        level_id: str,
        config: MemoryRenderConfig,
        *,
        same_level_mode: bool = False,
    ) -> str:
        source_level_ids = {
            str(item)
            for item in (
                self.source_metadata.get("source_level_ids")
                or self.source_metadata.get("source_train_level_ids")
                or []
            )
        }
        allow_same_level_rules = not source_level_ids or level_id in source_level_ids
        if same_level_mode:
            # V2: these heuristics were generated from THIS level's failures and
            # are only ever rendered for THIS level, so concrete, board-grounded
            # guidance (coordinates, directions, even short board fragments) is
            # exactly what we want. Do not drop on scope; just budget by count.
            item_cap = config.same_level_heuristic_render_items
            char_cap = config.same_level_heuristic_render_chars
            selected = [h for h in self.heuristics if h.strip()][:item_cap]
        else:
            item_cap = config.max_memory_items
            char_cap = config.max_memory_chars
            selected = [
                heuristic
                for heuristic in self.heuristics
                if classify_heuristic(heuristic)["scope"] == "global_allowed"
                or (
                    allow_same_level_rules
                    and classify_heuristic(heuristic)["scope"] == "same_level_only"
                )
            ][:item_cap]
        if not selected:
            return f"No same-level reflection heuristics are available for {level_id}."
        lines = [
            "Same-level reflection heuristics distilled from previous failures on this exact board:",
            f"level_id: {level_id}",
        ]
        lines.extend(f"{idx}. {heuristic}" for idx, heuristic in enumerate(selected, start=1))
        return truncate_text("\n".join(lines), char_cap)

    def classified_heuristics(self) -> list[dict[str, str]]:
        return [classify_heuristic(heuristic) for heuristic in self.heuristics]

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
    max_steps_per_memory: int | None = None,
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
    metadata = dict(data.get("metadata", {}))
    selected_steps = _select_steps(trajectory, max_steps)
    return {
        "level_id": data.get("level_id"),
        "status": data.get("status"),
        "step_count": data.get("step_count", len(trajectory)),
        "total_reward": data.get("total_reward"),
        "policy_mode": data.get("policy_mode"),
        "schema_version": metadata.get("schema_version"),
        "run_id": metadata.get("run_id"),
        "code_commit": metadata.get("code_commit"),
        "level_suite_hash": metadata.get("level_suite_hash"),
        "raw_plan_response": metadata.get("raw_plan_response"),
        "planned_pushes": metadata.get("planned_pushes", []),
        "expanded_actions": metadata.get("expanded_actions", []),
        "push_execution_log": metadata.get("push_execution_log", []),
        "failure_reason": metadata.get("failure_reason"),
        "failure_subtype": metadata.get("failure_subtype"),
        "failure_push_index": metadata.get("failure_push_index"),
        "initial_board": metadata.get("initial_board"),
        "final_board": metadata.get("final_board"),
        "board_before_failed_push": metadata.get("board_before_failed_push"),
        "board_after_last_successful_push": metadata.get("board_after_last_successful_push"),
        "v3_attempt_trace": metadata.get("v3_attempt_trace"),
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
        "semantic_phase": info.get("semantic_phase"),
        "push_index": info.get("push_index"),
        "planned_push": info.get("planned_push"),
        "next_state": step.get("next_state"),
    }


def _render_episode_summary(index: int, episode: dict[str, Any], max_steps: int) -> str:
    lines = [
        f"record_index: {index}",
        f"level_id: {episode.get('level_id')}",
        f"final_status: {episode.get('status')}",
        f"step_count: {episode.get('step_count')}",
        f"total_reward: {episode.get('total_reward')}",
        f"failure_reason: {episode.get('failure_reason')}",
        f"planned_pushes: {episode.get('planned_pushes', [])}",
        f"expanded_actions: {episode.get('expanded_actions', [])}",
    ]
    for push_log in list(episode.get("push_execution_log", []))[:max_steps]:
        lines.append(f"push_log: {push_log}")
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
                    f"solved={step.get('solved')}, "
                    f"phase={step.get('semantic_phase')}, "
                    f"push_index={step.get('push_index')}"
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
    metadata = getattr(memory, "source_metadata", {})
    if metadata.get("memory_scope") == "same_level":
        return
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


def classify_heuristic(heuristic: str) -> dict[str, str]:
    return {"heuristic": heuristic, "scope": heuristic_scope(heuristic)}


def _select_compact_same_level_records(records: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or not records:
        return []
    if len(records) <= max_items:
        return records
    by_subtype: dict[str, dict[str, Any]] = {}
    for record in reversed(records):
        subtype = str(record.get("failure_subtype") or record.get("failure_reason") or record.get("status"))
        by_subtype.setdefault(subtype, record)
    if len(by_subtype) >= max_items:
        return list(by_subtype.values())[:max_items]
    return [records[0], records[-1]][:max_items]


def _render_compact_same_level_evidence(index: int, episode: dict[str, Any]) -> str:
    failed_log = _failed_push_log(episode)
    lines = [
        f"evidence_index: {index}",
        f"final_status: {episode.get('status')}",
        f"failure_subtype: {episode.get('failure_subtype') or episode.get('failure_reason')}",
        f"verifier_reason: {episode.get('failure_reason')}",
        f"failure_push_index: {episode.get('failure_push_index')}",
        "initial_board:",
        str(episode.get("initial_board")),
        "board_before_failed_push:",
        str(episode.get("board_before_failed_push")),
        "failed_push:",
        str(failed_log.get("model_intent") or failed_log.get("intent") if failed_log else None),
        f"resolved_box_before_push: {failed_log.get('resolved_box_before_push') if failed_log else None}",
        f"standing_cell_required: {failed_log.get('standing_cell_required') if failed_log else None}",
        f"destination_cell: {failed_log.get('destination_cell') if failed_log else None}",
        "board_after_last_successful_push:",
        str(episode.get("board_after_last_successful_push")),
    ]
    concise_log = _concise_push_log(episode)
    if concise_log:
        lines.append(f"concise_push_log: {concise_log}")
    return "\n".join(lines)


def _failed_push_log(episode: dict[str, Any]) -> dict[str, Any] | None:
    logs = episode.get("push_execution_log")
    if not isinstance(logs, list):
        return None
    failure_push_index = episode.get("failure_push_index")
    if isinstance(failure_push_index, int):
        for log in logs:
            if isinstance(log, dict) and log.get("push_index") == failure_push_index:
                return log
    for log in reversed(logs):
        if isinstance(log, dict) and log.get("status") == "failed":
            return log
    return logs[-1] if logs and isinstance(logs[-1], dict) else None


def _concise_push_log(episode: dict[str, Any]) -> list[dict[str, Any]]:
    logs = episode.get("push_execution_log")
    if not isinstance(logs, list):
        return []
    concise = []
    for log in logs[-4:]:
        if not isinstance(log, dict):
            continue
        concise.append(
            {
                "push_index": log.get("push_index"),
                "model_intent": log.get("model_intent") or log.get("intent"),
                "status": log.get("status") or log.get("result"),
                "failure_subtype": log.get("failure_subtype"),
            }
        )
    return concise
