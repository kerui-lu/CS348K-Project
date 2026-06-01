from __future__ import annotations

import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sokoban_memory.llm_cache import stable_hash
from sokoban_memory.types import Level

V3_TRAJECTORY_SCHEMA_VERSION = "v3_trajectory_v1"
EXECUTOR_VERSION = "full_path_executor_v3"
PROMPT_RENDERER_VERSION = "full_path_prompt_renderer_v3"
MEMORY_RENDERER_VERSION = "memory_renderer_v3"

PROMPT_LEAKAGE_PATTERNS = (
    "reference_solution",
    "solver_min_pushes",
    "solver_min_steps",
    "solver_min_path",
    "optimal push sequence",
    "optimal_push_sequence",
)


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    code_commit: str
    level_suite_path: str | None
    level_suite_hash: str | None


def make_run_identity(level_suite_path: str | Path | None = None, run_id: str | None = None) -> RunIdentity:
    path = Path(level_suite_path) if level_suite_path else None
    return RunIdentity(
        run_id=run_id or uuid.uuid4().hex,
        code_commit=_current_code_commit(),
        level_suite_path=str(path) if path else None,
        level_suite_hash=_hash_path(path) if path and path.exists() else None,
    )


def failure_subtype(
    *,
    status: str,
    failure_reason: str | None,
    raw_output: str | None = None,
    error_message: str | None = None,
) -> str | None:
    if status == "success":
        return None
    if status in {"deadlock", "timeout", "plan_exhausted", "budget_exhausted", "api_error"}:
        return status
    reason = failure_reason or ""
    if reason == "invalid_push_plan_parse":
        text = (raw_output or "").strip()
        message = error_message or ""
        if not text:
            return "empty_output"
        if _looks_truncated_json(text):
            return "truncated_output"
        if "not valid JSON" in message or "Expecting" in message:
            return "json_parse_error"
        return "schema_error"
    if reason in {"box_coordinate_missing", "unknown_box_id"}:
        return "wrong_box_reference"
    if reason.startswith("box_destination_blocked"):
        return "blocked_destination"
    if reason.startswith("required_push_position_blocked"):
        return "blocked_standing_cell"
    if reason == "required_push_position_unreachable":
        return "unreachable_standing_cell"
    if reason == "push_did_not_move_box":
        return "schema_error"
    return reason or status


def progress_metrics(level: Level, initial_board: str, trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    boards = [initial_board]
    boards.extend(
        step["next_state"]
        for step in trajectory
        if isinstance(step, dict) and isinstance(step.get("next_state"), str)
    )
    counts = [_boxes_on_targets(board) for board in boards]
    total_boxes = max(1, len(level.boxes))
    first_deadlock_index = _first_deadlock_step_index(trajectory)
    event_limit = first_deadlock_index + 1 if first_deadlock_index is not None else len(trajectory)
    event_boards = boards[: event_limit + 1]
    events = 0
    previous = _boxes_on_targets(event_boards[0]) if event_boards else 0
    for board in event_boards[1:]:
        current = _boxes_on_targets(board)
        events += max(0, current - previous)
        previous = current
    final_count = counts[-1] if counts else 0
    best_count = max(counts) if counts else 0
    return {
        "best_boxes_on_targets": best_count,
        "final_boxes_on_targets": final_count,
        "target_placement_events_before_first_deadlock": events,
        "best_goal_completion_rate": best_count / total_boxes,
        "final_goal_completion_rate": final_count / total_boxes,
        "normalized_target_placement_before_first_deadlock": events / total_boxes,
    }


def board_before_failed_push(attempt: dict[str, Any]) -> str | None:
    failure_push_index = attempt.get("failure_push_index")
    logs = attempt.get("push_execution_log")
    if isinstance(logs, list):
        if isinstance(failure_push_index, int):
            for log in logs:
                if isinstance(log, dict) and log.get("push_index") == failure_push_index:
                    board = log.get("board_before_push")
                    return board if isinstance(board, str) else None
        for log in reversed(logs):
            if isinstance(log, dict) and log.get("status") in {"failed", "deadlock"}:
                board = log.get("board_before_push")
                return board if isinstance(board, str) else None
    board = attempt.get("final_board")
    return board if isinstance(board, str) else None


def board_after_last_successful_push(attempt: dict[str, Any], initial_board: str) -> str:
    logs = attempt.get("push_execution_log")
    if isinstance(logs, list):
        for log in reversed(logs):
            if not isinstance(log, dict):
                continue
            board = log.get("board_after_push")
            if isinstance(board, str):
                return board
    final_board = attempt.get("final_board")
    return final_board if isinstance(final_board, str) else initial_board


def build_v3_attempt_trace(
    *,
    level: Level,
    run_identity: RunIdentity,
    condition: str,
    attempt: dict[str, Any],
    attempt_index: int,
    seed: int,
    model: str | None,
    prompt_version: str | None,
    max_output_tokens: int | None,
    initial_board: str,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    call_metadata = dict(attempt.get("call_metadata", {}))
    if not call_metadata:
        call_metadata = _first_call_metadata(trajectory)
    status = str(attempt.get("status", "unknown"))
    subtype = failure_subtype(
        status=status,
        failure_reason=attempt.get("failure_reason"),
        raw_output=attempt.get("raw_plan_response"),
        error_message=attempt.get("error_message"),
    )
    metrics = progress_metrics(level, initial_board, trajectory)
    return {
        "schema_version": V3_TRAJECTORY_SCHEMA_VERSION,
        "run_id": run_identity.run_id,
        "code_commit": run_identity.code_commit,
        "level_suite_path": run_identity.level_suite_path,
        "level_suite_hash": run_identity.level_suite_hash,
        "executor_version": EXECUTOR_VERSION,
        "prompt_renderer_version": PROMPT_RENDERER_VERSION,
        "memory_renderer_version": MEMORY_RENDERER_VERSION,
        "cache_key": call_metadata.get("cache_key"),
        "level_id": level.level_id,
        "split": level.split,
        "source_family": level.metadata.get("source_family"),
        "difficulty_bucket": level.metadata.get("difficulty_bucket"),
        "condition": condition,
        "attempt_index": attempt_index,
        "seed": seed,
        "model": model,
        "prompt_version": prompt_version,
        "prompt_hash": call_metadata.get("prompt_hash"),
        "max_output_tokens": max_output_tokens,
        "initial_board": initial_board,
        "final_board": attempt.get("final_board"),
        "board_before_failed_push": board_before_failed_push(attempt),
        "board_after_last_successful_push": board_after_last_successful_push(attempt, initial_board),
        "raw_plan_response": attempt.get("raw_plan_response"),
        "parsed_plan": attempt.get("planned_pushes", []),
        "expanded_actions": attempt.get("expanded_actions", []),
        "push_execution_log": attempt.get("push_execution_log", []),
        "status": status,
        "failure_reason": attempt.get("failure_reason"),
        "failure_subtype": subtype,
        "failure_push_index": attempt.get("failure_push_index"),
        **metrics,
    }


def assert_no_prompt_leakage(text: str) -> None:
    lowered = text.lower()
    found = [pattern for pattern in PROMPT_LEAKAGE_PATTERNS if pattern in lowered]
    if found:
        raise ValueError(f"Prompt contains solver/reference leakage terms: {found}")


def heuristic_scope(heuristic: str) -> str:
    text = heuristic.strip()
    lowered = text.lower()
    if not text:
        return "rejected"
    if any(pattern in lowered for pattern in PROMPT_LEAKAGE_PATTERNS):
        return "rejected"
    if _contains_board_row(text) or _contains_raw_direction_sequence(lowered):
        return "rejected"
    if _contains_coordinate(text) or "level_id" in lowered or "boxoban_" in lowered or "in this level" in lowered:
        return "same_level_only"
    return "global_allowed"


def _hash_path(path: Path) -> str:
    try:
        return stable_hash(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return stable_hash(path.read_bytes().hex())


def _current_code_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _looks_truncated_json(text: str) -> bool:
    return text.startswith(("[", "{")) and not text.endswith(("]", "}"))


def _boxes_on_targets(board: str) -> int:
    return board.count("*")


def _first_deadlock_step_index(trajectory: list[dict[str, Any]]) -> int | None:
    for index, step in enumerate(trajectory):
        info = step.get("info") if isinstance(step, dict) else None
        if isinstance(info, dict) and info.get("deadlocked") is True:
            return index
    return None


def _first_call_metadata(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        return {
            "cache_key": step.get("cache_key"),
            "prompt_hash": step.get("prompt_hash"),
        }
    return {}


def _contains_coordinate(text: str) -> bool:
    return bool(re.search(r"(?:\[[0-9]+,\s*[0-9]+\]|\([0-9]+,\s*[0-9]+\))", text))


def _contains_board_row(text: str) -> bool:
    return bool(re.search(r"#[# .$@+*]{4,}#?", text))


def _contains_raw_direction_sequence(lowered: str) -> bool:
    directions = re.findall(r"\b(?:up|down|left|right)\b", lowered)
    return len(directions) >= 3 and ("sequence" in lowered or "," in lowered or " then " in lowered)
