from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from sokoban_memory.env import DIRECTIONS, SokobanEnv
from sokoban_memory.types import Action, Position


class PushPlanParseError(ValueError):
    pass


@dataclass(frozen=True)
class PushIntent:
    push: Action
    box: Position | None = None
    box_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {"push": self.push}
        if self.box_id is not None:
            item["box_id"] = self.box_id
        if self.box is not None:
            item["box"] = [self.box.row, self.box.col]
        return item


@dataclass
class FullPathExecutionResult:
    status: str
    trajectory: list[dict[str, Any]]
    total_reward: float
    expanded_actions: list[Action]
    push_execution_log: list[dict[str, Any]]
    executed_push_count: int = 0
    failure_reason: str | None = None
    failure_push_index: int | None = None


def parse_push_plan(raw_output: str, box_count: int | None = None) -> list[PushIntent]:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise PushPlanParseError(f"plan is not valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, list):
        raise PushPlanParseError("plan must be a JSON array")

    intents = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise PushPlanParseError(f"step {index} must be an object")
        if "push" not in item:
            raise PushPlanParseError(f"step {index} is missing push")

        push = item["push"]
        if push not in DIRECTIONS:
            raise PushPlanParseError(f"step {index} has unsupported push direction: {push}")

        box_id = None
        if "box_id" in item:
            raw_box_id = item["box_id"]
            if not isinstance(raw_box_id, int) or isinstance(raw_box_id, bool) or raw_box_id < 0:
                raise PushPlanParseError(f"step {index} box_id must be a non-negative integer")
            if box_count is not None and raw_box_id >= box_count:
                raise PushPlanParseError(f"step {index} has unknown box_id: {raw_box_id}")
            box_id = raw_box_id

        box = None
        if "box" in item:
            raw_box = item["box"]
            if (
                not isinstance(raw_box, list)
                or len(raw_box) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in raw_box)
            ):
                raise PushPlanParseError(f"step {index} box must be [row, col] integers")
            box = Position(raw_box[0], raw_box[1])

        if box_id is None and box is None:
            raise PushPlanParseError(f"step {index} is missing box or box_id")

        intents.append(PushIntent(box=box, box_id=box_id, push=push))
    return intents


def shortest_player_path(env: SokobanEnv, target: Position) -> list[Action] | None:
    blocked = set(env.level.walls) | set(env.boxes)
    if target in blocked or env._is_blocked_cell(target):
        return None

    queue: deque[tuple[Position, list[Action]]] = deque([(env.player, [])])
    visited = {env.player}
    while queue:
        current, path = queue.popleft()
        if current == target:
            return path
        for action, (dr, dc) in DIRECTIONS.items():
            next_pos = current.moved(dr, dc)
            if next_pos in visited or next_pos in blocked or env._is_blocked_cell(next_pos):
                continue
            visited.add(next_pos)
            queue.append((next_pos, [*path, action]))
    return None


def execute_push_plan(
    env: SokobanEnv,
    plan: list[PushIntent],
    max_steps: int,
    raw_response: str,
    call_metadata: dict[str, Any],
) -> FullPathExecutionResult:
    trajectory: list[dict[str, Any]] = []
    expanded_actions: list[Action] = []
    push_execution_log: list[dict[str, Any]] = []
    total_reward = 0.0
    executed_push_count = 0
    box_positions_by_id = {
        box_id: position
        for box_id, position in enumerate(sorted(env.boxes))
    }

    for push_index, intent in enumerate(plan):
        if len(trajectory) >= max_steps:
            return _result(
                "timeout",
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
                "max_steps_reached_before_next_push",
                push_index,
            )

        log_entry = {
            "push_index": push_index,
            "intent": intent.to_dict(),
            "player_before": _position_dict(env.player),
            "boxes_before": [_position_dict(pos) for pos in sorted(env.boxes)],
            "box_positions_by_id_before": _box_positions_by_id_dict(box_positions_by_id),
        }
        push_execution_log.append(log_entry)

        resolved_box = _resolve_intent_box(intent, box_positions_by_id)
        if resolved_box is None:
            failure_reason = "unknown_box_id"
            log_entry["result"] = "invalid_plan"
            log_entry["failure_reason"] = failure_reason
            return _result(
                "invalid_plan",
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
                failure_reason,
                push_index,
            )
        log_entry["resolved_box"] = _position_dict(resolved_box)
        resolved_box_id = (
            intent.box_id
            if intent.box_id is not None
            else _box_id_for_position(box_positions_by_id, resolved_box)
        )
        if resolved_box_id is not None:
            log_entry["resolved_box_id"] = resolved_box_id

        dr, dc = DIRECTIONS[intent.push]
        required_player = resolved_box.moved(-dr, -dc)
        destination = resolved_box.moved(dr, dc)
        log_entry["required_player_position"] = _position_dict(required_player)
        log_entry["box_destination"] = _position_dict(destination)

        validation_error = _validate_push_intent(env, resolved_box, intent.push)
        if validation_error is not None:
            log_entry["result"] = "invalid_plan"
            log_entry["failure_reason"] = validation_error
            return _result(
                "invalid_plan",
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
                validation_error,
                push_index,
            )

        path_to_push = shortest_player_path(env, required_player)
        if path_to_push is None:
            failure_reason = "required_push_position_unreachable"
            log_entry["result"] = "invalid_plan"
            log_entry["failure_reason"] = failure_reason
            return _result(
                "invalid_plan",
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
                failure_reason,
                push_index,
            )
        log_entry["path_to_push"] = path_to_push

        for action in path_to_push:
            if len(trajectory) >= max_steps:
                return _result(
                    "timeout",
                    trajectory,
                    total_reward,
                    expanded_actions,
                    push_execution_log,
                    executed_push_count,
                    "max_steps_reached_while_walking_to_push",
                    push_index,
                )
            step_result = env.step(action)
            expanded_actions.append(action)
            total_reward += step_result.reward
            trajectory.append(
                _trajectory_step(
                    step_result=step_result,
                    step_idx=len(trajectory),
                    action=action,
                    raw_response=raw_response,
                    call_metadata=call_metadata,
                    push_index=push_index,
                    semantic_phase="walk_to_push",
                    intent=intent,
                    invalid_reason=None,
                )
            )
            terminal = _terminal_status(env)
            if terminal is not None:
                log_entry["result"] = terminal
                return _result(
                    terminal,
                    trajectory,
                    total_reward,
                    expanded_actions,
                    push_execution_log,
                    executed_push_count,
                )

        if len(trajectory) >= max_steps:
            return _result(
                "timeout",
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
                "max_steps_reached_before_push",
                push_index,
            )

        step_result = env.step(intent.push)
        expanded_actions.append(intent.push)
        total_reward += step_result.reward
        invalid_reason = None if step_result.info.get("pushed_box") else "push_did_not_move_box"
        trajectory.append(
            _trajectory_step(
                step_result=step_result,
                step_idx=len(trajectory),
                action=intent.push,
                raw_response=raw_response,
                call_metadata=call_metadata,
                push_index=push_index,
                semantic_phase="push",
                intent=intent,
                invalid_reason=invalid_reason,
            )
        )
        if invalid_reason is not None:
            log_entry["result"] = "invalid_plan"
            log_entry["failure_reason"] = invalid_reason
            return _result(
                "invalid_plan",
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
                invalid_reason,
                push_index,
            )

        executed_push_count += 1
        if resolved_box_id is not None:
            box_positions_by_id[resolved_box_id] = destination
        log_entry["result"] = "executed"
        log_entry["player_after"] = _position_dict(env.player)
        log_entry["boxes_after"] = [_position_dict(pos) for pos in sorted(env.boxes)]
        log_entry["box_positions_by_id_after"] = _box_positions_by_id_dict(box_positions_by_id)
        terminal = _terminal_status(env)
        if terminal is not None:
            log_entry["result"] = terminal
            return _result(
                terminal,
                trajectory,
                total_reward,
                expanded_actions,
                push_execution_log,
                executed_push_count,
            )

    if env.is_solved():
        status = "success"
    elif len(trajectory) >= max_steps:
        status = "timeout"
    else:
        status = "plan_exhausted"
    return _result(
        status,
        trajectory,
        total_reward,
        expanded_actions,
        push_execution_log,
        executed_push_count,
        None if status == "success" else status,
        None,
    )


def _resolve_intent_box(
    intent: PushIntent,
    box_positions_by_id: dict[int, Position],
) -> Position | None:
    if intent.box_id is not None:
        return box_positions_by_id.get(intent.box_id)
    return intent.box


def _validate_push_intent(env: SokobanEnv, box: Position, push: Action) -> str | None:
    if box not in env.boxes:
        return "box_coordinate_missing"
    dr, dc = DIRECTIONS[push]
    destination = box.moved(dr, dc)
    required_player = box.moved(-dr, -dc)
    if env._is_blocked_cell(destination):
        return "box_destination_blocked_by_wall_or_boundary"
    if destination in env.boxes:
        return "box_destination_blocked_by_box"
    if env._is_blocked_cell(required_player):
        return "required_push_position_blocked_by_wall_or_boundary"
    if required_player in env.boxes:
        return "required_push_position_blocked_by_box"
    return None


def _trajectory_step(
    *,
    step_result: Any,
    step_idx: int,
    action: Action,
    raw_response: str,
    call_metadata: dict[str, Any],
    push_index: int,
    semantic_phase: str,
    intent: PushIntent,
    invalid_reason: str | None,
) -> dict[str, Any]:
    return {
        "step": step_idx,
        "state": step_result.state_text,
        "raw_action": action,
        "parsed_action": action,
        "executed_action": action,
        "next_state": step_result.next_state_text,
        "reward": step_result.reward,
        "done": step_result.done,
        "response_text": raw_response,
        "prompt_hash": call_metadata.get("prompt_hash"),
        "prompt_char_count": call_metadata.get("prompt_char_count", 0),
        "memory_char_count": call_metadata.get("memory_char_count", 0),
        "non_memory_template_hash": call_metadata.get("non_memory_template_hash"),
        "model": call_metadata.get("model"),
        "temperature": call_metadata.get("temperature"),
        "max_output_tokens": call_metadata.get("max_output_tokens"),
        "memory_hash": call_metadata.get("memory_hash"),
        "cache_namespace": call_metadata.get("cache_namespace"),
        "cache_hit": call_metadata.get("cache_hit", False),
        "cache_key": call_metadata.get("cache_key"),
        "usage": call_metadata.get("usage", {}),
        "info": {
            **step_result.info,
            "invalid_reason": invalid_reason,
            "push_index": push_index,
            "semantic_phase": semantic_phase,
            "planned_push": intent.to_dict(),
        },
    }


def _terminal_status(env: SokobanEnv) -> str | None:
    if env.is_solved():
        return "success"
    deadlocked, _reason = env.is_deadlocked()
    if deadlocked:
        return "deadlock"
    return None


def _result(
    status: str,
    trajectory: list[dict[str, Any]],
    total_reward: float,
    expanded_actions: list[Action],
    push_execution_log: list[dict[str, Any]],
    executed_push_count: int,
    failure_reason: str | None = None,
    failure_push_index: int | None = None,
) -> FullPathExecutionResult:
    return FullPathExecutionResult(
        status=status,
        trajectory=trajectory,
        total_reward=total_reward,
        expanded_actions=expanded_actions,
        push_execution_log=push_execution_log,
        executed_push_count=executed_push_count,
        failure_reason=failure_reason,
        failure_push_index=failure_push_index,
    )


def _position_dict(position: Position) -> dict[str, int]:
    return asdict(position)


def _box_positions_by_id_dict(box_positions_by_id: dict[int, Position]) -> dict[str, list[int]]:
    return {
        f"B{box_id}": [position.row, position.col]
        for box_id, position in sorted(box_positions_by_id.items())
    }


def _box_id_for_position(box_positions_by_id: dict[int, Position], position: Position) -> int | None:
    for box_id, box_position in box_positions_by_id.items():
        if box_position == position:
            return box_id
    return None
