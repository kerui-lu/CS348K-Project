from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sokoban_memory.action_parser import choose_fallback_action, parse_action
from sokoban_memory.agents import BaseAgent, LLMBudgetExceeded
from sokoban_memory.env import DIRECTIONS, SokobanEnv
from sokoban_memory.full_path import execute_push_plan, parse_push_plan, PushPlanParseError
from sokoban_memory.logging_utils import save_episode, save_summary
from sokoban_memory.memory import get_memory_source_level_ids, validate_no_eval_memory_leak
from sokoban_memory.metrics import summarize_results
from sokoban_memory.prompts import level_metadata
from sokoban_memory.types import EpisodeResult, Level, Position

RULES_TEXT = "Move Up/Down/Left/Right. Push boxes only. No pulling. No pushing two boxes."


def run_episode(
    env: SokobanEnv,
    agent: BaseAgent,
    max_steps: int,
    seed: int,
    max_repair_attempts: int = 0,
) -> EpisodeResult:
    if getattr(agent, "policy_mode", "") == "full_path":
        return run_full_path_episode(
            env,
            agent,
            max_steps=max_steps,
            seed=seed,
            max_repair_attempts=max_repair_attempts,
        )

    rng = random.Random(seed)
    state_text = env.reset()
    trajectory: list[dict[str, Any]] = []
    invalid_move_count = 0
    total_reward = 0.0
    status = "timeout"
    budget_exhausted = False
    metadata: dict[str, Any] = {}
    start_llm_calls = getattr(agent, "llm_call_count", 0)
    start_token_cost = getattr(agent, "token_cost", 0.0)
    start_cache_hits = getattr(agent, "cache_hits", 0)
    start_cache_misses = getattr(agent, "cache_misses", 0)

    for step_idx in range(max_steps):
        legal_actions = env.legal_actions()
        context = {
            "step": step_idx,
            "legal_actions": legal_actions,
            "push_actions": _push_actions(env, legal_actions),
            "rules": RULES_TEXT,
            "memory": _agent_memory(agent),
        }
        try:
            raw_action = agent.select_action(state_text, context)
        except LLMBudgetExceeded as exc:
            status = "budget_exhausted"
            budget_exhausted = True
            metadata["failure_reason"] = "llm_call_budget_exhausted"
            metadata["budget_exhausted_at_step"] = step_idx
            metadata["budget_error"] = str(exc)
            break
        except Exception as exc:
            status = "api_error" if getattr(agent, "policy_mode", "") == "full_path" else "failure"
            metadata["failure_reason"] = "api_error" if status == "api_error" else "agent_error"
            metadata["error_type"] = type(exc).__name__
            metadata["error_message"] = str(exc)
            metadata["error_at_step"] = step_idx
            break

        call_metadata = dict(getattr(agent, "last_call_metadata", {}))
        try:
            parsed_action = parse_action(raw_action)

            if parsed_action not in legal_actions:
                invalid_move_count += 1
                action = choose_fallback_action(legal_actions, rng)
                invalid_reason = "invalid_or_illegal_action"
            else:
                action = parsed_action
                invalid_reason = None

            result = env.step(action)
        except ValueError as exc:
            status = "invalid_failure"
            metadata["failure_reason"] = "invalid_action_recovery_failed"
            metadata["error_type"] = type(exc).__name__
            metadata["error_message"] = str(exc)
            metadata["error_at_step"] = step_idx
            break
        total_reward += result.reward
        trajectory.append(
            {
                "step": step_idx,
                "state": state_text,
                "raw_action": raw_action,
                "parsed_action": parsed_action,
                "executed_action": action,
                "next_state": result.next_state_text,
                "reward": result.reward,
                "done": result.done,
                "response_text": raw_action,
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
                "info": {**result.info, "invalid_reason": invalid_reason},
            }
        )
        state_text = result.next_state_text

        if env.is_solved():
            status = "success"
            break
        deadlocked, _reason = env.is_deadlocked()
        if deadlocked:
            status = "deadlock"
            break

    return EpisodeResult(
        level_id=env.level.level_id,
        agent_type=agent.agent_type,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        step_count=len(trajectory),
        invalid_move_count=invalid_move_count,
        total_reward=total_reward,
        llm_call_count=getattr(agent, "llm_call_count", 0) - start_llm_calls,
        token_cost=getattr(agent, "token_cost", 0.0) - start_token_cost,
        trajectory=trajectory,
        policy_mode=getattr(agent, "policy_mode", "unknown"),
        model=getattr(agent, "model", None),
        prompt_version=getattr(agent, "prompt_version", None),
        memory_path=getattr(agent, "memory_path", None),
        memory_hash=getattr(agent, "memory_hash", None),
        memory_caps=agent.memory_caps(),
        temperature=getattr(agent, "temperature", None),
        max_output_tokens=getattr(agent, "max_output_tokens", None),
        cache_namespace=getattr(agent, "cache_namespace", None),
        level_split=getattr(env.level, "split", "unspecified"),
        level_tags=list(getattr(env.level, "tags", [])),
        optimal_steps=getattr(env.level, "optimal_steps", None),
        cache_hits=getattr(agent, "cache_hits", 0) - start_cache_hits,
        cache_misses=getattr(agent, "cache_misses", 0) - start_cache_misses,
        budget_exhausted=budget_exhausted,
        metadata=metadata,
    )


def run_full_path_episode(
    env: SokobanEnv,
    agent: BaseAgent,
    max_steps: int,
    seed: int,
    max_repair_attempts: int = 0,
) -> EpisodeResult:
    state_text = env.reset()
    start_llm_calls = getattr(agent, "llm_call_count", 0)
    start_token_cost = getattr(agent, "token_cost", 0.0)
    start_cache_hits = getattr(agent, "cache_hits", 0)
    start_cache_misses = getattr(agent, "cache_misses", 0)
    max_repair_attempts = max(0, max_repair_attempts)
    repair_attempts: list[dict[str, Any]] = []
    repair_feedback: str | None = None
    final_status = "invalid_plan"
    final_step_count = 0
    final_total_reward = 0.0
    final_trajectory: list[dict[str, Any]] = []
    final_metadata: dict[str, Any] = {}
    budget_exhausted = False

    for attempt_index in range(max_repair_attempts + 1):
        state_text = env.reset()
        context = {
            "rules": RULES_TEXT,
            "state_summary": _state_summary(env),
            "max_steps": max_steps,
            "memory": _agent_memory(agent),
            "repair_feedback": repair_feedback,
        }
        try:
            if hasattr(agent, "select_plan"):
                raw_plan = agent.select_plan(state_text, context)  # type: ignore[attr-defined]
            else:
                raw_plan = agent.select_action(state_text, context)
        except LLMBudgetExceeded as exc:
            final_status = "budget_exhausted"
            budget_exhausted = True
            final_metadata = {
                "failure_reason": "llm_call_budget_exhausted",
                "budget_exhausted_at_attempt": attempt_index,
                "budget_error": str(exc),
                "planned_pushes": [],
                "expanded_actions": [],
                "push_execution_log": [],
            }
            break
        except Exception as exc:
            final_status = "api_error"
            final_metadata = {
                "failure_reason": "api_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_at_attempt": attempt_index,
                "planned_pushes": [],
                "expanded_actions": [],
                "push_execution_log": [],
            }
            break

        call_metadata = dict(getattr(agent, "last_call_metadata", {}))
        try:
            plan = parse_push_plan(raw_plan, box_count=len(env.boxes))
            planned_pushes = [intent.to_dict() for intent in plan]
        except PushPlanParseError as exc:
            attempt = {
                "attempt_index": attempt_index,
                "raw_plan_response": raw_plan,
                "status": "invalid_plan",
                "failure_reason": "invalid_push_plan_parse",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "planned_pushes": [],
                "executed_push_count": 0,
                "planned_push_count": 0,
                "expanded_primitive_step_count": 0,
                "expanded_actions": [],
                "push_execution_log": [],
            }
            repair_attempts.append(attempt)
            if attempt_index < max_repair_attempts:
                repair_feedback = _repair_feedback_from_attempt(attempt)
                continue
            final_status = "invalid_plan"
            final_metadata = dict(attempt)
            final_metadata.pop("attempt_index", None)
            break

        env.reset()
        execution = execute_push_plan(
            env=env,
            plan=plan,
            max_steps=max_steps,
            raw_response=raw_plan,
            call_metadata=call_metadata,
        )
        attempt = {
            "attempt_index": attempt_index,
            "raw_plan_response": raw_plan,
            "status": execution.status,
            "planned_pushes": planned_pushes,
            "expanded_actions": execution.expanded_actions,
            "push_execution_log": execution.push_execution_log,
            "executed_push_count": execution.executed_push_count,
            "planned_push_count": len(planned_pushes),
            "expanded_primitive_step_count": len(execution.expanded_actions),
        }
        final_board = _final_board_from_trajectory(execution.trajectory)
        if final_board:
            attempt["final_board"] = final_board
        deadlock_reason = _deadlock_reason_from_trajectory(execution.trajectory)
        if deadlock_reason:
            attempt["deadlock_reason"] = deadlock_reason
        if execution.failure_reason:
            attempt["failure_reason"] = execution.failure_reason
        if execution.failure_push_index is not None:
            attempt["failure_push_index"] = execution.failure_push_index
        repair_attempts.append(attempt)

        final_status = execution.status
        final_step_count = len(execution.trajectory)
        final_total_reward = execution.total_reward
        final_trajectory = execution.trajectory
        final_metadata = dict(attempt)
        final_metadata.pop("attempt_index", None)
        if (
            execution.status == "success"
            or not _should_repair_full_path_status(execution.status)
            or attempt_index >= max_repair_attempts
        ):
            break
        repair_feedback = _repair_feedback_from_attempt(attempt)

    final_metadata.update(_repair_metadata(repair_attempts, final_status))

    return EpisodeResult(
        level_id=env.level.level_id,
        agent_type=agent.agent_type,
        seed=seed,
        status=final_status,  # type: ignore[arg-type]
        step_count=final_step_count,
        invalid_move_count=0,
        total_reward=final_total_reward,
        llm_call_count=getattr(agent, "llm_call_count", 0) - start_llm_calls,
        token_cost=getattr(agent, "token_cost", 0.0) - start_token_cost,
        trajectory=final_trajectory,
        policy_mode=getattr(agent, "policy_mode", "unknown"),
        model=getattr(agent, "model", None),
        prompt_version=getattr(agent, "prompt_version", None),
        memory_path=getattr(agent, "memory_path", None),
        memory_hash=getattr(agent, "memory_hash", None),
        memory_caps=agent.memory_caps(),
        temperature=getattr(agent, "temperature", None),
        max_output_tokens=getattr(agent, "max_output_tokens", None),
        cache_namespace=getattr(agent, "cache_namespace", None),
        level_split=getattr(env.level, "split", "unspecified"),
        level_tags=list(getattr(env.level, "tags", [])),
        optimal_steps=getattr(env.level, "optimal_steps", None),
        cache_hits=getattr(agent, "cache_hits", 0) - start_cache_hits,
        cache_misses=getattr(agent, "cache_misses", 0) - start_cache_misses,
        budget_exhausted=budget_exhausted,
        metadata=final_metadata,
    )


def run_experiment(
    levels: list[Level],
    agent: BaseAgent,
    episodes: int,
    max_steps: int,
    seed: int,
    results_dir: Path,
    max_repair_attempts: int = 0,
) -> dict[str, Any]:
    memory_store = getattr(agent, "memory_store", None)
    if memory_store is not None and getattr(agent, "agent_type", "") != "no_memory":
        validate_no_eval_memory_leak(levels, memory_store)

    results: list[EpisodeResult] = []
    for episode_idx in range(episodes):
        level = levels[episode_idx % len(levels)]
        episode_seed = seed + episode_idx
        env = SokobanEnv(level, seed=episode_seed)
        result = run_episode(
            env,
            agent,
            max_steps=max_steps,
            seed=episode_seed,
            max_repair_attempts=max_repair_attempts,
        )
        save_episode(result, results_dir)
        results.append(result)
        if result.status in {"budget_exhausted", "api_error", "invalid_failure"}:
            break

    summary = summarize_results(results)
    level_meta = level_metadata(levels)
    source_level_ids = (
        get_memory_source_level_ids(memory_store)
        if memory_store is not None
        else []
    )
    summary.update(
        {
            "requested_episodes": episodes,
            **level_meta,
            "agent_type": agent.agent_type,
            "policy_mode": getattr(agent, "policy_mode", "unknown"),
            "model": getattr(agent, "model", None),
            "temperature": getattr(agent, "temperature", None),
            "max_output_tokens": getattr(agent, "max_output_tokens", None),
            "prompt_version": getattr(agent, "prompt_version", None),
            "seed": seed,
            "max_steps": max_steps,
            "max_repair_attempts": max_repair_attempts,
            "results_dir": str(results_dir),
            "memory_path": getattr(agent, "memory_path", None),
            "memory_hash": getattr(agent, "memory_hash", None),
            "memory_source_level_ids": source_level_ids,
            "memory_caps": agent.memory_caps(),
            "max_llm_calls": getattr(agent, "max_llm_calls", None),
            "cache_namespace": getattr(agent, "cache_namespace", None),
        }
    )
    save_summary(summary, results_dir)
    return summary


def _push_actions(env: SokobanEnv, legal_actions: list[str]) -> list[str]:
    pushes = []
    for action in legal_actions:
        dr, dc = DIRECTIONS[action]  # type: ignore[index]
        if env.player.moved(dr, dc) in env.boxes:
            pushes.append(action)
    return pushes


def _agent_memory(agent: BaseAgent) -> Any:
    if hasattr(agent, "memory_store"):
        return getattr(agent, "memory_store")
    if hasattr(agent, "heuristic_memory"):
        return getattr(agent, "heuristic_memory")
    return None


def _state_summary(env: SokobanEnv) -> dict[str, Any]:
    boxes = [_position_list(pos) for pos in sorted(env.boxes)]
    return {
        "player": _position_list(env.player),
        "boxes": boxes,
        "indexed_boxes": {
            f"B{box_id}": box
            for box_id, box in enumerate(boxes)
        },
        "targets": [_position_list(pos) for pos in sorted(env.level.targets)],
    }


def _position_list(position: Position) -> list[int]:
    return [position.row, position.col]


def _should_repair_full_path_status(status: str) -> bool:
    return status in {"invalid_plan", "plan_exhausted", "deadlock"}


def _repair_metadata(repair_attempts: list[dict[str, Any]], final_status: str) -> dict[str, Any]:
    repair_attempt_count = max(0, len(repair_attempts) - 1)
    metadata: dict[str, Any] = {
        "repair_attempts": repair_attempts,
        "repair_attempt_count": repair_attempt_count,
        "success_after_repair": final_status == "success" and repair_attempt_count > 0,
        "success_without_repair": final_status == "success" and repair_attempt_count == 0,
    }
    if repair_attempts:
        first_attempt = repair_attempts[0]
        metadata["first_attempt_status"] = first_attempt.get("status")
        if first_attempt.get("failure_reason"):
            metadata["first_attempt_failure_reason"] = first_attempt.get("failure_reason")
    return metadata


def _repair_feedback_from_attempt(attempt: dict[str, Any]) -> str:
    status = str(attempt.get("status", "unknown"))
    failure_reason = attempt.get("failure_reason") or "none"
    lines = [
        "Previous attempt failed.",
        f"Failure status: {status}",
        f"Failure reason: {failure_reason}",
    ]
    if attempt.get("failure_push_index") is not None:
        lines.append(f"Failed push index: {attempt['failure_push_index']}")

    failed_log = _failed_push_log(attempt)
    if failed_log:
        if failed_log.get("intent") is not None:
            lines.append(f"Failed intent: {json.dumps(failed_log['intent'], sort_keys=True)}")
        if failed_log.get("resolved_box") is not None:
            lines.append(f"Resolved box: {_format_log_position(failed_log['resolved_box'])}")
        if failed_log.get("box_destination") is not None:
            lines.append(f"Destination: {_format_log_position(failed_log['box_destination'])}")
        if failed_log.get("required_player_position") is not None:
            lines.append(f"Required player standing cell: {_format_log_position(failed_log['required_player_position'])}")
        if failed_log.get("error"):
            lines.append(f"Why invalid: {failed_log['error']}.")

    if status == "plan_exhausted":
        lines.append("The plan executed legally but did not solve the puzzle.")
        lines.append("Include enough remaining pushes to place all boxes on targets.")
    elif status == "deadlock":
        deadlock_reason = _deadlock_reason(attempt)
        if deadlock_reason:
            lines.append(f"Deadlock reason: {deadlock_reason}")
        board = _final_board_from_attempt(attempt)
        if board:
            lines.append("Final failed board:")
            lines.append(board)
    elif failure_reason == "invalid_push_plan_parse":
        if attempt.get("error_message"):
            lines.append(f"Parser error: {attempt['error_message']}")
        lines.append("Return valid JSON only.")

    lines.append("Regenerate a complete plan from the original board. Do not repeat the failed push.")
    return "\n".join(lines)


def _failed_push_log(attempt: dict[str, Any]) -> dict[str, Any] | None:
    logs = attempt.get("push_execution_log")
    if not isinstance(logs, list) or not logs:
        return None
    failure_push_index = attempt.get("failure_push_index")
    if isinstance(failure_push_index, int):
        for log in logs:
            if isinstance(log, dict) and log.get("push_index") == failure_push_index:
                return log
    last_log = logs[-1]
    return last_log if isinstance(last_log, dict) else None


def _format_log_position(value: Any) -> str:
    if isinstance(value, dict) and isinstance(value.get("row"), int) and isinstance(value.get("col"), int):
        return f"[{value['row']}, {value['col']}]"
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return f"[{value[0]}, {value[1]}]"
    return json.dumps(value)


def _deadlock_reason(attempt: dict[str, Any]) -> str | None:
    if attempt.get("deadlock_reason"):
        return str(attempt["deadlock_reason"])
    for step in reversed(attempt.get("trajectory", []) if isinstance(attempt.get("trajectory"), list) else []):
        if not isinstance(step, dict):
            continue
        info = step.get("info")
        if isinstance(info, dict) and info.get("deadlock_reason"):
            return str(info["deadlock_reason"])
    return str(attempt.get("failure_reason")) if attempt.get("failure_reason") else None


def _final_board_from_attempt(attempt: dict[str, Any]) -> str | None:
    if isinstance(attempt.get("final_board"), str):
        return attempt["final_board"]
    trajectory = attempt.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return None
    last_step = trajectory[-1]
    if isinstance(last_step, dict) and isinstance(last_step.get("next_state"), str):
        return last_step["next_state"]
    return None


def _final_board_from_trajectory(trajectory: list[dict[str, Any]]) -> str | None:
    if not trajectory:
        return None
    last_step = trajectory[-1]
    if isinstance(last_step.get("next_state"), str):
        return last_step["next_state"]
    return None


def _deadlock_reason_from_trajectory(trajectory: list[dict[str, Any]]) -> str | None:
    for step in reversed(trajectory):
        info = step.get("info")
        if isinstance(info, dict) and info.get("deadlock_reason"):
            return str(info["deadlock_reason"])
    return None


def _full_path_error_result(
    *,
    env: SokobanEnv,
    agent: BaseAgent,
    seed: int,
    status: str,
    start_llm_calls: int,
    start_token_cost: float,
    start_cache_hits: int,
    start_cache_misses: int,
    metadata: dict[str, Any],
) -> EpisodeResult:
    return EpisodeResult(
        level_id=env.level.level_id,
        agent_type=agent.agent_type,
        seed=seed,
        status=status,  # type: ignore[arg-type]
        step_count=0,
        invalid_move_count=0,
        total_reward=0.0,
        llm_call_count=getattr(agent, "llm_call_count", 0) - start_llm_calls,
        token_cost=getattr(agent, "token_cost", 0.0) - start_token_cost,
        trajectory=[],
        policy_mode=getattr(agent, "policy_mode", "unknown"),
        model=getattr(agent, "model", None),
        prompt_version=getattr(agent, "prompt_version", None),
        memory_path=getattr(agent, "memory_path", None),
        memory_hash=getattr(agent, "memory_hash", None),
        memory_caps=agent.memory_caps(),
        temperature=getattr(agent, "temperature", None),
        max_output_tokens=getattr(agent, "max_output_tokens", None),
        cache_namespace=getattr(agent, "cache_namespace", None),
        level_split=getattr(env.level, "split", "unspecified"),
        level_tags=list(getattr(env.level, "tags", [])),
        optimal_steps=getattr(env.level, "optimal_steps", None),
        cache_hits=getattr(agent, "cache_hits", 0) - start_cache_hits,
        cache_misses=getattr(agent, "cache_misses", 0) - start_cache_misses,
        budget_exhausted=False,
        metadata=metadata,
    )
