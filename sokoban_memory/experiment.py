from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from sokoban_memory.action_parser import choose_fallback_action, parse_action
from sokoban_memory.agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_CACHE_NAMESPACE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    BaseAgent,
    LLMBudgetExceeded,
    make_agent,
)
from sokoban_memory.env import DIRECTIONS, SokobanEnv
from sokoban_memory.full_path import execute_push_plan, parse_push_plan, PushPlanParseError
from sokoban_memory.logging_utils import save_episode, save_summary
from sokoban_memory.memory import (
    HeuristicMemory,
    MemoryRenderConfig,
    RawTrajectoryMemory,
    get_memory_source_level_ids,
    validate_no_eval_memory_leak,
)
from sokoban_memory.metrics import summarize_results
from sokoban_memory.prompts import level_metadata
from sokoban_memory.reflection import (
    DEFAULT_SAME_LEVEL_REFLECTION_VERSION,
    SAME_LEVEL_REFLECTION_VERSIONS,
    generate_reflection_memory,
    generate_same_level_reflection_memory,
)
from sokoban_memory.types import EpisodeResult, Level, Position
from sokoban_memory.v3_trajectory import (
    RunIdentity,
    board_after_last_successful_push,
    board_before_failed_push,
    build_v3_attempt_trace,
    failure_subtype,
    make_run_identity,
    progress_metrics,
)

RULES_TEXT = (
    "Move Up/Down/Left/Right only; no diagonal moves. "
    "The player may walk into empty floor cells or target cells, but not into walls, boundaries, or boxes. "
    "To push a box, the player must stand in the cell opposite the push direction. "
    "The box moves one cell in the push direction only if that destination cell is empty floor or a target. "
    "No pulling, no pushing two boxes, and no pushing a box into a wall, boundary, or another box. "
    "Goal: all boxes must be on target cells."
)


def run_episode(
    env: SokobanEnv,
    agent: BaseAgent,
    max_steps: int,
    seed: int,
    max_repair_attempts: int = 0,
    run_identity: RunIdentity | None = None,
    condition: str | None = None,
) -> EpisodeResult:
    if getattr(agent, "policy_mode", "") == "full_path":
        return run_full_path_episode(
            env,
            agent,
            max_steps=max_steps,
            seed=seed,
            max_repair_attempts=max_repair_attempts,
            run_identity=run_identity,
            condition=condition,
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
    run_identity: RunIdentity | None = None,
    condition: str | None = None,
) -> EpisodeResult:
    state_text = env.reset()
    initial_board = state_text
    run_identity = run_identity or make_run_identity()
    condition = condition or getattr(agent, "agent_type", "unknown")
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
            "level_id": env.level.level_id,
            "state_summary": _state_summary(env),
            "max_steps": max_steps,
            "memory": _agent_memory(agent),
            "repair_feedback": repair_feedback,
        }
        iteration_context = getattr(agent, "iteration_context", None)
        if isinstance(iteration_context, dict):
            context.update(iteration_context)
        try:
            if hasattr(agent, "select_plan"):
                raw_plan = agent.select_plan(state_text, context)  # type: ignore[attr-defined]
            else:
                raw_plan = agent.select_action(state_text, context)
        except LLMBudgetExceeded as exc:
            final_status = "budget_exhausted"
            budget_exhausted = True
            final_metadata = {
                "schema_version": "v3_trajectory_v1",
                "failure_reason": "llm_call_budget_exhausted",
                "failure_subtype": "budget_exhausted",
                "budget_exhausted_at_attempt": attempt_index,
                "budget_error": str(exc),
                "planned_pushes": [],
                "expanded_actions": [],
                "push_execution_log": [],
                "initial_board": initial_board,
            }
            break
        except Exception as exc:
            final_status = "api_error"
            final_metadata = {
                "schema_version": "v3_trajectory_v1",
                "failure_reason": "api_error",
                "failure_subtype": "api_error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_at_attempt": attempt_index,
                "planned_pushes": [],
                "expanded_actions": [],
                "push_execution_log": [],
                "initial_board": initial_board,
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
                "failure_subtype": failure_subtype(
                    status="invalid_plan",
                    failure_reason="invalid_push_plan_parse",
                    raw_output=raw_plan,
                    error_message=str(exc),
                ),
                "call_metadata": call_metadata,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "planned_pushes": [],
                "executed_push_count": 0,
                "planned_push_count": 0,
                "expanded_primitive_step_count": 0,
                "expanded_actions": [],
                "push_execution_log": [],
                "initial_board": initial_board,
                "final_board": initial_board,
                "board_before_failed_push": initial_board,
                "board_after_last_successful_push": initial_board,
            }
            attempt["v3_attempt_trace"] = build_v3_attempt_trace(
                level=env.level,
                run_identity=run_identity,
                condition=condition,
                attempt=attempt,
                attempt_index=attempt_index,
                seed=seed,
                model=getattr(agent, "model", None),
                prompt_version=getattr(agent, "prompt_version", None),
                max_output_tokens=getattr(agent, "max_output_tokens", None),
                initial_board=initial_board,
                trajectory=[],
            )
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
            "call_metadata": call_metadata,
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
        else:
            attempt["final_board"] = initial_board
        deadlock_reason = _deadlock_reason_from_trajectory(execution.trajectory)
        if deadlock_reason:
            attempt["deadlock_reason"] = deadlock_reason
        if execution.failure_reason:
            attempt["failure_reason"] = execution.failure_reason
        attempt["failure_subtype"] = failure_subtype(
            status=execution.status,
            failure_reason=attempt.get("failure_reason"),
            raw_output=raw_plan,
        )
        if execution.failure_push_index is not None:
            attempt["failure_push_index"] = execution.failure_push_index
        attempt["initial_board"] = initial_board
        attempt["board_before_failed_push"] = board_before_failed_push(attempt)
        attempt["board_after_last_successful_push"] = board_after_last_successful_push(attempt, initial_board)
        attempt.update(progress_metrics(env.level, initial_board, execution.trajectory))
        attempt["v3_attempt_trace"] = build_v3_attempt_trace(
            level=env.level,
            run_identity=run_identity,
            condition=condition,
            attempt=attempt,
            attempt_index=attempt_index,
            seed=seed,
            model=getattr(agent, "model", None),
            prompt_version=getattr(agent, "prompt_version", None),
            max_output_tokens=getattr(agent, "max_output_tokens", None),
            initial_board=initial_board,
            trajectory=execution.trajectory,
        )
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
    if "v3_attempt_trace" not in final_metadata and repair_attempts:
        final_metadata["v3_attempt_trace"] = repair_attempts[-1].get("v3_attempt_trace")
    if final_metadata.get("v3_attempt_trace"):
        trace = final_metadata["v3_attempt_trace"]
        for key in (
            "schema_version",
            "run_id",
            "code_commit",
            "level_suite_path",
            "level_suite_hash",
            "executor_version",
            "prompt_renderer_version",
            "memory_renderer_version",
            "cache_key",
            "failure_subtype",
            "initial_board",
            "board_before_failed_push",
            "board_after_last_successful_push",
            "best_boxes_on_targets",
            "final_boxes_on_targets",
            "target_placement_events_before_first_deadlock",
            "best_goal_completion_rate",
            "final_goal_completion_rate",
            "normalized_target_placement_before_first_deadlock",
        ):
            final_metadata.setdefault(key, trace.get(key))

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
    level_suite_path: str | Path | None = None,
) -> dict[str, Any]:
    memory_store = getattr(agent, "memory_store", None)
    if memory_store is not None and getattr(agent, "agent_type", "") != "no_memory":
        validate_no_eval_memory_leak(levels, memory_store)

    results: list[EpisodeResult] = []
    run_identity = make_run_identity(level_suite_path)
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
            run_identity=run_identity,
            condition=agent.agent_type,
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
            "run_id": run_identity.run_id,
            "code_commit": run_identity.code_commit,
            "level_suite_path": run_identity.level_suite_path,
            "level_suite_hash": run_identity.level_suite_hash,
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


SAME_LEVEL_ITERATIVE_CONDITIONS = {
    "single_shot_no_memory",
    "generic_retry_feedback",
    "verifier_summary_retry",
    "raw_same_level_iterative",
    "heuristic_same_level_iterative",
}


def run_same_level_iterative_experiment(
    *,
    levels: list[Level],
    condition: str,
    attempts_per_level: int,
    max_steps: int,
    seed: int,
    results_dir: Path,
    model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client: Any | None = None,
    reflection_client: Any | None = None,
    max_llm_calls: int | None = 50,
    memory_config: MemoryRenderConfig | None = None,
    llm_cache_path: str | Path | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
    level_suite_path: str | Path | None = None,
    same_level_reflection_version: str = DEFAULT_SAME_LEVEL_REFLECTION_VERSION,
) -> dict[str, Any]:
    if condition not in SAME_LEVEL_ITERATIVE_CONDITIONS:
        raise ValueError(f"Unknown same-level iterative condition: {condition}")
    if attempts_per_level <= 0:
        raise ValueError("attempts_per_level must be positive.")
    if same_level_reflection_version not in SAME_LEVEL_REFLECTION_VERSIONS:
        raise ValueError(
            f"Unknown same_level_reflection_version: {same_level_reflection_version}"
        )
    same_level_render_mode = (
        condition == "heuristic_same_level_iterative"
        and same_level_reflection_version != "baseline"
    )

    config = memory_config or MemoryRenderConfig()
    run_identity = make_run_identity(level_suite_path)
    results: list[EpisodeResult] = []
    effective_attempts = 1 if condition == "single_shot_no_memory" else attempts_per_level

    for level_index, level in enumerate(levels):
        raw_memory = _same_level_raw_memory(level, condition)
        heuristic_memory = _same_level_heuristic_memory(level, condition)
        agent = _make_same_level_agent(
            condition=condition,
            raw_memory=raw_memory,
            heuristic_memory=heuristic_memory,
            model=model,
            api_key_env=api_key_env,
            client=client,
            max_llm_calls=max_llm_calls,
            memory_config=config,
            llm_cache_path=llm_cache_path,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_namespace=cache_namespace,
        )

        for attempt_index in range(effective_attempts):
            agent.iteration_context = {
                "iteration_attempt_index": attempt_index,
                "attempt_budget": effective_attempts,
                "same_level_iterative_condition": condition,
                "same_level_render_mode": same_level_render_mode,
            }
            if hasattr(agent, "memory_hash"):
                agent.memory_hash = getattr(getattr(agent, "memory_store", None), "memory_hash", None)
            attempt_seed = seed + (level_index * attempts_per_level) + attempt_index
            result = run_episode(
                SokobanEnv(level, seed=attempt_seed),
                agent,
                max_steps=max_steps,
                seed=attempt_seed,
                run_identity=run_identity,
                condition=condition,
            )
            result.metadata.update(
                {
                    "condition": condition,
                    "same_level_iterative": True,
                    "iteration_attempt_index": attempt_index,
                    "attempt_budget": effective_attempts,
                }
            )
            save_episode(result, results_dir)
            results.append(result)

            if result.status == "success":
                break
            if result.status in {"budget_exhausted", "api_error", "invalid_failure"}:
                break
            _update_same_level_memory_after_failure(
                condition=condition,
                level=level,
                raw_memory=raw_memory,
                result=result,
                agent=agent,
                model=model,
                api_key_env=api_key_env,
                reflection_client=reflection_client,
                llm_cache_path=llm_cache_path,
                max_llm_calls=max_llm_calls,
                memory_config=config,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                cache_namespace=cache_namespace,
                same_level_reflection_version=same_level_reflection_version,
            )

    summary = summarize_results(results)
    summary.update(
        {
            "experiment_mode": "same_level_iterative",
            "condition": condition,
            "same_level_reflection_version": same_level_reflection_version,
            "attempts_per_level": effective_attempts,
            "requested_levels": len(levels),
            **level_metadata(levels),
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "seed": seed,
            "max_steps": max_steps,
            "results_dir": str(results_dir),
            "run_id": run_identity.run_id,
            "code_commit": run_identity.code_commit,
            "level_suite_path": run_identity.level_suite_path,
            "level_suite_hash": run_identity.level_suite_hash,
            "memory_caps": config.to_dict(),
            "max_llm_calls": max_llm_calls,
            "cache_namespace": cache_namespace,
        }
    )
    save_summary(summary, results_dir)
    return summary


def _same_level_raw_memory(level: Level, condition: str) -> RawTrajectoryMemory:
    return RawTrajectoryMemory(
        source_metadata={
            "memory_scope": "same_level",
            "condition": condition,
            "source_level_ids": [level.level_id],
        }
    )


def _same_level_heuristic_memory(level: Level, condition: str) -> HeuristicMemory:
    return HeuristicMemory(
        heuristics=[],
        source_metadata={
            "memory_scope": "same_level",
            "condition": condition,
            "source_level_ids": [level.level_id],
        },
    )


def _make_same_level_agent(
    *,
    condition: str,
    raw_memory: RawTrajectoryMemory,
    heuristic_memory: HeuristicMemory,
    model: str,
    api_key_env: str,
    client: Any | None,
    max_llm_calls: int | None,
    memory_config: MemoryRenderConfig,
    llm_cache_path: str | Path | None,
    temperature: float,
    max_output_tokens: int,
    cache_namespace: str,
) -> BaseAgent:
    agent_name = {
        "single_shot_no_memory": "no_memory",
        "generic_retry_feedback": "generic_retry_feedback",
        "verifier_summary_retry": "verifier_summary_retry",
        "raw_same_level_iterative": "raw_trajectory_memory",
        "heuristic_same_level_iterative": "heuristic_same_level_iterative",
    }[condition]
    memory = heuristic_memory if condition == "heuristic_same_level_iterative" else raw_memory
    return make_agent(
        agent_name,
        memory=memory,
        model=model,
        api_key_env=api_key_env,
        client=client,
        max_llm_calls=max_llm_calls,
        memory_config=memory_config,
        llm_cache_path=llm_cache_path,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        cache_namespace=cache_namespace,
    )


def _update_same_level_memory_after_failure(
    *,
    condition: str,
    level: Level,
    raw_memory: RawTrajectoryMemory,
    result: EpisodeResult,
    agent: BaseAgent,
    model: str,
    api_key_env: str,
    reflection_client: Any | None,
    llm_cache_path: str | Path | None,
    max_llm_calls: int | None,
    memory_config: MemoryRenderConfig,
    temperature: float,
    max_output_tokens: int,
    cache_namespace: str,
    same_level_reflection_version: str = DEFAULT_SAME_LEVEL_REFLECTION_VERSION,
) -> None:
    if condition not in {"verifier_summary_retry", "raw_same_level_iterative", "heuristic_same_level_iterative"}:
        return
    raw_memory.add_episode(result)
    raw_memory.source_metadata.update(
        {
            "memory_scope": "same_level",
            "source_level_ids": [level.level_id],
        }
    )
    raw_memory.memory_hash = raw_memory.compute_hash()
    if condition in {"verifier_summary_retry", "raw_same_level_iterative"}:
        agent.memory_store = raw_memory  # type: ignore[attr-defined]
        agent.memory_hash = raw_memory.memory_hash  # type: ignore[attr-defined]
        return

    heuristic_memory = generate_same_level_reflection_memory(
        raw_memory,
        level_id=level.level_id,
        version=same_level_reflection_version,
        model=model,
        api_key_env=api_key_env,
        client=reflection_client,
        llm_cache_path=str(llm_cache_path) if llm_cache_path is not None else None,
        max_llm_calls=max_llm_calls,
        memory_config=memory_config,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        cache_namespace=cache_namespace,
    )
    heuristic_memory.source_metadata.update(
        {
            "memory_scope": "same_level",
            "condition": condition,
            "source_level_ids": [level.level_id],
        }
    )
    heuristic_memory.memory_hash = heuristic_memory.compute_hash()
    agent.memory_store = heuristic_memory  # type: ignore[attr-defined]
    agent.memory_hash = heuristic_memory.memory_hash  # type: ignore[attr-defined]


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
        intent = failed_log.get("model_intent") or failed_log.get("intent")
        if intent is not None:
            lines.append(f"Failed intent: {json.dumps(intent, sort_keys=True)}")
        resolved_box = failed_log.get("resolved_box_before_push") or failed_log.get("resolved_box")
        if resolved_box is not None:
            lines.append(f"Resolved box: {_format_log_position(resolved_box)}")
        destination = failed_log.get("destination_cell") or failed_log.get("box_destination")
        if destination is not None:
            lines.append(f"Destination: {_format_log_position(destination)}")
        standing_cell = failed_log.get("standing_cell_required") or failed_log.get("required_player_position")
        if standing_cell is not None:
            lines.append(f"Required player standing cell: {_format_log_position(standing_cell)}")
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
