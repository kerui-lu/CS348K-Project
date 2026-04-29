from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from sokoban_memory.action_parser import choose_fallback_action, parse_action
from sokoban_memory.agents import BaseAgent, LLMBudgetExceeded
from sokoban_memory.env import DIRECTIONS, SokobanEnv
from sokoban_memory.logging_utils import save_episode, save_summary
from sokoban_memory.memory import get_memory_source_level_ids, validate_no_eval_memory_leak
from sokoban_memory.metrics import summarize_results
from sokoban_memory.prompts import level_metadata
from sokoban_memory.types import EpisodeResult, Level

RULES_TEXT = "Move Up/Down/Left/Right. Push boxes only. No pulling. No pushing two boxes."


def run_episode(env: SokobanEnv, agent: BaseAgent, max_steps: int, seed: int) -> EpisodeResult:
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
            status = "api_error" if getattr(agent, "policy_mode", "") == "one_step" else "failure"
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


def run_experiment(
    levels: list[Level],
    agent: BaseAgent,
    episodes: int,
    max_steps: int,
    seed: int,
    results_dir: Path,
) -> dict[str, Any]:
    memory_store = getattr(agent, "memory_store", None)
    if memory_store is not None and getattr(agent, "agent_type", "") != "no_memory":
        validate_no_eval_memory_leak(levels, memory_store)

    results: list[EpisodeResult] = []
    for episode_idx in range(episodes):
        level = levels[episode_idx % len(levels)]
        episode_seed = seed + episode_idx
        env = SokobanEnv(level, seed=episode_seed)
        result = run_episode(env, agent, max_steps=max_steps, seed=episode_seed)
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
