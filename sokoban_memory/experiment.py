from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from sokoban_memory.action_parser import choose_fallback_action, parse_action
from sokoban_memory.agents import BaseAgent
from sokoban_memory.env import DIRECTIONS, SokobanEnv
from sokoban_memory.logging_utils import save_episode, save_summary
from sokoban_memory.metrics import summarize_results
from sokoban_memory.types import EpisodeResult, Level

RULES_TEXT = "Move Up/Down/Left/Right. Push boxes only. No pulling. No pushing two boxes."


def run_episode(env: SokobanEnv, agent: BaseAgent, max_steps: int, seed: int) -> EpisodeResult:
    rng = random.Random(seed)
    state_text = env.reset()
    trajectory: list[dict[str, Any]] = []
    invalid_move_count = 0
    total_reward = 0.0
    status = "timeout"

    for step_idx in range(max_steps):
        legal_actions = env.legal_actions()
        context = {
            "step": step_idx,
            "legal_actions": legal_actions,
            "push_actions": _push_actions(env, legal_actions),
            "rules": RULES_TEXT,
            "memory": _agent_memory(agent),
        }
        raw_action = agent.select_action(state_text, context)
        parsed_action = parse_action(raw_action)

        if parsed_action not in legal_actions:
            invalid_move_count += 1
            action = choose_fallback_action(legal_actions, rng)
            invalid_reason = "invalid_or_illegal_action"
        else:
            action = parsed_action
            invalid_reason = None

        result = env.step(action)
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
        llm_call_count=getattr(agent, "llm_call_count", 0),
        token_cost=getattr(agent, "token_cost", 0.0),
        trajectory=trajectory,
    )


def run_experiment(
    levels: list[Level],
    agent: BaseAgent,
    episodes: int,
    max_steps: int,
    seed: int,
    results_dir: Path,
) -> dict[str, Any]:
    results: list[EpisodeResult] = []
    for episode_idx in range(episodes):
        level = levels[episode_idx % len(levels)]
        episode_seed = seed + episode_idx
        env = SokobanEnv(level, seed=episode_seed)
        result = run_episode(env, agent, max_steps=max_steps, seed=episode_seed)
        save_episode(result, results_dir)
        results.append(result)

    summary = summarize_results(results)
    summary.update(
        {
            "agent_type": agent.agent_type,
            "seed": seed,
            "max_steps": max_steps,
            "results_dir": str(results_dir),
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

