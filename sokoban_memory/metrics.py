from __future__ import annotations

from sokoban_memory.types import EpisodeResult


def summarize_results(results: list[EpisodeResult]) -> dict[str, float | int]:
    episodes = len(results)
    total_steps = sum(r.step_count for r in results)
    invalid_moves = sum(r.invalid_move_count for r in results)
    if episodes == 0:
        return {
            "episodes": 0,
            "solve_rate": 0.0,
            "deadlock_rate": 0.0,
            "timeout_rate": 0.0,
            "failure_rate": 0.0,
            "average_steps": 0.0,
            "average_reward": 0.0,
            "invalid_move_count": 0,
            "invalid_move_rate": 0.0,
            "average_invalid_moves_per_episode": 0.0,
            "llm_call_count": 0,
            "token_cost": 0.0,
        }
    return {
        "episodes": episodes,
        "solve_rate": _rate(results, "success"),
        "deadlock_rate": _rate(results, "deadlock"),
        "timeout_rate": _rate(results, "timeout"),
        "failure_rate": _rate(results, "failure"),
        "average_steps": total_steps / episodes,
        "average_reward": sum(r.total_reward for r in results) / episodes,
        "invalid_move_count": invalid_moves,
        "invalid_move_rate": invalid_moves / total_steps if total_steps else 0.0,
        "average_invalid_moves_per_episode": invalid_moves / episodes,
        "llm_call_count": sum(r.llm_call_count for r in results),
        "token_cost": sum(r.token_cost for r in results),
    }


def _rate(results: list[EpisodeResult], status: str) -> float:
    return sum(1 for r in results if r.status == status) / len(results)

