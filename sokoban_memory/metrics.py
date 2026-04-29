from __future__ import annotations

from statistics import median
from typing import Any

from sokoban_memory.types import EpisodeResult

STATUSES = (
    "success",
    "deadlock",
    "timeout",
    "budget_exhausted",
    "api_error",
    "invalid_failure",
    "failure",
)


def summarize_results(results: list[EpisodeResult]) -> dict[str, Any]:
    episodes = len(results)
    total_steps = sum(r.step_count for r in results)
    invalid_moves = sum(r.invalid_move_count for r in results)
    status_counts = _status_counts(results)
    success_results = [r for r in results if r.status == "success"]
    efficiency_values = [_solution_efficiency(r) for r in success_results]
    efficiency_values = [value for value in efficiency_values if value is not None]
    steps_over_optimal = [_steps_over_optimal(r) for r in success_results]
    steps_over_optimal = [value for value in steps_over_optimal if value is not None]

    if episodes == 0:
        return {
            "episodes": 0,
            **_zero_counts_and_rates(),
            "average_steps": 0.0,
            "average_success_steps": 0.0,
            "average_reward": 0.0,
            "average_solution_efficiency": 0.0,
            "median_solution_efficiency": 0.0,
            "steps_over_optimal_average": 0.0,
            "solution_efficiency_count": 0,
            "solution_efficiency_skipped_count": 0,
            "invalid_move_count": 0,
            "invalid_move_rate": 0.0,
            "average_invalid_moves_per_episode": 0.0,
            "llm_call_count": 0,
            "token_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "budget_exhaustion_count": 0,
            "per_level": {},
        }

    return {
        "episodes": episodes,
        **status_counts,
        **{f"{status}_rate": status_counts[f"{status}_count"] / episodes for status in STATUSES},
        "solve_rate": status_counts["success_count"] / episodes,
        "average_steps": total_steps / episodes,
        "average_success_steps": _average([r.step_count for r in success_results]),
        "average_reward": sum(r.total_reward for r in results) / episodes,
        "average_solution_efficiency": _average(efficiency_values),
        "median_solution_efficiency": median(efficiency_values) if efficiency_values else 0.0,
        "steps_over_optimal_average": _average(steps_over_optimal),
        "solution_efficiency_count": len(efficiency_values),
        "solution_efficiency_skipped_count": len(success_results) - len(efficiency_values),
        "invalid_move_count": invalid_moves,
        "invalid_move_rate": invalid_moves / total_steps if total_steps else 0.0,
        "average_invalid_moves_per_episode": invalid_moves / episodes,
        "llm_call_count": sum(r.llm_call_count for r in results),
        "token_cost": sum(r.token_cost for r in results),
        "cache_hits": sum(r.cache_hits for r in results),
        "cache_misses": sum(r.cache_misses for r in results),
        "budget_exhaustion_count": status_counts["budget_exhausted_count"],
        "per_level": summarize_by_level(results),
    }


def summarize_by_level(results: list[EpisodeResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.level_id, []).append(result)

    per_level = {}
    for level_id, level_results in sorted(grouped.items()):
        attempts = len(level_results)
        status_counts = _status_counts(level_results)
        success_results = [r for r in level_results if r.status == "success"]
        efficiency_values = [_solution_efficiency(r) for r in success_results]
        efficiency_values = [value for value in efficiency_values if value is not None]
        per_level[level_id] = {
            "attempts": attempts,
            "successes": status_counts["success_count"],
            "deadlocks": status_counts["deadlock_count"],
            "timeouts": status_counts["timeout_count"],
            **status_counts,
            "solve_rate": status_counts["success_count"] / attempts if attempts else 0.0,
            "average_steps": _average([r.step_count for r in level_results]),
            "average_success_steps": _average([r.step_count for r in success_results]),
            "average_efficiency": _average(efficiency_values),
            "solution_efficiency_count": len(efficiency_values),
            "solution_efficiency_skipped_count": len(success_results) - len(efficiency_values),
        }
    return per_level


def _status_counts(results: list[EpisodeResult]) -> dict[str, int]:
    return {f"{status}_count": sum(1 for result in results if result.status == status) for status in STATUSES}


def _zero_counts_and_rates() -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for status in STATUSES:
        values[f"{status}_count"] = 0
        values[f"{status}_rate"] = 0.0
    values["solve_rate"] = 0.0
    return values


def _solution_efficiency(result: EpisodeResult) -> float | None:
    if result.status != "success" or not result.optimal_steps or result.step_count <= 0:
        return None
    return result.optimal_steps / result.step_count


def _steps_over_optimal(result: EpisodeResult) -> int | None:
    if result.status != "success" or not result.optimal_steps:
        return None
    return result.step_count - result.optimal_steps


def _average(values: list[float | int]) -> float:
    return sum(values) / len(values) if values else 0.0

