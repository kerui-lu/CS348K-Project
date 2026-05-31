from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from sokoban_memory.types import EpisodeResult

STATUSES = (
    "success",
    "deadlock",
    "timeout",
    "invalid_plan",
    "plan_exhausted",
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
    planned_push_counts = [_metadata_int(r, "planned_push_count") for r in results]
    executed_push_counts = [_metadata_int(r, "executed_push_count") for r in results]
    expanded_step_counts = [_metadata_int(r, "expanded_primitive_step_count") for r in results]
    repair_attempt_counts = [_metadata_int(r, "repair_attempt_count") for r in results]
    success_after_repair_count = sum(1 for r in results if r.metadata.get("success_after_repair") is True)
    success_without_repair_count = sum(1 for r in results if r.metadata.get("success_without_repair") is True)
    first_attempt_invalid_plan_count = sum(
        1 for r in results if r.metadata.get("first_attempt_status") == "invalid_plan"
    )
    failure_subtype_counts = _failure_subtype_counts(results)
    progress_summary = _progress_summary(results)
    iterative_summary = _iterative_summary(results)
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
            "planned_push_count": 0,
            "executed_push_count": 0,
            "expanded_primitive_step_count": 0,
            "average_planned_pushes_per_episode": 0.0,
            "average_executed_pushes_per_episode": 0.0,
            "repair_attempt_count": 0,
            "average_repair_attempts_per_episode": 0.0,
            "success_after_repair_count": 0,
            "success_without_repair_count": 0,
            "first_attempt_invalid_plan_count": 0,
            "failure_subtype_counts": {},
            "average_best_goal_completion_rate": 0.0,
            "average_final_goal_completion_rate": 0.0,
            "average_target_placement_events_before_first_deadlock": 0.0,
            "solve_rate_at_1": 0.0,
            "solve_rate_at_k": 0.0,
            "cumulative_solved_by_attempt": {},
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
        "planned_push_count": sum(planned_push_counts),
        "executed_push_count": sum(executed_push_counts),
        "expanded_primitive_step_count": sum(expanded_step_counts),
        "average_planned_pushes_per_episode": sum(planned_push_counts) / episodes,
        "average_executed_pushes_per_episode": sum(executed_push_counts) / episodes,
        "repair_attempt_count": sum(repair_attempt_counts),
        "average_repair_attempts_per_episode": sum(repair_attempt_counts) / episodes,
        "success_after_repair_count": success_after_repair_count,
        "success_without_repair_count": success_without_repair_count,
        "first_attempt_invalid_plan_count": first_attempt_invalid_plan_count,
        "failure_subtype_counts": failure_subtype_counts,
        **progress_summary,
        **iterative_summary,
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
            "planned_push_count": sum(_metadata_int(r, "planned_push_count") for r in level_results),
            "executed_push_count": sum(_metadata_int(r, "executed_push_count") for r in level_results),
            "expanded_primitive_step_count": sum(
                _metadata_int(r, "expanded_primitive_step_count") for r in level_results
            ),
            "repair_attempt_count": sum(_metadata_int(r, "repair_attempt_count") for r in level_results),
            "success_after_repair_count": sum(
                1 for r in level_results if r.metadata.get("success_after_repair") is True
            ),
            "success_without_repair_count": sum(
                1 for r in level_results if r.metadata.get("success_without_repair") is True
            ),
            "first_attempt_invalid_plan_count": sum(
                1 for r in level_results if r.metadata.get("first_attempt_status") == "invalid_plan"
            ),
            "failure_subtype_counts": _failure_subtype_counts(level_results),
            **_progress_summary(level_results),
            **_iterative_summary(level_results),
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


def _metadata_int(result: EpisodeResult, key: str) -> int:
    value = result.metadata.get(key, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _metadata_float(result: EpisodeResult, key: str) -> float | None:
    value = result.metadata.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _failure_subtype_counts(results: list[EpisodeResult]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        subtype = result.metadata.get("failure_subtype")
        if isinstance(subtype, str) and subtype:
            counts[subtype] += 1
    return dict(sorted(counts.items()))


def _progress_summary(results: list[EpisodeResult]) -> dict[str, float]:
    best_rates = [
        value for result in results
        if (value := _metadata_float(result, "best_goal_completion_rate")) is not None
    ]
    final_rates = [
        value for result in results
        if (value := _metadata_float(result, "final_goal_completion_rate")) is not None
    ]
    placement_events = [
        value
        for result in results
        if (value := _metadata_float(result, "target_placement_events_before_first_deadlock")) is not None
    ]
    normalized_events = [
        value
        for result in results
        if (value := _metadata_float(result, "normalized_target_placement_before_first_deadlock")) is not None
    ]
    return {
        "average_best_goal_completion_rate": _average(best_rates),
        "average_final_goal_completion_rate": _average(final_rates),
        "average_target_placement_events_before_first_deadlock": _average(placement_events),
        "average_normalized_target_placement_before_first_deadlock": _average(normalized_events),
    }


def _iterative_summary(results: list[EpisodeResult]) -> dict[str, Any]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.level_id, []).append(result)
    if not grouped:
        return {
            "solve_rate_at_1": 0.0,
            "solve_rate_at_k": 0.0,
            "cumulative_solved_by_attempt": {},
        }

    max_attempt_index = 0
    first_attempt_successes = 0
    any_attempt_successes = 0
    first_success_index_by_level: dict[str, int] = {}
    for level_id, level_results in grouped.items():
        indexed = []
        for fallback_index, result in enumerate(level_results):
            attempt_index = result.metadata.get("iteration_attempt_index")
            if not isinstance(attempt_index, int) or isinstance(attempt_index, bool):
                attempt_index = fallback_index
            indexed.append((attempt_index, result))
            max_attempt_index = max(max_attempt_index, attempt_index)
        successes = [attempt_index for attempt_index, result in indexed if result.status == "success"]
        if successes:
            first_success = min(successes)
            first_success_index_by_level[level_id] = first_success
            any_attempt_successes += 1
            if first_success == 0:
                first_attempt_successes += 1

    level_count = len(grouped)
    cumulative = {
        str(attempt_number): (
            sum(
                1
                for first_success in first_success_index_by_level.values()
                if first_success < attempt_number
            )
            / level_count
        )
        for attempt_number in range(1, max_attempt_index + 2)
    }
    return {
        "solve_rate_at_1": first_attempt_successes / level_count,
        "solve_rate_at_k": any_attempt_successes / level_count,
        "cumulative_solved_by_attempt": cumulative,
    }
