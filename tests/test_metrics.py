from sokoban_memory.metrics import summarize_results
from sokoban_memory.types import EpisodeResult


def make_result(
    level_id: str,
    status: str,
    step_count: int,
    optimal_steps: int | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        level_id=level_id,
        agent_type="test_agent",
        seed=0,
        status=status,  # type: ignore[arg-type]
        step_count=step_count,
        invalid_move_count=0,
        total_reward=0.0,
        llm_call_count=0,
        token_cost=0.0,
        trajectory=[],
        optimal_steps=optimal_steps,
    )


def test_summary_counts_and_efficiency_only_for_successes():
    results = [
        make_result("l1", "success", 4, optimal_steps=2),
        make_result("l1", "deadlock", 3, optimal_steps=2),
        make_result("l2", "timeout", 5),
        make_result("l2", "success", 6),
    ]

    summary = summarize_results(results)

    assert summary["success_count"] == 2
    assert summary["deadlock_count"] == 1
    assert summary["timeout_count"] == 1
    assert summary["solve_rate"] == 0.5
    assert summary["average_success_steps"] == 5.0
    assert summary["average_solution_efficiency"] == 0.5
    assert summary["median_solution_efficiency"] == 0.5
    assert summary["steps_over_optimal_average"] == 2.0
    assert summary["solution_efficiency_count"] == 1
    assert summary["solution_efficiency_skipped_count"] == 1


def test_per_level_breakdown():
    summary = summarize_results([
        make_result("l1", "success", 4, optimal_steps=2),
        make_result("l1", "deadlock", 3, optimal_steps=2),
        make_result("l2", "timeout", 5),
    ])

    assert summary["per_level"]["l1"]["attempts"] == 2
    assert summary["per_level"]["l1"]["successes"] == 1
    assert summary["per_level"]["l1"]["deadlocks"] == 1
    assert summary["per_level"]["l1"]["solve_rate"] == 0.5
    assert summary["per_level"]["l1"]["average_efficiency"] == 0.5
    assert summary["per_level"]["l2"]["timeouts"] == 1

