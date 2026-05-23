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


def test_summary_reports_full_path_statuses_and_push_counts():
    invalid = make_result("l1", "invalid_plan", 0)
    invalid.metadata = {
        "planned_push_count": 2,
        "executed_push_count": 0,
        "expanded_primitive_step_count": 0,
    }
    exhausted = make_result("l1", "plan_exhausted", 4)
    exhausted.metadata = {
        "planned_push_count": 2,
        "executed_push_count": 2,
        "expanded_primitive_step_count": 4,
    }

    summary = summarize_results([invalid, exhausted])

    assert summary["invalid_plan_count"] == 1
    assert summary["plan_exhausted_count"] == 1
    assert summary["invalid_plan_rate"] == 0.5
    assert summary["plan_exhausted_rate"] == 0.5
    assert summary["planned_push_count"] == 4
    assert summary["executed_push_count"] == 2
    assert summary["expanded_primitive_step_count"] == 4


def test_summary_reports_repair_counters():
    repaired = make_result("l1", "success", 1)
    repaired.metadata = {
        "repair_attempt_count": 1,
        "success_after_repair": True,
        "success_without_repair": False,
        "first_attempt_status": "invalid_plan",
    }
    first_try = make_result("l2", "success", 1)
    first_try.metadata = {
        "repair_attempt_count": 0,
        "success_after_repair": False,
        "success_without_repair": True,
        "first_attempt_status": "success",
    }
    failed_after_repair = make_result("l3", "invalid_plan", 0)
    failed_after_repair.metadata = {
        "repair_attempt_count": 1,
        "success_after_repair": False,
        "success_without_repair": False,
        "first_attempt_status": "invalid_plan",
    }

    summary = summarize_results([repaired, first_try, failed_after_repair])

    assert summary["repair_attempt_count"] == 2
    assert summary["average_repair_attempts_per_episode"] == 2 / 3
    assert summary["success_after_repair_count"] == 1
    assert summary["success_without_repair_count"] == 1
    assert summary["first_attempt_invalid_plan_count"] == 2
    assert summary["per_level"]["l1"]["success_after_repair_count"] == 1


def test_summary_reports_partial_progress_score():
    partial = make_result("l1", "invalid_plan", 3)
    partial.trajectory = [
        {
            "state": "######\n#@$$.#\n#  . #\n######",
            "next_state": "######\n#@*$ #\n#  . #\n######",
        }
    ]
    partial.metadata = {
        "final_board": "######\n#@*$ #\n#  . #\n######",
    }
    stuck = make_result("l2", "timeout", 3)
    stuck.trajectory = [
        {
            "state": "#####\n#@$.#\n#####",
            "next_state": "#####\n#@$.#\n#####",
        }
    ]
    summary = summarize_results([partial, stuck])

    assert summary["average_final_goal_completion"] == 0.25
    assert summary["average_best_goal_completion"] == 0.25
    assert summary["partial_progress_score"] == 0.25
    assert summary["partial_progress_rate_25"] == 0.5
    assert summary["partial_progress_rate_50"] == 0.5
    assert summary["partial_progress_rate_75"] == 0.0
