import json
import subprocess
import sys

from evaluate_results import evaluate_result_dirs, validate_episode_dict


def write_episode(path, level_id, agent_type, status, step_count, solved=False, deadlocked=False):
    data = {
        "level_id": level_id,
        "agent_type": agent_type,
        "seed": 0,
        "status": status,
        "step_count": step_count,
        "invalid_move_count": 0,
        "total_reward": 0.0,
        "llm_call_count": 0,
        "token_cost": 0.0,
        "trajectory": [
            {
                "step": 0,
                "state": "s",
                "executed_action": "Right",
                "next_state": "n",
                "reward": 0.0,
                "done": status in {"success", "deadlock"},
                "info": {"solved": solved, "deadlocked": deadlocked},
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_evaluate_result_dirs_reports_agent_and_level_metrics(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    write_episode(results_dir / "success.json", "simple_001", "agent_a", "success", 1, solved=True)
    write_episode(results_dir / "deadlock.json", "wall_push_001", "agent_b", "deadlock", 4, deadlocked=True)
    (results_dir / "summary.json").write_text("{}", encoding="utf-8")

    report = evaluate_result_dirs([results_dir], levels_path=None)

    assert report["valid_episode_count"] == 2
    assert report["overall"]["success_count"] == 1
    assert report["overall"]["deadlock_count"] == 1
    assert report["by_agent"]["agent_a"]["solve_rate"] == 1.0
    assert report["per_level"]["simple_001"]["successes"] == 1


def test_evaluate_result_dirs_uses_level_reference_lengths(tmp_path):
    results_dir = tmp_path / "results"
    levels_path = tmp_path / "levels.json"
    results_dir.mkdir()
    write_episode(results_dir / "success.json", "level_ref", "agent_a", "success", 4, solved=True)
    levels_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level_id": "level_ref",
                        "split": "eval",
                        "tags": ["easy_simple_push"],
                        "optimal_steps": 2,
                        "grid": ["#####", "#@$.#", "#####"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_result_dirs([results_dir], levels_path=levels_path)

    assert report["overall"]["average_solution_efficiency"] == 0.5
    assert report["overall"]["steps_over_optimal_average"] == 2.0


def test_validate_episode_detects_malformed_success(tmp_path):
    path = tmp_path / "bad.json"
    bad = {
        "level_id": "l1",
        "agent_type": "agent",
        "seed": 0,
        "status": "success",
        "step_count": 1,
        "invalid_move_count": 0,
        "total_reward": 0.0,
        "llm_call_count": 0,
        "token_cost": 0.0,
        "trajectory": [{"info": {"solved": False}}],
    }

    errors = validate_episode_dict(bad, path)

    assert errors
    assert "not marked solved" in errors[0]["message"]


def test_evaluate_results_cli_writes_report(tmp_path):
    results_dir = tmp_path / "results"
    output_path = tmp_path / "evaluation_summary.json"
    results_dir.mkdir()
    write_episode(results_dir / "success.json", "simple_001", "agent_a", "success", 1, solved=True)

    completed = subprocess.run(
        [
            sys.executable,
            "evaluate_results.py",
            "--results_dir",
            str(results_dir),
            "--output",
            str(output_path),
            "--fail_on_validation_error",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["overall"]["solve_rate"] == 1.0
    assert "per_level" in report

