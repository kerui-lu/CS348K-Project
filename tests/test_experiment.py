import json
import subprocess
import sys

from sokoban_memory.agents import RuleBasedAgent
from sokoban_memory.env import SokobanEnv
from sokoban_memory.experiment import run_episode
from sokoban_memory.levels import load_levels


def test_run_episode_finishes_within_max_steps():
    level = load_levels("levels/simple.json")[0]
    result = run_episode(SokobanEnv(level), RuleBasedAgent(seed=0), max_steps=5, seed=0)
    assert result.step_count <= 5
    assert result.status in {"success", "deadlock", "timeout", "failure"}
    assert result.trajectory


def test_cli_runs_complete_episode(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "run_experiment.py",
            "--agent",
            "rule_based",
            "--episodes",
            "1",
            "--max_steps",
            "20",
            "--results_dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["episodes"] == 1
    assert (tmp_path / "summary.json").exists()
    episode_files = [p for p in tmp_path.glob("*.json") if p.name != "summary.json"]
    assert len(episode_files) == 1

