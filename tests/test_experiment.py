import json
import subprocess
import sys

from sokoban_memory.agents import RuleBasedAgent
from sokoban_memory.agents import LLMAgent
from sokoban_memory.env import SokobanEnv
from sokoban_memory.experiment import run_episode
from sokoban_memory.levels import load_levels
from sokoban_memory.metrics import summarize_results


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


class FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, output_text: str):
        self.responses = FakeResponses(output_text)


class RaisingResponses:
    def create(self, **kwargs):
        raise RuntimeError("api unavailable")


class RaisingClient:
    def __init__(self):
        self.responses = RaisingResponses()


def test_llm_budget_guard_prevents_api_call():
    level = load_levels("levels/simple.json")[0]
    client = FakeClient("Right")
    agent = LLMAgent(client=client, max_llm_calls=0)

    result = run_episode(SokobanEnv(level), agent, max_steps=5, seed=0)

    assert result.status == "budget_exhausted"
    assert result.budget_exhausted is True
    assert result.metadata["failure_reason"] == "llm_call_budget_exhausted"
    assert result.llm_call_count == 0
    assert result.step_count == 0
    assert client.responses.calls == []


def test_llm_api_error_gets_separate_status():
    level = load_levels("levels/simple.json")[0]
    agent = LLMAgent(client=RaisingClient(), max_llm_calls=5)

    result = run_episode(SokobanEnv(level), agent, max_steps=5, seed=0)

    assert result.status == "api_error"
    assert result.metadata["failure_reason"] == "api_error"
    assert result.metadata["error_type"] == "RuntimeError"


def test_summary_reports_v2_outcome_rates():
    level = load_levels("levels/simple.json")[0]
    budget_result = run_episode(SokobanEnv(level), LLMAgent(client=FakeClient("Right"), max_llm_calls=0), 1, 0)
    api_result = run_episode(SokobanEnv(level), LLMAgent(client=RaisingClient(), max_llm_calls=5), 1, 1)

    summary = summarize_results([budget_result, api_result])

    assert summary["budget_exhausted_rate"] == 0.5
    assert summary["api_error_rate"] == 0.5
    assert summary["invalid_failure_rate"] == 0.0
