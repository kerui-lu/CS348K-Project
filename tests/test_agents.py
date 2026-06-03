import os

import pytest

from run_experiment import build_parser
from sokoban_memory.agents import (
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    LLMAgent,
    MIN_MAX_OUTPUT_TOKENS,
    NoMemoryAgent,
    PROMPT_VERSION,
    RawTrajectoryMemoryAgent,
    ReflectionHeuristicAgent,
    _load_dotenv,
)
from sokoban_memory.experiment import RULES_TEXT
from sokoban_memory.memory import HeuristicMemory, RawTrajectoryMemory


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


def test_llm_agent_returns_full_path_response_text_and_tracks_call_count():
    client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    agent = LLMAgent(client=client)

    plan = agent.select_plan(
        "#####\n#@$.#\n#####",
        {
            "rules": "Move Up/Down/Left/Right.",
            "state_summary": {"player": [1, 1], "boxes": [[1, 2]], "targets": [[1, 3]]},
            "max_steps": 20,
            "memory": None,
        },
    )

    assert plan == '[{"box": [1, 2], "push": "Right"}]'
    assert agent.llm_call_count == 1
    assert client.responses.calls[0]["model"] == DEFAULT_LLM_MODEL
    assert "Current board" in client.responses.calls[0]["input"]
    assert f"Prompt version: {PROMPT_VERSION}" in client.responses.calls[0]["input"]
    assert "Return JSON only" in client.responses.calls[0]["input"]


def test_llm_agent_select_action_aliases_full_path_plan():
    agent = LLMAgent(client=FakeClient('[{"box": [1, 2], "push": "Right"}]'))

    raw_plan = agent.select_action(
        "#####\n# @ #\n#####",
        {
            "rules": "Move Up/Down/Left/Right.",
            "state_summary": {"player": [1, 2], "boxes": [], "targets": []},
            "max_steps": 20,
            "memory": None,
        },
    )

    assert raw_plan == '[{"box": [1, 2], "push": "Right"}]'


def test_cli_accepts_llm_model_and_api_key_env_args():
    args = build_parser().parse_args(
        [
            "--agent",
            "llm",
            "--model",
            "gpt-5-nano",
            "--api_key_env",
            "OPENAI_API_KEY",
        ]
    )

    assert args.agent == "llm"
    assert args.model == "gpt-5-nano"
    assert args.api_key_env == "OPENAI_API_KEY"


def test_cli_accepts_reasoning_effort_override():
    args = build_parser().parse_args(["--agent", "llm", "--reasoning_effort", "medium"])

    assert args.reasoning_effort == "medium"


def test_cli_accepts_v2_budget_memory_and_cache_args():
    assert MIN_MAX_OUTPUT_TOKENS == 16384
    args = build_parser().parse_args(
        [
            "--agent",
            "raw_trajectory_memory",
            "--memory_path",
            "memory_banks/raw.json",
            "--max_llm_calls",
            "7",
            "--max_memory_items",
            "2",
            "--max_steps_per_memory",
            "4",
            "--max_memory_chars",
            "500",
            "--llm_cache_path",
            ".llm_cache/test",
            "--temperature",
            "0",
            "--max_output_tokens",
            str(MIN_MAX_OUTPUT_TOKENS),
            "--cache_namespace",
            "main",
        ]
    )

    assert args.agent == "raw_trajectory_memory"
    assert args.memory_path == "memory_banks/raw.json"
    assert args.max_llm_calls == 7
    assert args.max_memory_items == 2
    assert args.max_steps_per_memory == 4
    assert args.max_memory_chars == 500
    assert args.llm_cache_path == ".llm_cache/test"
    assert args.temperature == 0
    assert args.max_output_tokens == MIN_MAX_OUTPUT_TOKENS
    assert args.cache_namespace == "main"


def test_cli_rejects_output_token_cap_below_minimum():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--agent", "llm", "--max_output_tokens", "8192"])


def test_cli_accepts_same_level_iterative_mode():
    args = build_parser().parse_args(
        [
            "--experiment_mode",
            "same_level_iterative",
            "--condition",
            "raw_same_level_iterative",
            "--attempt_budget",
            "5",
        ]
    )

    assert args.experiment_mode == "same_level_iterative"
    assert args.condition == "raw_same_level_iterative"
    assert args.attempt_budget == 5


def test_cli_defaults_to_low_cost_llm_model():
    args = build_parser().parse_args(["--agent", "llm"])

    assert args.model == "gpt-5-nano"


def test_v2_agent_prompts_differ_only_by_memory_condition():
    no_memory_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    raw_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    reflection_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    raw_memory = RawTrajectoryMemory(
        episodes=[
            {
                "level_id": "trap",
                "status": "deadlock",
                "step_count": 1,
                "total_reward": -5.1,
                "planned_pushes": [{"box": [1, 2], "push": "Right"}],
                "expanded_actions": ["Right"],
                "steps": [
                    {
                        "step": 0,
                        "state": "#@$",
                        "raw_action": "Right",
                        "parsed_action": "Right",
                        "executed_action": "Right",
                        "reward": -5.1,
                        "invalid_reason": None,
                        "pushed_box": True,
                        "deadlocked": True,
                        "solved": False,
                        "next_state": "# @$",
                    }
                ],
            }
        ]
    )
    heuristic_memory = HeuristicMemory(["Do not push a box into a non-target corner."])
    context = {
        "rules": RULES_TEXT,
        "state_summary": {"player": [1, 1], "boxes": [[1, 2]], "targets": [[1, 3]]},
        "max_steps": 20,
    }

    NoMemoryAgent(client=no_memory_client).select_action("#@$.#", context)
    RawTrajectoryMemoryAgent(memory_store=raw_memory, client=raw_client).select_action("#@$.#", context)
    ReflectionHeuristicAgent(memory_store=heuristic_memory, client=reflection_client).select_action(
        "#@$.#",
        context,
    )

    no_memory_prompt = no_memory_client.responses.calls[0]["input"]
    raw_prompt = raw_client.responses.calls[0]["input"]
    reflection_prompt = reflection_client.responses.calls[0]["input"]
    assert "Condition: none" in no_memory_prompt
    assert "Prompt version: full_path_v2_1" in no_memory_prompt
    assert "No past experience is available" in no_memory_prompt
    assert "space empty floor" in no_memory_prompt
    assert "walk into empty floor cells or target cells" in no_memory_prompt
    assert "stand in the cell opposite the push direction" in no_memory_prompt
    assert "destination cell is empty floor or a target" in no_memory_prompt
    assert "If pushing Right from box [r, c]" in no_memory_prompt
    assert "Only write a push if the box exists there now" in no_memory_prompt
    assert "Planning checklist:" in no_memory_prompt
    assert "until every box is on a target" in no_memory_prompt
    assert "Do not stop after one push" in no_memory_prompt
    assert "use the box's updated coordinate" in no_memory_prompt
    assert "boxes are interchangeable" in no_memory_prompt
    assert "any box may go to any target" in no_memory_prompt
    assert "avoid moving a box off a target unless necessary" in no_memory_prompt
    assert '"box": [row, col]' in no_memory_prompt
    assert '"box_id": 0' not in no_memory_prompt
    assert '"player_after": [row, col]' not in no_memory_prompt
    assert '"box_after": [row, col]' not in no_memory_prompt
    assert "Return JSON only" in no_memory_prompt
    assert "reference_solution" not in no_memory_prompt
    assert "solver_min_pushes" not in no_memory_prompt
    assert "solver_min_steps" not in no_memory_prompt
    assert '{"box": [4, 4], "push": "Right"}' in no_memory_prompt
    assert "Prior trajectory records" in raw_prompt
    assert "executed_action: Right" in raw_prompt
    assert "raw_action" not in raw_prompt.split("Memory context:", 1)[1]
    assert "parsed_action" not in raw_prompt.split("Memory context:", 1)[1]
    assert "Reflection heuristics distilled" in reflection_prompt
    assert "Do not push a box into a non-target corner." in reflection_prompt
    assert "state:" not in reflection_prompt.split("Memory context:", 1)[1]
    assert "temperature" not in no_memory_client.responses.calls[0]
    assert no_memory_client.responses.calls[0]["reasoning"] == {"effort": "minimal"}
    assert no_memory_client.responses.calls[0]["max_output_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS


def test_gpt52_defaults_to_low_reasoning_effort():
    client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    agent = LLMAgent(client=client, model="gpt-5.2")

    agent.select_plan(
        "#####\n#@$.#\n#####",
        {
            "rules": "Move Up/Down/Left/Right.",
            "state_summary": {"player": [1, 1], "boxes": [[1, 2]], "targets": [[1, 3]]},
            "max_steps": 20,
            "memory": None,
        },
    )

    assert client.responses.calls[0]["reasoning"] == {"effort": "low"}
    assert agent.last_call_metadata["reasoning_effort"] == "low"


def test_explicit_reasoning_effort_changes_request_and_cache_key():
    low_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    medium_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    low_agent = LLMAgent(client=low_client, model="gpt-5.2")
    medium_agent = LLMAgent(client=medium_client, model="gpt-5.2", reasoning_effort="medium")
    context = {
        "rules": "Move Up/Down/Left/Right.",
        "state_summary": {"player": [1, 1], "boxes": [[1, 2]], "targets": [[1, 3]]},
        "max_steps": 20,
        "memory": None,
    }

    low_agent.select_plan("#####\n#@$.#\n#####", context)
    medium_agent.select_plan("#####\n#@$.#\n#####", context)

    assert low_client.responses.calls[0]["reasoning"] == {"effort": "low"}
    assert medium_client.responses.calls[0]["reasoning"] == {"effort": "medium"}
    assert medium_agent.last_call_metadata["reasoning_effort"] == "medium"
    assert low_agent.last_call_metadata["cache_key"] != medium_agent.last_call_metadata["cache_key"]


def test_non_gpt5_llm_agent_sends_temperature():
    client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    agent = LLMAgent(client=client, model="gpt-4.1-mini", temperature=0.2)

    agent.select_plan(
        "#####\n#@$.#\n#####",
        {
            "rules": "Move Up/Down/Left/Right.",
            "state_summary": {"player": [1, 1], "boxes": [[1, 2]], "targets": [[1, 3]]},
            "max_steps": 20,
            "memory": None,
        },
    )

    assert client.responses.calls[0]["temperature"] == 0.2


def test_v2_prompts_share_non_memory_template():
    raw_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    reflection_client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    raw_memory = RawTrajectoryMemory(
        episodes=[
            {
                "level_id": "trap",
                "status": "deadlock",
                "step_count": 1,
                "total_reward": -5.1,
                "steps": [{"step": 0, "state": "#@$", "executed_action": "Right", "next_state": "# @$"}],
            }
        ]
    )
    heuristic_memory = HeuristicMemory(["Do not push a box into a non-target corner."])
    context = {
        "rules": "Move.",
        "state_summary": {"player": [1, 1], "boxes": [[1, 2]], "targets": [[1, 3]]},
        "max_steps": 20,
    }

    raw_agent = RawTrajectoryMemoryAgent(memory_store=raw_memory, client=raw_client)
    reflection_agent = ReflectionHeuristicAgent(memory_store=heuristic_memory, client=reflection_client)
    raw_agent.select_action("#@$.#", context)
    reflection_agent.select_action("#@$.#", context)

    assert (
        raw_agent.last_call_metadata["non_memory_template_hash"]
        == reflection_agent.last_call_metadata["non_memory_template_hash"]
    )


def test_load_dotenv_sets_missing_values_without_overriding_existing_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=from_file\n"
        "EXISTING_KEY=from_file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING_KEY", "from_shell")

    _load_dotenv(env_path)

    assert os.environ["OPENAI_API_KEY"] == "from_file"
    assert os.environ["EXISTING_KEY"] == "from_shell"
