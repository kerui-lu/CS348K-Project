import os

from run_experiment import build_parser
from sokoban_memory.action_parser import parse_action
from sokoban_memory.agents import DEFAULT_LLM_MODEL, LLMAgent, _load_dotenv


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


def test_llm_agent_returns_response_text_and_tracks_call_count():
    client = FakeClient("Right")
    agent = LLMAgent(client=client)

    action = agent.select_action(
        "#####\n#@$.#\n#####",
        {
            "rules": "Move Up/Down/Left/Right.",
            "legal_actions": ["Right"],
            "push_actions": ["Right"],
            "memory": None,
        },
    )

    assert action == "Right"
    assert agent.llm_call_count == 1
    assert client.responses.calls[0]["model"] == DEFAULT_LLM_MODEL
    assert "Current board" in client.responses.calls[0]["input"]


def test_llm_agent_natural_language_output_still_uses_existing_parser():
    agent = LLMAgent(client=FakeClient("I choose Up"))

    raw_action = agent.select_action(
        "#####\n# @ #\n#####",
        {
            "rules": "Move Up/Down/Left/Right.",
            "legal_actions": ["Up"],
            "push_actions": [],
            "memory": None,
        },
    )

    assert parse_action(raw_action) == "Up"


def test_cli_accepts_llm_model_and_api_key_env_args():
    args = build_parser().parse_args(
        [
            "--agent",
            "llm",
            "--model",
            "gpt-5.4-mini",
            "--api_key_env",
            "OPENAI_API_KEY",
        ]
    )

    assert args.agent == "llm"
    assert args.model == "gpt-5.4-mini"
    assert args.api_key_env == "OPENAI_API_KEY"


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
