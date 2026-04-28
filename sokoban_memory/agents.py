from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sokoban_memory.types import Action

DEFAULT_LLM_MODEL = "gpt-5.4-mini"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
PLACEHOLDER_OPENAI_API_KEY = "PLACEHOLDER_OPENAI_API_KEY"
DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class BaseAgent(ABC):
    agent_type = "base"
    llm_call_count = 0
    token_cost = 0.0

    @abstractmethod
    def select_action(self, state_text: str, context: dict[str, Any]) -> str:
        raise NotImplementedError


class RuleBasedAgent(BaseAgent):
    agent_type = "rule_based"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.llm_call_count = 0
        self.token_cost = 0.0

    def select_action(self, state_text: str, context: dict[str, Any]) -> str:
        legal_actions: list[Action] = context.get("legal_actions", [])
        if not legal_actions:
            return "Up"
        # Stable preference makes debugging easier; randomness only breaks ties.
        push_actions = context.get("push_actions", [])
        if push_actions:
            return self.rng.choice(push_actions)
        return self.rng.choice(legal_actions)


class NoMemoryAgent(RuleBasedAgent):
    agent_type = "no_memory"


class RawTrajectoryMemoryAgent(RuleBasedAgent):
    agent_type = "raw_trajectory_memory"

    def __init__(self, memory_store: Any, seed: int | None = None):
        super().__init__(seed=seed)
        self.memory_store = memory_store


class ReflectionHeuristicAgent(RuleBasedAgent):
    agent_type = "reflection_heuristic"

    def __init__(self, heuristic_memory: Any, seed: int | None = None):
        super().__init__(seed=seed)
        self.heuristic_memory = heuristic_memory


class LLMAgent(BaseAgent):
    agent_type = "llm"

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        client: Any | None = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.llm_call_count = 0
        self.token_cost = 0.0
        self.client = client if client is not None else self._make_openai_client(api_key_env)

    def select_action(self, state_text: str, context: dict[str, Any]) -> str:
        prompt = self._build_prompt(state_text, context)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        self.llm_call_count += 1
        return self._extract_text(response).strip()

    def _build_prompt(self, state_text: str, context: dict[str, Any]) -> str:
        legal_actions = context.get("legal_actions", [])
        push_actions = context.get("push_actions", [])
        memory = context.get("memory")
        memory_text = "None" if memory is None else str(memory)
        return (
            "You are playing Sokoban.\n\n"
            f"Rules:\n{context.get('rules', '')}\n\n"
            "Board symbols:\n"
            "# wall\n"
            ". target\n"
            "$ box\n"
            "* box on target\n"
            "@ player\n"
            "+ player on target\n\n"
            f"Current board:\n{state_text}\n\n"
            f"Legal actions: {legal_actions}\n"
            f"Actions that push a box: {push_actions}\n"
            f"Memory: {memory_text}\n\n"
            "Return exactly one action from this set: Up, Down, Left, Right.\n"
            "Do not include explanation, punctuation, or multiple actions."
        )

    def _make_openai_client(self, api_key_env: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required for --agent llm. "
                "Install dependencies with: python3 -m pip install -r requirements.txt"
            ) from exc

        _load_dotenv(DEFAULT_DOTENV_PATH)
        api_key = os.getenv(api_key_env, PLACEHOLDER_OPENAI_API_KEY)
        return OpenAI(api_key=api_key)

    def _extract_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        output = getattr(response, "output", None)
        if output:
            first_output = output[0]
            content = getattr(first_output, "content", None)
            if content:
                first_content = content[0]
                text = getattr(first_content, "text", None)
                if isinstance(text, str):
                    return text
        return str(response)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def make_agent(
    agent_name: str,
    seed: int | None = None,
    memory: Any = None,
    model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> BaseAgent:
    if agent_name == "rule_based":
        return RuleBasedAgent(seed=seed)
    if agent_name == "no_memory":
        return NoMemoryAgent(seed=seed)
    if agent_name in {"raw", "raw_trajectory", "raw_trajectory_memory"}:
        return RawTrajectoryMemoryAgent(memory_store=memory, seed=seed)
    if agent_name in {"reflection", "reflection_heuristic"}:
        return ReflectionHeuristicAgent(heuristic_memory=memory, seed=seed)
    if agent_name == "llm":
        return LLMAgent(model=model, api_key_env=api_key_env)
    raise ValueError(f"Unknown agent: {agent_name}")
