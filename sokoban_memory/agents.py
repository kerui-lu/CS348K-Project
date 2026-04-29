from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from sokoban_memory.llm_cache import LLMResponseCache, text_hash
from sokoban_memory.memory import HeuristicMemory, MemoryRenderConfig, RawTrajectoryMemory
from sokoban_memory.prompts import render_one_step_prompt
from sokoban_memory.types import Action

DEFAULT_LLM_MODEL = "gpt-5-nano"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 8
DEFAULT_CACHE_NAMESPACE = "main"
PLACEHOLDER_OPENAI_API_KEY = "PLACEHOLDER_OPENAI_API_KEY"
DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
PROMPT_VERSION = "one_step_v2"
POLICY_MODE_ONE_STEP = "one_step"
POLICY_MODE_NON_LLM = "non_llm"


class LLMBudgetExceeded(RuntimeError):
    pass


class BaseAgent(ABC):
    agent_type = "base"
    policy_mode = POLICY_MODE_NON_LLM
    model: str | None = None
    prompt_version: str | None = None
    memory_path: str | None = None
    memory_hash: str | None = None

    def __init__(self) -> None:
        self.llm_call_count = 0
        self.token_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_call_metadata: dict[str, Any] = {}

    @abstractmethod
    def select_action(self, state_text: str, context: dict[str, Any]) -> str:
        raise NotImplementedError

    def memory_caps(self) -> dict[str, Any]:
        return {}


class RuleBasedAgent(BaseAgent):
    agent_type = "rule_based"

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = random.Random(seed)

    def select_action(self, state_text: str, context: dict[str, Any]) -> str:
        legal_actions: list[Action] = context.get("legal_actions", [])
        self.last_call_metadata = {
            "policy_mode": self.policy_mode,
            "prompt_hash": None,
            "prompt_char_count": 0,
            "memory_char_count": 0,
            "cache_hit": False,
            "cache_key": None,
            "usage": {},
        }
        if not legal_actions:
            return "Up"
        push_actions = context.get("push_actions", [])
        if push_actions:
            return self.rng.choice(push_actions)
        return self.rng.choice(legal_actions)


class OneStepLLMAgent(BaseAgent):
    agent_type = "one_step_llm"
    memory_condition = "none"
    policy_mode = POLICY_MODE_ONE_STEP
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        memory_store: RawTrajectoryMemory | HeuristicMemory | None = None,
        memory_config: MemoryRenderConfig | None = None,
        model: str = DEFAULT_LLM_MODEL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        client: Any | None = None,
        max_llm_calls: int | None = 50,
        llm_cache_path: str | Path | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
        memory_path: str | Path | None = None,
    ):
        super().__init__()
        self.memory_store = memory_store
        self.memory_config = memory_config or MemoryRenderConfig()
        self.model = model
        self.api_key_env = api_key_env
        self.max_llm_calls = max_llm_calls
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache_namespace = cache_namespace
        self.memory_path = str(memory_path) if memory_path else None
        self.memory_hash = getattr(memory_store, "memory_hash", None)
        self.cache = LLMResponseCache(llm_cache_path, namespace=cache_namespace)
        self.client = client if client is not None else self._make_openai_client(api_key_env)

    def select_action(self, state_text: str, context: dict[str, Any]) -> str:
        prompt, memory_text, non_memory_template = self._build_prompt(state_text, context)
        prompt_hash = text_hash(prompt)
        request = {
            "model": self.model,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "prompt_version": self.prompt_version,
            "policy_mode": self.policy_mode,
            "cache_namespace": self.cache_namespace,
        }
        cache_key = self.cache.make_key(request)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            output_text = str(cached.get("output_text", "")).strip()
            self.last_call_metadata = self._call_metadata(
                prompt_hash=prompt_hash,
                prompt=prompt,
                memory_text=memory_text,
                non_memory_template=non_memory_template,
                cache_hit=True,
                cache_key=cache_key,
                usage=dict(cached.get("usage", {})),
            )
            return output_text

        if self.max_llm_calls is not None and self.llm_call_count >= self.max_llm_calls:
            raise LLMBudgetExceeded(
                f"LLM call budget exhausted: {self.llm_call_count}/{self.max_llm_calls}"
            )

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        self.llm_call_count += 1
        self.cache_misses += 1
        output_text = self._extract_text(response).strip()
        usage = self._extract_usage(response)
        self.cache.set(
            cache_key,
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "cache_namespace": self.cache_namespace,
                "prompt_hash": prompt_hash,
                "output_text": output_text,
                "usage": usage,
            },
        )
        self.last_call_metadata = self._call_metadata(
            prompt_hash=prompt_hash,
            prompt=prompt,
            memory_text=memory_text,
            non_memory_template=non_memory_template,
            cache_hit=False,
            cache_key=cache_key,
            usage=usage,
        )
        return output_text

    def memory_caps(self) -> dict[str, Any]:
        return self.memory_config.to_dict()

    def _build_prompt(self, state_text: str, context: dict[str, Any]) -> tuple[str, str, str]:
        legal_actions = context.get("legal_actions", [])
        push_actions = context.get("push_actions", [])
        memory_text = self._render_memory()
        rendered = render_one_step_prompt(
            policy_mode=self.policy_mode,
            prompt_version=self.prompt_version,
            rules=context.get("rules", ""),
            state_text=state_text,
            legal_actions=legal_actions,
            push_actions=push_actions,
            memory_condition=self.memory_condition,
            memory_text=memory_text,
        )
        return rendered.prompt, rendered.memory_text, rendered.non_memory_template

    def _render_memory(self) -> str:
        if self.memory_condition == "none" or self.memory_store is None:
            return "No past experience is available for this condition."
        if hasattr(self.memory_store, "render"):
            return self.memory_store.render(self.memory_config)
        return str(self.memory_store)

    def _call_metadata(
        self,
        prompt_hash: str,
        prompt: str,
        memory_text: str,
        non_memory_template: str,
        cache_hit: bool,
        cache_key: str,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "policy_mode": self.policy_mode,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "prompt_version": self.prompt_version,
            "prompt_hash": prompt_hash,
            "non_memory_template_hash": text_hash(non_memory_template),
            "prompt_char_count": len(prompt),
            "memory_char_count": len(memory_text),
            "memory_hash": self.memory_hash,
            "cache_namespace": self.cache_namespace,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "usage": usage,
        }

    def _make_openai_client(self, api_key_env: str) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The OpenAI SDK is required for LLM agents. "
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

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        return _jsonable(usage) if usage is not None else {}


class NoMemoryAgent(OneStepLLMAgent):
    agent_type = "no_memory"
    memory_condition = "none"

    def __init__(self, **kwargs: Any):
        super().__init__(memory_store=None, **kwargs)


class RawTrajectoryMemoryAgent(OneStepLLMAgent):
    agent_type = "raw_trajectory_memory"
    memory_condition = "raw_trajectory_memory"


class ReflectionHeuristicAgent(OneStepLLMAgent):
    agent_type = "reflection_heuristic"
    memory_condition = "reflection_heuristic"


class LLMAgent(NoMemoryAgent):
    agent_type = "llm"


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


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def make_agent(
    agent_name: str,
    seed: int | None = None,
    memory: Any = None,
    model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client: Any | None = None,
    max_llm_calls: int | None = 50,
    memory_config: MemoryRenderConfig | None = None,
    llm_cache_path: str | Path | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
    memory_path: str | Path | None = None,
) -> BaseAgent:
    if agent_name == "rule_based":
        return RuleBasedAgent(seed=seed)
    if agent_name == "no_memory":
        return NoMemoryAgent(
            model=model,
            api_key_env=api_key_env,
            client=client,
            max_llm_calls=max_llm_calls,
            memory_config=memory_config,
            llm_cache_path=llm_cache_path,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_namespace=cache_namespace,
        )
    if agent_name in {"raw", "raw_trajectory", "raw_trajectory_memory"}:
        return RawTrajectoryMemoryAgent(
            memory_store=memory,
            model=model,
            api_key_env=api_key_env,
            client=client,
            max_llm_calls=max_llm_calls,
            memory_config=memory_config,
            llm_cache_path=llm_cache_path,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_namespace=cache_namespace,
            memory_path=memory_path,
        )
    if agent_name in {"reflection", "reflection_heuristic"}:
        return ReflectionHeuristicAgent(
            memory_store=memory,
            model=model,
            api_key_env=api_key_env,
            client=client,
            max_llm_calls=max_llm_calls,
            memory_config=memory_config,
            llm_cache_path=llm_cache_path,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_namespace=cache_namespace,
            memory_path=memory_path,
        )
    if agent_name == "llm":
        return LLMAgent(
            model=model,
            api_key_env=api_key_env,
            client=client,
            max_llm_calls=max_llm_calls,
            memory_config=memory_config,
            llm_cache_path=llm_cache_path,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_namespace=cache_namespace,
        )
    raise ValueError(f"Unknown agent: {agent_name}")
