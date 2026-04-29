from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from sokoban_memory.agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_CACHE_NAMESPACE,
    DEFAULT_DOTENV_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    LLMBudgetExceeded,
    _load_dotenv,
)
from sokoban_memory.llm_cache import LLMResponseCache, text_hash
from sokoban_memory.memory import (
    HeuristicMemory,
    MemoryRenderConfig,
    RawTrajectoryMemory,
    get_memory_source_level_ids,
)

REFLECTION_PROMPT_VERSION = "reflection_v2"


def reflect_on_failure(trajectory: list[dict[str, Any]]) -> list[str]:
    heuristics = [
        "Do not push a box into a non-target corner.",
        "Avoid pushing a box against a wall unless the target is along that wall.",
        "Before pushing a box, check whether the box can still reach a target.",
    ]
    if any(step.get("info", {}).get("deadlocked") for step in trajectory):
        heuristics.insert(0, "The latest failure ended in a detected deadlock; avoid the final push pattern.")
    return heuristics


def generate_reflection_memory(
    raw_memory: RawTrajectoryMemory,
    model: str = DEFAULT_LLM_MODEL,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    client: Any | None = None,
    llm_cache_path: str | None = None,
    max_llm_calls: int | None = 1,
    memory_config: MemoryRenderConfig | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    cache_namespace: str = DEFAULT_CACHE_NAMESPACE,
) -> HeuristicMemory:
    config = memory_config or MemoryRenderConfig()
    prompt = build_reflection_prompt(raw_memory, config)
    prompt_hash = text_hash(prompt)
    source_train_level_ids = get_memory_source_level_ids(raw_memory)
    request = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": REFLECTION_PROMPT_VERSION,
        "task": "failure_reflection",
        "cache_namespace": cache_namespace,
    }
    cache = LLMResponseCache(llm_cache_path, namespace=cache_namespace)
    cache_key = cache.make_key(request)
    cached = cache.get(cache_key)
    cache_hit = cached is not None

    if cached is not None:
        output_text = str(cached.get("output_text", ""))
    else:
        if max_llm_calls is not None and max_llm_calls <= 0:
            raise LLMBudgetExceeded("Reflection LLM call budget exhausted before generation.")
        llm_client = client if client is not None else _make_openai_client(api_key_env)
        response = llm_client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        output_text = _extract_text(response).strip()
        cache.set(
            cache_key,
            {
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "cache_namespace": cache_namespace,
                "prompt_hash": prompt_hash,
                "output_text": output_text,
                "usage": _extract_usage(response),
            },
        )

    heuristics = parse_heuristics(output_text)
    return HeuristicMemory(
        heuristics=heuristics,
        source_metadata={
            "source_raw_memory_hash": raw_memory.memory_hash,
            "reflection_model": model,
            "reflection_prompt_version": REFLECTION_PROMPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_train_level_ids": source_train_level_ids,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "cache_namespace": cache_namespace,
            "prompt_hash": prompt_hash,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "memory_caps": config.to_dict(),
        },
    )


def build_reflection_prompt(raw_memory: RawTrajectoryMemory, config: MemoryRenderConfig) -> str:
    return (
        "You are analyzing failed Sokoban trajectories.\n"
        f"Prompt version: {REFLECTION_PROMPT_VERSION}\n\n"
        "Goal: distill concise heuristic rules that could help a future one-step Sokoban agent avoid similar failures.\n"
        "Use prescriptive rules. Do not replay the trajectories.\n"
        "Return a JSON array of strings only.\n\n"
        f"Failed trajectory memory:\n{raw_memory.render(config)}"
    )


def parse_heuristics(output_text: str) -> list[str]:
    text = output_text.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("heuristics", [])
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]

    heuristics = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            heuristics.append(cleaned)
    return heuristics


def _make_openai_client(api_key_env: str) -> Any:
    from openai import OpenAI

    _load_dotenv(DEFAULT_DOTENV_PATH)
    return OpenAI(api_key=os.getenv(api_key_env))


def _extract_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    return str(response)


def _extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return dict(getattr(usage, "__dict__", {}))
