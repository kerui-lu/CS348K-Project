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
    reasoning_config,
    responses_create_kwargs,
)
from sokoban_memory.llm_cache import LLMResponseCache, text_hash
from sokoban_memory.memory import (
    HEURISTIC_MEMORY_SCHEMA_VERSION,
    HeuristicMemory,
    MemoryRenderConfig,
    RawTrajectoryMemory,
    classify_heuristic,
    get_memory_source_level_ids,
    truncate_text,
)

REFLECTION_PROMPT_VERSION = "reflection_v2"
V3_GLOBAL_REFLECTION_PROMPT_VERSION = "reflection_v3_global_train_failures_v1"


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
    reasoning = reasoning_config(model)
    if reasoning is not None:
        request["reasoning"] = reasoning
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
            **responses_create_kwargs(
                model=model,
                input_text=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
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
        "Goal: distill concise heuristic rules that could help a future full-path push planner avoid similar failures.\n"
        "The future planner outputs JSON push intents such as {\"box\": [3, 4], \"push\": \"Right\"}; a local verifier expands reachable walking paths.\n"
        "Use prescriptive rules. Do not replay the trajectories.\n"
        "Return a JSON array of strings only.\n\n"
        f"Failed trajectory memory:\n{raw_memory.render(config)}"
    )


def generate_v3_global_reflection_memory(
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
    config = memory_config or MemoryRenderConfig(max_memory_items=999, max_memory_chars=30000)
    prompt = build_v3_global_reflection_prompt(raw_memory, config)
    prompt_hash = text_hash(prompt)
    source_train_level_ids = get_memory_source_level_ids(raw_memory)
    failure_subtype_distribution = _failure_subtype_distribution(raw_memory)
    request = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": V3_GLOBAL_REFLECTION_PROMPT_VERSION,
        "task": "v3_global_failure_reflection",
        "cache_namespace": cache_namespace,
    }
    reasoning = reasoning_config(model)
    if reasoning is not None:
        request["reasoning"] = reasoning
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
            **responses_create_kwargs(
                model=model,
                input_text=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
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
    scope_counts = _heuristic_scope_counts(heuristics)
    return HeuristicMemory(
        heuristics=heuristics,
        source_metadata={
            "source_raw_memory_hash": raw_memory.memory_hash,
            "reflection_model": model,
            "reflection_prompt_version": V3_GLOBAL_REFLECTION_PROMPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_train_level_ids": source_train_level_ids,
            "raw_failure_count_used": len(raw_memory.episodes),
            "failure_subtype_distribution": failure_subtype_distribution,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "cache_namespace": cache_namespace,
            "prompt_hash": prompt_hash,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "memory_caps": config.to_dict(),
            "heuristic_scope_counts": scope_counts,
            "schema_version": HEURISTIC_MEMORY_SCHEMA_VERSION,
        },
    )


def build_v3_global_reflection_prompt(raw_memory: RawTrajectoryMemory, config: MemoryRenderConfig) -> str:
    return (
        "You are analyzing all failed train split one-shot Sokoban attempts for a train-to-eval experiment.\n"
        f"Prompt version: {V3_GLOBAL_REFLECTION_PROMPT_VERSION}\n\n"
        "Goal: distill cross-level heuristic rules that could help a future full-path push planner on held-out eval levels.\n"
        "The future planner outputs JSON push intents such as {\"box\": [3, 4], \"push\": \"Right\"}; a local verifier expands reachable walking paths.\n"
        "Use only abstract, transferable rules. Do not mention level IDs, exact coordinates, copied board rows, raw push sequences, or solver/reference solutions.\n"
        "Do not replay the train trajectories. Do not write rules that only apply to one specific board.\n"
        "Return a JSON array of concise strings only.\n\n"
        f"Failed train evidence:\n{render_v3_global_failure_evidence(raw_memory, config)}"
    )


def render_v3_global_failure_evidence(raw_memory: RawTrajectoryMemory, config: MemoryRenderConfig) -> str:
    selected = raw_memory.episodes[: config.max_memory_items]
    if not selected:
        return "No failed train attempts are available."
    sections = [
        "All failed train one-shot attempts:",
        f"failure_count_rendered: {len(selected)}",
    ]
    for index, episode in enumerate(selected, start=1):
        sections.append(_render_v3_global_failure_item(index, episode))
    return truncate_text("\n\n".join(sections), config.max_memory_chars)


def _render_v3_global_failure_item(index: int, episode: dict[str, Any]) -> str:
    trace = episode.get("v3_attempt_trace") if isinstance(episode.get("v3_attempt_trace"), dict) else {}
    failed_log = _failed_push_log(episode)
    lines = [
        f"failure_index: {index}",
        f"level_id: {episode.get('level_id')}",
        f"source_family: {trace.get('source_family')}",
        f"difficulty_bucket: {trace.get('difficulty_bucket')}",
        f"status: {episode.get('status')}",
        f"failure_subtype: {episode.get('failure_subtype') or episode.get('failure_reason')}",
        f"failure_reason: {episode.get('failure_reason')}",
        f"failure_push_index: {episode.get('failure_push_index')}",
        f"best_boxes_on_targets: {trace.get('best_boxes_on_targets')}",
        f"final_boxes_on_targets: {trace.get('final_boxes_on_targets')}",
        f"failed_intent: {_log_value(failed_log, 'model_intent') or _log_value(failed_log, 'intent')}",
        f"resolved_box_before_push: {_log_value(failed_log, 'resolved_box_before_push')}",
        f"standing_cell_required: {_log_value(failed_log, 'standing_cell_required')}",
        f"destination_cell: {_log_value(failed_log, 'destination_cell')}",
        "board_before_failed_push:",
        str(episode.get("board_before_failed_push")),
        "board_after_last_successful_push:",
        str(episode.get("board_after_last_successful_push")),
    ]
    concise_log = _concise_push_log(episode)
    if concise_log:
        lines.append(f"concise_push_log: {concise_log}")
    return "\n".join(lines)


def _failed_push_log(episode: dict[str, Any]) -> dict[str, Any] | None:
    logs = episode.get("push_execution_log")
    if not isinstance(logs, list):
        return None
    failure_push_index = episode.get("failure_push_index")
    if isinstance(failure_push_index, int):
        for log in logs:
            if isinstance(log, dict) and log.get("push_index") == failure_push_index:
                return log
    for log in reversed(logs):
        if isinstance(log, dict) and log.get("status") in {"failed", "deadlock"}:
            return log
    return logs[-1] if logs and isinstance(logs[-1], dict) else None


def _concise_push_log(episode: dict[str, Any]) -> list[dict[str, Any]]:
    logs = episode.get("push_execution_log")
    if not isinstance(logs, list):
        return []
    concise = []
    for log in logs:
        if not isinstance(log, dict):
            continue
        status = log.get("status") or log.get("result")
        if status not in {"failed", "deadlock"} and len(concise) >= 6:
            continue
        concise.append(
            {
                "push_index": log.get("push_index"),
                "intent": log.get("model_intent") or log.get("intent"),
                "status": status,
                "failure_subtype": log.get("failure_subtype"),
            }
        )
    return concise[-8:]


def _log_value(log: dict[str, Any] | None, key: str) -> Any:
    return log.get(key) if isinstance(log, dict) else None


def _failure_subtype_distribution(raw_memory: RawTrajectoryMemory) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in raw_memory.episodes:
        key = str(episode.get("failure_subtype") or episode.get("failure_reason") or episode.get("status"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _heuristic_scope_counts(heuristics: list[str]) -> dict[str, int]:
    counts = {"global_allowed": 0, "same_level_only": 0, "rejected": 0}
    for heuristic in heuristics:
        scope = classify_heuristic(heuristic)["scope"]
        counts[scope] = counts.get(scope, 0) + 1
    return counts


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
