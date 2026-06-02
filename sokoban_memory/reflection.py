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

# --- Same-level reflection (V2) ---------------------------------------------
# These versions are used ONLY by the `heuristic_same_level_iterative` path.
# Unlike the cross-level reflection above (which deliberately produces generic,
# transferable rules), same-level reflection is meant to diagnose the specific
# failed attempt on this exact board and prescribe concrete, board-grounded
# corrective guidance for the next retry on the same level.
SAME_LEVEL_REFLECTION_VERSIONS = (
    "baseline",
    "v1_specific",
    "v2_complete_plan",
    "v3_hybrid_verifier",
)
DEFAULT_SAME_LEVEL_REFLECTION_VERSION = "baseline"


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


def _same_level_reflection_version_tag(version: str) -> str:
    return f"same_level_reflection_{version}"


_SAME_LEVEL_BASE_INTRO = (
    "You are debugging repeated failures on ONE specific Sokoban level.\n"
    "A full-path planner keeps trying the SAME board and keeps failing.\n"
    "A local verifier executes each push intent {\"box\": [r, c], \"push\": \"Up|Down|Left|Right\"}\n"
    "by walking the player to the required standing cell (one step opposite the push\n"
    "direction) and then pushing. To push Right from box [r, c] the player must stand\n"
    "at [r, c-1]; Left -> [r, c+1]; Down -> [r-1, c]; Up -> [r+1, c]. The push is rejected\n"
    "if that standing cell cannot be reached by walking around current walls and boxes,\n"
    "or if the destination cell is blocked.\n\n"
)

_SAME_LEVEL_OUTPUT_CONTRACT = (
    "\nReturn a JSON array of 3 to 5 short strings, nothing else. Each string is one\n"
    "actionable corrective rule for the NEXT attempt on THIS board. You MAY and SHOULD\n"
    "reference concrete coordinates, specific boxes, specific targets, and specific push\n"
    "directions from the evidence. Do not restate generic Sokoban tips the planner already\n"
    "knows; every rule must be tied to what actually went wrong on THIS board.\n"
)


def build_same_level_reflection_prompt(
    raw_memory: RawTrajectoryMemory,
    level_id: str,
    config: MemoryRenderConfig,
    version: str = "v1_specific",
) -> str:
    evidence = raw_memory.render_same_level_failure_evidence(level_id, config)
    if version == "baseline":
        return build_reflection_prompt(raw_memory, config)

    instructions = [
        "Diagnose the SPECIFIC reason the latest attempt failed: name the failed push index, the box coordinate, the required standing cell, and why the verifier rejected it (or why the plan stalled).",
        "Then prescribe how to fix it on the next attempt for this same board.",
        "If the failure was an unreachable standing cell, say which box must be moved first, and in which direction, to open the approach, using concrete coordinates.",
        "Never re-emit the exact push that just failed.",
    ]
    if version in ("v2_complete_plan", "v3_hybrid_verifier"):
        instructions += [
            "The plan must be COMPLETE: keep pushing until EVERY box is on a target. Several past attempts stopped early (plan_exhausted), so state the full box-to-target assignment and the push order needed to finish, so the next plan does not stop after a few pushes.",
        ]
    if version == "v3_hybrid_verifier":
        instructions = [
            "Make the FIRST rule a literal restatement of the verifier's rejection from the most recent attempt: the failed push index, the exact box coordinate, the exact required standing cell, and that it was unreachable/blocked. The planner must treat that specific push as forbidden until the approach is opened.",
        ] + instructions

    return (
        _SAME_LEVEL_BASE_INTRO
        + f"reflection_version: {_same_level_reflection_version_tag(version)}\n"
        + f"level_id: {level_id}\n\n"
        + "Instructions:\n- "
        + "\n- ".join(instructions)
        + _SAME_LEVEL_OUTPUT_CONTRACT
        + f"\nEvidence:\n{evidence}"
    )


def generate_same_level_reflection_memory(
    raw_memory: RawTrajectoryMemory,
    level_id: str,
    version: str = "v1_specific",
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
    """Same-level reflection: concrete, board-grounded corrective heuristics.

    Mirrors `generate_reflection_memory` but uses the same-level evidence
    renderer and a version-specific prompt. `version="baseline"` falls back to
    the legacy cross-level generic reflection for reproduction.
    """
    if version == "baseline":
        return generate_reflection_memory(
            raw_memory,
            model=model,
            api_key_env=api_key_env,
            client=client,
            llm_cache_path=llm_cache_path,
            max_llm_calls=max_llm_calls,
            memory_config=memory_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            cache_namespace=cache_namespace,
        )

    config = memory_config or MemoryRenderConfig()
    prompt = build_same_level_reflection_prompt(raw_memory, level_id, config, version)
    prompt_hash = text_hash(prompt)
    prompt_version = _same_level_reflection_version_tag(version)
    source_level_ids = get_memory_source_level_ids(raw_memory)
    request = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": prompt_version,
        "task": "same_level_failure_reflection",
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
            "reflection_prompt_version": prompt_version,
            "same_level_reflection_version": version,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_level_ids": source_level_ids or [level_id],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "cache_namespace": cache_namespace,
            "prompt_hash": prompt_hash,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "memory_caps": config.to_dict(),
        },
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
