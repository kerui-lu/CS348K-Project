from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sokoban_memory.agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_CACHE_NAMESPACE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    MIN_MAX_OUTPUT_TOKENS,
)
from sokoban_memory.memory import MemoryRenderConfig, RawTrajectoryMemory
from sokoban_memory.reflection import generate_v3_global_reflection_memory


def _max_output_tokens_at_least_min(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max_output_tokens must be an integer.") from exc
    if parsed < MIN_MAX_OUTPUT_TOKENS:
        raise argparse.ArgumentTypeError(
            f"--max_output_tokens must be at least {MIN_MAX_OUTPUT_TOKENS}."
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild V3 train-to-eval global heuristics from a raw train failure bank."
    )
    parser.add_argument("--raw_memory_path", default="memory_banks/v3_train_one_shot_raw_failures.json")
    parser.add_argument(
        "--heuristic_memory_path",
        default="memory_banks/v3_train_one_shot_global_heuristics.json",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api_key_env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--llm_cache_path", default=None)
    parser.add_argument("--max_llm_calls", type=int, default=1)
    parser.add_argument("--max_memory_items", type=int, default=999)
    parser.add_argument("--max_steps_per_memory", type=int, default=999)
    parser.add_argument("--max_memory_chars", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--max_output_tokens",
        type=_max_output_tokens_at_least_min,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = rebuild_v3_global_heuristics(args)
    print(json.dumps(summary, indent=2))


def rebuild_v3_global_heuristics(args: argparse.Namespace) -> dict[str, object]:
    raw_path = Path(args.raw_memory_path)
    raw_memory = RawTrajectoryMemory()
    raw_memory.load(raw_path)
    config = MemoryRenderConfig(
        max_memory_items=args.max_memory_items,
        max_steps_per_memory=args.max_steps_per_memory,
        max_memory_chars=args.max_memory_chars,
    )
    heuristic_memory = generate_v3_global_reflection_memory(
        raw_memory,
        model=args.model,
        api_key_env=args.api_key_env,
        llm_cache_path=args.llm_cache_path,
        max_llm_calls=args.max_llm_calls,
        memory_config=config,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        cache_namespace=args.cache_namespace,
    )
    heuristic_path = Path(args.heuristic_memory_path)
    heuristic_memory.save(heuristic_path)
    return {
        "raw_memory_path": str(raw_path),
        "heuristic_memory_path": str(heuristic_path),
        "raw_failure_count_used": len(raw_memory.episodes),
        "heuristic_count": len(heuristic_memory.heuristics),
        "heuristic_scope_counts": heuristic_memory.source_metadata.get("heuristic_scope_counts", {}),
        "source_raw_memory_hash": heuristic_memory.source_metadata.get("source_raw_memory_hash"),
        "prompt_hash": heuristic_memory.source_metadata.get("prompt_hash"),
        "cache_hit": heuristic_memory.source_metadata.get("cache_hit"),
        "memory_caps": config.to_dict(),
    }


if __name__ == "__main__":
    main()
