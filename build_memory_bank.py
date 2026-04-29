from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_memory.agents import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_CACHE_NAMESPACE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    make_agent,
)
from sokoban_memory.env import SokobanEnv
from sokoban_memory.experiment import run_episode
from sokoban_memory.levels import load_levels
from sokoban_memory.logging_utils import save_episode, save_summary
from sokoban_memory.memory import HeuristicMemory, MemoryRenderConfig, build_raw_memory_bank
from sokoban_memory.metrics import summarize_results
from sokoban_memory.prompts import level_metadata
from sokoban_memory.reflection import generate_reflection_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen V2 Sokoban memory banks.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--levels", default="levels/v2_pilot.json")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results/v2_memory_build")
    parser.add_argument("--raw_memory_path", default="memory_banks/raw_failures.json")
    parser.add_argument("--heuristic_memory_path", default="memory_banks/reflection_heuristics.json")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api_key_env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--max_llm_calls", type=int, default=50)
    parser.add_argument("--max_memory_items", type=int, default=3)
    parser.add_argument("--max_steps_per_memory", type=int, default=6)
    parser.add_argument("--max_memory_chars", type=int, default=4000)
    parser.add_argument("--llm_cache_path", default=None)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--cache_namespace", default=DEFAULT_CACHE_NAMESPACE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_memory_bank_build(args)
    print(json.dumps(summary, indent=2))


def run_memory_bank_build(
    args: argparse.Namespace,
    agent_client: object | None = None,
    reflection_client: object | None = None,
) -> dict:
    all_levels = load_levels(args.levels)
    levels = [level for level in all_levels if level.split == "train"]
    if not levels:
        raise ValueError("Memory bank build requires at least one level with split='train'.")
    source_train_level_ids = [level.level_id for level in levels]
    memory_config = MemoryRenderConfig(
        max_memory_items=args.max_memory_items,
        max_steps_per_memory=args.max_steps_per_memory,
        max_memory_chars=args.max_memory_chars,
    )
    agent = make_agent(
        "no_memory",
        seed=args.seed,
        model=args.model,
        api_key_env=args.api_key_env,
        client=agent_client,
        max_llm_calls=args.max_llm_calls,
        memory_config=memory_config,
        llm_cache_path=args.llm_cache_path,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        cache_namespace=args.cache_namespace,
    )

    results = []
    results_dir = Path(args.results_dir)
    for episode_idx in range(args.episodes):
        level = levels[episode_idx % len(levels)]
        episode_seed = args.seed + episode_idx
        result = run_episode(SokobanEnv(level, seed=episode_seed), agent, args.max_steps, episode_seed)
        save_episode(result, results_dir)
        results.append(result)
        if result.budget_exhausted:
            break

    raw_memory = build_raw_memory_bank(
        results,
        source_metadata={
            "builder": "build_memory_bank.py",
            "source_agent": "no_memory",
            "policy_mode": getattr(agent, "policy_mode", None),
            "model": args.model,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "cache_namespace": args.cache_namespace,
            "levels": args.levels,
            "source_train_level_ids": source_train_level_ids,
            "seed": args.seed,
            "requested_episodes": args.episodes,
            "completed_episodes": len(results),
            "max_steps": args.max_steps,
            "memory_caps": memory_config.to_dict(),
        },
        max_steps_per_memory=args.max_steps_per_memory,
    )
    raw_path = Path(args.raw_memory_path)
    raw_memory.save(raw_path)

    remaining_calls = None
    if args.max_llm_calls is not None:
        remaining_calls = max(0, args.max_llm_calls - getattr(agent, "llm_call_count", 0))
    if raw_memory.episodes:
        heuristic_memory = generate_reflection_memory(
            raw_memory,
            model=args.model,
            api_key_env=args.api_key_env,
            client=reflection_client,
            llm_cache_path=args.llm_cache_path,
            max_llm_calls=remaining_calls,
            memory_config=memory_config,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            cache_namespace=args.cache_namespace,
        )
    else:
        heuristic_memory = HeuristicMemory(
            heuristics=[],
            source_metadata={
                "source_raw_memory_hash": raw_memory.memory_hash,
                "reflection_model": args.model,
                "no_failed_episodes": True,
                "source_train_level_ids": source_train_level_ids,
                "memory_caps": memory_config.to_dict(),
            },
        )
    heuristic_path = Path(args.heuristic_memory_path)
    heuristic_memory.save(heuristic_path)

    summary = summarize_results(results)
    level_meta = level_metadata(levels)
    summary.update(
        {
            "requested_episodes": args.episodes,
            **level_meta,
            "agent_type": "no_memory",
            "model": args.model,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "cache_namespace": args.cache_namespace,
            "levels": args.levels,
            "source_train_level_ids": source_train_level_ids,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "max_llm_calls": args.max_llm_calls,
            "raw_memory_path": str(raw_path),
            "raw_memory_hash": raw_memory.memory_hash,
            "raw_memory_item_count": len(raw_memory.episodes),
            "heuristic_memory_path": str(heuristic_path),
            "heuristic_memory_hash": heuristic_memory.memory_hash,
            "heuristic_count": len(heuristic_memory.heuristics),
            "memory_caps": memory_config.to_dict(),
        }
    )
    save_summary(summary, results_dir)
    return summary


if __name__ == "__main__":
    main()
