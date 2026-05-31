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
from sokoban_memory.levels import load_levels
from sokoban_memory.memory import HeuristicMemory, MemoryRenderConfig, RawTrajectoryMemory
from sokoban_memory.experiment import (
    SAME_LEVEL_ITERATIVE_CONDITIONS,
    run_experiment,
    run_same_level_iterative_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CS348K Sokoban memory prototype experiments.")
    parser.add_argument("--agent", default="rule_based", choices=[
        "rule_based",
        "no_memory",
        "raw",
        "raw_trajectory",
        "raw_trajectory_memory",
        "reflection",
        "reflection_heuristic",
        "generic_retry_feedback",
        "verifier_summary_retry",
        "heuristic_same_level_iterative",
        "llm",
    ])
    parser.add_argument("--experiment_mode", default="standard", choices=["standard", "same_level_iterative"])
    parser.add_argument("--condition", default=None, choices=sorted(SAME_LEVEL_ITERATIVE_CONDITIONS))
    parser.add_argument("--attempt_budget", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--levels", default="levels/simple.json")
    parser.add_argument("--level_split", default=None, choices=["train", "eval", "unspecified"])
    parser.add_argument(
        "--level_id",
        action="append",
        default=None,
        help="Run only the specified level id. May be passed multiple times.",
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_repair_attempts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--memory_path", default=None)
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
    parser = build_parser()
    args = parser.parse_args()

    levels = load_levels(args.levels)
    if args.level_split:
        levels = [level for level in levels if level.split == args.level_split]
        if not levels:
            raise ValueError(f"No levels found for split: {args.level_split}")
    if args.level_id:
        requested_level_ids = set(args.level_id)
        levels = [level for level in levels if level.level_id in requested_level_ids]
        missing_level_ids = requested_level_ids - {level.level_id for level in levels}
        if missing_level_ids:
            missing = ", ".join(sorted(missing_level_ids))
            raise ValueError(f"No levels found for level_id: {missing}")
    memory = None
    if args.agent.startswith("raw"):
        memory = RawTrajectoryMemory()
        if args.memory_path:
            memory.load(Path(args.memory_path))
    elif args.agent.startswith("reflection"):
        memory = HeuristicMemory()
        if args.memory_path:
            memory.load(Path(args.memory_path))

    memory_config = MemoryRenderConfig(
        max_memory_items=args.max_memory_items,
        max_steps_per_memory=args.max_steps_per_memory,
        max_memory_chars=args.max_memory_chars,
    )
    if args.experiment_mode == "same_level_iterative":
        condition = args.condition or _condition_from_agent(args.agent)
        summary = run_same_level_iterative_experiment(
            levels=levels,
            condition=condition,
            attempts_per_level=args.attempt_budget,
            max_steps=args.max_steps,
            seed=args.seed,
            results_dir=Path(args.results_dir),
            model=args.model,
            api_key_env=args.api_key_env,
            max_llm_calls=args.max_llm_calls,
            memory_config=memory_config,
            llm_cache_path=args.llm_cache_path,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            cache_namespace=args.cache_namespace,
            level_suite_path=args.levels,
        )
        print(json.dumps(summary, indent=2))
        return

    agent = make_agent(
        args.agent,
        seed=args.seed,
        memory=memory,
        model=args.model,
        api_key_env=args.api_key_env,
        max_llm_calls=args.max_llm_calls,
        memory_config=memory_config,
        llm_cache_path=args.llm_cache_path,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        cache_namespace=args.cache_namespace,
        memory_path=args.memory_path,
    )
    summary = run_experiment(
        levels=levels,
        agent=agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        results_dir=Path(args.results_dir),
        max_repair_attempts=args.max_repair_attempts,
        level_suite_path=args.levels,
    )
    print(json.dumps(summary, indent=2))


def _condition_from_agent(agent_name: str) -> str:
    if agent_name == "no_memory":
        return "single_shot_no_memory"
    if agent_name == "generic_retry_feedback":
        return "generic_retry_feedback"
    if agent_name == "verifier_summary_retry":
        return "verifier_summary_retry"
    if agent_name in {"raw", "raw_trajectory", "raw_trajectory_memory"}:
        return "raw_same_level_iterative"
    if agent_name in {"reflection", "reflection_heuristic", "heuristic_same_level_iterative"}:
        return "heuristic_same_level_iterative"
    raise ValueError(f"Agent {agent_name!r} cannot be used as a same-level iterative condition.")


if __name__ == "__main__":
    main()
