from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_memory.agents import DEFAULT_API_KEY_ENV, DEFAULT_LLM_MODEL, make_agent
from sokoban_memory.levels import load_levels
from sokoban_memory.memory import HeuristicMemory, RawTrajectoryMemory
from sokoban_memory.experiment import run_experiment


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
        "llm",
    ])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--levels", default="levels/simple.json")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--memory_path", default=None)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--api_key_env", default=DEFAULT_API_KEY_ENV)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    levels = load_levels(args.levels)
    memory = None
    if args.agent.startswith("raw"):
        memory = RawTrajectoryMemory()
        if args.memory_path:
            memory.load(Path(args.memory_path))
    elif args.agent.startswith("reflection"):
        memory = HeuristicMemory()
        if args.memory_path:
            memory.load(Path(args.memory_path))

    agent = make_agent(
        args.agent,
        seed=args.seed,
        memory=memory,
        model=args.model,
        api_key_env=args.api_key_env,
    )
    summary = run_experiment(
        levels=levels,
        agent=agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        results_dir=Path(args.results_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
