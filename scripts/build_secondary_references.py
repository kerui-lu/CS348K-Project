#!/usr/bin/env python3
"""Solve levels locally and write secondary reference plans (no primary ref reuse)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sokoban_memory.levels import load_levels
from sokoban_memory.solver import bfs_solve, verify_solution

DEFAULT_OUTPUT = "levels/v2_pilot_secondary_references.json"


def _select_levels(levels, *, only_missing_primary: bool, level_ids: set[str] | None, tag: str | None):
    selected = []
    for level in levels:
        if level_ids and level.level_id not in level_ids:
            continue
        if tag and tag not in level.tags:
            continue
        if only_missing_primary and level.reference_solution:
            continue
        selected.append(level)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="levels/v2_pilot.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only-missing-primary",
        action="store_true",
        help="solve only levels without reference_solution (default targets boxoban_medium)",
    )
    parser.add_argument("--tag", default=None, help="only levels containing this tag")
    parser.add_argument("--level-id", action="append", default=[], dest="level_ids")
    parser.add_argument("--max-nodes", type=int, default=5_000_000)
    parser.add_argument(
        "--prune-deadlocks",
        action="store_true",
        help="skip successor states that look deadlocked (faster but can miss solutions)",
    )
    parser.add_argument(
        "--include-primary-comparison",
        action="store_true",
        help="record whether secondary matches primary length/path when primary exists",
    )
    args = parser.parse_args()

    levels = load_levels(args.levels)
    tag = args.tag
    only_missing = args.only_missing_primary
    if not args.level_ids and not tag and not only_missing:
        only_missing = True
        tag = "boxoban_medium"

    selected = _select_levels(
        levels,
        only_missing_primary=only_missing,
        level_ids=set(args.level_ids) if args.level_ids else None,
        tag=tag,
    )
    if not selected:
        print("No levels matched selection.", file=sys.stderr)
        sys.exit(1)

    solutions: dict[str, dict] = {}
    failures = 0
    for level in selected:
        print(f"solving {level.level_id} ...", flush=True)
        result = bfs_solve(
            level,
            max_nodes=args.max_nodes,
            prune_deadlocks=args.prune_deadlocks,
        )
        entry: dict = {
            "status": "solved" if result.solution is not None else "failed",
            "reason": result.reason,
            "nodes_expanded": result.nodes_expanded,
            "solver": "bfs",
        }
        if result.solution is not None:
            verified = verify_solution(level, result.solution)
            entry["reference_solution"] = result.solution
            entry["steps"] = len(result.solution)
            entry["verified_in_env"] = verified
            if not verified:
                entry["status"] = "failed"
                entry["reason"] = "verification_failed"
                failures += 1
            elif args.include_primary_comparison and level.reference_solution:
                primary = level.reference_solution
                entry["primary_steps"] = len(primary)
                entry["same_length_as_primary"] = len(primary) == len(result.solution)
                entry["same_path_as_primary"] = primary == result.solution
        else:
            failures += 1
        solutions[level.level_id] = entry
        status = entry["status"]
        steps = entry.get("steps", "-")
        print(f"  -> {status} steps={steps} nodes={result.nodes_expanded}")

    payload = {
        "source_levels": str(args.levels),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "solver": "bfs",
        "selection": {
            "only_missing_primary": only_missing,
            "tag": tag,
            "level_ids": args.level_ids,
        },
        "solutions": solutions,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path} ({len(solutions)} levels, {failures} failures)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
