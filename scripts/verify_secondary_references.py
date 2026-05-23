#!/usr/bin/env python3
"""Replay secondary reference plans from v2_pilot_secondary_references.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sokoban_memory.levels import load_levels
from sokoban_memory.solver import verify_solution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="levels/v2_pilot.json")
    parser.add_argument(
        "--secondary",
        default="levels/v2_pilot_secondary_references.json",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.secondary).read_text(encoding="utf-8"))
    by_id = {level.level_id: level for level in load_levels(args.levels)}
    solutions = data.get("solutions", {})

    failures = 0
    for level_id, entry in solutions.items():
        if entry.get("status") != "solved":
            print(f"[SKIP] {level_id}: {entry.get('reason')}")
            failures += 1
            continue
        level = by_id.get(level_id)
        if level is None:
            print(f"[FAIL] {level_id}: unknown level")
            failures += 1
            continue
        path = entry["reference_solution"]
        ok = verify_solution(level, path)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {level_id} ({len(path)} steps)")
        if not ok:
            failures += 1

    if failures:
        sys.exit(1)
    print(f"\n{len(solutions) - failures}/{len(solutions)} verified")


if __name__ == "__main__":
    main()
