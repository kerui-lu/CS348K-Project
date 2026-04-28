from __future__ import annotations

from typing import Any


def reflect_on_failure(trajectory: list[dict[str, Any]]) -> list[str]:
    heuristics = [
        "Do not push a box into a non-target corner.",
        "Avoid pushing a box against a wall unless the target is along that wall.",
        "Before pushing a box, check whether the box can still reach a target.",
    ]
    if any(step.get("info", {}).get("deadlocked") for step in trajectory):
        heuristics.insert(0, "The latest failure ended in a detected deadlock; avoid the final push pattern.")
    return heuristics

