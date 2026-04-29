from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sokoban_memory.types import Level, Position


def load_levels(path: str | Path) -> list[Level]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_levels = data["levels"] if isinstance(data, dict) and "levels" in data else data
    if not isinstance(raw_levels, list):
        raise ValueError("Level file must contain a list or an object with a 'levels' list.")
    return [_parse_level(item) for item in raw_levels]


def _parse_level(raw: dict[str, Any]) -> Level:
    level_id = str(raw["level_id"])
    grid = raw["grid"]
    tags = raw.get("tags", [])
    split = str(raw.get("split", "unspecified"))
    if not grid or not all(isinstance(row, str) for row in grid):
        raise ValueError(f"{level_id}: grid must be a non-empty list of strings.")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError(f"{level_id}: tags must be a list of strings.")
    if split not in {"train", "eval", "unspecified"}:
        raise ValueError(f"{level_id}: split must be train, eval, or unspecified.")

    height = len(grid)
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError(f"{level_id}: all grid rows must have the same width.")

    walls: set[Position] = set()
    targets: set[Position] = set()
    boxes: set[Position] = set()
    player: Position | None = None

    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            pos = Position(r, c)
            if char == "#":
                walls.add(pos)
            elif char == ".":
                targets.add(pos)
            elif char == "$":
                boxes.add(pos)
            elif char == "*":
                boxes.add(pos)
                targets.add(pos)
            elif char == "@":
                _ensure_single_player(level_id, player)
                player = pos
            elif char == "+":
                _ensure_single_player(level_id, player)
                player = pos
                targets.add(pos)
            elif char == " ":
                continue
            else:
                raise ValueError(f"{level_id}: unsupported grid character {char!r}.")

    if player is None:
        raise ValueError(f"{level_id}: expected exactly one player.")
    if not boxes:
        raise ValueError(f"{level_id}: expected at least one box.")
    if len(targets) < len(boxes):
        raise ValueError(f"{level_id}: expected at least as many targets as boxes.")

    return Level(
        level_id=level_id,
        width=width,
        height=height,
        walls=walls,
        targets=targets,
        boxes=boxes,
        player=player,
        tags=tags,
        split=split,
    )


def _ensure_single_player(level_id: str, player: Position | None) -> None:
    if player is not None:
        raise ValueError(f"{level_id}: expected exactly one player.")
