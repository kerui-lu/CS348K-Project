from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sokoban_memory.levels import _parse_level
from sokoban_memory.solver import estimate_solution_cost, legal_pushes, reachable_floor_count
from sokoban_memory.types import Level

BOXOBAN_VALID_CHARS = {"#", " ", "@", "$", ".", "*", "+"}


@dataclass(frozen=True)
class BoxobanPuzzle:
    source_index: int
    grid: list[str]


def parse_boxoban_text(text: str) -> list[BoxobanPuzzle]:
    puzzles: list[BoxobanPuzzle] = []
    current_index: int | None = None
    current_grid: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith(";"):
            _append_puzzle(puzzles, current_index, current_grid)
            current_grid = []
            current_index = _parse_separator_index(line)
            continue
        if line:
            current_grid.append(line)

    _append_puzzle(puzzles, current_index, current_grid)
    return puzzles


def load_boxoban_text_file(path: str | Path) -> list[BoxobanPuzzle]:
    return parse_boxoban_text(Path(path).read_text(encoding="utf-8"))


def canonical_grid_key(grid: list[str]) -> str:
    return "\n".join(grid)


def canonical_grid_hash(grid: list[str]) -> str:
    return hashlib.sha256(canonical_grid_key(grid).encode("utf-8")).hexdigest()[:16]


def grid_to_level(
    *,
    level_id: str,
    grid: list[str],
    split: str,
    tags: list[str],
    source: str,
) -> Level:
    return _parse_level(
        {
            "level_id": level_id,
            "grid": grid,
            "split": split,
            "tags": tags,
            "source": source,
        }
    )


def level_entry(
    *,
    level_id: str,
    grid: list[str],
    split: str,
    tags: list[str],
    source: str,
    source_family: str,
    source_split: str,
    source_file: str,
    source_index: int,
    difficulty_bucket: str,
    compute_solver: bool = True,
    solver_max_states: int = 50_000,
) -> dict[str, Any]:
    level = grid_to_level(level_id=level_id, grid=grid, split=split, tags=tags, source=source)
    metadata = structural_features(level)
    if compute_solver:
        solver_result = estimate_solution_cost(level, max_states=solver_max_states)
        metadata.update(
            {
                "solver_min_pushes": solver_result.min_pushes,
                "solver_min_steps": solver_result.min_steps,
                "solver_status": solver_result.status,
                "solver_explored_states": solver_result.explored_states,
            }
        )
    else:
        metadata.update(
            {
                "solver_min_pushes": None,
                "solver_min_steps": None,
                "solver_status": "not_run",
                "solver_explored_states": 0,
            }
        )

    metadata.update(
        {
            "source_family": source_family,
            "source_split": source_split,
            "source_file": source_file,
            "source_index": source_index,
            "difficulty_bucket": difficulty_bucket,
            "canonical_grid_hash": canonical_grid_hash(grid),
        }
    )
    metadata["difficulty_score"] = difficulty_score(metadata)

    return {
        "level_id": level_id,
        "split": split,
        "tags": tags,
        "source": source,
        "grid": grid,
        **metadata,
    }


def structural_features(level: Level) -> dict[str, Any]:
    total_cells = level.width * level.height
    static_free_cells = total_cells - len(level.walls)
    reachable_count = reachable_floor_count(level)
    legal_push_count = len(legal_pushes(level))
    return {
        "num_boxes": len(level.boxes),
        "num_targets": len(level.targets),
        "wall_density": round(len(level.walls) / total_cells, 4) if total_cells else 0.0,
        "player_reachable_ratio": round(reachable_count / static_free_cells, 4) if static_free_cells else 0.0,
        "initial_legal_push_count": legal_push_count,
    }


def difficulty_score(features: dict[str, Any]) -> float:
    reachable = _float(features.get("player_reachable_ratio"))
    wall_density = _float(features.get("wall_density"))
    legal_pushes_value = min(_float(features.get("initial_legal_push_count")), 12.0)
    solver_pushes = features.get("solver_min_pushes")
    solver_component = 0.6
    if isinstance(solver_pushes, int) and solver_pushes >= 0:
        solver_component = min(solver_pushes, 80) / 80
    score = (
        0.35 * (1.0 - reachable)
        + 0.25 * wall_density
        + 0.25 * (1.0 - legal_pushes_value / 12.0)
        + 0.15 * solver_component
    )
    return round(score, 4)


def lightweight_difficulty_score(features: dict[str, Any]) -> float:
    draft = dict(features)
    draft["solver_min_pushes"] = None
    return difficulty_score(draft)


def is_standard_boxoban_grid(grid: list[str]) -> bool:
    if len(grid) != 10 or any(len(row) != 10 for row in grid):
        return False
    if any(char not in BOXOBAN_VALID_CHARS for row in grid for char in row):
        return False
    level = grid_to_level(
        level_id="candidate",
        grid=grid,
        split="unspecified",
        tags=[],
        source="candidate",
    )
    return len(level.boxes) == 4 and len(level.targets) == 4


def _append_puzzle(
    puzzles: list[BoxobanPuzzle],
    source_index: int | None,
    grid: list[str],
) -> None:
    if not grid:
        return
    if source_index is None:
        source_index = len(puzzles)
    puzzles.append(BoxobanPuzzle(source_index=source_index, grid=list(grid)))


def _parse_separator_index(line: str) -> int:
    raw_index = line[1:].strip()
    if not raw_index:
        return 0
    try:
        return int(raw_index.split()[0])
    except ValueError:
        return 0


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
