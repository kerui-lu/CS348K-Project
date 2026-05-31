from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Iterable

from sokoban_memory.env import DIRECTIONS
from sokoban_memory.types import Action, Level, Position


@dataclass(frozen=True)
class SolverResult:
    status: str
    min_pushes: int | None = None
    min_steps: int | None = None
    explored_states: int = 0


def estimate_solution_cost(
    level: Level,
    *,
    max_states: int = 50_000,
    max_pushes: int = 120,
) -> SolverResult:
    """Estimate exact min pushes/steps within a bounded push-state search.

    This is used only for benchmark metadata. If the search cap is reached, the
    caller still gets a deterministic structural suite with solver_status=capped.
    """

    walls = {(pos.row, pos.col) for pos in level.walls}
    targets = frozenset((pos.row, pos.col) for pos in level.targets)
    initial_boxes = tuple(sorted((pos.row, pos.col) for pos in level.boxes))
    initial_player = (level.player.row, level.player.col)
    height = level.height
    width = level.width

    if set(initial_boxes).issubset(targets):
        return SolverResult(status="solved", min_pushes=0, min_steps=0, explored_states=1)

    tie_breaker = count()
    queue: list[tuple[int, int, int, tuple[int, int], tuple[tuple[int, int], ...]]] = []
    heappush(queue, (0, 0, next(tie_breaker), initial_player, initial_boxes))
    best: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], tuple[int, int]] = {}
    explored_states = 0

    while queue and explored_states < max_states:
        pushes, steps, _counter, player, boxes = heappop(queue)
        state_key = (player, boxes)
        if best.get(state_key, (10**9, 10**9)) < (pushes, steps):
            continue
        explored_states += 1

        if set(boxes).issubset(targets):
            return SolverResult(
                status="solved",
                min_pushes=pushes,
                min_steps=steps,
                explored_states=explored_states,
            )
        if pushes >= max_pushes:
            continue

        box_set = set(boxes)
        reachable = _reachable_distances(player, box_set, walls, height, width)
        for box in boxes:
            for action, (dr, dc) in DIRECTIONS.items():
                standing = (box[0] - dr, box[1] - dc)
                destination = (box[0] + dr, box[1] + dc)
                if standing not in reachable:
                    continue
                if _blocked(destination, box_set, walls, height, width):
                    continue

                new_box_set = set(boxes)
                new_box_set.remove(box)
                new_box_set.add(destination)
                new_boxes = tuple(sorted(new_box_set))
                new_player = box
                new_pushes = pushes + 1
                new_steps = steps + reachable[standing] + 1
                new_state_key = (new_player, new_boxes)
                if best.get(new_state_key, (10**9, 10**9)) <= (new_pushes, new_steps):
                    continue
                best[new_state_key] = (new_pushes, new_steps)
                heappush(queue, (new_pushes, new_steps, next(tie_breaker), new_player, new_boxes))

    status = "capped" if queue else "unsolved"
    return SolverResult(status=status, explored_states=explored_states)


def _reachable_distances(
    start: tuple[int, int],
    boxes: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    height: int,
    width: int,
) -> dict[tuple[int, int], int]:
    queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    distances = {start: 0}
    while queue:
        current, distance = queue.popleft()
        for _action, (dr, dc) in DIRECTIONS.items():
            next_pos = (current[0] + dr, current[1] + dc)
            if next_pos in distances or _blocked(next_pos, boxes, walls, height, width):
                continue
            distances[next_pos] = distance + 1
            queue.append((next_pos, distance + 1))
    return distances


def legal_pushes(level: Level) -> list[dict[str, object]]:
    walls = {(pos.row, pos.col) for pos in level.walls}
    boxes = {(pos.row, pos.col) for pos in level.boxes}
    reachable = _reachable_distances((level.player.row, level.player.col), boxes, walls, level.height, level.width)
    pushes = []
    for box in sorted(boxes):
        for action, (dr, dc) in DIRECTIONS.items():
            standing = (box[0] - dr, box[1] - dc)
            destination = (box[0] + dr, box[1] + dc)
            if standing not in reachable:
                continue
            if _blocked(destination, boxes, walls, level.height, level.width):
                continue
            pushes.append(
                {
                    "box": [box[0], box[1]],
                    "push": action,
                    "stand": [standing[0], standing[1]],
                    "dest": [destination[0], destination[1]],
                }
            )
    return pushes


def reachable_floor_count(level: Level) -> int:
    walls = {(pos.row, pos.col) for pos in level.walls}
    boxes = {(pos.row, pos.col) for pos in level.boxes}
    reachable = _reachable_distances((level.player.row, level.player.col), boxes, walls, level.height, level.width)
    return len(reachable)


def _blocked(
    pos: tuple[int, int],
    boxes: Iterable[tuple[int, int]],
    walls: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    row, col = pos
    return row < 0 or row >= height or col < 0 or col >= width or pos in walls or pos in boxes
