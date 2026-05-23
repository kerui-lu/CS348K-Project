"""Breadth-first Sokoban solver for offline reference plans."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from sokoban_memory.env import DIRECTIONS, SokobanEnv
from sokoban_memory.types import Action, Level, Position

ACTIONS: tuple[Action, ...] = ("Up", "Down", "Left", "Right")


@dataclass(frozen=True)
class SearchState:
    player: Position
    boxes: frozenset[Position]


@dataclass
class SolveResult:
    solution: list[Action] | None
    nodes_expanded: int
    reason: str


def _state_key(state: SearchState) -> tuple[int, int, frozenset[Position]]:
    return (state.player.row, state.player.col, state.boxes)


def _is_blocked(level: Level, pos: Position) -> bool:
    return (
        pos.row < 0
        or pos.col < 0
        or pos.row >= level.height
        or pos.col >= level.width
        or pos in level.walls
    )


def _is_solved(level: Level, boxes: frozenset[Position]) -> bool:
    return bool(boxes) and boxes.issubset(level.targets)


def _successors(level: Level, state: SearchState) -> list[tuple[Action, SearchState]]:
    out: list[tuple[Action, SearchState]] = []
    for action in ACTIONS:
        dr, dc = DIRECTIONS[action]
        next_player = state.player.moved(dr, dc)
        if _is_blocked(level, next_player):
            continue
        boxes = set(state.boxes)
        if next_player in boxes:
            next_box = next_player.moved(dr, dc)
            if _is_blocked(level, next_box) or next_box in boxes:
                continue
            boxes.remove(next_player)
            boxes.add(next_box)
        next_state = SearchState(player=next_player, boxes=frozenset(boxes))
        out.append((action, next_state))
    return out


def _is_deadlocked(level: Level, state: SearchState) -> bool:
    env = SokobanEnv(level)
    env.player = state.player
    env.boxes = set(state.boxes)
    deadlocked, _ = env.is_deadlocked()
    return deadlocked


def bfs_solve(
    level: Level,
    *,
    max_nodes: int = 5_000_000,
    prune_deadlocks: bool = False,
) -> SolveResult:
    """Return a shortest primitive-action plan under this env's rules."""
    start = SearchState(player=level.player, boxes=frozenset(level.boxes))
    if _is_solved(level, start.boxes):
        return SolveResult(solution=[], nodes_expanded=0, reason="already_solved")

    queue: deque[tuple[SearchState, list[Action]]] = deque([(start, [])])
    seen = {_state_key(start)}
    nodes = 0

    while queue:
        state, path = queue.popleft()
        nodes += 1
        if nodes > max_nodes:
            return SolveResult(solution=None, nodes_expanded=nodes, reason="node_limit")

        for action, next_state in _successors(level, state):
            if prune_deadlocks and _is_deadlocked(level, next_state):
                continue
            if _is_solved(level, next_state.boxes):
                return SolveResult(
                    solution=path + [action],
                    nodes_expanded=nodes,
                    reason="solved",
                )
            key = _state_key(next_state)
            if key in seen:
                continue
            seen.add(key)
            queue.append((next_state, path + [action]))

    return SolveResult(solution=None, nodes_expanded=nodes, reason="exhausted")


def verify_solution(level: Level, solution: list[Action]) -> bool:
    env = SokobanEnv(level)
    env.reset()
    for action in solution:
        if action not in env.legal_actions():
            return False
        env.step(action)
    return env.is_solved()
