from __future__ import annotations

import copy
import random
from typing import Any

from sokoban_memory.types import Action, Level, Position, StepResult

DIRECTIONS: dict[Action, tuple[int, int]] = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class SokobanEnv:
    def __init__(self, level: Level, seed: int | None = None):
        self.level = copy.deepcopy(level)
        self.rng = random.Random(seed)
        self.player = level.player
        self.boxes = set(level.boxes)

    def reset(self) -> str:
        self.player = self.level.player
        self.boxes = set(self.level.boxes)
        return self.render_text()

    def step(self, action: Action) -> StepResult:
        state_text = self.render_text()
        if action not in DIRECTIONS:
            raise ValueError(f"Unsupported action: {action}")

        dr, dc = DIRECTIONS[action]
        next_player = self.player.moved(dr, dc)
        reward = -0.1
        moved = False
        pushed_box = False
        hit = None

        if self._is_blocked_cell(next_player):
            hit = "wall_or_boundary"
        elif next_player in self.boxes:
            next_box = next_player.moved(dr, dc)
            if self._is_blocked_cell(next_box):
                hit = "box_blocked_by_wall_or_boundary"
            elif next_box in self.boxes:
                hit = "box_blocked_by_box"
            else:
                old_box_on_target = next_player in self.level.targets
                new_box_on_target = next_box in self.level.targets
                self.boxes.remove(next_player)
                self.boxes.add(next_box)
                self.player = next_player
                moved = True
                pushed_box = True
                if new_box_on_target and not old_box_on_target:
                    reward += 1.0
                elif old_box_on_target and not new_box_on_target:
                    reward -= 1.0
        else:
            self.player = next_player
            moved = True

        solved = self.is_solved()
        deadlocked, deadlock_reason = self.is_deadlocked()
        if solved:
            reward += 10.0
        elif deadlocked:
            reward -= 5.0

        next_state_text = self.render_text()
        return StepResult(
            state_text=state_text,
            next_state_text=next_state_text,
            action=action,
            parsed_action=action,
            reward=reward,
            done=solved or deadlocked,
            info={
                "moved": moved,
                "pushed_box": pushed_box,
                "hit": hit,
                "solved": solved,
                "deadlocked": deadlocked,
                "deadlock_reason": deadlock_reason,
                "legal_actions_after": self.legal_actions(),
            },
        )

    def legal_actions(self) -> list[Action]:
        return [action for action in DIRECTIONS if self._can_execute(action)]

    def render_text(self) -> str:
        rows = []
        for r in range(self.level.height):
            chars = []
            for c in range(self.level.width):
                pos = Position(r, c)
                if pos in self.level.walls:
                    char = "#"
                elif pos == self.player and pos in self.level.targets:
                    char = "+"
                elif pos == self.player:
                    char = "@"
                elif pos in self.boxes and pos in self.level.targets:
                    char = "*"
                elif pos in self.boxes:
                    char = "$"
                elif pos in self.level.targets:
                    char = "."
                else:
                    char = " "
                chars.append(char)
            rows.append("".join(chars))
        return "\n".join(rows)

    def is_solved(self) -> bool:
        return bool(self.boxes) and self.boxes.issubset(self.level.targets)

    def is_deadlocked(self) -> tuple[bool, str | None]:
        if self.is_solved():
            return False, None
        for box in self.boxes:
            if box in self.level.targets:
                continue
            up = self._is_blocked_cell(box.moved(-1, 0))
            down = self._is_blocked_cell(box.moved(1, 0))
            left = self._is_blocked_cell(box.moved(0, -1))
            right = self._is_blocked_cell(box.moved(0, 1))
            if (up or down) and (left or right):
                return True, f"box_at_non_target_corner:{box.row},{box.col}"
        return False, None

    def clone_state(self) -> dict[str, Any]:
        return {
            "player": {"row": self.player.row, "col": self.player.col},
            "boxes": [{"row": p.row, "col": p.col} for p in sorted(self.boxes)],
            "state_text": self.render_text(),
        }

    def _can_execute(self, action: Action) -> bool:
        dr, dc = DIRECTIONS[action]
        next_player = self.player.moved(dr, dc)
        if self._is_blocked_cell(next_player):
            return False
        if next_player not in self.boxes:
            return True
        next_box = next_player.moved(dr, dc)
        return not self._is_blocked_cell(next_box) and next_box not in self.boxes

    def _is_blocked_cell(self, pos: Position) -> bool:
        return (
            pos.row < 0
            or pos.row >= self.level.height
            or pos.col < 0
            or pos.col >= self.level.width
            or pos in self.level.walls
        )

