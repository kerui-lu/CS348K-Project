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
        for box in sorted(self.boxes):
            if box in self.level.targets:
                continue
            if self._box_in_static_corner(box):
                return True, f"box_at_non_target_corner:{box.row},{box.col}"
        for box in sorted(self.boxes):
            if box in self.level.targets:
                continue
            reason = self._box_against_wall_no_target_no_exit(box)
            if reason:
                return True, reason
        reason = self._box_in_2x2_freeze()
        if reason:
            return True, reason
        reason = self._two_box_freeze()
        if reason:
            return True, reason
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

    def _is_static_free_cell(self, pos: Position) -> bool:
        return not self._is_blocked_cell(pos)

    def _box_in_static_corner(self, box: Position) -> bool:
        up = self._is_blocked_cell(box.moved(-1, 0))
        down = self._is_blocked_cell(box.moved(1, 0))
        left = self._is_blocked_cell(box.moved(0, -1))
        right = self._is_blocked_cell(box.moved(0, 1))
        return (up or down) and (left or right)

    def _box_against_wall_no_target_no_exit(self, box: Position) -> str | None:
        wall_directions: tuple[tuple[str, tuple[int, int]], ...] = (
            ("Up", (-1, 0)),
            ("Down", (1, 0)),
            ("Left", (0, -1)),
            ("Right", (0, 1)),
        )
        for wall_name, (wall_dr, wall_dc) in wall_directions:
            wall_side = box.moved(wall_dr, wall_dc)
            if not self._is_blocked_cell(wall_side):
                continue

            segment = self._wall_segment_with_possible_exit(box, wall_dr, wall_dc)
            if any(pos in self.level.targets for pos in segment):
                continue
            if any(self._wall_segment_position_has_exit(pos, wall_dr, wall_dc) for pos in segment):
                continue
            return f"box_against_wall_no_target_no_exit:{box.row},{box.col},wall={wall_name}"
        return None

    def _wall_segment_with_possible_exit(
        self,
        box: Position,
        wall_dr: int,
        wall_dc: int,
    ) -> list[Position]:
        parallel_dirs = ((0, -1), (0, 1)) if wall_dr else ((-1, 0), (1, 0))
        segment = [box]
        for dr, dc in parallel_dirs:
            current = box.moved(dr, dc)
            while self._is_static_free_cell(current):
                segment.append(current)
                if not self._is_blocked_cell(current.moved(wall_dr, wall_dc)):
                    break
                current = current.moved(dr, dc)
        return sorted(set(segment))

    def _wall_segment_position_has_exit(self, pos: Position, wall_dr: int, wall_dc: int) -> bool:
        wall_side = pos.moved(wall_dr, wall_dc)
        away_side = pos.moved(-wall_dr, -wall_dc)
        return self._is_static_free_cell(wall_side) and self._is_static_free_cell(away_side)

    def _box_in_2x2_freeze(self) -> str | None:
        candidate_blocks: set[tuple[int, int]] = set()
        for box in self.boxes:
            candidate_blocks.update(
                {
                    (box.row - 1, box.col - 1),
                    (box.row - 1, box.col),
                    (box.row, box.col - 1),
                    (box.row, box.col),
                }
            )
        for top, left in sorted(candidate_blocks):
            cells = [
                Position(top, left),
                Position(top, left + 1),
                Position(top + 1, left),
                Position(top + 1, left + 1),
            ]
            included_boxes = sorted(pos for pos in cells if pos in self.boxes)
            non_target_boxes = [pos for pos in included_boxes if pos not in self.level.targets]
            if not non_target_boxes:
                continue
            if all(self._is_blocked_cell(pos) or pos in self.boxes for pos in cells):
                box = non_target_boxes[0]
                return f"box_in_2x2_freeze:{box.row},{box.col}"
        return None

    def _two_box_freeze(self) -> str | None:
        for box in sorted(self.boxes):
            if box in self.level.targets:
                continue
            for dr, dc in ((0, 1), (1, 0)):
                other = box.moved(dr, dc)
                if other not in self.boxes or other in self.level.targets:
                    continue
                if self._box_has_theoretical_push(box, other):
                    continue
                if self._box_has_theoretical_push(other, box):
                    continue
                first, second = sorted([box, other])
                return f"two_box_freeze:{first.row},{first.col};{second.row},{second.col}"
        return None

    def _box_has_theoretical_push(self, box: Position, paired_box: Position) -> bool:
        for dr, dc in DIRECTIONS.values():
            destination = box.moved(dr, dc)
            player_side = box.moved(-dr, -dc)
            if destination == paired_box or player_side == paired_box:
                continue
            if self._is_blocked_cell(destination) or self._is_blocked_cell(player_side):
                continue
            return True
        return False
