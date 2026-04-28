from __future__ import annotations

import random
import re

from sokoban_memory.types import Action

ACTIONS: tuple[Action, ...] = ("Up", "Down", "Left", "Right")
_ACTION_BY_LOWER = {action.lower(): action for action in ACTIONS}


def parse_action(raw_output: str) -> Action | None:
    text = raw_output.strip().lower()
    if text in _ACTION_BY_LOWER:
        return _ACTION_BY_LOWER[text]

    matches = re.findall(r"\b(up|down|left|right)\b", text)
    if len(matches) == 1:
        return _ACTION_BY_LOWER[matches[0]]
    return None


def choose_fallback_action(legal_actions: list[Action], rng: random.Random) -> Action:
    if not legal_actions:
        raise ValueError("No legal actions available for fallback.")
    return rng.choice(legal_actions)

