import random

from sokoban_memory.action_parser import choose_fallback_action, parse_action
from sokoban_memory.agents import BaseAgent
from sokoban_memory.env import SokobanEnv
from sokoban_memory.experiment import run_episode
from sokoban_memory.types import Level, Position


def test_parse_action_variants():
    assert parse_action("Up") == "Up"
    assert parse_action("move up") == "Up"
    assert parse_action("Action: Left") == "Left"


def test_parse_action_rejects_invalid_or_ambiguous_text():
    assert parse_action("teleport") is None
    assert parse_action("up then left") is None


def test_choose_fallback_action_is_legal():
    action = choose_fallback_action(["Left", "Right"], random.Random(0))
    assert action in {"Left", "Right"}


class InvalidAgent(BaseAgent):
    agent_type = "invalid_agent"

    def select_action(self, state_text, context):
        return "teleport"


def test_invalid_action_fallback_increments_count():
    level = Level(
        level_id="invalid_fallback",
        width=5,
        height=4,
        walls={
            Position(0, 0), Position(0, 1), Position(0, 2), Position(0, 3), Position(0, 4),
            Position(3, 0), Position(3, 1), Position(3, 2), Position(3, 3), Position(3, 4),
            Position(1, 0), Position(2, 0), Position(1, 4), Position(2, 4),
        },
        targets={Position(1, 3)},
        boxes={Position(2, 2)},
        player=Position(1, 1),
    )
    result = run_episode(SokobanEnv(level), InvalidAgent(), max_steps=1, seed=1)
    assert result.invalid_move_count == 1
    assert result.trajectory[0]["info"]["invalid_reason"] == "invalid_or_illegal_action"

