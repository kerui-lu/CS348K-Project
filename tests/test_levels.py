import json

import pytest

from sokoban_memory.env import SokobanEnv
from sokoban_memory.levels import load_levels


def test_reference_solution_is_parsed(tmp_path):
    level_path = tmp_path / "levels.json"
    level_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level_id": "ref_001",
                        "split": "train",
                        "tags": ["easy_simple_push"],
                        "optimal_steps": 1,
                        "reference_solution": ["Right"],
                        "grid": [
                            "#####",
                            "#@$.#",
                            "#####",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    level = load_levels(level_path)[0]

    assert level.reference_solution == ["Right"]


def test_reference_solution_rejects_invalid_actions(tmp_path):
    level_path = tmp_path / "levels.json"
    level_path.write_text(
        json.dumps(
            {
                "levels": [
                    {
                        "level_id": "bad_ref_001",
                        "optimal_steps": 1,
                        "reference_solution": ["Jump"],
                        "grid": [
                            "#####",
                            "#@$.#",
                            "#####",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference_solution contains unsupported actions"):
        load_levels(level_path)


def test_v2_pilot_levels_have_balanced_splits_and_valid_reference_solutions():
    levels = load_levels("levels/v2_pilot.json")
    train_levels = [level for level in levels if level.split == "train"]
    eval_levels = [level for level in levels if level.split == "eval"]

    assert len(train_levels) >= 6
    assert len(eval_levels) >= 6

    for level in levels:
        assert level.reference_solution is not None
        assert len(level.reference_solution) == level.optimal_steps

        env = SokobanEnv(level)
        env.reset()
        for action in level.reference_solution:
            step = env.step(action)
        assert env.is_solved(), level.level_id
        assert step.done is True
        assert step.info["solved"] is True
