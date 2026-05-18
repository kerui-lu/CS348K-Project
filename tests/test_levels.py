import json

from sokoban_memory.levels import load_levels


def test_reference_solution_is_preserved_without_validation(tmp_path):
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

    level = load_levels(level_path)[0]

    assert level.reference_solution == ["Jump"]


def test_v2_pilot_levels_have_expanded_splits_and_no_duplicate_grids():
    levels = load_levels("levels/v2_pilot.json")
    train_levels = [level for level in levels if level.split == "train"]
    eval_levels = [level for level in levels if level.split == "eval"]
    level_ids = [level.level_id for level in levels]
    grid_keys = ["\n".join(level.grid) for level in levels]

    assert len(train_levels) >= 12
    assert len(eval_levels) >= 12
    assert len(levels) >= 24
    assert len(level_ids) == len(set(level_ids))
    assert len(grid_keys) == len(set(grid_keys))

    imported = [level for level in levels if "boxoban_medium" in level.tags]
    assert len([level for level in imported if level.split == "train"]) == 6
    assert len([level for level in imported if level.split == "eval"]) == 6

    for level in imported:
        assert level.width == 10
        assert level.height == 10
        assert level.optimal_steps is None
        assert level.reference_solution is None
        assert level.source.startswith("google-deepmind/boxoban-levels medium/")
