import json
from collections import Counter, defaultdict

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


def test_v3_boxoban_balanced_suite_shape_and_metadata():
    raw = _load_raw_levels("levels/v3_boxoban_balanced.json")
    levels = load_levels("levels/v3_boxoban_balanced.json")
    by_id = {level.level_id: level for level in levels}
    ids = [item["level_id"] for item in raw]
    grid_keys = ["\n".join(item["grid"]) for item in raw]

    assert len(raw) == 48
    assert len(ids) == len(set(ids))
    assert len(grid_keys) == len(set(grid_keys))
    assert Counter(item["split"] for item in raw) == {"train": 24, "eval": 24}
    assert {
        (item["split"], item["source_split"])
        for item in raw
    } == {("train", "train"), ("eval", "valid")}

    expected_bucket_counts = {
        (split, family, bucket): 4
        for split in ("train", "eval")
        for family in ("unfiltered", "medium")
        for bucket in ("open", "middle", "constrained")
    }
    assert Counter(
        (item["split"], item["source_family"], item["difficulty_bucket"])
        for item in raw
    ) == expected_bucket_counts

    for item in raw:
        level = by_id[item["level_id"]]
        assert level.width == 10
        assert level.height == 10
        assert item["num_boxes"] == 4
        assert item["num_targets"] == 4
        assert len(level.boxes) == 4
        assert len(level.targets) == 4
        assert "boxoban" in level.tags
        assert not any(tag.startswith("easy_") for tag in level.tags)
        assert item["source"].startswith("google-deepmind/boxoban-levels ")
        assert item["source_file"].endswith(".txt")
        assert item["canonical_grid_hash"]
        assert item["solver_status"] in {"solved", "capped", "unsolved"}
        assert item.get("reference_solution") is None
        assert item.get("optimal_steps") is None


def test_v3_boxoban_ood_suite_shape_and_split():
    raw = _load_raw_levels("levels/v3_boxoban_ood.json")
    balanced_hashes = {item["canonical_grid_hash"] for item in _load_raw_levels("levels/v3_boxoban_balanced.json")}
    ids = [item["level_id"] for item in raw]
    hashes = [item["canonical_grid_hash"] for item in raw]

    assert len(raw) == 16
    assert len(ids) == len(set(ids))
    assert len(hashes) == len(set(hashes))
    assert not balanced_hashes.intersection(hashes)
    assert Counter(item["source_family"] for item in raw) == {"hard": 8, "medium": 8}
    assert all(item["split"] == "eval" for item in raw)
    assert all("ood" in item["tags"] for item in raw)


def test_v3_boxoban_train_eval_difficulty_summaries_are_matched():
    raw = _load_raw_levels("levels/v3_boxoban_balanced.json")
    grouped = defaultdict(list)
    for item in raw:
        grouped[(item["split"], item["source_family"])].append(item)

    for family in ("unfiltered", "medium"):
        train = grouped[("train", family)]
        eval_levels = grouped[("eval", family)]
        assert abs(_mean(train, "wall_density") - _mean(eval_levels, "wall_density")) <= 0.05
        assert abs(_mean(train, "player_reachable_ratio") - _mean(eval_levels, "player_reachable_ratio")) <= 0.12
        assert abs(_mean(train, "initial_legal_push_count") - _mean(eval_levels, "initial_legal_push_count")) <= 2.0
        assert abs(_mean(train, "difficulty_score") - _mean(eval_levels, "difficulty_score")) <= 0.08


def _load_raw_levels(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["levels"]


def _mean(items, key):
    return sum(item[key] for item in items) / len(items)
