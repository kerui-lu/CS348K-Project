from sokoban_memory.memory import MemoryRenderConfig, RawTrajectoryMemory
from sokoban_memory.reflection import (
    build_same_level_reflection_prompt,
    build_v3_global_reflection_prompt,
    generate_reflection_memory,
    generate_same_level_reflection_memory,
    generate_v3_global_reflection_memory,
    parse_heuristics,
)


class FakeResponses:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeClient:
    def __init__(self, output_text: str):
        self.responses = FakeResponses(output_text)


def make_raw_memory():
    return RawTrajectoryMemory(
        source_metadata={"source_train_level_ids": ["train_trap"]},
        episodes=[
            {
                "level_id": "trap",
                "status": "deadlock",
                "step_count": 1,
                "total_reward": -5.1,
                "steps": [{"step": 0, "state": "#@$", "executed_action": "Right", "next_state": "# @$"}],
            }
        ]
    )


def make_same_level_raw_memory():
    return RawTrajectoryMemory(
        source_metadata={"memory_scope": "same_level", "source_level_ids": ["lvl_a"]},
        episodes=[
            {
                "level_id": "lvl_a",
                "status": "invalid_plan",
                "failure_reason": "required_push_position_unreachable",
                "failure_subtype": "unreachable_standing_cell",
                "failure_push_index": 2,
                "initial_board": "#####\n#@$.#\n#####",
                "board_before_failed_push": "#####\n#@$.#\n#####",
                "board_after_last_successful_push": "#####\n#@$.#\n#####",
                "push_execution_log": [
                    {
                        "push_index": 2,
                        "model_intent": {"box": [1, 2], "push": "Left"},
                        "status": "failed",
                        "resolved_box_before_push": {"row": 1, "col": 2},
                        "standing_cell_required": {"row": 1, "col": 3},
                        "destination_cell": {"row": 1, "col": 1},
                        "failure_subtype": "unreachable_standing_cell",
                    }
                ],
            }
        ],
    )


def test_same_level_reflection_prompt_is_failure_specific():
    raw = make_same_level_raw_memory()
    config = MemoryRenderConfig()
    prompt = build_same_level_reflection_prompt(raw, "lvl_a", config, version="v1_specific")
    assert "ONE specific Sokoban level" in prompt
    assert "lvl_a" in prompt
    # Concrete failure evidence must be present for the reflector to use.
    assert "standing_cell_required" in prompt
    assert "Never re-emit the exact push that just failed." in prompt

    hybrid = build_same_level_reflection_prompt(raw, "lvl_a", config, version="v3_hybrid_verifier")
    assert "literal restatement of the verifier" in hybrid

    complete = build_same_level_reflection_prompt(raw, "lvl_a", config, version="v2_complete_plan")
    assert "COMPLETE" in complete


def test_same_level_reflection_baseline_matches_legacy(tmp_path):
    raw = make_same_level_raw_memory()
    config = MemoryRenderConfig(max_memory_chars=2000)
    legacy = build_same_level_reflection_prompt(raw, "lvl_a", config, version="baseline")
    from sokoban_memory.reflection import build_reflection_prompt

    assert legacy == build_reflection_prompt(raw, config)


def test_generate_same_level_reflection_uses_cache_and_version(tmp_path):
    cache_path = tmp_path / "cache"
    client = FakeClient('["Box [1,2] cannot be pushed Left: standing cell [1,3] is blocked; push it Right instead."]')
    second = FakeClient('["should not run"]')
    mem = generate_same_level_reflection_memory(
        make_same_level_raw_memory(),
        level_id="lvl_a",
        version="v1_specific",
        client=client,
        llm_cache_path=str(cache_path),
        cache_namespace="test_same_level_reflection",
    )
    cached = generate_same_level_reflection_memory(
        make_same_level_raw_memory(),
        level_id="lvl_a",
        version="v1_specific",
        client=second,
        llm_cache_path=str(cache_path),
        max_llm_calls=0,
        cache_namespace="test_same_level_reflection",
    )
    assert mem.heuristics and "[1,3]" in mem.heuristics[0]
    assert cached.heuristics == mem.heuristics
    assert len(client.responses.calls) == 1
    assert second.responses.calls == []
    assert mem.source_metadata["same_level_reflection_version"] == "v1_specific"
    assert mem.source_metadata["reflection_prompt_version"] == "same_level_reflection_v1_specific"


def test_parse_heuristics_accepts_json_array_and_bullets():
    assert parse_heuristics('["Rule one.", "Rule two."]') == ["Rule one.", "Rule two."]
    assert parse_heuristics("- Rule one.\n2. Rule two.") == ["Rule one.", "Rule two."]


def test_reflection_generation_uses_cache(tmp_path):
    cache_path = tmp_path / "cache"
    first_client = FakeClient('["Do not push boxes into non-target corners."]')
    second_client = FakeClient('["This should not be called."]')

    first = generate_reflection_memory(
        make_raw_memory(),
        client=first_client,
        llm_cache_path=str(cache_path),
        memory_config=MemoryRenderConfig(max_memory_chars=1000),
    )
    second = generate_reflection_memory(
        make_raw_memory(),
        client=second_client,
        llm_cache_path=str(cache_path),
        max_llm_calls=0,
        memory_config=MemoryRenderConfig(max_memory_chars=1000),
    )

    assert first.heuristics == ["Do not push boxes into non-target corners."]
    assert second.heuristics == first.heuristics
    assert len(first_client.responses.calls) == 1
    assert second_client.responses.calls == []
    assert second.source_metadata["cache_hit"] is True
    assert second.source_metadata["source_train_level_ids"] == ["train_trap"]
    assert second.source_metadata["reflection_prompt_version"] == "reflection_v2"
    assert "generated_at_utc" in second.source_metadata


def make_v3_raw_memory():
    return RawTrajectoryMemory(
        source_metadata={"source_train_level_ids": ["train_a", "train_b"]},
        episodes=[
            {
                "level_id": "train_a",
                "status": "invalid_plan",
                "failure_reason": "required_push_position_unreachable",
                "failure_subtype": "unreachable_standing_cell",
                "failure_push_index": 2,
                "board_before_failed_push": "#####\n# @ #\n# $ #\n# . #\n#####",
                "board_after_last_successful_push": "#####\n# @ #\n# $ #\n# . #\n#####",
                "push_execution_log": [
                    {
                        "push_index": 2,
                        "model_intent": {"box": [2, 2], "push": "Down"},
                        "resolved_box_before_push": [2, 2],
                        "standing_cell_required": [1, 2],
                        "destination_cell": [3, 2],
                        "status": "failed",
                        "failure_subtype": "unreachable_standing_cell",
                    }
                ],
                "v3_attempt_trace": {
                    "source_family": "medium",
                    "difficulty_bucket": "open",
                    "best_boxes_on_targets": 0,
                    "final_boxes_on_targets": 0,
                },
            },
            {
                "level_id": "train_b",
                "status": "deadlock",
                "failure_reason": "box_in_2x2_freeze:3,4",
                "failure_subtype": "deadlock",
                "failure_push_index": 1,
                "board_before_failed_push": "#####\n#@$ #\n#   #\n# . #\n#####",
                "board_after_last_successful_push": "#####\n# @$#\n#   #\n# . #\n#####",
                "push_execution_log": [
                    {
                        "push_index": 1,
                        "model_intent": {"box": [1, 2], "push": "Right"},
                        "resolved_box_before_push": [1, 2],
                        "standing_cell_required": [1, 1],
                        "destination_cell": [1, 3],
                        "status": "deadlock",
                        "failure_subtype": "deadlock",
                    }
                ],
                "v3_attempt_trace": {
                    "source_family": "unfiltered",
                    "difficulty_bucket": "constrained",
                    "best_boxes_on_targets": 1,
                    "final_boxes_on_targets": 0,
                },
            },
        ],
    )


def test_v3_global_reflection_prompt_renders_all_train_failures():
    raw_memory = make_v3_raw_memory()
    prompt = build_v3_global_reflection_prompt(
        raw_memory,
        MemoryRenderConfig(max_memory_items=999, max_memory_chars=30000),
    )

    assert "failure_count_rendered: 2" in prompt
    assert "level_id: train_a" in prompt
    assert "level_id: train_b" in prompt
    assert "source_family: medium" in prompt
    assert "difficulty_bucket: constrained" in prompt
    assert "board_before_failed_push:" in prompt
    assert "standing_cell_required: [1, 2]" in prompt


def test_v3_global_reflection_generation_records_scope_counts_and_source_hash():
    raw_memory = make_v3_raw_memory()
    client = FakeClient(
        '["Check standing-cell reachability before each push.", '
        '"In this level, avoid [3, 4].", '
        '"Use sequence Right, Up, Left."]'
    )

    memory = generate_v3_global_reflection_memory(
        raw_memory,
        client=client,
        memory_config=MemoryRenderConfig(max_memory_items=999, max_memory_chars=30000),
    )

    assert len(client.responses.calls) == 1
    assert memory.source_metadata["raw_failure_count_used"] == 2
    assert memory.source_metadata["source_raw_memory_hash"] == raw_memory.memory_hash
    assert memory.source_metadata["heuristic_scope_counts"] == {
        "global_allowed": 1,
        "same_level_only": 1,
        "rejected": 1,
    }
    assert memory.source_metadata["failure_subtype_distribution"] == {
        "deadlock": 1,
        "unreachable_standing_cell": 1,
    }
