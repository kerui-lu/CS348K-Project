from sokoban_memory.memory import MemoryRenderConfig, RawTrajectoryMemory
from sokoban_memory.reflection import (
    build_same_level_reflection_prompt,
    generate_reflection_memory,
    generate_same_level_reflection_memory,
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
