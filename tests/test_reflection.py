from sokoban_memory.memory import MemoryRenderConfig, RawTrajectoryMemory
from sokoban_memory.reflection import generate_reflection_memory, parse_heuristics


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
