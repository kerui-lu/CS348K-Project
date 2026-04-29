from sokoban_memory.memory import (
    HeuristicMemory,
    MemoryRenderConfig,
    RAW_RENDER_BANNED_WORDS,
    RawTrajectoryMemory,
    compress_episode,
)


def test_raw_memory_renderer_respects_caps_and_keeps_step_fields():
    memory = RawTrajectoryMemory(
        episodes=[
            {
                "level_id": "trap_1",
                "status": "deadlock",
                "step_count": 2,
                "total_reward": -5.2,
                "steps": [
                    {
                        "step": 0,
                        "state": "state0",
                        "raw_action": "Right",
                        "parsed_action": "Right",
                        "executed_action": "Right",
                        "reward": -0.1,
                        "invalid_reason": None,
                        "pushed_box": True,
                        "deadlocked": False,
                        "solved": False,
                        "next_state": "state1",
                    },
                    {
                        "step": 1,
                        "state": "state1",
                        "raw_action": "Up",
                        "parsed_action": "Up",
                        "executed_action": "Up",
                        "reward": -5.1,
                        "invalid_reason": None,
                        "pushed_box": True,
                        "deadlocked": True,
                        "solved": False,
                        "next_state": "state2",
                    },
                ],
            }
        ]
    )

    rendered = memory.render(MemoryRenderConfig(max_memory_items=1, max_steps_per_memory=1, max_memory_chars=500))

    assert "record_index: 1" in rendered
    assert "executed_action: Up" in rendered
    assert "deadlocked=True" in rendered
    assert "raw_action" not in rendered
    assert "parsed_action" not in rendered
    for banned in RAW_RENDER_BANNED_WORDS:
        assert banned not in rendered.lower()


def test_memory_renderers_apply_same_character_cap():
    config = MemoryRenderConfig(max_memory_items=3, max_steps_per_memory=6, max_memory_chars=80)
    raw_memory = RawTrajectoryMemory(
        episodes=[
            {
                "level_id": "long",
                "status": "timeout",
                "step_count": 1,
                "total_reward": -0.1,
                "steps": [{"step": 0, "state": "x" * 200, "next_state": "y" * 200}],
            }
        ]
    )
    heuristic_memory = HeuristicMemory(["Use a concise rule. " * 20])

    assert len(raw_memory.render(config)) <= 80
    assert len(heuristic_memory.render(config)) <= 80


def test_raw_memory_compression_selects_last_n_steps():
    episode = {
        "level_id": "trap",
        "status": "deadlock",
        "step_count": 4,
        "total_reward": -5.4,
        "trajectory": [
            {"step": 0, "state": "s0", "executed_action": "Up", "next_state": "s1", "info": {}},
            {"step": 1, "state": "s1", "executed_action": "Down", "next_state": "s2", "info": {}},
            {"step": 2, "state": "s2", "executed_action": "Left", "next_state": "s3", "info": {}},
            {"step": 3, "state": "s3", "executed_action": "Right", "next_state": "s4", "info": {}},
        ],
    }

    compressed = compress_episode(episode, max_steps=2)

    assert [step["step"] for step in compressed["steps"]] == [2, 3]
