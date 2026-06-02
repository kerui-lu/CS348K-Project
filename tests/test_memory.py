from sokoban_memory.memory import (
    HeuristicMemory,
    MemoryRenderConfig,
    RAW_RENDER_BANNED_WORDS,
    RawTrajectoryMemory,
    classify_heuristic,
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


def test_same_level_raw_memory_renders_only_matching_level_and_compact_boards():
    memory = RawTrajectoryMemory(
        episodes=[
            {
                "level_id": "other",
                "status": "invalid_plan",
                "failure_reason": "box_destination_blocked_by_box",
                "failure_subtype": "blocked_destination",
                "initial_board": "other_initial",
                "board_before_failed_push": "other_failed",
                "board_after_last_successful_push": "other_last",
            },
            {
                "level_id": "target",
                "status": "invalid_plan",
                "failure_reason": "box_destination_blocked_by_wall_or_boundary",
                "failure_subtype": "blocked_destination",
                "failure_push_index": 0,
                "initial_board": "target_initial",
                "board_before_failed_push": "target_failed",
                "board_after_last_successful_push": "target_last",
                "push_execution_log": [
                    {
                        "push_index": 0,
                        "model_intent": {"box": [1, 2], "push": "Up"},
                        "resolved_box_before_push": [1, 2],
                        "standing_cell_required": [2, 2],
                        "destination_cell": [0, 2],
                        "status": "failed",
                    }
                ],
            },
        ]
    )

    rendered = memory.render_for_level(
        "target",
        MemoryRenderConfig(max_memory_items=3, max_steps_per_memory=6, max_memory_chars=2000),
    )

    assert "target_initial" in rendered
    assert "target_failed" in rendered
    assert "target_last" in rendered
    assert "other_initial" not in rendered
    assert "resolved_box_before_push: [1, 2]" in rendered


def test_verifier_summary_and_raw_evidence_are_distinct():
    memory = RawTrajectoryMemory(
        episodes=[
            {
                "level_id": "target",
                "status": "invalid_plan",
                "failure_reason": "required_push_position_unreachable",
                "failure_subtype": "unreachable_standing_cell",
                "failure_push_index": 2,
                "initial_board": "initial",
                "board_before_failed_push": "failed",
                "board_after_last_successful_push": "last_success",
            }
        ]
    )
    config = MemoryRenderConfig(max_memory_items=3, max_steps_per_memory=6, max_memory_chars=2000)

    verifier = memory.render_verifier_summary_for_level("target", config)
    raw = memory.render_for_level("target", config)

    assert "failed_push_index: 2" in verifier
    assert "board_before_failed_push" not in verifier
    assert "board_before_failed_push" in raw
    assert len(raw) > len(verifier)


def test_heuristic_classifier_and_cross_level_rendering_scopes():
    memory = HeuristicMemory(
        [
            "Do not push boxes into non-target corners.",
            "In this level, avoid pushing the box at [3, 4] right.",
            "Use the sequence Right, Up, Left, Left.",
            "Follow the solver_min_steps metadata when planning.",
        ]
    )

    assert classify_heuristic(memory.heuristics[0])["scope"] == "global_allowed"
    assert classify_heuristic(memory.heuristics[1])["scope"] == "same_level_only"
    assert classify_heuristic(memory.heuristics[2])["scope"] == "rejected"
    assert classify_heuristic(memory.heuristics[3])["scope"] == "rejected"
    cross_level = memory.render(MemoryRenderConfig(max_memory_items=3, max_steps_per_memory=6, max_memory_chars=2000))
    same_level = memory.render_for_level(
        "target",
        MemoryRenderConfig(max_memory_items=3, max_steps_per_memory=6, max_memory_chars=2000),
    )

    assert "non-target corners" in cross_level
    assert "[3, 4]" not in cross_level
    assert "sequence Right" not in cross_level
    assert "solver_min_steps" not in cross_level
    assert "[3, 4]" in same_level
    assert "sequence Right" not in same_level


def test_same_level_heuristic_rendering_respects_source_level_ids():
    memory = HeuristicMemory(
        [
            "Do not push boxes into non-target corners.",
            "In this level, avoid pushing the box at [3, 4] right.",
        ],
        source_metadata={"source_level_ids": ["level_a"]},
    )
    config = MemoryRenderConfig(max_memory_items=3, max_steps_per_memory=6, max_memory_chars=2000)

    matching = memory.render_for_level("level_a", config)
    non_matching = memory.render_for_level("level_b", config)

    assert "[3, 4]" in matching
    assert "[3, 4]" not in non_matching
    assert "non-target corners" in non_matching
