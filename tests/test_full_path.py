import pytest

from sokoban_memory.agents import LLMAgent
from sokoban_memory.env import SokobanEnv
from sokoban_memory.experiment import run_episode
from sokoban_memory.full_path import PushPlanParseError, execute_push_plan, parse_push_plan, shortest_player_path
from sokoban_memory.types import Level, Position


class FakeResponses:
    def __init__(self, output_text: str | list[str]):
        self.output_texts = output_text if isinstance(output_text, list) else [output_text]
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output_text = self.output_texts[min(len(self.calls) - 1, len(self.output_texts) - 1)]
        return type("FakeResponse", (), {"output_text": output_text})()


class FakeClient:
    def __init__(self, output_text: str | list[str]):
        self.responses = FakeResponses(output_text)


def make_level(grid: list[str], level_id: str = "test") -> Level:
    walls = set()
    targets = set()
    boxes = set()
    player = None
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            pos = Position(r, c)
            if char == "#":
                walls.add(pos)
            elif char == ".":
                targets.add(pos)
            elif char == "$":
                boxes.add(pos)
            elif char == "*":
                boxes.add(pos)
                targets.add(pos)
            elif char == "@":
                player = pos
            elif char == "+":
                player = pos
                targets.add(pos)
    assert player is not None
    return Level(level_id, len(grid[0]), len(grid), walls, targets, boxes, player)


def test_parse_push_plan_accepts_strict_json():
    plan = parse_push_plan('[{"box": [1, 2], "push": "Right"}]')

    assert plan[0].box == Position(1, 2)
    assert plan[0].push == "Right"


def test_parse_push_plan_accepts_box_id_schema():
    plan = parse_push_plan(
        '[{"box_id": 0, "push": "Right", "player_after": [1, 2], "box_after": [1, 3]}]',
        box_count=1,
    )

    assert plan[0].box_id == 0
    assert plan[0].box is None
    assert plan[0].push == "Right"


def test_parse_push_plan_ignores_structured_trace_fields():
    plan = parse_push_plan(
        '[{"box": [1, 2], "push": "Right", '
        '"stand": [1, 1], '
        '"after": {"player": [1, 2], "boxes": [[1, 3]]}}]'
    )

    assert len(plan) == 1
    assert plan[0].box == Position(1, 2)
    assert plan[0].push == "Right"


@pytest.mark.parametrize(
    "raw_output",
    [
        '[{"box_id": "0", "push": "Right"}]',
        '[{"box_id": -1, "push": "Right"}]',
        '[{"box_id": true, "push": "Right"}]',
    ],
)
def test_parse_push_plan_rejects_invalid_box_id(raw_output):
    with pytest.raises(PushPlanParseError):
        parse_push_plan(raw_output, box_count=1)


def test_parse_push_plan_rejects_unknown_box_id_when_box_count_is_known():
    with pytest.raises(PushPlanParseError, match="unknown box_id"):
        parse_push_plan('[{"box_id": 2, "push": "Right"}]', box_count=1)


@pytest.mark.parametrize(
    "raw_output",
    [
        "Push box at (1,2) Right",
        '{"box": [1, 2], "push": "Right"}',
        '[{"push": "Right"}]',
        '[{"box": [1, 2], "push": "Jump"}]',
        '[{"box": [1, "2"], "push": "Right"}]',
    ],
)
def test_parse_push_plan_rejects_malformed_outputs(raw_output):
    with pytest.raises(PushPlanParseError):
        parse_push_plan(raw_output)


def test_shortest_player_path_reaches_push_position_deterministically():
    env = SokobanEnv(make_level([
        "#####",
        "#@  #",
        "# $.#",
        "#####",
    ]))
    env.reset()

    path = shortest_player_path(env, Position(2, 1))

    assert path == ["Down"]


def test_shortest_player_path_returns_none_for_unreachable_position():
    env = SokobanEnv(make_level([
        "#####",
        "#@# #",
        "# # #",
        "# # #",
        "#####",
    ]))
    env.reset()

    assert shortest_player_path(env, Position(1, 3)) is None


def test_execute_push_plan_solves_simple_level():
    env = SokobanEnv(make_level([
        "#####",
        "#@$.#",
        "#####",
    ]))
    env.reset()
    plan = parse_push_plan('[{"box": [1, 2], "push": "Right"}]')

    result = execute_push_plan(env, plan, max_steps=10, raw_response="[]", call_metadata={})

    assert result.status == "success"
    assert result.expanded_actions == ["Right"]
    assert result.executed_push_count == 1


def test_execute_push_plan_solves_simple_level_with_box_id():
    env = SokobanEnv(make_level([
        "#####",
        "#@$.#",
        "#####",
    ]))
    env.reset()
    plan = parse_push_plan('[{"box_id": 0, "push": "Right"}]', box_count=1)

    result = execute_push_plan(env, plan, max_steps=10, raw_response="[]", call_metadata={})

    assert result.status == "success"
    assert result.expanded_actions == ["Right"]
    assert result.executed_push_count == 1
    assert result.push_execution_log[0]["resolved_box"] == {"row": 1, "col": 2}
    assert result.push_execution_log[0]["box_positions_by_id_after"] == {"B0": [1, 3]}


def test_execute_push_plan_tracks_box_id_across_multiple_pushes():
    env = SokobanEnv(make_level([
        "#######",
        "#     #",
        "# @   #",
        "# $   #",
        "#    .#",
        "#     #",
        "#######",
    ]))
    env.reset()
    plan = parse_push_plan(
        "["
        '{"box_id": 0, "push": "Down"},'
        '{"box_id": 0, "push": "Right"},'
        '{"box_id": 0, "push": "Right"},'
        '{"box_id": 0, "push": "Right"}'
        "]",
        box_count=1,
    )

    result = execute_push_plan(env, plan, max_steps=20, raw_response="[]", call_metadata={})

    assert result.status == "success"
    assert result.executed_push_count == 4
    assert result.expanded_actions == ["Down", "Left", "Down", "Right", "Right", "Right"]
    assert result.push_execution_log[-1]["box_positions_by_id_after"] == {"B0": [4, 5]}


def test_run_episode_full_path_missing_box_is_invalid_plan():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    agent = LLMAgent(client=FakeClient('[{"box": [2, 2], "push": "Right"}]'))

    result = run_episode(SokobanEnv(level), agent, max_steps=10, seed=0)

    assert result.status == "invalid_plan"
    assert result.metadata["failure_reason"] == "box_coordinate_missing"
    assert result.metadata["planned_pushes"] == [{"box": [2, 2], "push": "Right"}]


def test_run_episode_full_path_blocked_push_is_invalid_plan():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    agent = LLMAgent(client=FakeClient('[{"box": [1, 2], "push": "Up"}]'))

    result = run_episode(SokobanEnv(level), agent, max_steps=10, seed=0)

    assert result.status == "invalid_plan"
    assert result.metadata["failure_reason"] == "box_destination_blocked_by_wall_or_boundary"


def test_run_episode_full_path_empty_plan_is_plan_exhausted():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    agent = LLMAgent(client=FakeClient("[]"))

    result = run_episode(SokobanEnv(level), agent, max_steps=10, seed=0)

    assert result.status == "plan_exhausted"
    assert result.metadata["planned_push_count"] == 0
    assert result.metadata["executed_push_count"] == 0


def test_run_episode_full_path_push_can_deadlock():
    level = make_level([
        "#####",
        "# $@#",
        "#   #",
        "#  .#",
        "#####",
    ])
    agent = LLMAgent(client=FakeClient('[{"box": [1, 2], "push": "Left"}]'))

    result = run_episode(SokobanEnv(level), agent, max_steps=10, seed=0)

    assert result.status == "deadlock"
    assert result.metadata["executed_push_count"] == 1
    assert result.trajectory[-1]["info"]["deadlocked"] is True


def test_run_episode_full_path_push_can_create_wall_no_exit_deadlock():
    level = make_level([
        "#######",
        "#     #",
        "# @$  #",
        "#     #",
        "#######",
    ])
    agent = LLMAgent(client=FakeClient('[{"box": [2, 3], "push": "Up"}]'))

    result = run_episode(SokobanEnv(level), agent, max_steps=10, seed=0)

    assert result.status == "deadlock"
    assert result.metadata["executed_push_count"] == 1
    assert result.trajectory[-1]["info"]["deadlock_reason"] == (
        "box_against_wall_no_target_no_exit:1,3,wall=Up"
    )


def test_run_episode_full_path_calls_llm_once_per_episode():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    client = FakeClient('[{"box": [1, 2], "push": "Right"}]')
    agent = LLMAgent(client=client)

    result = run_episode(SokobanEnv(level), agent, max_steps=10, seed=0)

    assert result.status == "success"
    assert result.llm_call_count == 1
    assert len(client.responses.calls) == 1


def test_run_episode_full_path_repair_can_turn_invalid_plan_into_success():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    client = FakeClient([
        '[{"box_id": 0, "push": "Up"}]',
        '[{"box_id": 0, "push": "Right"}]',
    ])
    agent = LLMAgent(client=client)

    result = run_episode(
        SokobanEnv(level),
        agent,
        max_steps=10,
        seed=0,
        max_repair_attempts=1,
    )

    assert result.status == "success"
    assert result.llm_call_count == 2
    assert len(client.responses.calls) == 2
    assert result.metadata["repair_attempt_count"] == 1
    assert result.metadata["success_after_repair"] is True
    assert result.metadata["first_attempt_status"] == "invalid_plan"
    assert [attempt["status"] for attempt in result.metadata["repair_attempts"]] == [
        "invalid_plan",
        "success",
    ]
    repair_prompt = client.responses.calls[1]["input"]
    assert "Current board:" in repair_prompt
    assert "Output contract:" in repair_prompt
    assert "Legal push candidates on this board" in repair_prompt
    assert "Repair feedback:" in repair_prompt
    assert "Failure reason: box_destination_blocked_by_wall_or_boundary" in repair_prompt
    assert 'Failed intent: {"box_id": 0, "push": "Up"}' in repair_prompt
    assert "Legal alternatives near this state:" in repair_prompt
    assert '"push": "Right"' in repair_prompt
    assert "Regenerate a complete plan from the original board" in repair_prompt


def test_run_episode_full_path_repair_disabled_matches_single_attempt_flow():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    client = FakeClient([
        '[{"box_id": 0, "push": "Up"}]',
        '[{"box_id": 0, "push": "Right"}]',
    ])
    agent = LLMAgent(client=client)

    result = run_episode(
        SokobanEnv(level),
        agent,
        max_steps=10,
        seed=0,
        max_repair_attempts=0,
    )

    assert result.status == "invalid_plan"
    assert result.llm_call_count == 1
    assert len(client.responses.calls) == 1
    assert result.metadata["repair_attempt_count"] == 0
    assert result.metadata["success_after_repair"] is False
    assert len(result.metadata["repair_attempts"]) == 1


def test_run_episode_full_path_repair_budget_exhausted_returns_last_failure():
    level = make_level([
        "#####",
        "#@$.#",
        "#####",
    ])
    client = FakeClient([
        '[{"box_id": 0, "push": "Up"}]',
        '[{"box_id": 0, "push": "Up"}]',
    ])
    agent = LLMAgent(client=client)

    result = run_episode(
        SokobanEnv(level),
        agent,
        max_steps=10,
        seed=0,
        max_repair_attempts=1,
    )

    assert result.status == "invalid_plan"
    assert result.llm_call_count == 2
    assert result.metadata["repair_attempt_count"] == 1
    assert [attempt["status"] for attempt in result.metadata["repair_attempts"]] == [
        "invalid_plan",
        "invalid_plan",
    ]


def test_run_episode_full_path_repair_from_current_state_chains_partial_plans():
    level = make_level([
        "#######",
        "#@ $ .#",
        "#######",
    ])
    client = FakeClient([
        '[{"box_id": 0, "push": "Right"}]',
        '[{"box_id": 0, "push": "Right"}]',
    ])
    agent = LLMAgent(client=client)

    result = run_episode(
        SokobanEnv(level),
        agent,
        max_steps=20,
        seed=0,
        max_repair_attempts=1,
        repair_from_current_state=True,
    )

    assert result.status == "success"
    assert result.llm_call_count == 2
    assert len(client.responses.calls) == 2
    assert result.metadata["repair_attempt_count"] == 1
    assert [attempt["status"] for attempt in result.metadata["repair_attempts"]] == [
        "plan_exhausted",
        "success",
    ]
    repair_prompt = client.responses.calls[1]["input"]
    assert "Regenerate a complete plan from the current board" in repair_prompt
