import json

from sokoban_memory.logging_utils import save_episode
from sokoban_memory.types import EpisodeResult


def test_save_episode_outputs_expected_json(tmp_path):
    result = EpisodeResult(
        level_id="simple_001",
        agent_type="rule_based",
        seed=42,
        status="timeout",
        step_count=1,
        invalid_move_count=0,
        total_reward=-0.1,
        llm_call_count=0,
        token_cost=0.0,
        trajectory=[
            {
                "state": "state",
                "raw_action": "Right",
                "parsed_action": "Right",
                "executed_action": "Right",
                "next_state": "next",
                "reward": -0.1,
                "done": False,
                "info": {},
            }
        ],
    )
    path = save_episode(result, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["level_id"] == "simple_001"
    assert data["agent_type"] == "rule_based"
    assert data["seed"] == 42
    assert data["status"] == "timeout"
    assert data["step_count"] == 1
    assert data["invalid_move_count"] == 0
    assert {"state", "executed_action", "next_state", "reward", "done", "info"}.issubset(
        data["trajectory"][0]
    )

