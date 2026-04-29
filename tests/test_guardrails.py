import pytest

from sokoban_memory.agents import RawTrajectoryMemoryAgent
from sokoban_memory.experiment import run_experiment
from sokoban_memory.memory import RawTrajectoryMemory
from sokoban_memory.types import Level, Position


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


def make_eval_level(level_id: str = "eval_001") -> Level:
    return Level(
        level_id=level_id,
        width=5,
        height=4,
        walls={
            Position(0, 0), Position(0, 1), Position(0, 2), Position(0, 3), Position(0, 4),
            Position(3, 0), Position(3, 1), Position(3, 2), Position(3, 3), Position(3, 4),
            Position(1, 0), Position(2, 0), Position(1, 4), Position(2, 4),
        },
        targets={Position(1, 3)},
        boxes={Position(2, 2)},
        player=Position(1, 1),
        tags=["wall_trap"],
        split="eval",
    )


def test_eval_memory_with_no_level_overlap_runs(tmp_path):
    memory = RawTrajectoryMemory(source_metadata={"source_train_level_ids": ["train_001"]})
    agent = RawTrajectoryMemoryAgent(memory_store=memory, client=FakeClient("Right"), max_llm_calls=2)

    summary = run_experiment([make_eval_level()], agent, episodes=1, max_steps=1, seed=0, results_dir=tmp_path)

    assert summary["episodes"] == 1
    assert summary["memory_source_level_ids"] == ["train_001"]


def test_eval_memory_overlap_fails_hard(tmp_path):
    memory = RawTrajectoryMemory(source_metadata={"source_train_level_ids": ["eval_001"]})
    agent = RawTrajectoryMemoryAgent(memory_store=memory, client=FakeClient("Right"), max_llm_calls=2)

    with pytest.raises(ValueError, match="contains eval level IDs"):
        run_experiment([make_eval_level()], agent, episodes=1, max_steps=1, seed=0, results_dir=tmp_path)


def test_eval_memory_missing_source_metadata_fails_hard(tmp_path):
    memory = RawTrajectoryMemory()
    agent = RawTrajectoryMemoryAgent(memory_store=memory, client=FakeClient("Right"), max_llm_calls=2)

    with pytest.raises(ValueError, match="missing source_train_level_ids"):
        run_experiment([make_eval_level()], agent, episodes=1, max_steps=1, seed=0, results_dir=tmp_path)

