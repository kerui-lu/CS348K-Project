from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Action = Literal["Up", "Down", "Left", "Right"]
EpisodeStatus = Literal[
    "success",
    "deadlock",
    "timeout",
    "budget_exhausted",
    "api_error",
    "invalid_failure",
    "failure",
]


@dataclass(frozen=True, order=True)
class Position:
    row: int
    col: int

    def moved(self, dr: int, dc: int) -> "Position":
        return Position(self.row + dr, self.col + dc)


@dataclass
class Level:
    level_id: str
    width: int
    height: int
    walls: set[Position]
    targets: set[Position]
    boxes: set[Position]
    player: Position
    tags: list[str] = field(default_factory=list)
    split: str = "unspecified"
    optimal_steps: int | None = None


@dataclass
class StepResult:
    state_text: str
    next_state_text: str
    action: str
    parsed_action: Action | None
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class EpisodeResult:
    level_id: str
    agent_type: str
    seed: int
    status: EpisodeStatus
    step_count: int
    invalid_move_count: int
    total_reward: float
    llm_call_count: int
    token_cost: float
    trajectory: list[dict[str, Any]]
    policy_mode: str = "non_llm"
    model: str | None = None
    prompt_version: str | None = None
    memory_path: str | None = None
    memory_hash: str | None = None
    memory_caps: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_output_tokens: int | None = None
    cache_namespace: str | None = None
    level_split: str = "unspecified"
    level_tags: list[str] = field(default_factory=list)
    optimal_steps: int | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    budget_exhausted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
