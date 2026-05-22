from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptRenderResult:
    prompt: str
    memory_text: str
    non_memory_template: str


def render_full_path_prompt(
    *,
    policy_mode: str,
    prompt_version: str,
    rules: str,
    state_text: str,
    state_summary: dict[str, Any],
    max_steps: int,
    memory_condition: str,
    memory_text: str,
    repair_feedback: str | None = None,
) -> PromptRenderResult:
    memory_block = (
        "Memory context:\n"
        f"Condition: {memory_condition}\n"
        f"{memory_text}"
    )
    sections = [
        "You are playing Sokoban.",
        f"Policy mode: {policy_mode}",
        f"Prompt version: {prompt_version}",
        "",
        "Rules:",
        rules,
        "",
        "Board symbols:",
        "# wall",
        "space empty floor",
        ". target",
        "$ box",
        "* box on target",
        "@ player",
        "+ player on target",
        "",
        "Current board:",
        state_text,
        "",
        "Coordinate system:",
        "Use 0-indexed [row, col] coordinates. Row increases downward; col increases rightward.",
        "",
        "Current coordinates:",
        f"Player: {state_summary.get('player')}",
        f"Boxes: {state_summary.get('boxes')}",
        f"Targets: {state_summary.get('targets')}",
        f"Maximum primitive steps after local expansion: {max_steps}",
        "",
        "Planning task:",
        "Return a complete high-level push plan that continues until every box is on a target.",
        "Do not stop after one push or after a locally useful partial plan.",
        "Do not output walking moves.",
        "Each plan item names a box at its current [row, col] at that point in the plan and the direction to push it.",
        "After each push, mentally update the board before choosing the next push.",
        "For later pushes of the same box, use the box's updated coordinate, not its original coordinate.",
        "The local verifier will check whether the player can reach the required push position and will expand each push into shortest walking moves plus one push.",
        "A box on a target still counts as a box.",
        "",
        "Push legality examples:",
        "If pushing Right from box [r, c], the player must stand at [r, c-1], and the box destination is [r, c+1].",
        "If pushing Left from box [r, c], the player must stand at [r, c+1], and the box destination is [r, c-1].",
        "If pushing Down from box [r, c], the player must stand at [r-1, c], and the box destination is [r+1, c].",
        "If pushing Up from box [r, c], the player must stand at [r+1, c], and the box destination is [r-1, c].",
        "Only write a push if the box exists there now, the player standing cell is not blocked, and the destination cell is not blocked.",
        "",
        "Planning checklist:",
        "1) Legality: every listed push must satisfy the standing-cell and destination-cell rules.",
        "2) Reachability: only propose pushes whose standing cell can be reached by walking around current boxes and walls.",
        "3) Progress: push boxes toward target cells; boxes are interchangeable, so any box may go to any target.",
        "4) Safety: avoid immediate deadlocks, and avoid moving a box off a target unless necessary.",
        "5) Completion: the final push should leave all boxes on target cells.",
        "",
        memory_block,
        "",
        "Output contract:",
        "Return JSON only, with no markdown and no explanation.",
        "The JSON must be an array of objects.",
        'Each object must have exactly this shape: {"box": [row, col], "push": "Up|Down|Left|Right"}.',
        "Example with one box moving through updated coordinates:",
        "[",
        '  {"box": [3, 2], "push": "Down"},',
        '  {"box": [4, 2], "push": "Right"},',
        '  {"box": [4, 3], "push": "Right"},',
        '  {"box": [4, 4], "push": "Right"}',
        "]",
    ]
    if repair_feedback:
        sections.extend([
            "",
            "Repair feedback:",
            repair_feedback,
        ])
    prompt = "\n".join(sections)
    non_memory_template = prompt.replace(memory_block, "Memory context:\n<MEMORY_BLOCK>")
    return PromptRenderResult(
        prompt=prompt,
        memory_text=memory_text,
        non_memory_template=non_memory_template,
    )


def level_metadata(levels: list[Any]) -> dict[str, Any]:
    return {
        "level_ids": [level.level_id for level in levels],
        "level_splits": {level.level_id: getattr(level, "split", "unspecified") for level in levels},
        "level_tags": {level.level_id: list(getattr(level, "tags", [])) for level in levels},
        "level_optimal_steps": {
            level.level_id: getattr(level, "optimal_steps", None)
            for level in levels
        },
    }
