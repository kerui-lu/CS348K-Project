from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptRenderResult:
    prompt: str
    memory_text: str
    non_memory_template: str


def render_one_step_prompt(
    *,
    policy_mode: str,
    prompt_version: str,
    rules: str,
    state_text: str,
    legal_actions: list[str],
    push_actions: list[str],
    memory_condition: str,
    memory_text: str,
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
        ". target",
        "$ box",
        "* box on target",
        "@ player",
        "+ player on target",
        "",
        "Current board:",
        state_text,
        "",
        f"Legal actions: {legal_actions}",
        f"Actions that push a box: {push_actions}",
        "",
        "Decision checklist (apply for this single move):",
        "1) Safety: avoid any move likely to create an immediate deadlock.",
        "2) Progress: prefer pushes that move a box closer to a target.",
        "3) Anti-loop: avoid actions that only undo recent movement without progress.",
        "4) Tie-break: if multiple actions are similar, choose the one that repositions for a future useful push.",
        "",
        memory_block,
        "",
        "Output contract:",
        "Choose the next single action for the current board only.",
        "Return exactly one action from this set: Up, Down, Left, Right.",
        "Do not include explanation, punctuation, or multiple actions.",
    ]
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
