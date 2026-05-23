# Week 8 Checkpoint: Full-Path Evaluation and Failure Analysis

## Summary

This checkpoint updates the Week 6 evaluation scaffold with real intermediate results from the `full_path_kerui` branch. The project has moved from one-step LLM actions to full-path push-intent planning: the LLM proposes a complete JSON push plan, and the local executor verifies each push with Sokoban rules and BFS reachability before executing primitive moves.

The main result is diagnostic rather than performance-positive: the full-path infrastructure works, but prompt-only full-plan generation is not yet reliable enough for Sokoban. The dominant failure mode is semantic invalidity, not JSON formatting.

## Evaluation Template

The final evaluation will compare memory conditions under matched settings:

- `no_memory`
- `raw_trajectory_memory`
- `reflection_heuristic`

Each run uses the same levels, model, seed, temperature, output cap, step cap, memory budget, and prompt version. The current level suite has 12 train and 12 eval levels in `levels/v2_pilot.json`.

Primary metrics:

- `solve_rate`
- `invalid_plan_rate`
- `deadlock_rate`
- `plan_exhausted_rate`
- `timeout_rate`

Secondary diagnostics:

- `planned_push_count`
- `executed_push_count`
- `expanded_primitive_step_count`
- `repair_attempt_count`
- `success_after_repair_count`
- failure reason counts such as blocked destination, unreachable standing cell, and missing box coordinate

Planned presentation artifacts:

- Outcome table comparing prompt variants and memory conditions
- Failure-reason table for invalid plans
- Board examples showing a successful plan and representative invalid pushes
- Trajectory logs for success and failure case studies

## Intermediate Results

All runs below used the 12 train levels, `no_memory`, `gpt-5-nano`, temperature 0, and local full-path validation.

| Run | Prompt / scaffold | Success | Invalid plan | Deadlock | Plan exhausted | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `full_path_v2_train_prompt_check` | coordinate push plan | 2/12 | 9/12 | 1/12 | 0/12 | Best simple baseline so far |
| `full_path_v3_train_prompt_check` | v2 plus explicit legality self-check wording | 1/12 | 11/12 | 0/12 | 0/12 | More legality text did not help |
| `full_path_v4_trace_prompt_check` | v2 plus structured state trace fields | 2/12 | 9/12 | 1/12 | 0/12 | No valid-rate gain; more parse risk |
| `full_path_v5_indexed_trace_prompt_check` | indexed boxes plus compact trace | 1/12 | 10/12 | 0/12 | 1/12 | Reduced coordinate burden but still illegal pushes |
| `full_path_v5_repair1_train_check` | v5 plus one verifier-guided repair attempt | 1/12 | 10/12 | 1/12 | 0/12 | 11 repairs, 0 success-after-repair |
| `full_path_v2_1_rules_prompt_check` | v2 plus clearer rules and push-legality examples | 1/12 | 11/12 | 0/12 | 0/12 | Better prompt grounding did not improve success |

The most useful finding is that simply adding prompt wording is not enough. The model often understands the output format, but still fails local Sokoban validity checks.

## Representative Failures

`simple_001` is solved, but the harder train levels expose consistent invalid-plan patterns.

- `two_box_basic_001`: the first push tries to push a box into another box. The model proposes `{"box": [2, 3], "push": "Right"}` even though `[2, 4]` already contains a box.
- `corner_trap_001`: after several legal pushes, the model refers to a box coordinate that no longer contains a box. This shows that dynamic coordinate tracking is unreliable.
- Boxoban train/medium levels: many first or early pushes use a blocked destination, a blocked player standing cell, or a standing cell that BFS says is unreachable.

Common failure reasons:

- `box_destination_blocked_by_box`
- `box_destination_blocked_by_wall_or_boundary`
- `required_push_position_blocked_by_box`
- `required_push_position_unreachable`
- `box_coordinate_missing`

## What This Answers

The current branch answers several evaluation and system-design questions:

- The full-path local verifier and BFS executor are implemented and catch invalid LLM plans.
- Evaluation no longer depends on reference solutions; solve and failure rates apply to every level.
- Raw trajectory and reflection memory can be evaluated under the same full-path runner once the base planner is valid enough.
- Prompt-only full-path planning is currently not reliable enough to support a fair memory comparison.
- Verifier feedback alone, with one repair attempt, did not recover failed plans in the latest sanity run.

## Remaining Problems and Next Step

The main unresolved problem is that the LLM is asked to perform too much grid-level legality reasoning internally. It often proposes impossible pushes even when the prompt states the rules.

The next implementation step should expose locally computed legal push options to the LLM. Instead of asking the model to infer all legal pushes from the grid, the local code should provide a compact list such as:

```json
[
  {"box": [2, 4], "push": "Down", "stand": [1, 4], "dest": [3, 4]},
  {"box": [3, 2], "push": "Right", "stand": [3, 1], "dest": [3, 3]}
]
```

Then the LLM can plan over verified legal push choices. This should directly target the current invalid-plan bottleneck and make the later memory comparison more meaningful.

Until then, full-path results should be presented as an evaluation-pipeline and failure-analysis milestone, not as final evidence that one memory representation is better than another.
