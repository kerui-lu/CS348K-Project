# Week 6 Evaluation Checkpoint TODO

This document summarizes what is already complete for the first checkpoint and what still needs to be done before the project has a complete evaluation package.

## Current State

The evaluation framework is mostly in place, but the project still needs enough experimental material and actual evaluation data.

Already completed:

- Automatic outcome detection:
  - `success`
  - `deadlock`
  - `timeout`
  - `api_error`
  - `budget_exhausted`
  - `invalid_failure`
- Per-episode trajectory logging.
- `evaluate_results.py` for aggregating experiment results.
- Core metrics:
  - `solve_rate`
  - `deadlock_count`
  - `timeout_count`
  - `average_success_steps`
  - `solution_efficiency`
  - `per_level`
- Unit tests, currently passing.
- Train/eval split guardrail.
- Memory leak checking.
- Raw memory and reflection memory framework.

## Remaining TODOs

### 1. Expand the Level Suite

The current `levels/v2_pilot.json` has only:

- 2 train levels
- 1 eval level

This is enough to test the pipeline, but not enough for meaningful agent comparison.

Recommended target:

```text
train: 6-8 levels
eval: 6-8 levels
```

The level suite should cover tags such as:

```text
easy_simple_push
corner_trap
wall_trap
narrow_corridor
requires_repositioning
two_box_basic
multi_box_optional
```

Each level should include:

```json
"split": "train or eval",
"tags": ["..."],
"optimal_steps": 4
```

Ideally, each level should also include:

```json
"reference_solution": ["Right", "Down"]
```

### 2. Build Enough Memory Data

The memory framework exists, but we do not yet have enough real memory data.

We still need to generate:

```text
memory_banks/raw_failures.json
memory_banks/reflection_heuristics.json
```

These should be built from:

```text
NoMemoryAgent failures on train levels
```

Missing pieces:

- More train levels.
- Real NoMemoryAgent failure episodes on train levels.
- Raw memory built from those failures.
- Reflection heuristics generated from the same raw failures.

### 3. Run Real Three-Agent Comparisons

We have not yet run the full comparison among:

```text
NoMemoryAgent
RawTrajectoryMemoryAgent
ReflectionHeuristicAgent
```

To answer the research question, all three agents should be evaluated under identical conditions:

```text
same eval levels
same seeds
same model
same temperature
same max_output_tokens
same max_steps
same memory budget
```

Then use `evaluate_results.py` to compare:

```text
solve_rate
deadlock_rate
timeout_rate
average_solution_efficiency
average_success_steps
invalid_move_rate
```

### 4. Improve Deadlock Detection Later

The current deadlock detector mainly catches:

```text
box in a non-target corner
```

This is acceptable for the Week 6 checkpoint because it is automatic and tested.

Later, we may want to add more Sokoban deadlock patterns:

- box against a wall with no target on that wall
- two boxes stuck together
- frozen box pattern
- tunnel trap

This is a later improvement, not a blocker for the first checkpoint.

### 5. Make Solution Efficiency More Reliable

Right now, `optimal_steps` is manually written in the level file.

This works for an initial evaluation, but a stronger version would add:

```json
"reference_solution": ["Right", "Down"]
```

Then automatically check:

```text
len(reference_solution) == optimal_steps
reference_solution actually solves the level
```

This would make solution efficiency more trustworthy.

### 6. Create a Clear Week 6 Checkpoint Artifact

The code exists, but for the CA/instructor it would be useful to have a concise checkpoint artifact, such as:

```text
docs/week6_checkpoint.md
```

or:

```text
docs/week6_eval_summary.md
```

This document should summarize:

- What the evaluation pipeline checks.
- What metrics are computed.
- A small local sanity result.
- That the checkpoint focuses on evaluation quality, not final LLM performance yet.

Since `results/` is ignored, it is better to include key numbers in a markdown summary rather than committing raw result JSON files.

## Recommended Next Steps

1. Expand the level suite to around 12 levels:
   - 6 train
   - 6 eval
2. Add `reference_solution` and automatic validation for levels.
3. Run local rule-based evaluation sanity checks without API calls.
4. Generate an evaluation summary using `evaluate_results.py`.
5. Write a Week 6 checkpoint summary document.
6. Only after those pass, run a real LLM smoke test.
7. Then build memory banks.
8. Finally run the three-agent comparison:

```text
NoMemoryAgent vs RawTrajectoryMemoryAgent vs ReflectionHeuristicAgent
```

## One-Sentence Summary

The evaluation pipeline is implemented, but we still need a larger tagged level suite, real train-failure memory banks, three-agent comparison results, and a clearer checkpoint summary artifact before the project has a complete first-checkpoint evaluation package.
