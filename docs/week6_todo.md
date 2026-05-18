# Week 6 Evaluation Checkpoint TODO

This document summarizes what is already complete for the first checkpoint and what still needs to be done before the project has a complete evaluation package.

Week 7 status is now summarized in `docs/week7_update.md`. The branch currently uses the `full_path_v2` prompt again, with an optional verifier-guided repair loop available through `--max_repair_attempts`.

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

The current `levels/v2_pilot.json` has:

- 12 train levels
- 12 eval levels

This is enough for a stronger pilot comparison than the original checkpoint, while still small enough for low-cost LLM runs.

The suite covers tags such as:

```text
easy_simple_push
corner_trap
wall_trap
narrow_corridor
requires_repositioning
two_box_basic
multi_box_optional
boxoban_medium
```

Each level should include at minimum:

```json
"split": "train or eval",
"tags": ["..."]
```

Reference solutions are no longer required or validated. Some older calibration levels still include `optimal_steps` and historical `reference_solution` metadata, while imported Boxoban medium levels intentionally omit both.

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

### 4. Improve Deadlock Detection

The original Week 6 detector mainly caught:

```text
box in a non-target corner
```

On branch `full_path_kerui`, this has been upgraded with conservative local checks:

- box against a wall segment with no target and no exit away from the wall
- 2x2 wall/box freeze pattern
- conservative two-box freeze

We intentionally do not add static dead-square detection on this branch.

Later possible work:

- narrow tunnel/corridor trap
- multi-box recursive freeze

### 5. Treat Solution Efficiency As Optional

Some older calibration levels have manually written `optimal_steps`, but newly imported Boxoban medium levels do not.

Primary evaluation should rely on solve rate and failure-type rates. Efficiency metrics remain opportunistic and are computed only when `optimal_steps` exists.

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

1. Run local rule-based evaluation sanity checks without API calls.
2. Generate an evaluation summary using `evaluate_results.py`.
3. Write or refresh the checkpoint summary document.
4. Only after those pass, run a real LLM smoke test.
5. Then build memory banks.
6. Finally run the three-agent comparison:

```text
NoMemoryAgent vs RawTrajectoryMemoryAgent vs ReflectionHeuristicAgent
```

## One-Sentence Summary

The evaluation pipeline is implemented, but we still need a larger tagged level suite, real train-failure memory banks, three-agent comparison results, and a clearer checkpoint summary artifact before the project has a complete first-checkpoint evaluation package.
