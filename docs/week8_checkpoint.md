# Week 8 Checkpoint: Full-Path Evaluation and Failure Analysis

## Summary

This checkpoint updates the earlier Week 8 `main` narrative with the additional work that happened afterward on top of `full_path_kerui`. The core setup remains full-path push-intent planning: the LLM proposes semantic pushes, and local code verifies legality/reachability before executing primitive moves. During this cycle we tightened legal-push/box-id planning guidance, improved failure observability with annotated GIFs and cleaner deadlock short-circuit logging, fixed a deadlock false positive, and added partial-progress metrics.

The core conclusion is still diagnostic: infrastructure and measurement improved substantially, but strict solve performance on hard eval levels remains constrained by semantic planning errors (`invalid_plan`, `deadlock`) rather than JSON-format failures.

## Evaluation Template

The final memory comparison still uses matched settings across:

- `no_memory`
- `raw_trajectory_memory`
- `reflection_heuristic`

Each run uses fixed levels/settings with the same model, seed, output budget, step budget, and memory budget.

Primary metrics:

- `solve_rate`
- `invalid_plan_count`
- `deadlock_count`
- `plan_exhausted_count`
- `timeout_count`

Secondary diagnostics:

- `planned_push_count`
- `executed_push_count`
- `expanded_primitive_step_count`
- `repair_attempt_count`
- `success_after_repair_count`
- failure reason counts (blocked destination, unreachable stand position, missing box coordinate, etc.)

Week 8 additions on top of earlier checkpoint template:

- `partial_progress_score`
- `average_best_goal_completion`

## Intermediate Results

### A) Earlier prompt-sweep baseline (from previous `main` week8 checkpoint)

All runs in this table used 12 train levels, `no_memory`, `gpt-5-nano`, temperature 0.

| Run | Prompt / scaffold | Success | Invalid plan | Deadlock | Plan exhausted | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `full_path_v2_train_prompt_check` | coordinate push plan | 2/12 | 9/12 | 1/12 | 0/12 | Best simple baseline in this sweep |
| `full_path_v3_train_prompt_check` | v2 + legality self-check wording | 1/12 | 11/12 | 0/12 | 0/12 | More legality text did not help |
| `full_path_v4_trace_prompt_check` | v2 + structured trace fields | 2/12 | 9/12 | 1/12 | 0/12 | No valid-rate gain |
| `full_path_v5_indexed_trace_prompt_check` | indexed boxes + compact trace | 1/12 | 10/12 | 0/12 | 1/12 | Less coordinate burden, still many illegal pushes |
| `full_path_v5_repair1_train_check` | v5 + one repair attempt | 1/12 | 10/12 | 1/12 | 0/12 | 11 repairs, 0 success-after-repair |
| `full_path_v2_1_rules_prompt_check` | v2 + clearer push-legality examples | 1/12 | 11/12 | 0/12 | 0/12 | Better grounding, no solve lift |

Interpretation carried forward:

- Prompt wording alone was insufficient to fix semantic Sokoban validity.

### B) Later matched memory comparisons and on-top changes

Short run IDs:

- `W6_STEP_BASE` -> `results/v2_task3_post_evaluation_summary.json`
- `W7_FP_BOXID` -> `results/lmgame_closer_eval_boxid_20260522_132732/evaluation_summary.json`
- `W7_FP_BOXID_DLOOK` -> `results/deadlock_lookahead_eval_20260522_2109/evaluation_summary.json`

| Run ID | Agent | Episodes | Solve Rate | Invalid Plan | Deadlock | Plan Exhausted | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| `W6_STEP_BASE` | `no_memory` | 6 | 0.000 | 0 | 2 | 0 | 4 |
| `W6_STEP_BASE` | `raw_trajectory_memory` | 6 | 0.000 | 0 | 1 | 0 | 5 |
| `W6_STEP_BASE` | `reflection_heuristic` | 6 | 0.000 | 0 | 1 | 0 | 5 |
| `W7_FP_BOXID` | `no_memory` | 12 | 0.083 | 3 | 7 | 1 | 0 |
| `W7_FP_BOXID` | `raw_trajectory_memory` | 12 | 0.083 | 3 | 8 | 0 | 0 |
| `W7_FP_BOXID` | `reflection_heuristic` | 12 | 0.083 | 4 | 7 | 0 | 0 |
| `W7_FP_BOXID_DLOOK` | `no_memory` | 12 | 0.000 | 5 | 5 | 2 | 0 |
| `W7_FP_BOXID_DLOOK` | `raw_trajectory_memory` | 12 | 0.083 | 5 | 3 | 3 | 0 |
| `W7_FP_BOXID_DLOOK` | `reflection_heuristic` | 12 | 0.083 | 5 | 4 | 2 | 0 |

| Run ID | Agent | Solve Rate | Partial Progress Score | Avg Best Goal Completion |
|---|---|---:|---:|---:|
| `W7_FP_BOXID` | `no_memory` | 0.083 | 0.208 | 0.208 |
| `W7_FP_BOXID` | `raw_trajectory_memory` | 0.083 | 0.188 | 0.188 |
| `W7_FP_BOXID` | `reflection_heuristic` | 0.083 | 0.208 | 0.208 |
| `W7_FP_BOXID_DLOOK` | `no_memory` | 0.000 | 0.271 | 0.271 |
| `W7_FP_BOXID_DLOOK` | `raw_trajectory_memory` | 0.083 | 0.333 | 0.333 |
| `W7_FP_BOXID_DLOOK` | `reflection_heuristic` | 0.083 | 0.375 | 0.375 |

Interpretation:

- Relative to old Week 6 one-step baseline, full-path + box-id era reached non-zero solve rate.
- In the direct on-top comparison (`W7_FP_BOXID` -> `W7_FP_BOXID_DLOOK`), strict solve stayed flat while partial progress improved.
- Memory-enabled variants currently look better than `no_memory` overall, but `raw_trajectory_memory` vs `reflection_heuristic` remains unresolved on strict solve rate.
- The tried-and-reverted unstuck fallback is excluded from this evidence because it increased noisy behavior without solve-rate lift on target reruns.

### C) Artifacts and tooling produced during this cycle

The checkpoint evidence is backed by concrete artifacts generated during these runs:

- Failure comparison GIF panels:
  - `docs/failure_gifs/lmgame_boxid_20260522/`
  - `docs/failure_gifs/deadlock_lookahead_eval_20260522_2109/`
  - `docs/failure_gifs/deadlock_fix_retest_20260522/`
  - `docs/failure_gifs/invalid_step_overlay_20260527/` (regenerated with player-direction arrows for failed moves)
- Reference replay GIF set:
  - `docs/reference_gifs/`
- Supporting scripts used to generate/verify these outputs:
  - `scripts/render_episode_gifs.py`
  - `scripts/render_reference_gifs.py`
  - `scripts/verify_reference_solutions.py`
  - `scripts/build_secondary_references.py`
  - `scripts/verify_secondary_references.py`

## Representative Failures

`simple_001` can be solved, but hard levels still expose repeated semantic invalid-plan patterns.

- early pushes into blocked destinations
- required standing cell blocked or unreachable
- stale or missing box coordinate during longer plans

Common failure reasons:

- `box_destination_blocked_by_box`
- `box_destination_blocked_by_wall_or_boundary`
- `required_push_position_blocked_by_box`
- `required_push_position_unreachable`
- `box_coordinate_missing`

## What This Answers

This updated checkpoint now supports the following claims:

- Full-path executor + verifier + failure taxonomy are stable and reproducible.
- Week 8 added stronger observability (annotated GIFs), deadlock-heuristic correction, and partial-progress metrics.
- Memory-enabled modes currently outperform no-memory on combined signals, but strict winner between raw vs reflection memory is not yet established.
- Unstuck fallback is excluded from final evidence because it was tested and reverted.

## Remaining Problems and Next Step

The main unresolved issue remains semantic legality and long-horizon consistency, not JSON shape. Prompt wording iterations alone were not enough; legality scaffolding and box-id guidance improved diagnostics and partial progress but did not yet produce robust hard-level solve gains.

Post-checkpoint guardrail note: the legal-push candidate contract is now enforced in the executor, not only stated in the prompt, and repair feedback now blocks repeating the same genuinely failed `(box_id, push)` pair. This targets the repeated invalid-push/deadlock patterns seen in the GIF review without reintroducing the noisy unstuck fallback.

Next steps should prioritize memory-winner resolution under matched seeds:

1. Run expanded seed-matched eval for all three memory modes on same level list.
2. Report strict winner by `solve_rate`; break ties with `deadlock_count`, then `partial_progress_score`.
3. Keep per-mode failure slicing to show which memory representation reduces which failure class.

Until that is complete, results should be framed as a strong evaluation/failure-analysis milestone rather than final proof of best memory representation.
