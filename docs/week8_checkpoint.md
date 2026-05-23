# Week 8 Checkpoint (May 22, 2026)

## Goal

Improve Sokoban solve quality by reducing `invalid_plan` and `deadlock` failures in our full-path LLM setup, while keeping comparisons reproducible across:

- `no_memory`
- `raw_trajectory_memory`
- `reflection_heuristic`

Detailed change log: `docs/progress_log_2026-05-22.md`.

---

## LMGame Background and What We Learned

LMGame-Bench motivated this direction: Sokoban typically needs harness support (state scaffolding, structured interaction, memory) to move beyond near-zero performance.

What we carried into this repo:

- Structured full-path plan format (with box IDs and local validation).
- Memory-conditioned variants and side-by-side evaluation.
- Repair loop with failure feedback.
- Better observability (episode JSON + failure GIFs).

Main learning so far: LMGame-style scaffolding helps with progress signals, but our current blocker is still execution-quality on long-horizon pushes (illegal push proposals and deadlock-prone plans), so solve rate remains low on hard eval levels.

---

## Evaluation Plan (Finalized)

### Quantitative outputs

- Strict metrics:
  - `solve_rate`
  - `invalid_plan_count`
  - `deadlock_count`
  - `plan_exhausted_count`
  - `timeout_count`
- Progress metrics:
  - `partial_progress_score`
  - `average_best_goal_completion`

### What partial progress means

For each episode:

- `goal_completion_ratio = boxes_on_target / total_boxes`
- Track the **best** ratio reached over the episode.
- `partial_progress_score` is the average of those best ratios across episodes.

This captures meaningful intermediate progress that `solve_rate` (binary) misses.

### Qualitative outputs

- Failure GIFs per run and agent.
- Overlay shows terminal failure reason (invalid instruction or deadlock reason).

### Short run IDs used below

- `W6_STEP_BASE` -> `results/v2_task3_post_evaluation_summary.json`
- `W7_FP_BOXID` -> `results/lmgame_closer_eval_boxid_20260522_132732/evaluation_summary.json`
- `W7_FP_BOXID_DLOOK` -> `results/deadlock_lookahead_eval_20260522_2109/evaluation_summary.json`

---

## What We Tried on Top of Kerui Full-Path

1. Failure-trace analysis pipeline:
   - Correlated GIF behavior with episode metadata and push execution logs.
2. Repair-loop/data integrity fixes:
   - Cleaned edge cases so deadlock short-circuit episodes still serialize valid trajectory/state for tooling.
3. Deadlock lookahead filter:
   - Removed push candidates that immediately enter deadlock states.
4. Deadlock false-positive fix:
   - Fixed wall-segment deadlock heuristic and added regression test.
5. Partial progress scoring:
   - Added metrics and tests to report non-binary improvement.
6. Unstuck fallback prototype:
   - Implemented and tested, then fully reverted because outcomes worsened (more bad terminal behavior).
7. GIF explainability upgrade:
   - Added clear failure subtitle (failed instruction or deadlock reason) in rendered GIFs.

---

## Intermediate Results

## Table A — Strict outcomes

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

Takeaway: full-path + box ID improved from zero-solve baseline to non-zero solve rate; deadlock lookahead improved some failure composition but did not yet raise solve rate on this set.

## Table B — Partial progress

| Run ID | Agent | Solve Rate | Partial Progress Score | Avg Best Goal Completion |
|---|---|---:|---:|---:|
| `W7_FP_BOXID` | `no_memory` | 0.083 | 0.208 | 0.208 |
| `W7_FP_BOXID` | `raw_trajectory_memory` | 0.083 | 0.188 | 0.188 |
| `W7_FP_BOXID` | `reflection_heuristic` | 0.083 | 0.208 | 0.208 |
| `W7_FP_BOXID_DLOOK` | `no_memory` | 0.000 | 0.271 | 0.271 |
| `W7_FP_BOXID_DLOOK` | `raw_trajectory_memory` | 0.083 | 0.333 | 0.333 |
| `W7_FP_BOXID_DLOOK` | `reflection_heuristic` | 0.083 | 0.375 | 0.375 |

Takeaway: progress score rises in lookahead runs, especially with memory, even where strict solves do not.

## Memory Options Status (Current Evidence)

Question: which memory option is best (`no_memory`, `raw_trajectory_memory`, or `reflection_heuristic`)?

Current readout from available runs:

- On strict `solve_rate`, `raw_trajectory_memory` and `reflection_heuristic` are tied at `0.083` in both full-path runs.
- `no_memory` is less stable: it drops to `0.000` in `W7_FP_BOXID_DLOOK`.
- On partial progress, ranking in lookahead run is:
  - `reflection_heuristic` (`0.375`) > `raw_trajectory_memory` (`0.333`) > `no_memory` (`0.271`).
- On deadlocks in lookahead run:
  - `raw_trajectory_memory` (`3`) is slightly better than `reflection_heuristic` (`4`), both better than `no_memory` (`5`).

Bottom line: evidence currently supports **memory-enabled variants over no-memory**, but we do **not** yet have enough seed-matched data to declare `raw_trajectory_memory` vs `reflection_heuristic` winner on strict solves.

## Artifacts

- `docs/failure_gifs/lmgame_boxid_20260522/`
- `docs/failure_gifs/deadlock_lookahead_eval_20260522_2109/`
- `docs/reference_gifs/`

---

## What Is Working vs Not Working

Working:

- Reproducible multi-agent evaluation with explicit failure taxonomy.
- Stronger diagnostics (episode metadata + annotated GIFs).
- Partial-progress metrics reveal useful movement hidden by solve-only reporting.

Not yet working:

- Solve rate remains low on difficult eval levels.
- `invalid_plan` and `deadlock` still dominate terminal outcomes.
- Memory variants do not yet create strong, consistent solve-rate separation.

---

## Clean Next Steps

1. Memory winner experiment (highest priority):
   - Run `no_memory`, `raw_trajectory_memory`, and `reflection_heuristic` on the same expanded seed set and level list.
   - Report paired per-level winner counts and overall solve-rate deltas.
2. Tie-break metrics for memory modes:
   - For each memory mode, report `solve_rate`, `partial_progress_score`, `invalid_plan_count`, and `deadlock_count`.
   - Use this to separate "more progress" from "more actual solves."
3. Failure-mode slicing by memory type:
   - Build a compact table showing which failure class each memory mode reduces most.
4. Prompt/executor changes aimed at memory comparison validity:
   - Prioritize fixes that reduce `invalid_plan` noise so memory effects are easier to detect.
5. Final project decision rule:
   - Choose memory mode by strict solve-rate first; break ties with deadlock rate, then partial-progress score.
