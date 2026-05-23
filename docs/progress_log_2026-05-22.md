# Sokoban Improvement Log (May 22, 2026)

Date: 2026-05-22  
Scope: Week 6 to Week 7 iteration cycle on full-path Sokoban planning

## Goals

- Improve strict solve performance for `no_memory`, `raw_trajectory_memory`, and `reflection_heuristic`.
- Reduce early failures caused by invalid push plans and avoidable deadlocks.
- Make failure analysis legible through reproducible artifacts (episode JSON + GIF replay).
- Track progress beyond solve rate with partial-completion style metrics.

## What We Are Optimizing

- Primary objective: increase `solve_rate` on eval levels.
- Secondary objectives:
  - reduce `invalid_plan`, `deadlock`, and `plan_exhausted` rates,
  - improve planning stability under repair attempts,
  - increase partial board progress even on unsolved episodes.

## End-to-End Change Log

### 1) Reference correctness and replay tooling

- Verified reference solutions in `levels/v2_pilot.json` by replaying through `SokobanEnv`.
- Added scripts to support reproducible verification/rendering workflows:
  - `scripts/verify_reference_solutions.py`
  - `scripts/render_reference_gifs.py`
- Generated reference GIF assets for visual baseline checks.

### 2) Failure analysis workflow improvements

- Added/used episode replay tooling for failed trajectories.
- Built comparative memory-condition GIF sets for:
  - `no_memory`
  - `raw_trajectory_memory`
  - `reflection_heuristic`
- Standardized run inspection around:
  - episode status,
  - `metadata.failure_reason`,
  - `planned_pushes`,
  - `push_execution_log`,
  - final board state.

### 3) Repair-loop and deadlock short-circuit fixes

- File: `sokoban_memory/experiment.py`
- Changes:
  - Added explicit short-circuit when board is already deadlocked before a repair attempt.
  - Added synthetic trajectory row for short-circuit deadlock cases so logs remain valid.
- Impact:
  - removed ambiguous empty-trajectory deadlock artifacts,
  - made evaluation validation and GIF rendering more stable.

### 4) One-step deadlock lookahead for legal push scaffolding

- File: `sokoban_memory/experiment.py`
- Changes:
  - Added immediate deadlock simulation filter in legal push candidate generation.
  - Avoids presenting pushes to the model that immediately deadlock after execution.
- Impact:
  - improves candidate quality fed into prompts,
  - reduces one class of obvious deadlock suggestions.

### 5) Deadlock heuristic false-positive correction

- File: `sokoban_memory/env.py`
- Problem:
  - `box_against_wall_no_target_no_exit` was over-conservative in some states (notably `boxoban_medium_valid_000_002`) and could short-circuit before any model action.
- Fix:
  - adjusted wall-segment exit logic to avoid false positives on one-sided openings.
- Tests:
  - Added regression in `tests/test_env.py` using the problematic board shape.
  - Updated full-path tests in `tests/test_full_path.py` to match corrected semantics.

### 6) Partial-progress metrics (beyond solve rate)

- File: `sokoban_memory/metrics.py`
- Added overall and per-level metrics:
  - `average_final_goal_completion`
  - `average_best_goal_completion`
  - `partial_progress_score`
  - `partial_progress_rate_25`
  - `partial_progress_rate_50`
  - `partial_progress_rate_75`
- Test coverage:
  - Added metric test cases in `tests/test_metrics.py`.

### 7) Local unstuck fallback during full-path execution

- Files:
  - `sokoban_memory/full_path.py`
  - `sokoban_memory/experiment.py`
  - `run_experiment.py`
  - `tests/test_full_path.py`
- New behavior:
  - Optional executor fallback (`--enable_unstuck_fallback`) when planned push is invalid or required push position is unreachable.
  - Chooses a reachable legal push using local scoring (novelty/mobility/reward heuristic), rejects immediate deadlocks, and continues plan execution.
- Logging additions:
  - `unstuck_applied_count`
  - recovery records in `push_execution_log`:
    - `invalid_plan_recovered_with_unstuck`
    - `unreachable_recovered_with_unstuck`

### 8) GIF annotation upgrades for debugging

- File: `scripts/render_episode_gifs.py` (recreated and upgraded)
- Added clear terminal subtitles:
  - explicit final failed instruction for `invalid_plan`,
  - explicit deadlock reason for `deadlock`.
- Output now preserves quick diagnosis directly in the artifact without requiring separate JSON reading.

## Recent Focused Rerun Artifacts

- Run set: `results/unstuck_rerun_fullnet_20260522_215842`
- Annotated GIF set: `docs/failure_gifs/unstuck_fullnet_20260522_215842`
- Compared levels:
  - `boxoban_eval_000_006`
  - `boxoban_medium_valid_000_002`
- Compared agent types:
  - `no_memory`
  - `raw_trajectory_memory`
  - `reflection_heuristic`

### Observed outcomes from the focused rerun

- Final attempt status for all six runs was `deadlock` (not `invalid_plan`).
- Unstuck fallback triggered in some intermediate attempts (visible in episode JSON logs), but did not yet convert these selected seeds to success.
- GIF overlays now make final failure mode explicit (deadlock reason or failed instruction depending on terminal status).

## Current Progress Summary

- Infrastructure progress: strong.
  - Full-path executor, repair loop, deadlock filtering, partial metrics, and visualization are now substantially more diagnosable.
- Scientific progress: moderate.
  - Failure signatures are better isolated and easier to compare.
  - Partial progress can now be measured directly.
- Solve-rate progress on hard eval seeds: limited so far.
  - Remaining bottlenecks are semantic planning quality and deadlock avoidance under deeper sequences.

## Open Issues / Next Steps

- Improve unstuck scoring with stronger penalties for creating non-target corner risk.
- Add richer recovery feedback in repair prompts (include candidate ranking and local hazard notes).
- Add stuck-loop detection from repeated board hashes and trigger unstuck earlier.
- Add comparative before/after report on identical seeds:
  - baseline vs deadlock-lookahead vs unstuck-fallback.
- Keep reporting both strict solve metrics and partial-progress metrics in all checkpoints.

