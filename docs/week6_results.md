# Week 6 Checkpoint (Evaluation Focus)

This checkpoint documents the current evaluation status for the Sokoban memory project.  
Scope is evaluation pipeline quality and controlled comparison setup, not final model performance.

## What This Checkpoint Covers

- Automatic outcome labeling for each episode (`success`, `deadlock`, `timeout`, `budget_exhausted`, `api_error`, `invalid_failure`)
- Per-episode trajectory logging and aggregate summary generation
- Split-aware level usage (`train` for memory build, `eval` for comparison)
- Frozen-memory comparison protocol across three agents:
  - `no_memory`
  - `raw_trajectory_memory`
  - `reflection_heuristic`

## Step Status (Week 6 TODO)

- Step 1 (expand level suite): **done** (`levels/v2_pilot.json` includes 6 train + 6 eval levels)
- Step 2 (build memory data): **done**
  - `memory_banks/raw_failures.json` (16 failure records from train runs)
  - `memory_banks/reflection_heuristics.json` (8 generated heuristics from the same raw failures)
- Step 3 (three-agent comparison): **done** with matched settings and consolidated report
- Step 4 (deadlock detector improvements): **deferred intentionally**
- Step 5 (efficiency reliability hardening): **deferred intentionally**
- Step 6 (clear checkpoint artifact): **this document**

## Memory-Build Artifact (Train Split)

Memory bank build source metadata confirms:

- Source agent: `no_memory`
- Levels file: `levels/v2_pilot.json`
- Source train level IDs recorded in metadata
- Requested/completed memory build episodes: `20 / 20`
- Raw memory items: `16`
- Reflection heuristics: `8`

Files:

- `memory_banks/raw_failures.json`
- `memory_banks/reflection_heuristics.json`

## Three-Agent Evaluation Run (Eval Split)

Main comparison report:

- `results/v2_task3_large_evaluation_summary.json`

Input result directories:

- `results/v2_task3_large_no_memory`
- `results/v2_task3_large_raw`
- `results/v2_task3_large_reflection`

Matched conditions used across agents:

- Same level set: `levels/v2_pilot.json` with `level_split=eval`
- Same seed: `42`
- Same model: `gpt-4.1-mini`
- Same temperature: `0`
- Same output cap: `max_output_tokens=256`
- Same horizon: `max_steps=100`
- Same memory caps: `max_memory_items=3`, `max_steps_per_memory=6`, `max_memory_chars=4000`

## Validation and Sanity Checks

From `results/v2_task3_large_evaluation_summary.json`:

- `episode_file_count = 60`
- `valid_episode_count = 60`
- `validation_error_count = 0`
- `api_error_rate = 0.0`
- `budget_exhausted_rate = 0.0`
- `invalid_failure_rate = 0.0`

Interpretation: the evaluation pipeline executed end-to-end with clean validation and no infrastructure/API failure contamination in the final run.

## Current Comparison Snapshot

Per-agent headline metrics from the large eval run:

- `no_memory`: `solve_rate=0.0`, `deadlock_rate=0.15`, `timeout_rate=0.85`
- `raw_trajectory_memory`: `solve_rate=0.0`, `deadlock_rate=0.15`, `timeout_rate=0.85`
- `reflection_heuristic`: `solve_rate=0.0`, `deadlock_rate=0.35`, `timeout_rate=0.65`

Overall note: no agent solved eval levels in this checkpoint run, so this snapshot is a protocol/evaluation milestone rather than a final performance claim.

## Deferred Work (By Request)

The following were intentionally skipped for now and remain open:

- Step 4: richer deadlock pattern detection beyond current corner-based checks
- Step 5: stronger solution-efficiency reliability checks tied to validated reference solutions

## Checkpoint Conclusion

Week 6 evaluation infrastructure is operational and reproducible: memory banks are generated from train failures, three matched eval conditions were executed, and consolidated validation-clean summaries were produced.  
This establishes a solid evaluation checkpoint while deferring deadlock-detector expansion and efficiency-validation hardening to a later iteration.
