# Initial LLM Difficulty Snapshot (0% Solve Era)

This document isolates the earliest evaluation results showing how hard Sokoban was for the initial LLM setups.

## Zero-Solve Results

Source: `docs/week6_results.md` (historical Week 6 one-step checkpoint summary).

Matched settings in that run:
- levels: `levels/v2_pilot.json` (`level_split=eval`)
- model: `gpt-4.1-mini`
- temperature: `0`
- `max_output_tokens=256`
- `max_steps=100`
- memory caps: `max_memory_items=3`, `max_steps_per_memory=6`, `max_memory_chars=4000`
- summary artifact: `results/v2_task3_large_evaluation_summary.json`

| Run Phase | Agent condition | Solve rate | Deadlock rate | Timeout rate | Interpretation |
|---|---|---:|---:|---:|---|
| Week 6 large eval | `no_memory` | **0.0%** | 15% | 85% | No solved episodes despite full budgeted rollout. |
| Week 6 large eval | `raw_trajectory_memory` | **0.0%** | 15% | 85% | Raw failure replay did not improve early solve outcome. |
| Week 6 large eval | `reflection_heuristic` | **0.0%** | 35% | 65% | Heuristic memory changed failure mix but still produced zero solves. |

## Why this matters

- The early bottleneck was not evaluation infrastructure (validation clean, no API/budget contamination in the reported run).
- The bottleneck was planning quality under Sokoban constraints: early systems could not convert memory into actual solves.
- This 0% phase is the baseline motivation for later full-path + verifier + repair and same-level retry experiments.

## Evidence references

- `docs/week6_results.md` (headline metrics and run conditions)
- `results/v2_task3_large_evaluation_summary.json` (historical summary artifact path listed in Week 6 doc)
