# Level Suite Design

## Summary

This project now separates debugging levels from the final memory-evaluation benchmark.

- `levels/v2_pilot.json` is a pilot suite for verifier debugging, prompt checks, and small ablations.
- `levels/v3_boxoban_balanced.json` is the main train-to-eval benchmark.
- `levels/v3_boxoban_ood.json` is an optional harder out-of-distribution evaluation suite.

The final benchmark should use Boxoban-based levels only. Hand-written calibration levels remain useful for sanity checks, but they are not part of the aggregate generalization benchmark.

## Pilot Suite

`levels/v2_pilot.json` contains 12 train and 12 eval levels. It mixes hand-written calibration puzzles, earlier curated Boxoban examples, and imported Boxoban medium levels.

Intended use:

- Debug local Sokoban rules, deadlock detection, and full-path execution.
- Run low-cost prompt sanity checks.
- Inspect representative failures before launching larger experiments.

Not intended use:

- Do not use this suite for final train-to-eval generalization claims.
- Do not compare final memory conditions only on this suite, because train and eval are not difficulty-balanced.

## Balanced Boxoban Benchmark

`levels/v3_boxoban_balanced.json` is the main benchmark for memory generalization experiments. It contains 48 levels:

| Split | Source family | Count | Buckets |
| --- | --- | ---: | --- |
| train | `unfiltered/train` | 12 | 4 open, 4 middle, 4 constrained |
| train | `medium/train` | 12 | 4 open, 4 middle, 4 constrained |
| eval | `unfiltered/valid` | 12 | 4 open, 4 middle, 4 constrained |
| eval | `medium/valid` | 12 | 4 open, 4 middle, 4 constrained |

All levels are 10x10 Boxoban puzzles with 4 boxes and 4 targets. The suite is selected deterministically with seed `348`, rejects duplicate canonical grids, and preserves the official train/valid split separation.

Intended use:

- Build train failures and reflection heuristics from `split=train` only.
- Evaluate held-out performance on `split=eval` only.
- Report both aggregate metrics and stratum metrics by `source_family` and `difficulty_bucket`.
- Use heuristic memory for train-to-eval generalization.

Memory-use policy:

- Raw trajectory memory should not be used cross-level in the main train-to-eval benchmark.
- Raw trajectory memory is appropriate for same-level iterative retry experiments.
- Heuristic memory can be used for both same-level iterative retry and train-to-eval generalization.

## OOD Suite

`levels/v3_boxoban_ood.json` is an optional evaluation-only suite with 16 levels:

| Split | Source | Count | Label |
| --- | --- | ---: | --- |
| eval | held-out `medium/valid` | 8 | `ood_medium_valid_hardest` |
| eval | `hard` | 8 | `ood_hard` |

Hard levels are reported separately as OOD because the official Boxoban hard set does not provide train/eval split separation. These levels should not be mixed into the matched balanced benchmark aggregate.

## Metadata

Each Boxoban level includes structural metadata for filtering, sanity checks, and stratum-level reporting:

- `num_boxes`
- `num_targets`
- `wall_density`
- `player_reachable_ratio`
- `initial_legal_push_count`
- `solver_min_pushes`
- `solver_min_steps`
- `solver_status`
- `source_family`
- `source_split`
- `source_file`
- `source_index`
- `difficulty_bucket`
- `canonical_grid_hash`

`solver_min_pushes` and `solver_min_steps` come from a bounded local solver used only for benchmark metadata. If the solver reaches its cap, the level remains valid and records `solver_status = "capped"`.

## Reporting

For final reports, use `levels/v3_boxoban_balanced.json` as the main benchmark and report:

- Overall solve and failure rates.
- Results by source family: `unfiltered` and `medium`.
- Results by difficulty bucket: `open`, `middle`, and `constrained`.
- OOD results separately from `levels/v3_boxoban_ood.json`.

Primary metrics remain solve rate and failure-type rates. Efficiency metrics are secondary and should only be used where the required metadata is available.

## V3 Memory Evaluation Design

The V3 benchmark separates two experimental questions:

- Same-level adaptation: after failing on a level, can the agent improve on repeated attempts for that same level?
- Train-to-eval generalization: can abstract heuristics generated from train failures improve held-out eval performance?

Same-level iterative experiments treat train and eval levels symmetrically. Each level receives a fixed attempt budget `K`, with early stopping on success. The supported conditions are:

| Condition | Memory shown on retry | Cross-level use |
| --- | --- | --- |
| `single_shot_no_memory` | none | no |
| `generic_retry_feedback` | only says the previous same-level attempt failed | no |
| `verifier_summary_retry` | failed push index, failure subtype, and verifier reason | no |
| `raw_same_level_iterative` | compact raw evidence from prior failures on the same `level_id` | no |
| `heuristic_same_level_iterative` | same-level heuristics distilled from prior failures on the same `level_id` | no |

Train-to-eval experiments should use only heuristic memory across levels. Raw trajectory memory is not a train-to-eval condition because it contains puzzle-specific board states, failed pushes, and local coordinates.

Cross-level heuristic memory uses scope classification:

- `global_allowed`: abstract rules that may be rendered on held-out eval levels.
- `same_level_only`: coordinate-specific or board-specific rules that may be rendered only for the same `level_id`.
- `rejected`: copied action sequences, solution-like text, solver paths, or reference-solution-like content that must not be rendered.

## V3 Trajectory Records

Every V3 full-path attempt records an audit trace with:

- run identity: run ID, code commit, level suite path, level suite hash, cache key, prompt hash
- board states: initial board, final board, board before failed push, board after last successful push
- execution details: raw plan response, parsed push plan, expanded primitive actions, state-relative push execution log
- outcome details: status, failure reason, failure subtype, failure push index
- progress metrics: best boxes on targets, final boxes on targets, and target-placement events before the first deadlock

Prompt-facing raw memory is intentionally smaller than the audit trace. It renders at most two same-level failures, prefers distinct failure subtypes, and avoids long primitive walking traces by default.
