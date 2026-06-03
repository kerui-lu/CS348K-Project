# Week 9 Checkpoint: Level-Suite Design

## Summary

Week 9 adds a final Boxoban-based level-suite design separate from the pilot suite. The project now has a stratified benchmark for train-to-eval memory generalization, plus an optional harder OOD evaluation suite.

The new level-suite design is documented in `docs/level_suite_design.md`.

## Progress

The current branch now distinguishes three level files:

- `levels/v2_pilot.json`: debugging and prompt-ablation suite only.
- `levels/v3_boxoban_balanced.json`: main benchmark with 24 train and 24 eval levels.
- `levels/v3_boxoban_ood.json`: optional eval-only OOD suite with 16 levels.

The balanced benchmark is reproducible with fixed seed `348`. It uses official Boxoban source families and preserves train/valid split separation:

| Split | Source family | Count | Difficulty buckets |
| --- | --- | ---: | --- |
| train | `unfiltered` | 12 | 4 open, 4 middle, 4 constrained |
| train | `medium` | 12 | 4 open, 4 middle, 4 constrained |
| eval | `unfiltered` | 12 | 4 open, 4 middle, 4 constrained |
| eval | `medium` | 12 | 4 open, 4 middle, 4 constrained |

## Difficulty Buckets And Metrics

The `open`, `middle`, and `constrained` labels are project-defined analysis strata, not official Boxoban labels. They are computed from structural features of each candidate level.

Computed metrics:

- `wall_density`: fraction of board cells that are walls.
- `player_reachable_ratio`: fraction of non-wall cells reachable by the player from the initial state while treating boxes as blocked.
- `initial_legal_push_count`: number of legal box pushes available at the initial state.
- `solver_min_pushes` / `solver_min_steps`: bounded local solver estimate when the level is solved within the search cap.
- `solver_status`: `solved`, `capped`, or `unsolved`.

Bucket construction:

- Candidates are sorted within each source family and split by a lightweight structural difficulty score.
- The lowest third becomes `open`, the middle third becomes `middle`, and the highest third becomes `constrained`.
- Four levels are sampled from each bucket for each source family and split.

Selected balanced-suite averages:

| Bucket | Avg. wall density | Avg. reachable ratio | Avg. initial legal pushes | Avg. difficulty score |
| --- | ---: | ---: | ---: | ---: |
| `open` | 0.618 | 0.892 | 10.38 | 0.306 |
| `middle` | 0.682 | 0.577 | 6.62 | 0.482 |
| `constrained` | 0.690 | 0.063 | 1.25 | 0.790 |

The OOD suite has 8 held-out medium-valid levels and 8 hard levels. Hard levels are marked as OOD because the official hard set does not include a train/eval split.

## Evaluation Update

Evaluation reporting now supports stratum-level summaries. This allows final results to be read by:

- overall aggregate
- source family
- difficulty bucket
- OOD group, reported separately

This structure supports the intended comparison between:

- no memory
- same-level raw trajectory retry
- same-level heuristic retry
- train-to-eval heuristic generalization

Raw trajectory memory should not be used cross-level for the main train-to-eval benchmark. Heuristic memory can be used cross-level because it stores abstracted rules rather than level-specific replay traces.

## Current Limitation

The main unresolved issue remains LLM full-path legality. The local verifier catches invalid plans, but the model still often proposes pushes that are illegal, unreachable, or inconsistent with updated box positions.

Because of this, the level-suite work should be treated as benchmark infrastructure progress, not as a new performance claim. The next experimental step is to stabilize planner validity, then run matched memory comparisons on `levels/v3_boxoban_balanced.json`.

## Next Steps

- Use `levels/v3_boxoban_balanced.json` for the main memory comparison.
- Report `levels/v3_boxoban_ood.json` separately as a harder generalization check.
- Keep `levels/v2_pilot.json` for fast debugging and prompt iteration.
- Run final tables by source family and difficulty bucket, not only by overall aggregate.
