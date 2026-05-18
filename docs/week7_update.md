# Week 7 Update: Full-Path Prompt and Repair Status

## Current Branch Status

Branch `full_path_kerui` now uses the full-path push-intent architecture:

- LLM-backed agents generate a complete push plan for the whole puzzle.
- The local executor validates each push and expands it into primitive moves with BFS reachability.
- Evaluation no longer depends on reference solutions.
- `levels/v2_pilot.json` contains 12 train levels and 12 eval levels.
- Deadlock detection includes conservative local corner, wall/no-target/no-exit, 2x2 freeze, and two-box freeze checks.

The active prompt has been reverted to `full_path_v2`. It asks for strict JSON push intents:

```json
[
  {"box": [3, 2], "push": "Down"},
  {"box": [4, 2], "push": "Right"}
]
```

Coordinates are still dynamic: after a push, later plan items must use the box's updated coordinate.

## Week 7 Experiments

We tested several prompt variants after the first full-path implementation:

- `full_path_v2`: coordinate-based push intents.
- `full_path_v3`: added explicit local legality self-check wording.
- `full_path_v4`: added structured state trace fields.
- `full_path_v5`: switched to indexed boxes and compact trace fields.

The later prompts did not improve the overall valid-plan rate. The indexed-box prompt removed some coordinate bookkeeping burden, but it also produced many plans whose first or early pushes were locally illegal. The repair loop was then added as an explicit scaffold:

```text
attempt 0: generate a full plan from the original board
execute locally
if invalid_plan / plan_exhausted / deadlock and repair budget remains:
  append verifier feedback
  regenerate a complete plan from the original board
```

The repair budget is controlled by `--max_repair_attempts` and defaults to `0`, so old runs remain comparable unless repair is explicitly enabled.

## Current Problem

The main bottleneck is not JSON parsing anymore. The main bottleneck is semantic Sokoban validity:

- The model often proposes a push that is blocked by a wall, boundary, or another box.
- The model sometimes chooses a box coordinate that is no longer correct after previous pushes.
- The model frequently stops with a partial plan or creates a deadlock.
- Even when verifier feedback is given, one repair attempt did not recover any additional train successes in the latest sanity run.

Latest sanity run:

```text
results/full_path_v5_repair1_train_check
episodes: 12
success_count: 1
invalid_plan_count: 10
deadlock_count: 1
success_after_repair_count: 0
repair_attempt_count: 11
```

Because `full_path_v5` plus repair did not improve results, the active prompt is back to `full_path_v2` for a simpler and more interpretable baseline.

## Next Recommended Step

The next improvement should make repair more actionable rather than only re-prompting generally. Good candidates:

- Include the local legal push set for the current board in the prompt.
- During repair, include the legal alternatives near the failed box.
- Add a local planner-assisted repair that rejects impossible first pushes before spending another API call.
- Keep the repair budget fixed across `no_memory`, `raw_trajectory_memory`, and `reflection_heuristic` for fair comparison.

Until those changes are tested, full-path results should be described as an evaluation-pipeline milestone, not as final evidence about memory quality.
