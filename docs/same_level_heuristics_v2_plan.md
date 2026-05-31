# Same-Level Heuristic Retry — Improvement Plan (V2)

Branch: `same_level_heuristics_v2` (off `full_path_kerui`).

This document records the diagnosis, the planned improvement versions, the
experiment protocol, and a running log of what was changed and why. It is meant
to be read top-to-bottom as a lab notebook.

## 1. Baseline result being improved

From `docs/v3_memory_evaluation_report.md`, eval split, `K = 3`, `gpt-5.2`,
reasoning `low`, `max_output_tokens = 16384`, prompt `full_path_v2_1`:

| Condition | Solve@1 | Solve@K | Solved levels | Invalid-plan attempts | Plan-exhausted attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| single-shot baseline | 33.3% | 33.3% | 8 | 13 | 1 |
| `verifier_summary_retry` | 29.2% | 54.2% | 13 | 30 | 1 |
| `raw_same_level_iterative` | 25.0% | 54.2% | 13 | 34 | 4 |
| **`heuristic_same_level_iterative`** | 20.8% | **50.0%** | **12** | 31 | **9** |

The heuristic condition is the **weakest same-level retry** condition: lower
solve@K (50% vs 54.2%) and a much higher `plan_exhausted` count (9 vs 1–4).

## 2. Evidence (what the generated heuristics actually look like)

Reconstructed in `analysis/heuristic_same_level_reconstruction.json`. Concrete
findings from that artifact:

- Every generated heuristic on every level is classified `global_allowed`
  (generic, transferable). **Zero** heuristics reference the specific board,
  the specific failed push, specific boxes, or coordinates.
- The model emits **10–12 heuristics per failure**, but the planner render cap
  (`max_memory_items = 3`) shows **only the first 3**, which are the most
  generic ("verify the required standing cell is reachable…"). 7–9 are dropped.
- The heuristic sets are near-duplicates across unrelated levels: they restate
  the same abstract Sokoban principles (`verify standing cell reachable`,
  `treat narrow corridors as one-way`, `preserve connectivity`,
  `avoid pushing to boundary`). These are things the planner already "knows."
- The advice is heavily **cautionary / avoidance-oriented** ("avoid pushing into
  corridors / corners / boundary / chokepoints"). On levels where the failure
  was `plan_exhausted`, this caution likely makes the planner emit shorter,
  incomplete plans — consistent with the elevated `plan_exhausted` count (9).

## 3. Root-cause hypotheses

- **H1 — Wrong prompt for the job.** Same-level heuristic generation reuses the
  *cross-level* `build_reflection_prompt`, whose instructions are
  "Use prescriptive rules. Do not replay the trajectories." That prompt is
  designed to produce generic, transferable rules for train→eval generalization.
  For *same-level* retry we want the opposite: a concrete diagnosis of why *this*
  attempt on *this* board failed, with corrective guidance grounded in the
  actual boxes/coordinates/failed push. The current design discards exactly the
  signal (`verifier_summary` / `raw`) that makes the other two conditions win.

- **H2 — The reflection is starved of failure detail.** `generate_reflection_memory`
  builds its prompt from `raw_memory.render(config)` capped at
  `max_memory_items = 3`, `max_steps_per_memory = 6`, `max_memory_chars = 4000`.
  The full failure evidence (initial board, board before the failed push, the
  exact failed intent, the verifier reason, executed vs planned pushes) is
  truncated. The reflector can't see precisely what went wrong.

- **H3 — The render cap throws away most heuristics and keeps the worst ones.**
  10–12 generated, 3 rendered, first-3 are the vaguest. Even if good concrete
  rules were generated later in the list, they never reach the planner.

- **H4 — `heuristic_scope` discards concrete content.** `heuristic_scope`
  classifies any heuristic containing a board row or a 3+ direction sequence as
  `rejected` (never rendered), and anything with coordinates as
  `same_level_only`. For *same-level* retry, concrete coordinates/directions are
  precisely what we want, but the generic prompt avoids them anyway (so all come
  out `global_allowed`), and the render path would drop the concrete ones if
  they appeared.

- **H5 — Over-caution drives `plan_exhausted`.** Generic avoidance advice without
  board context biases the planner toward incomplete plans. We need to keep the
  "emit a COMPLETE plan that solves the level" pressure while fixing legality.

## 4. Planned improvement versions

All versions keep the identical run protocol (Section 5). Only the same-level
*reflection* path changes. Each is selectable via
`--same_level_reflection_version`.

- **`baseline`** — existing behavior (generic cross-level prompt). For
  reproduction / control.

- **`v1_specific`** — Dedicated same-level reflection prompt. Feeds the *full*
  same-level failure evidence (large char budget; H2). Instructs the model to
  (a) diagnose the specific failed push and verifier reason, (b) produce a
  short list (≤5) of concrete, board-grounded corrective rules that may name
  specific boxes / coordinates / directions (H1, H4), and (c) not repeat the
  failed push. Planner renders up to 5 same-level heuristics, keeping concrete
  content (H3, H4).

- **`v2_complete_plan`** — `v1_specific` plus strong anti-`plan_exhausted`
  pressure (H5): the reflection must also state, concretely, how to extend the
  plan so every box ends on a target (e.g., box→target assignment, ordering).

- **`v3_hybrid_verifier`** — `v1_specific` plus the exact verifier rejection
  restated as the first, highest-priority rule (failed push index, the
  unreachable/blocked standing cell). This fuses the winning `verifier_summary`
  signal with distilled corrective heuristics.

## 5. Fixed run protocol (matches the V3 report)

- model `gpt-5.2`, reasoning `low`, `temperature 0`, `max_output_tokens 16384`
- prompt version `full_path_v2_1`, `max_steps 100`
- suite `levels/v3_boxoban_balanced.json`, `--level_split eval`
- `--experiment_mode same_level_iterative`,
  `--condition heuristic_same_level_iterative`, `--attempt_budget 3`
- distinct `--cache_namespace` per version so caches don't collide
- distinct `--results_dir` per version; evaluate with `evaluate_results.py`

Cost control: each new version is first smoke-tested on a small stratified
subset of eval levels (the medium/middle + medium/open levels that had the
worst invalid/plan-exhausted behavior) before committing to the full 24.

## 6. Running log

(Entries appended per change/run below.)

### Entry 0 — diagnosis complete (this document)

Branch created, environment set up (`.venv`, `.env` from `API_key.txt`),
baseline `pytest` green (102 passed). Diagnosis and version plan recorded above.
</content>
</invoke>
