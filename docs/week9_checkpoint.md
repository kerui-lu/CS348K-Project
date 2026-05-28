# Week 9 Checkpoint: Memory Effects Under Full-Path Sokoban Planning

Date: 2026-05-27  
Project branch context: full-path Sokoban pipeline with legal-push scaffolding, box-id guidance, deadlock diagnostics, and partial-progress metrics.

## 1) What We Are Trying to Show

The final project question is not just "can an LLM solve Sokoban?" but:

1. Under a fixed full-path harness, which memory strategy is more useful?
   - `no_memory`
   - `raw_trajectory_memory`
   - `reflection_heuristic`
2. Do memory strategies improve strict solve outcomes, or only intermediate progress?
3. Which failure classes are reduced by each memory strategy?

Target deliverable for final report:
- a defensible ranking of memory options under matched conditions,
- with evidence from strict metrics, progress metrics, and failure-type analysis.

---

## 2) Working Hypotheses

## H1: Memory beats no-memory under a legality-aware harness

Rationale:
- With legal-push scaffolding and richer verifier feedback, memory should help action selection quality instead of being drowned by basic legality errors.

Predictions:
- lower `invalid_plan_count` and/or `deadlock_count` for memory modes versus `no_memory`;
- higher `partial_progress_score`;
- possibly higher `solve_rate` as sample size increases.

## H2: Reflection memory should outperform raw trajectory memory on deadlock avoidance

Rationale:
- Raw trajectories preserve factual history but may include noisy details.
- Reflection heuristics compress "what to avoid" and should generalize better across related layouts.

Predictions:
- `reflection_heuristic` should reduce deadlock-related failures more than `raw_trajectory_memory`;
- if strict solves tie, reflection may still win on deadlock rate or progress consistency.

## H3: Better harness quality is a prerequisite for fair memory comparison

Rationale:
- If semantic invalidity dominates, memory differences are masked.
- Therefore, prompt/executor quality improvements are not "extra"; they are required to measure memory effects.

Predictions:
- after legality/box-id/deadlock-filter improvements, memory separation signals should become clearer.

---

## 3) What Current Results Mean

From Week 8 evidence:
- Full-path + box-id era moved from zero-solve historical baseline to non-zero solves on the eval slice.
- Deadlock lookahead changed failure composition and improved partial progress, but strict solve gains remained small.
- Memory-enabled modes currently look better than no-memory on combined indicators.
- `raw_trajectory_memory` vs `reflection_heuristic` remains unresolved on strict solve rate.

Post-guardrail matched eval (`PG_GUARDRAILS_20260527`):

Settings:
- 12 eval episodes per memory mode from `levels/v2_pilot.json`
- seeds 42-53, `gpt-5-nano`, temperature 0
- `max_steps=50`, `max_repair_attempts=2`, `max_output_tokens=4096`
- memory caps: 3 items, 6 steps/item, 4000 chars
- validation: 36 valid episode files, 0 validation errors
- summary artifact: `docs/post_guardrail_eval_20260527_summary.json`

| Run ID | Agent | Episodes | Solve Rate | Invalid Plan | Deadlock | Plan Exhausted | Partial Progress | Repair Attempts | Success After Repair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `PG_GUARDRAILS_20260527` | `no_memory` | 12 | 0.000 | 10 | 1 | 1 | 0.083 | 24 | 0 |
| `PG_GUARDRAILS_20260527` | `raw_trajectory_memory` | 12 | 0.083 | 10 | 1 | 0 | 0.146 | 22 | 0 |
| `PG_GUARDRAILS_20260527` | `reflection_heuristic` | 12 | 0.083 | 9 | 2 | 0 | 0.146 | 23 | 1 |

Interpretation:
- The project has crossed from "pipeline construction" into "measurement-sensitive comparison."
- We now have enough instrumentation to make nuanced claims (strict + partial + failure taxonomy), but not yet enough matched samples to claim a final winner.
- After hard legality enforcement and repeated-failed-push repair blocking, memory-enabled modes still beat `no_memory` on strict solve rate and partial progress in this matched run.
- `raw_trajectory_memory` and `reflection_heuristic` tie on solve rate and partial progress; reflection has one fewer invalid-plan failure but one more deadlock, so strict winner remains unresolved.

---

## 4) Research Grounding

This direction is supported by LMGame-Bench-style findings:
- Sokoban often remains near-zero without strong harness support.
- Structured interaction/perception/memory scaffolding can produce meaningful gains.

How this informs our setup:
- We already adopted several harness-aligned components (structured prompts, verifier loop, memory variants, artifact-driven diagnostics).
- Our remaining gap is long-horizon semantic consistency in full-path plans.

Practical implication:
- To claim memory differences credibly, we must keep harness settings fixed and reduce confounds from basic legality failures.

References:
- `docs/lmgame_gap_analysis_and_plan.md`
- [LMGame-Bench paper](https://arxiv.org/pdf/2505.15146)
- [GamingAgent repository](https://github.com/lmgame-org/GamingAgent)

---

## 5) Week 9 Experimental Plan (Actionable)

## A. Core "memory winner" matrix (highest priority)

Run matched eval for all three modes with identical:
- levels/split,
- seeds,
- model and prompt version,
- repair budget,
- memory budget (where applicable).

Report:
- `solve_rate` (primary),
- `deadlock_count`, `invalid_plan_count`, `plan_exhausted_count`,
- `partial_progress_score`,
- per-level winner counts.

Decision rule:
1. highest `solve_rate`;
2. if tied, lower `deadlock_count`;
3. if still tied, higher `partial_progress_score`.

## B. Heuristic/setup toggles to test for interesting conclusions

Keep memory mode fixed within each toggle block and ablate one factor at a time:

1. Deadlock lookahead ON vs OFF
   - Question: does it trade strict solves for safer partial progress?
2. Repair budget small sweep (`max_repair_attempts` = 0, 1, 2)
   - Question: does extra repair produce true recovery or just longer failures?
3. Memory budget sweep (`max_memory_chars` small/medium/large)
   - Question: is there a "too much memory" regime that harms plan quality?
4. Prompt contract strictness (box-id emphasis variants, same legality list)
   - Question: does stronger identity anchoring reduce stale-coordinate failures?

These toggles can produce publishable conclusions even if absolute solve rates remain modest:
- safety-performance tradeoffs,
- memory-capacity sensitivity,
- where reflection vs raw memory helps most.

## C. Harness fixes tried after invalid-step review

| Issue observed | Change tried | Result / status |
|---|---|---|
| GIFs showed terminal failure but not the attempted move clearly. | Regenerated failure GIFs with a red direction arrow on the player for the attempted action, while keeping text overlays minimal. | Curated arrow examples are in `docs/failure_gifs/invalid_step_overlay_20260527/curated/`. The older clean comparison panels were restored because they are easier to scan than the text-heavy intermediate overlays. |
| Legal-push prompt instructions were advisory rather than enforced. | Added executor-side legal-candidate gating: a push must match the current reachable `legal_push_candidates` set before execution continues. | Implemented, tested, and included in `PG_GUARDRAILS_20260527`; validation errors were 0/36. |
| Repair attempts could repeat the exact failed `(box_id, push)` pair. | Added a repair guardrail that bans the last genuinely failed `(box_id, push)` pair on the next repair attempt. | Implemented, tested, and included in `PG_GUARDRAILS_20260527`; legal `plan_exhausted` continuations remain allowed, and reflection had 1 success-after-repair. |

Result interpretation:
- Both requested guardrails are now in code, not just in the prompt.
- Post-guardrail eval does not eliminate invalid plans; instead, it makes invalidity stricter and earlier by rejecting non-current legal candidates.
- Memory-enabled modes still look better than `no_memory`, but raw vs reflection is still a tie on the primary metric.

---

## 6) Proposed New Analyses for Week 9 Writeup

To deepen conclusions beyond aggregate counts:

1. First-failure-position analysis
   - distribution of first invalid push index by memory mode.
2. Per-level failure-shift map
   - for each level, which failure type changed most from no-memory to raw/reflection.
3. Recovery effectiveness
   - `success_after_repair_count / repair_attempt_count` by mode and level group.
4. Stability metric
   - variance across seeds for each mode (strict + partial metrics).

If memory does not lift strict solves strongly, these analyses still show whether it makes plans safer, earlier-valid, and more stable.

---

## 7) What Would Count as an Interesting Week 9 Conclusion

Any of the following are meaningful:

- **Conclusion type A (direct winner):** one memory mode wins on strict solve rate under matched settings.
- **Conclusion type B (tradeoff):** reflection reduces deadlocks most, raw improves early validity, neither dominates on strict solves.
- **Conclusion type C (capacity effect):** memory helps up to a budget threshold, then degrades due to context noise.
- **Conclusion type D (harness interaction):** memory gains appear only when deadlock lookahead and box-id legality scaffolding are enabled.

This gives us multiple valid scientific outcomes instead of a single all-or-nothing success criterion.

---

## 8) Risks and Guardrails

Risks:
- Over-claiming from small sample sizes.
- Confounding memory effects with prompt/harness changes.
- Reporting only strict solve rate and missing real movement.

Guardrails:
- seed-matched comparisons,
- one-factor-at-a-time ablations,
- fixed decision rule for memory winner,
- always report strict + partial + failure-taxonomy metrics together.

---

## 9) Week 9 Deliverables

1. Updated results table for matched memory comparison.
2. Short ablation table for toggles (lookahead, repair budget, memory budget).
3. One per-level failure-shift table.
4. Curated GIF panel (success / invalid-plan / deadlock) from the same matched run.
5. Final memory recommendation with explicit confidence level and caveats.
