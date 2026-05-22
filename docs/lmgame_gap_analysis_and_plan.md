# LMGame-Bench vs Our Sokoban Pipeline

Date: 2026-05-22  
Branch context: `aditri-full-path` (synced with `full_path_kerui` + local repair chaining option)

## Purpose

This document summarizes:

1. How LMGame-Bench/GamingAgent evaluates Sokoban and what performance it reports.
2. How our Week 6 / Week 7 setup differs.
3. What changes we already implemented to move closer.
4. What to implement next to get stronger, more publishable baseline/results for this project.

## External References

- GamingAgent repository: [https://github.com/lmgame-org/GamingAgent](https://github.com/lmgame-org/GamingAgent)
- lmgame-Bench paper (arXiv): [https://arxiv.org/pdf/2505.15146](https://arxiv.org/pdf/2505.15146)

## 1) What LMGame-Bench reports for Sokoban

From the paper:

- Many models are near zero on Sokoban without harness support.
- Harness support yields non-zero Sokoban scores for top models (example values from Table 1):
  - `o3`: `2.0 -> 8.0` (No harness -> Harness)
  - `o4-mini`: `1.3 -> 5.3`
  - `gemini-2.5-pro`: `1.0 -> 4.3`
- Random baseline on their Sokoban score is `0.0`.
- Broad harness effect across games: without harness, many runs fail to beat random; with harness, most runs do.

Important interpretation detail:

- Their primary Sokoban metric is a game score/progression-style metric in their benchmark framework.
- Our primary metric is strict per-episode `solve_rate` plus status breakdown (`success`, `deadlock`, `timeout`, `invalid_plan`, etc.).
- Therefore, their "better" and our "better" are related but not directly numerically equivalent.

## 2) Test flow and pipeline differences (the critical gap)

## A. Environment and interface

LMGame-Bench:

- Gym-style loop with explicit action schema (`push up/down/left/right`, `up/down/left/right`, `no_op`).
- Fixed config in Sokoban env wrapper (`max_steps_episode`, stuck/unchanged-step termination).

Our project:

- Week 6 `main`: one-step primitive actions from LLM.
- Week 7 `full_path`: LLM emits full push-plan JSON, local verifier expands/executes.
- Strong local deadlock checks and detailed failure labeling.

Gap:

- LMGame uses an iterative harness-oriented action flow with progress scoring.
- We currently rely on a strict full-plan generation that often fails due to semantic invalidity of pushes.

## B. Harness modules

LMGame-Bench:

- Perception module (symbolic extraction from game state).
- Memory module (recent trajectory + reflection).
- Prompt standardization/optimization workflow.

Our project:

- We already provide textual board plus coordinate summary.
- We have raw trajectory memory and reflection heuristics.
- Prompt variants tested (`full_path_v2`-`v5`) but no robust lift yet.

Gap:

- Our bottleneck is not parsing; it is semantic push validity under hard levels.
- Their harness explicitly stabilizes perception + memory + prompting together.

## C. Metrics and reporting

LMGame-Bench:

- Emphasizes benchmark scores and model separation under harnessed/unharnessed settings.

Our project:

- Emphasizes strict `solve_rate` and failure taxonomy under matched conditions.

Gap:

- Our metrics are excellent for scientific diagnosis, but strict solve-only reporting makes partial improvements harder to show.

## 3) Our Week 6 / Week 7 result snapshot

Week 6 historical (one-step, matched three-agent eval):

- `no_memory`: `solve_rate=0.0`
- `raw_trajectory_memory`: `solve_rate=0.0`
- `reflection_heuristic`: `solve_rate=0.0`

Week 7/full-path status:

- Full-path executor + repair scaffold added.
- Latest sanity run in docs: `1/12` train success, heavy `invalid_plan`, repair not recovering additional success.

Latest local A/B check on this branch (repair chaining experiment):

- `repair_from_current_state=false`: `1/8` solves (`0.125`)
- `repair_from_current_state=true`: `1/8` solves (`0.125`)
- Dominant failure remains `invalid_plan` reasons (blocked/unreachable/missing box).

## 4) Changes already implemented to get closer

Implemented in this branch:

1. Full-path planning and verifier execution pipeline is active.
2. Optional verifier-guided repair loop exists (`--max_repair_attempts`).
3. New option added: `--repair_from_current_state`
   - Allows repair attempts to continue from the post-plan board instead of always resetting.
   - Added metadata plumbing and tests.
4. GPT-5 compatibility handling for Responses API temperature behavior is in place.

What this improved:

- Better architectural alignment with harness-style iterative correction.
- Better experiment control for A/B and failure analysis.

What did not improve yet:

- Solve rate is still low; semantic invalid plans remain the dominant bottleneck.

## 5) What to implement next (priority order)

Goal: make results more valuable and publishable by showing reliable baseline gains and clearer diagnostics.

## Priority 1: Local legal-push scaffolding in prompt (highest ROI)

Add to full-path prompt context:

- Current legal push set per box (`box_id`, push direction, required player cell, destination).
- For repair prompts, include legal alternatives near failed intent.

Why:

- Directly targets current top failure mode (`invalid_plan` due to impossible pushes).

Expected impact:

- Lower `invalid_plan_rate`.
- Higher executable plan fraction.

## Priority 2: Two-phase planner baseline

Add an explicit two-phase policy:

1. Propose only next `K` pushes (short horizon, e.g., 2-4).
2. Verify and execute.
3. Replan from updated board.

Why:

- Reduces stale-coordinate and long-horizon drift from single full-plan generation.

Expected impact:

- Better robustness than one-shot full plans.
- More controlled failure localization.

## Priority 3: Add progress-aligned secondary metrics

Keep strict solve-rate primary, but add:

- `boxes_on_target_max`
- `boxes_on_target_final`
- `valid_push_ratio`
- `executable_plan_ratio`
- `first_invalid_push_index` distribution

Why:

- Lets us show meaningful improvements even before solve-rate moves strongly.
- Makes comparisons to harness papers more interpretable.

## Priority 4: Stronger and fairer baselines

Run a fixed matrix:

- Agents: `no_memory`, `raw_trajectory_memory`, `reflection_heuristic`
- Policy variants: current full-path vs legal-push-scaffolded variant
- Same levels, seeds, `max_repair_attempts`, model, and LLM budget
- Sufficient episodes (at least 20 per condition; ideally 60 total+ for eval)

Why:

- Converts ad-hoc sanity checks into defensible baseline evidence.

## Priority 5: Reflection quality upgrade

Current reflection is often too generic. Improve by generating heuristics from:

- Critical transition snippets (first deadlock-causing push, first unrecoverable invalid segment)
- Structured failure reasons already logged

Why:

- Reflection becomes state-conditioned and actionable, not generic advice.

## 6) Suggested minimum "showable" package for project milestone

To produce a credible milestone quickly:

1. Implement Priority 1 (legal push set in prompts + repair alternatives).
2. Run 20-episode eval per condition (matched settings).
3. Report:
   - `solve_rate`
   - `invalid_plan_rate`
   - `deadlock_rate`
   - `valid_push_ratio` (new)
   - per-level breakdown
4. Include one qualitative case study where repaired/legality-informed planning prevents a previously invalid first push.

This creates a clear narrative:

- "We moved from mostly semantically invalid plans to more executable plans; solve-rate movement follows."

## 7) Risks and interpretation guardrails

- Do not over-claim cross-paper parity: metric definitions and environments differ.
- Keep Week 6 one-step and Week 7 full-path results separate in plots/tables.
- Keep API/budget failures out of comparisons (or report separately).
- Prefer fixed seeds and fixed prompt versions for every reported table.

## 8) Immediate next action recommendation

Implement legal push candidate injection in the full-path prompt and repair feedback, then rerun a matched 3-agent eval with adequate call budget.

That is the most direct path from "pipeline milestone" to "performance evidence" in this repo.

## 9) Implementation status and first matched run (2026-05-22)

This section records what was actually implemented and measured on branch `aditri-full-path`.

Implemented:

1. Legal push scaffolding in full-path prompts:
   - Added `legal_push_candidates` to full-path context at every planning attempt.
   - Prompt now includes a "Legal push candidates on this board" block.
2. Repair alternatives:
   - On failed attempts, repair feedback now includes "Legal alternatives near this state" based on current board legality, prioritized around failed box when possible.
3. Matched 3-condition evaluation:
   - Conditions: `no_memory`, `raw_trajectory_memory`, `reflection_heuristic`
   - Same model/settings/levels/seed/repair budget across all three conditions.

Validation:

- Test suite passed after implementation (`83 passed`).
- Evaluation validation errors: `0`.

Matched run artifact:

- `results/lmgame_closer_eval_20260522_130617/evaluation_summary.json`

Headline results from that run:

- `no_memory`: `solve_rate=0.0`, `invalid_plan_rate=0.3333`, `deadlock_rate=0.5833`
- `raw_trajectory_memory`: `solve_rate=0.0`, `invalid_plan_rate=0.25`, `deadlock_rate=0.5833`
- `reflection_heuristic`: `solve_rate=0.0`, `invalid_plan_rate=0.25`, `deadlock_rate=0.75`

Interpretation:

- Solve rate is still zero on this hard eval slice.
- Compared with `no_memory`, both memory conditions reduced `invalid_plan_rate` in this run (`0.3333` -> `0.25`), indicating some semantic-validity improvement.
- Reflection memory reduced invalid plans but shifted failures toward deadlocks in this run.

Context note:

- In an earlier no-memory run before this scaffolding (`results/ablation_auth_20260522_123722/no_chain/summary.json`), `invalid_plan_rate` was `0.875` on an 8-episode sample.  
  The current matched no-memory run shows `0.3333` on 12 episodes, suggesting the legal-push scaffold likely improved executable-plan quality. Keep this as indicative, not definitive, because run slices differ.

## 10) Next fix: box_id-first planning prompt (2026-05-22)

Implemented follow-up fix:

1. Prompt contract now strongly prefers `box_id`-based plan items:
   - Preferred: `{"box_id": integer, "push": "Up|Down|Left|Right"}`
   - Fallback: `{"box": [row, col], "push": "..."}`
2. Repair feedback now explicitly reminds the model to reuse listed `box_id` values.

Why:

- Coordinates can become stale across long plans.
- Stable box identity should reduce semantic drift and malformed mid-plan references.

Matched run artifact:

- `results/lmgame_closer_eval_boxid_20260522_132732/evaluation_summary.json`

Direct comparison vs prior legal-push-scaffold run:

- Old: `results/lmgame_closer_eval_20260522_130617/evaluation_summary.json`
- New: `results/lmgame_closer_eval_boxid_20260522_132732/evaluation_summary.json`

Per-agent headline deltas (old -> new):

- `no_memory`:
  - `solve_rate`: `0.0 -> 0.0833` (`0 -> 1` success)
  - `invalid_plan_rate`: `0.3333 -> 0.25`
  - `deadlock_rate`: `0.5833 -> 0.5833` (unchanged)
- `raw_trajectory_memory`:
  - `solve_rate`: `0.0 -> 0.0833` (`0 -> 1` success)
  - `invalid_plan_rate`: `0.25 -> 0.25` (unchanged)
  - `deadlock_rate`: `0.5833 -> 0.6667` (higher)
- `reflection_heuristic`:
  - `solve_rate`: `0.0 -> 0.0833` (`0 -> 1` success)
  - `invalid_plan_rate`: `0.25 -> 0.3333` (higher)
  - `deadlock_rate`: `0.75 -> 0.5833` (lower)

Interpretation:

- First non-zero solves now appear across all three conditions in this matched slice.
- The fix appears to improve practical solvability in this sample, but failure redistribution differs by memory type.
- This is encouraging but still a small sample; run a larger matched eval before final claims.

