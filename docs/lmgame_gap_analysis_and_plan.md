# LMGame-Bench vs Our Sokoban Pipeline

Date: 2026-05-22  
Branch context: `aditri-full-path` (synced with `full_path_kerui` + local repair chaining option)

Update note: the latest comprehensive implementation/progress log is in:

- `docs/progress_log_2026-05-22.md`

## Purpose

This document summarizes:

1. How LMGame-Bench/GamingAgent evaluates Sokoban and what performance it reports.
2. How our Week 6 / Week 7 setup differs.
3. What changes we already implemented to move closer.
4. **Complete attempt log** to improve solve rate (section 11).
5. **Why LMGame avoids our failure mode** structurally (section 12).
6. **Paper-inspired next steps** for this repo (section 13).

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

Completed since this list was first written (see sections 9–11 for measured outcomes):

- Priority 1 (legal-push scaffolding): **done**
- Box-id-first planning prompt: **done** (first non-zero solves on eval slice)

## Priority 1 (current): Short-horizon replanning loop

Add an explicit two-phase policy aligned with LMGame’s stepwise control loop:

1. Propose only next `K` legal pushes (e.g., 2–4), preferably chosen from `legal_push_candidates`.
2. Verify and execute locally.
3. Replan from the updated board until solve, deadlock, or budget.

Why:

- Reduces stale-coordinate and long-horizon drift from single full-plan generation.
- Matches the paper’s iterative “observe → act → verify” harness rather than one-shot full plans.

Expected impact:

- Higher `executed_push_count / planned_push_count` ratio.
- Better robustness on Boxoban eval levels (not only tutorial levels).

## Priority 2: Constrained action selection (paper-style action menu)

Instead of free-form JSON over the whole board, require each step to pick from the enumerated legal set (or `no_op` / movement primitives if we add a hybrid mode).

Why:

- LMGame’s Sokoban interface uses a small fixed action vocabulary; the env rejects illegal semantics before they compound across a long plan.

Expected impact:

- Large drop in `invalid_plan` and `first_attempt_invalid_plan_count`.

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

## Priority 4: Reflection quality upgrade (paper-style state-conditioned memory)

Current reflection heuristics are often generic. Regenerate from:

- Critical transition snippets (first deadlock push, first invalid segment).
- Structured failure reasons already logged in episode JSON.

Why:

- LMGame reflection is tied to how state changed after each action, not generic “avoid corners” advice.

## Priority 5: Stronger and fairer baselines

Run a fixed matrix:

- Agents: `no_memory`, `raw_trajectory_memory`, `reflection_heuristic`
- Policy variants: current full-path vs legal-push-scaffolded variant
- Same levels, seeds, `max_repair_attempts`, model, and LLM budget
- Sufficient episodes (at least 20 per condition; ideally 60 total+ for eval)

Why:

- Converts ad-hoc sanity checks into defensible baseline evidence.

## 6) Suggested minimum "showable" package for project milestone

To produce a credible milestone quickly:

1. Implement short-horizon replanning (section 13, Step 1).
2. Run 20-episode eval per condition (matched settings).
3. Report:
   - `solve_rate` (strict) + per-level breakdown
   - `invalid_plan_rate`, `deadlock_rate`
   - `executed_push_count / planned_push_count`
   - `boxes_on_target_max` / progress proxy
4. Include one qualitative case study on a **Boxoban eval** level (not only `wall_push_001`).

Narrative target:

- "We shifted from monolithic invalid full plans to stepwise legal actions; progress metrics move before strict solve rate on hard levels."

## 7) Risks and interpretation guardrails

- Do not over-claim cross-paper parity: metric definitions and environments differ.
- Keep Week 6 one-step and Week 7 full-path results separate in plots/tables.
- Keep API/budget failures out of comparisons (or report separately).
- Prefer fixed seeds and fixed prompt versions for every reported table.

## 8) Immediate next action recommendation

Run **short-horizon replanning** (Priority 1) with the existing legal-push + `box_id` prompt, then rerun a matched 3-agent eval at **≥20 episodes per condition** on the eval split.

Report both strict `solve_rate` and progress-style metrics (`executed_push_count / planned_push_count`, boxes-on-target max/final) so improvements are visible even before full solves appear on hard Boxoban levels.

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
- Important caveat: all three successes in the box-id run are on `wall_push_001` (tutorial-style level, `solve_rate=1.0` per level). **Zero** solves on Boxoban eval/medium levels in that run.

---

## 11) Complete log: attempts to improve solve rate

This section is the authoritative chronological record of what we tried, why, and what changed. It complements Week 6/7 docs (`docs/week6_results.md`, `docs/week7_update.md`).

### A. Evaluation setup (held constant for fair comparisons)

| Setting | Value |
| --- | --- |
| Levels file | `levels/v2_pilot.json` (12 train + 12 eval) |
| Eval split for headline comparisons | `level_split=eval` (12 levels × 1 episode per agent per run unless noted) |
| Agents | `no_memory`, `raw_trajectory_memory`, `reflection_heuristic` |
| Memory banks | `memory_banks/raw_failures.json`, `memory_banks/reflection_heuristics.json` (built from train failures) |
| Model (recent LMGame-closer runs) | `gpt-4.1-mini`, `temperature=0` |
| Repair budget (recent runs) | `max_repair_attempts=2` |
| Seed | `42` |

### B. Attempt chronology

| # | Phase | What we changed | Hypothesis | Measured outcome | Artifact / notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Week 6 baseline | One-step primitive actions (`main`); ASCII grid + coordinates; optional memory injection | Memory reduces repeated mistakes on hard eval | **0% solve** all agents; **85% timeout**, 15% deadlock (`no_memory` / raw) | `results/v2_task3_large_evaluation_summary.json`, `docs/week6_results.md` |
| 2 | Week 7 architecture | Full-path push-intent planning + local BFS executor + richer deadlock checks | Planning whole puzzle beats greedy one-step | Pipeline works; semantic invalidity dominates | `docs/week7_update.md` |
| 3 | Prompt sweep | `full_path_v2` → `v3` (legality wording) → `v4` (state trace) → `v5` (indexed boxes) | Better prompts reduce invalid first pushes | **No robust lift**; v5 still many illegal early pushes | Reverted active prompt to `full_path_v2` per `docs/week7_update.md` |
| 4 | Repair loop | `--max_repair_attempts=1` with verifier feedback; regenerate full plan from **original** board | Feedback recovers from first invalid push | Train sanity: **1/12** solve, **10** `invalid_plan`, **0** `success_after_repair` | `results/full_path_v5_repair1_train_check` |
| 5 | Repair chaining | `--repair_from_current_state` (continue from post-partial-plan board vs reset) | Chaining partial progress improves solve rate | A/B on 8 episodes: **1/8** both arms (**0.125**); no gain | `results/ablation_auth_20260522_123722/` |
| 6 | Legal-push scaffold | Inject `legal_push_candidates` into every plan/repair prompt; repair lists nearby legal alternatives | Model stops proposing impossible pushes | Matched 36-ep eval: **0% solve**; overall `invalid_plan_rate` **0.278** (down from ~0.33 `no_memory`); `executed/planned` pushes still low (~**0.14**) | `results/lmgame_closer_eval_20260522_130617/` |
| 7 | Box-id prompt | Prefer `{"box_id", "push"}` from legal list; repair reminds to reuse `box_id` | Stable box identity fixes coordinate drift | Matched 36-ep eval: overall **8.33% solve** (3/36); per-agent **8.33%** each; **all successes on `wall_push_001` only**; Boxoban eval/medium still **0%** | `results/lmgame_closer_eval_boxid_20260522_132732/` |
| 8 | Memory comparison (within #6–7) | Same harness, three memory types | Heuristic memory beats raw on invalid/deadlock | Mixed: raw/reflection sometimes lower `invalid_plan_rate`, but **no consistent solve lift**; reflection can shift failures toward deadlock | Same result dirs as #6–7 |

### C. What actually moved vs what did not

**Moved (secondary metrics):**

- `invalid_plan_rate` for `no_memory` appears lower with legal-push scaffold than an earlier 8-ep ablation without it (**0.875 → 0.333** on different slices — indicative only).
- First **non-zero** strict solves after box-id prompt (**3/36**), all on one tutorial level.
- `success_without_repair_count = 3`, `success_after_repair_count = 0` in box-id run → repairs are not the source of wins yet.

**Did not move (primary metric on hard eval):**

- **0%** solve on Boxoban eval/medium levels after all Week 7 harness work.
- Memory type still does not separate clearly on solve rate at this sample size.
- Dominant failure mix remains **`deadlock` + `invalid_plan`**, not API/infra (`api_error_rate = 0`).

### D. Root cause summary (why solve rate stays near zero on hard levels)

1. **Planning horizon mismatch:** We ask for a full push sequence in one shot; the model drifts after the first few pushes even with legal candidates listed.
2. **Semantic errors survive prompting:** Legal list is advisory; the model can still emit off-menu pushes or wrong `box_id`.
3. **Strict metric:** Partial progress (boxes moved, valid pushes executed) does not count as success; LMGame’s score would give partial credit.
4. **Repair regenerates whole plans** and has not converted invalid-first-attempt episodes into solves on hard levels.

---

## 12) Why LMGame-Bench does not run into this issue the same way

LMGame/GamingAgent and our repo both use symbolic state, but the **control loop and success definition** differ. That is the main reason their Sokoban numbers are non-zero under harness while ours stayed at 0% on hard eval for a long time.

### Structural differences

| Dimension | LMGame-Bench (paper + GamingAgent) | Our pipeline (Week 7 branch) |
| --- | --- | --- |
| Control loop | **Stepwise** Gym loop: one action per LLM call, env advances | **Batch plan:** LLM emits full push JSON, local verifier executes until invalid/deadlock/solve |
| Action interface | Fixed vocabulary: `up/down/left/right`, `push up/...`, `no_op` | Free-form push intents (`box` / `box_id` + direction) over multiple steps |
| Invalidity handling | Illegal step is an environment concern each turn; agent re-observes | Invalidity often appears **mid-plan**; entire attempt may collapse after one bad push |
| Metric | **Game score / progression** (partial credit for boxes on targets, etc.) | **Strict solve_rate** + failure taxonomy |
| Harness stack | **Perception + memory + prompt optimization** evaluated ablated | Perception-like ASCII+coords yes; memory yes; **no** systematic prompt opt loop |
| Horizon | Short decision each step; state refreshed every turn | Long horizon in one JSON array; coordinate/`box_id` drift across steps |
| Paper finding | >3/4 models score **0** without harness; top models gain with harness (e.g. o3 **2.0→8.0**) | gpt-4.1-mini **0%** on hard eval through Week 6; **8.33%** overall only after box-id, concentrated on one easy level |

### Mechanism (plain language)

LMGame does not rely on the model producing a **globally consistent multi-push program** in one response. Each turn it:

1. Converts backend state to a **standardized symbolic observation** (perception module).
2. Asks for **one** action from a small menu.
3. Lets the environment apply physics and scoring.
4. Updates **transient memory** and optional **reflection** from the last transition.

So the model’s job is “what is the best **next** legal action?” not “emit a 10-step push script that stays valid after each box moves.” Invalid plans still happen, but they are **localized to one step**, recovered on the next observation, and often still earn **progress score** even when the level is unsolved.

Our harness is stronger for **diagnosis** (exact `invalid_plan` vs `deadlock` reasons, repair logs, planned vs executed push counts). LMGame’s harness is stronger for **getting any learning signal** on hard puzzles under a bounded action menu.

### What we already match from their paper

- Symbolic board + coordinates (perception-like input).
- Transient trajectory memory ≈ our raw failure trajectories.
- Reflection-style memory ≈ our heuristic bank (quality still weaker than theirs).
- External verification (our BFS executor + deadlock checks ≈ their env step validation).

### What we still do not match (and causes the solve-rate gap)

- **Iterative replanning** instead of monolithic full-path JSON.
- **Progress-aligned reporting** alongside strict solve rate.
- **Constrained action selection** from a per-step legal menu.
- **Prompt optimization workflow** (they treat prompting as a first-class harness component).

---

## 13) Next steps inspired by the LMGame paper

Mapped from [LMGame-Bench (arXiv:2505.15146)](https://arxiv.org/pdf/2505.15146) and [GamingAgent](https://github.com/lmgame-org/GamingAgent) to concrete work in this repo. Order reflects expected ROI for **our** memory-comparison goals.

### Step 1 — Adopt their control loop shape (highest ROI)

Implement **short-horizon replanning** on top of existing `legal_push_candidates` + `box_id`:

- Each LLM call: choose 1–`K` pushes from the legal list only.
- Execute locally, refresh board, append to trajectory, repeat.
- Keep `max_repair_attempts` for parse/execution failures within a chunk.

*Paper alignment:* stepwise agent–environment interaction with perception refresh each turn.

*Success criteria:* `executed_push_count / planned_push_count` ↑; first Boxoban eval level with non-zero solve; optional comparison vs current full-path mode on same 36-episode matrix.

### Step 2 — Add LMGame-style progress metrics

Log and report alongside `solve_rate`:

- `boxes_on_target_max`, `boxes_on_target_final`
- `valid_push_ratio` (accepted pushes / proposed pushes)
- `progress_score` proxy (e.g. fraction of boxes on targets at episode end)

*Paper alignment:* their Sokoban score rewards progression, not only terminal solve.

*Why for us:* lets the report show memory/harness improvements even when strict solve is still 0 on hard levels.

### Step 3 — Harden perception / action menu (perception > memory)

Paper reports perception matters more than memory for Sokoban. We should:

- Keep ASCII + coordinate summary (already done).
- Add explicit **player position, box ids, target set, and per-box “reachable push directions”** every turn (tighten what we already send as `legal_push_candidates`).
- Optionally add a **single-action mode**: model returns an index into `legal_push_candidates` (or `no_op`) to eliminate free-form JSON errors.

*Paper alignment:* standardized symbolic perception + small action space.

### Step 4 — Improve reflection memory like their transition analysis

Regenerate `reflection_heuristics.json` from:

- State before/after each failed push.
- Verifier `failure_reason` strings.
- One “critical transition” snippet per episode (first deadlock or first invalid push).

*Paper alignment:* reflection over how state changed and whether the action helped.

*Experiment:* rerun matched 3-agent eval; compare `deadlock_rate` and `invalid_plan_rate`, not only solve rate.

### Step 5 — Fair baseline matrix (science)

Run at **≥20 episodes per condition** with frozen:

- levels, seed, model, `max_repair_attempts`, replan horizon `K`, prompt version.

Compare rows: `full_path` vs `replan_K` × memory type.

*Paper alignment:* harnessed vs unharnessed style comparison, but our variable is **memory representation** under the same harness.

### Step 6 — Optional prompt optimization pass

LMGame uses prompt optimization as harness infrastructure. Lightweight version for us:

- Fix a small dev subset of train levels.
- Sweep 2–3 prompt templates (menu-select vs free JSON vs hybrid).
- Lock winner before large eval.

*Guardrail:* do not conflate prompt gains with memory gains — report them as separate ablation rows.

### What we should **not** claim without new runs

- Parity with LMGame Table 1 scores (different env, metric, and model suite).
- Memory wins on solve rate until hard-eval solves appear at n≥20 per condition.
- That box-id prompt “solves Sokoban” — current wins are isolated to `wall_push_001`.

### Minimum next experiment (one week)

1. Implement replan-`K=2` with menu selection from `legal_push_candidates`.
2. Run 36-episode matched eval (12 per agent) + report progress metrics.
3. Update this doc with a new row in section 11 and a comparison table vs `lmgame_closer_eval_boxid_20260522_132732`.

