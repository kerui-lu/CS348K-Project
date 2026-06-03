# The Impact of Memory Strategies on LLM-Based Sokoban Gameplay

**Aditri Patil** — apatil26@stanford.edu  
**Kerui Lu** — keruilu@stanford.edu

CS348K (Visual Computing Systems) · 2nd June 2026

---

## Main Takeaways

1. Overall, memory helps, but different memory representations improve different aspects of gameplay. Heuristics improve legality but make less progress. Raw memory trajectory keeps richer spatial context reducing deadlock failures. Compact summary provides a clearer signal. 

2. **Instance-specific feedback beats global rules.** Same-level retry at 54.2% vs. train-to-eval global heuristics at **37.5%** (top-3 rules). 

3. **Harness quality was the technical crux**—full-path JSON planning, local verifier (BFS + deadlock checks), token-budgeted memory renderers, and stratified levels—not raw API usage.
---

## Background and Setup

### What is Sokoban?

Sokoban is a grid puzzle where the goal is to push every box onto a target square. The player moves one cell at a time and can only push boxes into open spaces, never through walls or into another box (also  trying to say if 2 boxes adjacent cant push).

![Sokoban Example](reference_gifs/boxoban_eval_000_000_reference_solution_success.gif)

Here is an example of a Sokoban puzzle where the blue circle is the user who is pushing the the brown boxes onto the yellow target loactions. Any box can go on any target location.


### Why is Sokoban hard for LLMs?

 A single bad push can create an irreversible deadlock, so mistakes compound over a long horizon.

The main three core challenges:

1. **Irreversible moves lead to deadlocks** — As you can only push boxes, many moves lead to irreversable states like deadlocks. This figure shows various kinds of deadlocks found in the game: 

![Deadlock Examples](figures/deadlock_examples.png)

For example, in the figure above, all the deadlocks are because a single push leads to the box being trapped against a box and a wall or a box pushed against a wall. For example, Deadlock 3 is caused a non-target corner push which the box cannot escape and deadlock 5 is caused by a single push that creates a 2x2 freeze of boxes.

While it is possible to check for deadlocks before executing a push, a irreversible mistake 5 steps prior may lead the LLM to make a mistake that leads to a deadlock.

2. **Coordinate tracking of multiple boxes** — The LLM has to track various spatial aspects like the coordinates of the boxes, target locations, player position, and walls.
each push updates box coordinates; one wrong `[row, col]` invalidates the rest of the plan.




3. **Long-horizon planning** — solutions require many dependent pushes; early errors prune valid futures.

### Problem definition

| Aspect | Specification |
|--------|----------------|
| **Inputs** | ASCII board (walls, boxes, targets, player) plus game rules |
| **Outputs** | JSON array of semantic push intents: `{"box": [r, c], "push": "Up/Down/Left/Right"}` per step |
| **Goals** | All boxes on targets with **no invalid steps** during execution |
| **Constraints** | Fixed attempt budget, token budget, deterministic temperature-0 runs |
| **Primary metric** | **Solve rate** — fraction of levels fully solved |
| **Dataset** | `levels/v3_boxoban_balanced.json`: 24 train + 24 eval Boxoban levels (10×10, 4 boxes), stratified by source family and difficulty bucket |

### Research questions

1. How well does an LLM solve Sokoban **without** memory?
2. What **failure types** dominate?
3. Can **memory of prior failures** improve performance?
4. Which memory format helps most: **raw trajectory**, **compact verifier summary**, or **reflection heuristics**?
5. Can memory **generalize** from train failures to held-out eval levels?

**Central hypothesis (falsifiable):** *Instance-specific feedback on the current puzzle* (same-level retry) will outperform *abstract rules* distilled from other puzzles (train-to-eval heuristics).

### Technical crux

The hard part was not “call an LLM API” but **harness engineering**:

- Build a **deterministic executor** that expands each push into walking moves and rejects illegal geometry.
- **Diagnose failures** locally (unreachable standing cell, blocked destination, deadlock, truncated plan).
- **Compress failures into token-budgeted memory** without leaking full solution spoilers.
- **Isolate memory effects** with matched models, caches, and stratified benchmarks.

Early runs showed ~48% of failures are **invalid plans** (local legality), not high-level strategy errors. That shifted the project toward verifier-grounded memory rather than generic planning tips.

### Prior benchmarks

LMGame Bench (Zhang et al.) reports weak LLM Sokoban performance with visual + memory modules but does not compare memory *formats*. We use Boxoban levels and our own verifier-centric memory ablations.

---

## Approach

### Research motivation

Prior work motivates three memory families:

- **LMGame** — memory module for Sokoban without comparing representations.
- **Reflexion** — linguistic feedback after failure (coding agents).
- **Synapse** — retrieved trajectories as in-context examples for control tasks.

We ask: *which representation* helps Sokoban full-path planning under a fixed verifier?

### Four memory strategies

| Strategy | What is stored | Shown to planner on retry |
|----------|----------------|---------------------------|
| **No memory** | — | Empty-state message only |
| **Raw trajectory** | Compact failure evidence: boards, failed push, push log | Same-level raw evidence block |
| **Compact summary** | Failed push index, failure subtype, verifier reason | Verifier-summary block (~200 tokens) |
| **Reflection heuristic** | LLM-distilled rules from failures | Numbered heuristic strings (~300 tokens) |

**Same-level retry:** up to **K = 3** attempts per level; memory updates only after a failed attempt on *that* level.

**Train-to-eval:** failures on 24 train levels → global heuristic bank → same top-K rules on all 24 eval levels (no raw cross-level trajectories).

### System architecture

The planner LLM outputs one JSON push plan per attempt. A **local verifier** (not the LLM) parses each intent, checks standing-cell reachability with BFS, executes walking moves plus pushes, and records structured failure evidence. On retry, that evidence is rendered as raw trajectory memory, compact verifier summary, or—after a separate reflection LLM call—heuristic rules.

![Planner, verifier, and memory pipeline](figures/memory_dataflow.png)

*Figure 1. End-to-end pipeline (slides: “Our System” / “Memory Creation”). The LLM never executes moves; the harness does. Heuristic conditions add a reflection step between failures that distills rules for the next attempt.*

### Full-path planning loop

1. **Prompt** (`full_path_v2_1`): rules, board, coordinates, planning checklist, memory block, JSON-only output contract.
2. **LLM** returns a complete push plan (not single-step actions).
3. **Verifier** for each push: resolve box coordinate, compute required standing cell, BFS reachability, destination legality, execute, deadlock checks.
4. **Failure typing:** `invalid_plan`, `deadlock`, `plan_exhausted`, etc., with subtypes such as `unreachable_standing_cell`.
5. **Memory update** (retry conditions): append compressed episode; render memory for next attempt.

**Deadlock checks (local):** non-target corner, wall-segment trap, 2×2 freeze, two-box freeze.

### Level suite

Levels come from [Boxoban](https://github.com/google-deepmind/boxoban-levels). Within each source family (`medium`, `unfiltered`), candidates are sorted by a structural difficulty score and split into **open**, **middle**, and **constrained** thirds (4 levels each per split × family).

| Bucket | Avg. wall density | Avg. reachable ratio | Avg. initial legal pushes |
|--------|------------------:|---------------------:|--------------------------:|
| Open | 0.618 | 0.892 | 10.38 |
| Middle | 0.682 | 0.577 | 6.62 |
| Constrained | 0.690 | 0.063 | 1.25 |

### Hypotheses tested

| ID | Hypothesis | Outcome |
|----|------------|---------|
| **H1** | With verifier feedback, memory improves action quality vs. drowning in legality noise | **Supported** — see same-level solve-rate results below |
| **H2** | Reflection heuristics outperform raw memory on **deadlock avoidance** | **Not supported** — compact summary increases deadlock share vs. raw |
| **H3** | Fair memory comparison requires a working harness | **Supported** — token and prompt ablations preceded memory experiments |

### What we started with

We began from the course Sokoban scaffold (grid env, baseline agents) and the Week 6 one-step LLM policy. **Our work:** full-path JSON planning, V3 verifier executor, four memory conditions, stratified Boxoban benchmark, evaluation/plotting scripts, and reflection pipeline (`sokoban_memory/`).

---

## Evaluation and Results

### Definition of success

**Primary success:** Higher **solve rate** on held-out eval levels when memory is enabled under a fixed attempt budget.

**Secondary evidence:**

- Lower **invalid-plan** and **unreachable_standing_cell** rates
- Higher **best goal completion** (max boxes on targets before failure, averaged over levels)
- Interpretable **failure-subtype shifts** under each memory format

**Operational solved:** all boxes on targets, zero invalid executed steps (verifier-enforced).

### Experimental setup

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2`, reasoning effort **low** |
| `max_output_tokens` | **16384** (8192 caused `empty_output` — reasoning consumed the budget) |
| Temperature | 0 |
| Eval split | 24 levels in `v3_boxoban_balanced.json` |
| Same-level attempts | **K = 3** |
| Baseline | Single-shot `no_memory` (also 33.3% on train) |
| Reproducibility | Per-condition LLM response cache namespaces |

**Baselines compared:** single-shot no memory; same-level compact, raw, heuristic retry; train-to-eval global heuristics (top-3 and all-16 variants).

**Why no raw memory in train-to-eval:** raw trajectories are **level-specific**; cross-level prompts may only use abstracted `global_allowed` heuristics (design choice to avoid leaking eval layouts).

### How good are LLMs at Sokoban initially?

**Single-shot solve rate: 33.3%** (8/24) on both train and eval.

| Failure category | Eval count (of 24) |
|------------------|-------------------:|
| Invalid plan | 13 |
| Success | 8 |
| Deadlock | 2 |
| Plan exhausted | 1 |

**Dominant subtype:** `unreachable_standing_cell` — the model names a push whose required standing cell is not reachable given current walls/boxes. This is a **local legality** error, not necessarily wrong high-level strategy.

The harness visualizes where execution stops: invalid plans fail on the first illegal push; deadlocks occur after one or more successful pushes when the board enters a locally detected trap.

![Example invalid-plan and deadlock trajectories on the same board family](figures/trajectory_examples_labeled.png)

*Figure 2. Representative failure trajectories. Invalid plans (left) fail before meaningful progress when the verifier rejects a push. Deadlocks (right) execute several legal pushes then trap the board. This is the evidence raw memory compresses and compact summary abstracts.*

### Same-level retry results

We compare four conditions on the **eval split** with up to three attempts per level: single-shot baseline, compact verifier-summary retry, raw same-level trajectory retry, and same-level heuristic retry.

![Eval solve rate after up to K=3 attempts per level](figures/same_level_retry_condition_solve_rate_k3.png)

*Figure 3. **Solve rate @ K=3** on 24 eval levels. Compact summary and raw trajectory both reach **13/24 (54.2%)**, up from **8/24 (33.3%)** single-shot. Heuristic same-level reaches **12/24 (50.0%)**. This is the primary evidence that instance-specific memory helps under a fixed attempt budget.*

| Condition | Solve @ 1 | Solve @ 3 | Levels gained vs. baseline |
|-----------|----------:|----------:|---------------------------:|
| Single-shot (no memory) | 33.3% | 33.3% | — |
| Compact summary | 29.2% | **54.2%** | +5 |
| Raw trajectory | 25.0% | **54.2%** | +5 |
| Heuristic (same-level) | 20.8% | **50.0%** | +4 |

**Why solve @ 1 differs (not used to rank memory):** Attempt 1 on retry conditions still has **empty memory**, but the prompt `Condition:` label and empty-state text differ from the single-shot baseline, and attempt-0 seeds follow `seed + level_index × 3` instead of `seed + level_index`. We therefore treat **solve @ 3** as the fair memory comparison, matching the slide note that solve @ 1 is a confounded “primed single-shot” diagnostic.

**Why solve @ 3 matters:** Compact and raw **tie** at 54.2% but solve **different** level subsets—raw rescues at least one level compact misses. Heuristic lags slightly but still gains four levels over baseline, at the cost of an extra reflection LLM call per failure.

![Cumulative count of eval levels solved by attempt index](figures/same_level_retry_cumulative_solved_by_attempt.png)

*Figure 4. **Cumulative levels solved** (compact, raw, heuristic vs. flat baseline). Most improvement appears by **attempt 2**; attempt 3 shows diminishing returns. This matches the presentation narrative: memory helps quickly once verifier feedback is available, then plateaus.*

### Partial progress improves even when solve rate lags

Not every retry produces a full solve. **Best goal completion** tracks the maximum fraction of boxes placed on targets before failure, averaged across levels.

![Average best goal completion @ K=3 by memory condition](figures/same_level_retry_condition_avg_best_goal_completion_k3.png)

*Figure 5. **Avg. best goal completion @ K=3.** Memory raises partial progress even when a level never reaches 100% solved—supporting the slide claim that “compact improves partial progress.” Compact and raw sit above heuristic; all three beat the implicit single-shot baseline on this metric.*

### Failure analysis: what memory changes

Failures are still common after memory; the question is whether the **mix** of failure types shifts in a useful way.

![Failure subtype mix across all K=3 attempts, by condition](figures/same_level_retry_failure_subtype_mix_k3.png)

*Figure 6. **Overall failure subtype mix** (all attempts, K=3). **Unreachable standing cell** remains the largest bucket everywhere, confirming that local legality is still the bottleneck. Memory narrows that bucket relative to baseline attempts but does not eliminate it.*

![Failure status and subtype breakdown by memory condition](figures/same_level_retry_failure_subtypes_by_condition.png)

*Figure 7. **Failure subtypes stacked by condition.** Compact summary reduces unreachable-standing-cell share versus baseline (~60% vs. ~81% of failed attempts) but can increase **deadlock** versus raw—the model sometimes executes farther into the puzzle before trapping. Heuristic runs show more **plan_exhausted** (legal but incomplete plans), consistent with generic `reflection_v2` rules that improve wording without always lengthening the plan.*

![Invalid-plan subtypes only, by condition](figures/same_level_retry_invalid_plan_subtypes_by_condition.png)

*Figure 8. **Invalid-plan subtypes.** Most invalid plans are still unreachable standing cells; compact memory has the lowest share in this slice. Raw memory occasionally helps via context but can also introduce noise (higher invalid-plan count overall in the V3 run).*

**Unreachable standing cell (approximate share of failed attempts):**

| Condition | Share |
|-----------|------:|
| Baseline | ~81% |
| Compact summary | ~60% |
| Raw trajectory | ~69% |
| Heuristic | ~60% |

**Slide-aligned trade-offs:**

- **Compact summary** — best at cutting illegal pushes; verifier line is short and actionable (example below).
- **Raw trajectory** — same solve rate as compact but different rescued levels; more tokens of board context.
- **Heuristic** — improves general legality language; more `plan_exhausted`; extra reflection call.

Example **compact summary** injected on retry:

```
Verifier summary from the previous same-level attempt:
level_id: boxoban_medium_valid_000_092
failed_push_index: 5
failure_subtype: blocked_standing_cell
verifier_reason: required_push_position_blocked_by_box
```

### Representative distilled heuristics

Reflection on failures often produced rules aligned with observed errors (slides: top three themes):

1. **Standing-cell reachability** — verify the standing cell is reachable in the *current* layout before emitting a push.
2. **Corridor fragility** — do not park a box in a 1-wide corridor needed for later access.
3. **Self-blocking** — avoid pushes that wall off the only future approach square.

These improve **legality language** but underperformed compact/raw on **full solves** in our eval run when generated with generic `reflection_v2`.

### Train-to-eval heuristic generalization

Cross-level memory pools train failures into **global heuristics** rendered on every eval level. Raw trajectories are excluded cross-level by design.

![Eval solve rate for train-to-eval heuristic conditions](figures/train_to_eval_condition_solve_rate.png)

*Figure 9. **Train-to-eval solve rate.** Top-3 global heuristics improve from **8/24 to 9/24 (37.5%)**—only one extra level versus no-memory eval. Rendering **all 16** train-derived rules does not help (back to 33.3%), supporting the slide “more heuristics ≠ better performance.”*

| Condition | Eval solve rate | Levels solved |
|-----------|----------------:|--------------:|
| No-memory eval baseline | 33.3% | 8/24 |
| Train → eval (top 3 heuristics) | **37.5%** | 9/24 |
| Train → eval (all 16 heuristics) | 33.3% | 8/24 |

![Train-to-eval best goal completion by condition](figures/train_to_eval_avg_best_goal_completion.png)

*Figure 10. **Best goal completion (train-to-eval).** Global heuristics raise average partial progress (~51% → ~59% with top-3 rules) even when full solve rate barely moves. Memory is doing useful work—avoiding early catastrophes—without guaranteeing complete solutions.*

![Train-to-eval failure subtype mix](figures/train_to_eval_failure_subtype_mix.png)

*Figure 11. **Train-to-eval failure mix.** Unreachable standing cells still dominate; adding more heuristic text does not change the fundamental error mode. Cross-level abstraction lacks the coordinate-specific signal that same-level compact/raw provide.*

**Why transfer is limited (interpretation; partly speculative):**

- **Domain shift** between train and eval layouts within the same bucket.
- **Abstraction gap** — rules like “check reachability” still require expensive per-puzzle re-reasoning.
- **Prompt bloat** — 16 heuristics consume context without puzzle-specific coordinates.

### Synthesis: does memory help?

| Claim | Evidence |
|-------|----------|
| Memory helps on **same puzzle** | Figures 3–4: +5–6 levels at K=3 |
| **Instance-specific** beats **global** rules | 54.2% same-level vs. 37.5% train-to-eval (Figure 9) |
| **Format matters** | Figures 6–8: compact on legality, raw on complementary rescues, heuristic on partial legality |
| Bottleneck remains **local validity** | Unreachable standing cell still #1 in Figures 6–7 and 11 |

### Limitations and open questions

1. **Invalid-plan ceiling** — enumerate legal pushes from the verifier for the planner.
2. **k=1 confound** — not used to rank memory modes (see same-level retry section).
3. **Medium-stratum weakness** — medium-middle eval levels often stay at 0% solve.
4. **Heuristic generator** — try `v1_specific` / `v3_hybrid_verifier` reflection prompts.

---

## Team Responsibilities

**Aditri Patil:** Problem framing and hypotheses; full-path planner + verifier implementation; memory renderers (raw, compact, heuristic); prompt iteration (`full_path_v2_1`); V3 evaluation runs and analysis; figures and final report.

**Kerui Lu:** Architecture and level-suite design; stratified Boxoban selection; evaluation coordination; failure-mode debugging; memory-format design feedback.

---

## References

1. Zhang, C., et al. (2024). *LMGame Bench: How Good are LLMs at Playing Games?*
2. Google DeepMind. *Boxoban Levels.* https://github.com/google-deepmind/boxoban-levels
3. Shinn, N., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.*
4. Liang, P., et al. (2024). *Synapse: Trajectory-as-Exemplar Prompting for Robustness.*
5. CS348K-Project repository: `sokoban_memory/`, `levels/v3_boxoban_balanced.json`, `docs/v3_memory_evaluation_report.md`.

---

## Appendix (experimental details)

**Token ablation:** At 8192 max output tokens, `gpt-5.2` often returned no parseable plan (`empty_output`); 16384 was required for visible JSON.

**Full-path vs. one-step:** One-step policies lost long-horizon structure; full-path matches how humans plan Sokoban (sequence of pushes).

**K = 3:** Attempt 2 captures most gains; K=5+ risks context bloat from accumulated memory.

**Figure regeneration** (from repo root):

```bash
.venv/bin/python scripts/plot_same_level_k3_condition_comparison.py
.venv/bin/python scripts/plot_failure_subtypes_condition_comparison.py
.venv/bin/python scripts/plot_train_to_eval_condition_comparison.py
```

Outputs are written to `docs/figures/`.
