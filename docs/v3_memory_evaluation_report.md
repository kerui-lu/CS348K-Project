# V3 Sokoban Memory Evaluation Report

## Summary

This report documents the current V3 Sokoban memory-evaluation design and the results executed on `levels/v3_boxoban_balanced.json`.

The current executed results include the planner-validity gate and the first same-level retry condition, `verifier_summary_retry`. The gate shows that `gpt-5.2` with the active full-path prompt can solve some V3 Boxoban levels, but invalid local push plans remain the dominant failure mode. The verifier-summary retry improves held-out eval solve rate under a fixed attempt budget, but raw trajectory memory, heuristic same-level memory, and train-to-eval heuristic generalization are not yet complete.

Main result:

- Balanced benchmark: 48 levels, 24 train and 24 eval.
- Model: `gpt-5.2`.
- Reasoning effort: `low`.
- Prompt version: `full_path_v2_1`.
- Main token cap used after ablation: `16384`.
- Overall single-shot solve rate: `16/48 = 33.3%`.
- Overall invalid-plan rate: `23/48 = 47.9%`.
- Dominant failure subtype: `unreachable_standing_cell`.
- First same-level retry result: `verifier_summary_retry` on eval improves `solve_rate@K` from `8/24 = 33.3%` to `13/24 = 54.2%` with `K = 3`.

Because invalid plans are still common, same-level raw/heuristic retry and train-to-eval heuristic generalization should be interpreted separately from verifier-guided scaffolding.

## Experimental Design

### Benchmark Suites

The final benchmark uses Boxoban-based levels only.

| Suite | Use | Contents |
| --- | --- | --- |
| `levels/v2_pilot.json` | Debugging and prompt sanity checks only | mixed hand-written and pilot Boxoban levels |
| `levels/v3_boxoban_balanced.json` | Main benchmark | 24 train + 24 eval |
| `levels/v3_boxoban_ood.json` | Optional OOD eval | 16 eval-only levels |

The balanced benchmark is stratified by:

- split: `train`, `eval`
- source family: `medium`, `unfiltered`
- difficulty bucket: `open`, `middle`, `constrained`

Each split contains:

| Source family | Open | Middle | Constrained | Total |
| --- | ---: | ---: | ---: | ---: |
| `medium` | 4 | 4 | 4 | 12 |
| `unfiltered` | 4 | 4 | 4 | 12 |

### Memory Conditions

The intended V3 memory evaluation separates same-level adaptation from train-to-eval generalization.

Same-level adaptation conditions:

| Condition | Memory shown on retry |
| --- | --- |
| `single_shot_no_memory` | none |
| `generic_retry_feedback` | only says the previous same-level attempt failed |
| `verifier_summary_retry` | failed push index, failure subtype, and verifier reason |
| `raw_same_level_iterative` | compact same-level raw evidence |
| `heuristic_same_level_iterative` | heuristics generated from same-level failures |

Train-to-eval heuristic generalization conditions:

| Condition | Description |
| --- | --- |
| `no_memory_eval` | held-out eval without memory |
| `generic_sokoban_tips_eval` | fixed hand-written tips baseline |
| `train_one_shot_global_heuristic` | global heuristics from one train attempt per level |
| `train_iterated_global_heuristic` | global heuristics from iterative train failures |
| `eval_same_level_heuristic_adapt_only` | eval-only same-level heuristic adaptation |
| `train_iterated_global_plus_eval_adapt` | train-derived global heuristics plus eval same-level adaptation |

Raw trajectory memory is same-level-only. Cross-level prompts may render only `global_allowed` heuristic rules.

### Metrics

Primary metrics:

- `solve_rate`
- `invalid_plan_rate`
- `deadlock_rate`
- `plan_exhausted_rate`
- `timeout_rate`

Progress metrics:

- `best_goal_completion_rate`
- `final_goal_completion_rate`
- `target_placement_events_before_first_deadlock`
- `normalized_target_placement_before_first_deadlock`

Failure subtypes:

- `empty_output`
- `json_parse_error`
- `schema_error`
- `truncated_output`
- `wrong_box_reference`
- `blocked_destination`
- `blocked_standing_cell`
- `unreachable_standing_cell`
- `deadlock`
- `plan_exhausted`

## Executed Commands

Rule-based sanity:

```bash
.venv/bin/python run_experiment.py \
  --agent rule_based \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --episodes 24 \
  --max_steps 100 \
  --results_dir results/v3_rule_based_eval_sanity

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_rule_based_eval_sanity \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_rule_based_eval_sanity/evaluation_summary.json \
  --fail_on_validation_error
```

Token-cap smoke checks:

```bash
.venv/bin/python run_experiment.py \
  --agent no_memory \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split train \
  --episodes 1 \
  --max_steps 100 \
  --max_llm_calls 1 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_track0_smoke_gpt52_low_8192 \
  --temperature 0 \
  --max_output_tokens 8192 \
  --results_dir results/v3_track0_smoke_gpt52_low_8192

.venv/bin/python run_experiment.py \
  --agent no_memory \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split train \
  --episodes 1 \
  --max_steps 100 \
  --max_llm_calls 1 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_token_ablation_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_token_ablation_gpt52_low_16384
```

Track 0 planner-validity runs:

```bash
.venv/bin/python run_experiment.py \
  --agent no_memory \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split train \
  --episodes 24 \
  --max_steps 100 \
  --max_llm_calls 24 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_track0_train_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_track0_train_gpt52_low_16384

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_track0_train_gpt52_low_16384 \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_track0_train_gpt52_low_16384/evaluation_summary.json \
  --fail_on_validation_error

.venv/bin/python run_experiment.py \
  --agent no_memory \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --episodes 24 \
  --max_steps 100 \
  --max_llm_calls 24 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_track0_eval_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_track0_eval_gpt52_low_16384

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_track0_eval_gpt52_low_16384 \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_track0_eval_gpt52_low_16384/evaluation_summary.json \
  --fail_on_validation_error

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_track0_train_gpt52_low_16384 \
  --results_dir results/v3_track0_eval_gpt52_low_16384 \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_track0_balanced_gpt52_low_16384_summary.json \
  --fail_on_validation_error
```

Same-level verifier-summary retry:

```bash
.venv/bin/python run_experiment.py \
  --experiment_mode same_level_iterative \
  --condition verifier_summary_retry \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --attempt_budget 3 \
  --max_steps 100 \
  --max_llm_calls 3 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_eval_verifier_retry_k3_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_eval_verifier_retry_k3_gpt52_low_16384

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_eval_verifier_retry_k3_gpt52_low_16384 \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_eval_verifier_retry_k3_gpt52_low_16384/evaluation_summary.json \
  --fail_on_validation_error
```

## Token-Cap Smoke Results

The original `8192` token cap was not sufficient for `gpt-5.2` with reasoning effort `low`.

| Token cap | Episodes | Visible output | Status | Notes |
| ---: | ---: | --- | --- | --- |
| 8192 | 1 | no | `invalid_plan` | all 8192 output tokens were reasoning tokens |
| 16384 | 1 | yes | `invalid_plan` | generated 11 planned pushes, executed 3, failed at unreachable standing cell |

The 8192 smoke call recorded:

```json
{
  "output_tokens": 8192,
  "output_tokens_details": {"reasoning_tokens": 8192}
}
```

The 16384 smoke call recorded:

```json
{
  "output_tokens": 12833,
  "output_tokens_details": {"reasoning_tokens": 12649}
}
```

The main Track 0 run therefore used `max_output_tokens = 16384`.

## Track 0 Results

### Overall Single-Shot Results

| Split | Episodes | Success | Solve rate | Invalid plan | Deadlock | Plan exhausted | Avg. best goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 24 | 8 | 33.3% | 10 | 3 | 3 | 58.3% |
| eval | 24 | 8 | 33.3% | 13 | 2 | 1 | 51.0% |
| combined | 48 | 16 | 33.3% | 23 | 5 | 4 | 54.7% |

### Failure Subtypes

| Split | `unreachable_standing_cell` | `deadlock` | `plan_exhausted` |
| --- | ---: | ---: | ---: |
| train | 10 | 3 | 3 |
| eval | 13 | 2 | 1 |
| combined | 23 | 5 | 4 |

The dominant invalid-plan failure is `unreachable_standing_cell`: the LLM names a push whose required player standing cell cannot be reached from the current state while treating boxes and walls as blocked.

### Stratum Breakdown

| Stratum | Episodes | Solve rate | Invalid-plan rate | Deadlock rate | Plan-exhausted rate | Avg. best goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train / medium / constrained | 4 | 0.0% | 75.0% | 0.0% | 25.0% | 37.5% |
| train / medium / middle | 4 | 0.0% | 50.0% | 0.0% | 50.0% | 25.0% |
| train / medium / open | 4 | 25.0% | 25.0% | 50.0% | 0.0% | 62.5% |
| train / unfiltered / constrained | 4 | 75.0% | 25.0% | 0.0% | 0.0% | 81.2% |
| train / unfiltered / middle | 4 | 75.0% | 25.0% | 0.0% | 0.0% | 75.0% |
| train / unfiltered / open | 4 | 25.0% | 50.0% | 25.0% | 0.0% | 68.8% |
| eval / medium / constrained | 4 | 25.0% | 50.0% | 0.0% | 25.0% | 43.8% |
| eval / medium / middle | 4 | 0.0% | 100.0% | 0.0% | 0.0% | 37.5% |
| eval / medium / open | 4 | 0.0% | 75.0% | 25.0% | 0.0% | 31.2% |
| eval / unfiltered / constrained | 4 | 75.0% | 0.0% | 25.0% | 0.0% | 75.0% |
| eval / unfiltered / middle | 4 | 25.0% | 75.0% | 0.0% | 0.0% | 43.8% |
| eval / unfiltered / open | 4 | 75.0% | 25.0% | 0.0% | 0.0% | 75.0% |

## Same-Level Verifier-Summary Retry Results

The first memory/scaffold condition tested was `verifier_summary_retry` on the eval split with `K = 3`. This condition retries from the original board and shows a concise verifier summary from prior same-level failures.

| Condition | Levels | Attempts | Solve rate@1 | Solve rate@K | Successful levels | Invalid-plan attempts | Deadlock attempts | Plan-exhausted attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-shot eval baseline | 24 | 24 | 33.3% | 33.3% | 8 | 13 | 2 | 1 |
| `verifier_summary_retry`, K=3 | 24 | 52 | 29.2% | 54.2% | 13 | 30 | 8 | 1 |

The retry condition solved all 8 levels solved by the single-shot baseline and solved 5 additional eval levels:

- `boxoban_medium_valid_000_168`
- `boxoban_medium_valid_000_270`
- `boxoban_unfiltered_valid_000_127`
- `boxoban_unfiltered_valid_000_197`
- `boxoban_unfiltered_valid_000_289`

The cumulative solved-by-attempt curve was:

| Attempt cutoff | Solved level rate |
| ---: | ---: |
| 1 | 29.2% |
| 2 | 54.2% |
| 3 | 54.2% |

Failure subtype counts across all verifier-summary retry attempts:

| Failure subtype | Count |
| --- | ---: |
| `unreachable_standing_cell` | 25 |
| `deadlock` | 8 |
| `empty_output` | 3 |
| `blocked_standing_cell` | 1 |
| `wrong_box_reference` | 1 |
| `plan_exhausted` | 1 |

Interpretation:

- Verifier-summary retry improves same-level eval solve coverage from 8 to 13 solved levels.
- Most gains occurred by the second attempt; the third attempt did not add more solved levels in this run.
- The condition still produces many invalid plans, especially unreachable standing cells, so it improves solved-level coverage without fully solving the planner-legality problem.

## Interpretation

The V3 balanced benchmark is runnable end to end, and the current full-path LLM planner solves a nontrivial fraction of levels without memory. The first same-level retry scaffold improves eval solved-level coverage. However, planner validity is still not clean enough for a final train-to-eval heuristic-memory claim.

Key observations:

- `gpt-5.2` needs more than 8192 output tokens under reasoning effort `low`; otherwise it can return no visible plan.
- With 16384 output tokens, the model produces full push plans and solves 33.3% of both train and eval levels.
- The main failure mode is not malformed JSON; it is locally invalid Sokoban planning.
- The most common invalid-plan subtype is `unreachable_standing_cell`, meaning the proposed push is syntactically valid but not executable by the local verifier.
- Medium levels are harder than unfiltered levels under the current prompt. Eval medium middle levels had 0% solve rate and 100% invalid-plan rate.
- `verifier_summary_retry` raises eval `solve_rate@K` to 54.2%, but it does not remove invalid push proposals.

## Current Gate Decision

The current Track 0 and verifier-summary results trigger the planner-validity caution in the V3 design. Memory experiments can still be run, but any improvement must be interpreted carefully:

- If memory reduces `invalid_plan_rate`, it may be improving local legality rather than high-level Sokoban planning.
- If memory improves `best_goal_completion_rate` without improving solve rate, it may be helping partial progress but not complete planning.
- If train-derived global heuristics improve eval solve rate while also reducing invalid pushes, that would be stronger evidence of useful abstraction.

The next experimental step is to compare verifier-summary retry against raw same-level evidence and same-level heuristic memory before broad train-to-eval heuristic claims.

## Next Runs

Recommended next execution order:

1. `raw_same_level_iterative` on the eval split with `K = 3`.
2. `heuristic_same_level_iterative` on the eval split with `K = 3`.
3. If same-level retry improves validity, run the same conditions on train.
4. Generate train-derived global heuristics only after same-level heuristic generation produces stable, non-rejected rules.

These runs should keep the same fixed settings:

- model: `gpt-5.2`
- reasoning effort: `low`
- max output tokens: `16384`
- max steps: `100`
- prompt version: `full_path_v2_1`
- level suite: `levels/v3_boxoban_balanced.json`

## Artifact Index

Executed result directories:

- `results/v3_rule_based_eval_sanity`
- `results/v3_track0_smoke_gpt52_low_8192`
- `results/v3_token_ablation_gpt52_low_16384`
- `results/v3_track0_train_gpt52_low_16384`
- `results/v3_track0_eval_gpt52_low_16384`
- `results/v3_eval_verifier_retry_k3_gpt52_low_16384`

Combined evaluation summary:

- `results/v3_track0_balanced_gpt52_low_16384_summary.json`

Validation:

- `results/v3_track0_train_gpt52_low_16384/evaluation_summary.json`
- `results/v3_track0_eval_gpt52_low_16384/evaluation_summary.json`
- `results/v3_eval_verifier_retry_k3_gpt52_low_16384/evaluation_summary.json`
