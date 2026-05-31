# V3 Sokoban Memory Evaluation Report

## Summary

This report documents the current V3 Sokoban memory-evaluation design and the results executed on `levels/v3_boxoban_balanced.json`.

The current executed results include the planner-validity gate, three same-level retry conditions, and one train-to-eval global heuristic condition. The gate shows that `gpt-5.2` with the active full-path prompt can solve some V3 Boxoban levels, but invalid local push plans remain the dominant failure mode. All three same-level retry conditions improve held-out eval solve rate under a fixed attempt budget. The train-derived global heuristic condition gives a smaller cross-level improvement over the no-memory eval baseline.

Main result:

- Balanced benchmark: 48 levels, 24 train and 24 eval.
- Model: `gpt-5.2`.
- Reasoning effort: `low`.
- Prompt version: `full_path_v2_1`.
- Main token cap used after ablation: `16384`.
- Current code default and minimum `max_output_tokens` for future runs: `16384`.
- Overall single-shot solve rate: `16/48 = 33.3%`.
- Overall invalid-plan rate: `23/48 = 47.9%`.
- Dominant failure subtype: `unreachable_standing_cell`.
- Same-level verifier-summary retry: eval `solve_rate@K` improves from `8/24 = 33.3%` to `13/24 = 54.2%` with `K = 3`.
- Same-level raw trajectory retry: eval `solve_rate@K` also reaches `13/24 = 54.2%` with `K = 3`, but with a different solved-level set and a higher invalid-plan attempt count.
- Same-level heuristic retry: eval `solve_rate@K` reaches `12/24 = 50.0%` with `K = 3`.
- Train-to-eval one-shot global heuristic: eval solve rate reaches `9/24 = 37.5%`.

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

For train-to-eval heuristic conditions, the heuristic memory is pooled across train levels: train failures are distilled into one global heuristic bank, and every eval level receives the same rendered top-K `global_allowed` heuristics. This is separate from `heuristic_same_level_iterative`, where heuristics are generated from prior failures on the same level and are used only for that level's retries.

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

The commands in this section record already executed runs. Some historical commands use token caps below the current code minimum of `16384`; reruns should omit `--max_output_tokens` or set it to at least `16384`.

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

Same-level raw trajectory retry:

```bash
.venv/bin/python run_experiment.py \
  --experiment_mode same_level_iterative \
  --condition raw_same_level_iterative \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --attempt_budget 3 \
  --max_steps 100 \
  --max_llm_calls 3 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_eval_raw_same_level_k3_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_eval_raw_same_level_k3_gpt52_low_16384

.venv/bin/python run_experiment.py \
  --experiment_mode same_level_iterative \
  --condition raw_same_level_iterative \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --level_id boxoban_unfiltered_valid_000_286 \
  --attempt_budget 3 \
  --max_steps 100 \
  --max_llm_calls 3 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_eval_raw_same_level_k3_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --seed 111 \
  --results_dir results/v3_eval_raw_same_level_k3_gpt52_low_16384

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_eval_raw_same_level_k3_gpt52_low_16384_clean \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_eval_raw_same_level_k3_gpt52_low_16384_clean/evaluation_summary.json \
  --fail_on_validation_error
```

The final raw same-level evaluation uses `results/v3_eval_raw_same_level_k3_gpt52_low_16384_clean`, a clean copy of the raw result directory with one duplicate cached file removed. The duplicate occurred when the original long run wrote the final level after the missing-level check had already started a cached supplemental run.

Same-level heuristic retry:

```bash
.venv/bin/python run_experiment.py \
  --experiment_mode same_level_iterative \
  --condition heuristic_same_level_iterative \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --attempt_budget 3 \
  --max_steps 100 \
  --max_llm_calls 3 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_eval_heuristic_same_level_k3_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_eval_heuristic_same_level_k3_gpt52_low_16384

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_eval_heuristic_same_level_k3_gpt52_low_16384 \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_eval_heuristic_same_level_k3_gpt52_low_16384/evaluation_summary.json \
  --fail_on_validation_error
```

Train-to-eval one-shot global heuristic:

```bash
.venv/bin/python build_memory_bank.py \
  --levels levels/v3_boxoban_balanced.json \
  --episodes 24 \
  --max_steps 100 \
  --max_llm_calls 50 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_track0_train_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_train_one_shot_memory_build_gpt52_low_16384 \
  --raw_memory_path memory_banks/v3_train_one_shot_raw_failures.json \
  --heuristic_memory_path memory_banks/v3_train_one_shot_global_heuristics.json \
  --model gpt-5.2

.venv/bin/python run_experiment.py \
  --agent reflection_heuristic \
  --model gpt-5.2 \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --episodes 24 \
  --max_steps 100 \
  --max_llm_calls 24 \
  --memory_path memory_banks/v3_train_one_shot_global_heuristics.json \
  --max_memory_items 3 \
  --max_steps_per_memory 6 \
  --max_memory_chars 4000 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace v3_eval_train_one_shot_global_heuristic_gpt52_low_16384 \
  --temperature 0 \
  --max_output_tokens 16384 \
  --results_dir results/v3_eval_train_one_shot_global_heuristic_gpt52_low_16384

.venv/bin/python evaluate_results.py \
  --results_dir results/v3_eval_train_one_shot_global_heuristic_gpt52_low_16384 \
  --levels levels/v3_boxoban_balanced.json \
  --output results/v3_eval_train_one_shot_global_heuristic_gpt52_low_16384/evaluation_summary.json \
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

## Same-Level Raw Trajectory Retry Results

The second same-level retry condition tested was `raw_same_level_iterative` on the eval split with `K = 3`. This condition retries from the original board and renders compact same-level raw evidence from prior failed attempts, including the failed model intent, verifier reason, board before failure, and board after the last successful push.

| Condition | Levels | Attempts | Solve rate@1 | Solve rate@K | Successful levels | Invalid-plan attempts | Deadlock attempts | Plan-exhausted attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-shot eval baseline | 24 | 24 | 33.3% | 33.3% | 8 | 13 | 2 | 1 |
| `verifier_summary_retry`, K=3 | 24 | 52 | 29.2% | 54.2% | 13 | 30 | 8 | 1 |
| `raw_same_level_iterative`, K=3 | 24 | 55 | 25.0% | 54.2% | 13 | 34 | 4 | 4 |
| `heuristic_same_level_iterative`, K=3 | 24 | 55 | 20.8% | 50.0% | 12 | 31 | 3 | 9 |

Raw same-level retry solved 6 levels not solved by the single-shot eval baseline:

- `boxoban_medium_valid_000_168`
- `boxoban_medium_valid_000_270`
- `boxoban_medium_valid_000_275`
- `boxoban_unfiltered_valid_000_127`
- `boxoban_unfiltered_valid_000_197`
- `boxoban_unfiltered_valid_000_289`

Raw same-level retry lost one level solved by the single-shot eval baseline:

- `boxoban_medium_valid_000_290`

Compared with verifier-summary retry, raw same-level retry solved one additional level:

- `boxoban_medium_valid_000_275`

It also failed one level that verifier-summary retry solved:

- `boxoban_medium_valid_000_290`

The cumulative solved-by-attempt curve was:

| Attempt cutoff | Solved level rate |
| ---: | ---: |
| 1 | 25.0% |
| 2 | 45.8% |
| 3 | 54.2% |

Failure subtype counts across all raw same-level retry attempts:

| Failure subtype | Count |
| --- | ---: |
| `unreachable_standing_cell` | 29 |
| `deadlock` | 4 |
| `plan_exhausted` | 4 |
| `empty_output` | 2 |
| `blocked_destination` | 1 |
| `blocked_standing_cell` | 1 |
| `truncated_output` | 1 |

Stratum-level raw retry results:

| Eval stratum | Attempts | Solve rate@1 | Solve rate@K | Invalid-plan rate | Deadlock rate | Plan-exhausted rate | Avg. best goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium / constrained | 10 | 25.0% | 25.0% | 50.0% | 10.0% | 30.0% | 32.5% |
| medium / middle | 11 | 0.0% | 50.0% | 81.8% | 0.0% | 0.0% | 43.2% |
| medium / open | 12 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 35.4% |
| unfiltered / constrained | 8 | 25.0% | 75.0% | 12.5% | 37.5% | 12.5% | 37.5% |
| unfiltered / middle | 10 | 0.0% | 75.0% | 70.0% | 0.0% | 0.0% | 47.5% |
| unfiltered / open | 4 | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% |

Interpretation:

- Raw same-level retry reaches the same `solve_rate@K` as verifier-summary retry, but it does not clearly dominate verifier-summary retry.
- Raw evidence can rescue different levels than verifier-summary retry, suggesting that concrete same-level evidence is sometimes useful.
- Raw retry has more invalid-plan attempts than verifier-summary retry in this run, so compact raw evidence may add useful local context but also increases prompt complexity.

## Same-Level Heuristic Retry Results

The third same-level retry condition tested was `heuristic_same_level_iterative` on the eval split with `K = 3`. This condition generates concise same-level reflection heuristics from prior failures, then renders the accepted same-level heuristics on later attempts for the same level.

| Condition | Levels | Attempts | Solve rate@1 | Solve rate@K | Successful levels | Invalid-plan attempts | Deadlock attempts | Plan-exhausted attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single-shot eval baseline | 24 | 24 | 33.3% | 33.3% | 8 | 13 | 2 | 1 |
| `verifier_summary_retry`, K=3 | 24 | 52 | 29.2% | 54.2% | 13 | 30 | 8 | 1 |
| `raw_same_level_iterative`, K=3 | 24 | 55 | 25.0% | 54.2% | 13 | 34 | 4 | 4 |
| `heuristic_same_level_iterative`, K=3 | 24 | 55 | 20.8% | 50.0% | 12 | 31 | 3 | 9 |

Heuristic same-level retry solved 4 levels not solved by the single-shot eval baseline:

- `boxoban_medium_valid_000_168`
- `boxoban_medium_valid_000_270`
- `boxoban_unfiltered_valid_000_127`
- `boxoban_unfiltered_valid_000_289`

It did not lose any level solved by the single-shot eval baseline.

Compared with raw same-level retry, heuristic same-level retry solved one additional level:

- `boxoban_medium_valid_000_290`

It failed two levels solved by raw same-level retry:

- `boxoban_medium_valid_000_275`
- `boxoban_unfiltered_valid_000_197`

Compared with verifier-summary retry, heuristic same-level retry did not solve any additional level and missed one verifier-solved level:

- `boxoban_unfiltered_valid_000_197`

The cumulative solved-by-attempt curve was:

| Attempt cutoff | Solved level rate |
| ---: | ---: |
| 1 | 20.8% |
| 2 | 50.0% |
| 3 | 50.0% |

Failure subtype counts across all heuristic same-level retry attempts:

| Failure subtype | Count |
| --- | ---: |
| `unreachable_standing_cell` | 26 |
| `plan_exhausted` | 9 |
| `empty_output` | 4 |
| `deadlock` | 3 |
| `blocked_standing_cell` | 1 |

Stratum-level heuristic retry results:

| Eval stratum | Attempts | Solve rate@1 | Solve rate@K | Invalid-plan rate | Deadlock rate | Plan-exhausted rate | Avg. best goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium / constrained | 10 | 0.0% | 50.0% | 40.0% | 0.0% | 40.0% | 30.0% |
| medium / middle | 11 | 0.0% | 25.0% | 54.5% | 0.0% | 36.4% | 22.7% |
| medium / open | 12 | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 25.0% |
| unfiltered / constrained | 7 | 50.0% | 75.0% | 14.3% | 42.9% | 0.0% | 42.9% |
| unfiltered / middle | 9 | 0.0% | 75.0% | 55.6% | 0.0% | 11.1% | 61.1% |
| unfiltered / open | 6 | 75.0% | 75.0% | 50.0% | 0.0% | 0.0% | 70.8% |

Interpretation:

- Same-level heuristic retry improves over the single-shot baseline but does not outperform verifier-summary or raw same-level retry on solved-level coverage.
- It has fewer invalid-plan attempts than raw retry but more plan-exhausted attempts, suggesting that some generated heuristics improve legality while still failing to produce complete plans.
- The third attempt did not improve solved-level coverage in this run; all gains happened by attempt 2.

## Train-To-Eval Global Heuristic Results

The cross-level memory condition tested was `train_one_shot_global_heuristic`. The memory bank was built from one no-memory attempt on each of the 24 train levels. The train planner calls were cache hits from the Track 0 train run, producing 16 failed train records and 12 reflection heuristics. All 12 generated heuristics were classified as `global_allowed`; none were classified as `same_level_only` or `rejected`.

The eval run used `reflection_heuristic` with the train-derived heuristic memory. The prompt rendered at most 3 heuristic items per episode to keep the memory budget aligned with the other memory conditions.

| Eval condition | Levels | Attempts | Solve rate | Successful levels | Invalid-plan attempts | Deadlock attempts | Timeout attempts | Plan-exhausted attempts | Avg. best goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-memory eval baseline | 24 | 24 | 33.3% | 8 | 13 | 2 | 0 | 1 | 51.0% |
| train one-shot global heuristic | 24 | 24 | 37.5% | 9 | 12 | 1 | 1 | 1 | 59.4% |

The train-derived global heuristic condition solved 3 levels not solved by the no-memory eval baseline:

- `boxoban_medium_valid_000_168`
- `boxoban_medium_valid_000_270`
- `boxoban_unfiltered_valid_000_127`

It failed 2 levels solved by the no-memory eval baseline:

- `boxoban_medium_valid_000_290`
- `boxoban_unfiltered_valid_000_135`

Failure subtype counts:

| Failure subtype | Count |
| --- | ---: |
| `unreachable_standing_cell` | 12 |
| `deadlock` | 1 |
| `timeout` | 1 |
| `plan_exhausted` | 1 |

Stratum-level train-to-eval results:

| Eval stratum | Episodes | Solve rate | Invalid-plan rate | Deadlock rate | Timeout rate | Plan-exhausted rate | Avg. best goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium / constrained | 4 | 25.0% | 25.0% | 0.0% | 25.0% | 25.0% | 56.2% |
| medium / middle | 4 | 25.0% | 75.0% | 0.0% | 0.0% | 0.0% | 31.2% |
| medium / open | 4 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 37.5% |
| unfiltered / constrained | 4 | 75.0% | 0.0% | 25.0% | 0.0% | 0.0% | 75.0% |
| unfiltered / middle | 4 | 25.0% | 75.0% | 0.0% | 0.0% | 0.0% | 68.8% |
| unfiltered / open | 4 | 75.0% | 25.0% | 0.0% | 0.0% | 0.0% | 87.5% |

Interpretation:

- Train-derived global heuristics produce a small cross-level solve-rate gain over no memory: 9 solved eval levels instead of 8.
- The cross-level condition does not match same-level retry coverage, which reached 12-13 solved eval levels under `K = 3`.
- The invalid-plan rate remains high at 50.0%, so the train-derived heuristics do not solve the full-path legality problem.
- Average best goal completion improves from 51.0% to 59.4%, suggesting that the heuristics may help partial progress even when the final plan remains invalid or incomplete.

## Interpretation

The V3 balanced benchmark is runnable end to end, and the current full-path LLM planner solves a nontrivial fraction of levels without memory. The verifier-summary and raw same-level retry conditions both improve eval solved-level coverage to 13/24, while heuristic same-level retry improves it to 12/24. Train-derived global heuristic memory gives a smaller cross-level improvement to 9/24. Planner validity remains the main bottleneck.

Key observations:

- `gpt-5.2` needs more than 8192 output tokens under reasoning effort `low`; otherwise it can return no visible plan.
- With 16384 output tokens, the model produces full push plans and solves 33.3% of both train and eval levels.
- The main failure mode is not malformed JSON; it is locally invalid Sokoban planning.
- The most common invalid-plan subtype is `unreachable_standing_cell`, meaning the proposed push is syntactically valid but not executable by the local verifier.
- Medium levels are harder than unfiltered levels under the current prompt. Eval medium middle levels had 0% solve rate and 100% invalid-plan rate.
- `verifier_summary_retry` raises eval `solve_rate@K` to 54.2%, but it does not remove invalid push proposals.
- `raw_same_level_iterative` also reaches 54.2% eval `solve_rate@K`, with a different solved-level set and more invalid-plan attempts.
- `heuristic_same_level_iterative` reaches 50.0% eval `solve_rate@K`; it is useful relative to no memory, but it is not stronger than verifier-summary or raw retry in this run.
- Train-to-eval global heuristics produce a small eval solve-rate gain and a larger progress-metric gain, but do not substantially reduce invalid push proposals.

## Current Gate Decision

The current Track 0, same-level retry, and train-to-eval results trigger the planner-validity caution in the V3 design. Memory helps some levels, but the primary failure mode remains invalid local Sokoban plans:

- If memory reduces `invalid_plan_rate`, it may be improving local legality rather than high-level Sokoban planning.
- If memory improves `best_goal_completion_rate` without improving solve rate, it may be helping partial progress but not complete planning.
- Train-derived global heuristics improve eval solve rate slightly but do not reduce invalid pushes enough to support a strong generalization claim.

The executed V3 results support two narrower conclusions:

- Same-level feedback is useful as an instance-level repair scaffold, especially verifier-summary and raw evidence.
- Train-derived global heuristic memory shows limited cross-level transfer under the current planner and prompt.

## Optional Follow-Up Runs

The current report covers the main V3 comparison. Additional runs can be used as follow-up rather than as prerequisites for the current result table:

1. `train_iterated_global_heuristic`: build global heuristics from iterative train failures instead of one-shot train failures.
2. `train_iterated_global_plus_eval_adapt`: combine train-derived global heuristics with eval same-level heuristic adaptation.
3. `generic_sokoban_tips_eval`: compare against a fixed, hand-written global tips baseline.
4. OOD evaluation on `levels/v3_boxoban_ood.json`.

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
- `results/v3_eval_raw_same_level_k3_gpt52_low_16384`
- `results/v3_eval_raw_same_level_k3_gpt52_low_16384_clean`
- `results/v3_eval_heuristic_same_level_k3_gpt52_low_16384`
- `results/v3_train_one_shot_memory_build_gpt52_low_16384`
- `results/v3_eval_train_one_shot_global_heuristic_gpt52_low_16384`

Combined evaluation summary:

- `results/v3_track0_balanced_gpt52_low_16384_summary.json`

Memory artifacts:

- `memory_banks/v3_train_one_shot_raw_failures.json`
- `memory_banks/v3_train_one_shot_global_heuristics.json`

Validation:

- `results/v3_track0_train_gpt52_low_16384/evaluation_summary.json`
- `results/v3_track0_eval_gpt52_low_16384/evaluation_summary.json`
- `results/v3_eval_verifier_retry_k3_gpt52_low_16384/evaluation_summary.json`
- `results/v3_eval_raw_same_level_k3_gpt52_low_16384_clean/evaluation_summary.json`
- `results/v3_eval_heuristic_same_level_k3_gpt52_low_16384/evaluation_summary.json`
- `results/v3_eval_train_one_shot_global_heuristic_gpt52_low_16384/evaluation_summary.json`
