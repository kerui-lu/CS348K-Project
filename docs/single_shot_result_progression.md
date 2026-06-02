# Single-Shot Result Progression

This document summarizes the single-shot Sokoban planning results completed so far. All rows use:

- agent: `no_memory`
- policy mode: `full_path`
- temperature: `0`
- no repair loop
- local verifier/executor validation

The early `v2_pilot` runs use a smaller pilot/debug suite with calibration levels and pilot Boxoban imports. The V3 runs use the harder, balanced Boxoban benchmark in `levels/v3_boxoban_balanced.json`.

## Multi-Level Results

| Stage | Suite / split | Levels | Model | Reasoning effort | Max output tokens | Max steps | Prompt | Solve rate | Notes |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |
| Early pilot baseline | `v2_pilot` train | 12 | `gpt-5-nano` | default / not logged | 512 | 100 | `full_path_v2` | `2/12 = 16.7%` | Easier pilot/debug suite |
| Stronger model pilot | `v2_pilot` train | 12 | `gpt-5.2` | low | 8192 | 100 | `full_path_v2` | `5/12 = 41.7%` | Same pilot suite |
| Final benchmark train | `v3_boxoban_balanced` train | 24 | `gpt-5.2` | low | 16384 | 100 | `full_path_v2_1` | `8/24 = 33.3%` | Harder balanced Boxoban train split |
| Final benchmark eval | `v3_boxoban_balanced` eval | 24 | `gpt-5.2` | low | 16384 | 100 | `full_path_v2_1` | `8/24 = 33.3%` | Held-out matched eval split |
| Targeted rescue ablation | failed train subset from V3 low run | 16 | `gpt-5.2` | medium | 32768 | 512 | `full_path_v2_1` | `5/16 = 31.2%` | Only reruns levels that failed under V3 low reasoning |

## Token-Cap Smoke Checks

| Smoke check | Suite / split | Episodes | Model | Reasoning effort | Max output tokens | Max steps | Solve rate | Observation |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| V3 token smoke | `v3_boxoban_balanced` train | 1 | `gpt-5.2` | low | 8192 | 100 | `0/1 = 0.0%` | No visible JSON plan; all output tokens were reasoning tokens |
| V3 token smoke | `v3_boxoban_balanced` train | 1 | `gpt-5.2` | low | 16384 | 100 | `0/1 = 0.0%` | Visible plan produced, but failed local verification |

## Interpretation

The early `gpt-5-nano` result was low even on the easier pilot suite. Switching to `gpt-5.2` with a larger output budget improved the pilot result from `16.7%` to `41.7%`.

The V3 suite is harder and more balanced than the pilot suite, so the `33.3%` train and eval solve rates are the main single-shot benchmark results. The medium-reasoning run is a targeted rescue ablation over the 16 failed train levels from the V3 low-reasoning run; it should not be treated as a replacement for the original full-suite train aggregate.

## Source Artifacts

- `results/full_path_v2_train_prompt_check`
- `results/full_path_v2_gpt5_2_low_8192_train_check`
- `results/v3_track0_smoke_gpt52_low_8192`
- `results/v3_token_ablation_gpt52_low_16384`
- `results/v3_track0_train_gpt52_low_16384`
- `results/v3_track0_eval_gpt52_low_16384`
- `results/v3_train_failed_single_shot_gpt52_medium_32768_512`
