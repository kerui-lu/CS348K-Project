# CS348K Project: Sokoban Memory Prototype

Minimal Python prototype for comparing memory conditions in LLM-style Sokoban agents.

The first version of the agent stack is complete: the repo now has a deterministic Sokoban environment, rule-based and LLM agents, action parsing/fallback handling, trajectory logging, summary metrics, and a small baseline experiment path.

V2 adds the main research comparison: controlled one-step LLM agents with no memory, compressed raw trajectory memory, or reflection heuristic memory. V2.1 hardens the protocol with train/eval split checks, raw-memory guardrails, explicit API request settings, and separate outcome categories for budget/API failures.

## Current Status

### Week 6 checkpoint

See [`docs/week6_checkpoint.md`](docs/week6_checkpoint.md).

Completed in v1:

- Deterministic grid-based Sokoban environment with movement, pushing, wall collision, two-box blocking, solved-state detection, and simple corner deadlock detection.
- Agent interface plus working `rule_based`, `no_memory`, `raw_trajectory_memory`, `reflection_heuristic`, and `llm` agent types.
- OpenAI-backed LLM action selection through the Responses API.
- Robust action parsing for exact and short natural-language model outputs.
- Per-episode JSON trajectory logs and aggregate `summary.json` metrics.
- Unit tests for the environment, parser, logging, CLI, and LLM-agent wrapper.

Added in v2:

- Shared one-step LLM policy for `no_memory`, `raw_trajectory_memory`, and `reflection_heuristic`.
- Frozen offline memory-bank workflow for clean ablations.
- Compressed raw failed-trajectory memory renderer.
- Cached LLM reflection generation for heuristic memory.
- LLM call budget guard through `--max_llm_calls`.
- Optional response caching through `--llm_cache_path`.
- Prompt/memory/cache metadata in episode trajectories and summaries.
- V2.1 guardrails: tagged train/eval levels, fail-hard memory leak checks, raw memory with factual replay only, `temperature`/`max_output_tokens` logging, and separate `budget_exhausted`, `api_error`, and `invalid_failure` outcomes.

Memory conditions supported by the CLI:

- `no_memory`
- `raw_trajectory_memory`
- `reflection_heuristic`

`rule_based` remains available as a cheap non-LLM baseline and debugging agent.

The checked-in `levels/v2_pilot.json` now contains a 12-level evaluation suite with 6 `train` and 6 `eval` levels. It mixes small local calibration puzzles with a curated subset of levels adapted from DeepMind's Boxoban dataset, and each level includes `tags`, `optimal_steps`, and `reference_solution`.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run_experiment.py --agent rule_based --episodes 10 --max_steps 100
```

Episode JSON files and `summary.json` are written to `results/` by default.

For LLM runs, create a local `.env` file with:

```bash
OPENAI_API_KEY=your_api_key_here
```

Then run a low-cost smoke test:

```bash
python3 run_experiment.py --agent llm --episodes 1 --max_steps 20
```

## V2 Workflow

V2 uses an offline frozen-memory protocol:

1. Run no-memory LLM episodes on `split=train` levels and collect failed trajectories.
2. Save those failures as compressed raw trajectory memory.
3. Generate cached reflection heuristics from the same failures.
4. Evaluate all three one-step LLM agents on held-out `split=eval` levels with matched memory caps.

The runner refuses eval runs if a loaded memory bank was generated from any eval `level_id`.

Build memory banks with a small budget:

```bash
python3 build_memory_bank.py \
  --levels levels/v2_pilot.json \
  --episodes 5 \
  --max_steps 100 \
  --max_llm_calls 50 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace main \
  --temperature 0 \
  --max_output_tokens 8 \
  --raw_memory_path memory_banks/raw_failures.json \
  --heuristic_memory_path memory_banks/reflection_heuristics.json
```

Run matched pilot evaluations:

```bash
python3 run_experiment.py \
  --agent no_memory \
  --levels levels/v2_pilot.json \
  --level_split eval \
  --episodes 3 \
  --max_llm_calls 50 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace main \
  --temperature 0 \
  --max_output_tokens 8 \
  --results_dir results/v2_no_memory

python3 run_experiment.py \
  --agent raw_trajectory_memory \
  --levels levels/v2_pilot.json \
  --level_split eval \
  --episodes 3 \
  --memory_path memory_banks/raw_failures.json \
  --max_memory_items 3 \
  --max_steps_per_memory 6 \
  --max_memory_chars 4000 \
  --max_llm_calls 50 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace main \
  --temperature 0 \
  --max_output_tokens 8 \
  --results_dir results/v2_raw

python3 run_experiment.py \
  --agent reflection_heuristic \
  --levels levels/v2_pilot.json \
  --level_split eval \
  --episodes 3 \
  --memory_path memory_banks/reflection_heuristics.json \
  --max_memory_items 3 \
  --max_steps_per_memory 6 \
  --max_memory_chars 4000 \
  --max_llm_calls 50 \
  --llm_cache_path .llm_cache/responses \
  --cache_namespace main \
  --temperature 0 \
  --max_output_tokens 8 \
  --results_dir results/v2_reflection
```

The main policy mode is `one_step`: each API call sees only the current board, legal actions, rules, and the condition-specific memory context, then returns one next action.

For a memory-budget sweep, repeat the raw/reflection eval commands with `--max_memory_chars 1000`, `2000`, and `4000`, keeping all other settings fixed.

Use cache-enabled `cache_namespace=main` runs for auditable main experiments. For stochastic robustness checks, disable cache or use a separate namespace such as `--cache_namespace robustness_seed_1`.

## Week 6 Evaluation Checkpoint

The evaluation pipeline is automatic and tested. It covers success/failure checking, deadlock accounting, solve rate, and solution efficiency.

Outcome checks:

- `success`: all boxes are on targets; the final trajectory step must have `info.solved == true`.
- `deadlock`: the environment detects a non-target corner deadlock; the final step must have `info.deadlocked == true`.
- `timeout`: the episode reaches `--max_steps` without success or detected deadlock.
- `budget_exhausted`, `api_error`, and `invalid_failure`: tracked separately from gameplay failures.

Metrics:

- `solve_rate = success_count / episodes`
- `deadlock_count`, `timeout_count`, and rates for all outcome categories
- `average_success_steps`
- `solution_efficiency = optimal_steps / actual_steps` for successful episodes only
- `average_solution_efficiency`, `median_solution_efficiency`, and `steps_over_optimal_average`
- per-level attempts, successes, deadlocks, timeouts, solve rate, average steps, and average efficiency

`optimal_steps` is stored in level metadata for the pilot suite. Episodes without a reference length are skipped for efficiency metrics and counted in `solution_efficiency_skipped_count`.

A complete evaluation report should be read in this order:

1. Check `validation_error_count == 0`. If validation fails, the reported metrics are not trusted.
2. Check non-gameplay failure rates: `budget_exhausted_rate`, `api_error_rate`, and `invalid_failure_rate`.
3. Compare `solve_rate` as the primary task-success metric.
4. Compare `deadlock_rate` to measure whether the agent avoids irreversible Sokoban mistakes.
5. Compare `average_solution_efficiency` and `average_success_steps` to measure how efficient successful solutions are.
6. Inspect `per_level` to see which challenge types produce improvements or failures.

For the main research question, compare the three agents under identical eval levels, seeds, model, prompt version, memory budget, and API settings:

- `no_memory`: baseline LLM agent with no prior experience.
- `raw_trajectory_memory`: tests whether factual replay of failed trajectories helps.
- `reflection_heuristic`: tests whether abstracted failure rules help more than raw trajectory replay.

The strongest evidence for reflection memory would be higher `solve_rate`, lower `deadlock_rate`, and equal or better `average_solution_efficiency` than raw trajectory memory across the same held-out eval levels.

Report shape:

```json
{
  "overall": {
    "solve_rate": 0.0,
    "deadlock_count": 3,
    "average_success_steps": 0.0,
    "average_solution_efficiency": 0.0
  },
  "by_agent": {
    "no_memory": {},
    "raw_trajectory_memory": {},
    "reflection_heuristic": {}
  },
  "per_level": {
    "wall_push_001": {
      "attempts": 3,
      "successes": 0,
      "deadlocks": 3,
      "solve_rate": 0.0,
      "average_efficiency": 0.0
    }
  },
  "validation_errors": []
}
```

Generate a consolidated evaluation report from one or more result directories:

```bash
python3 evaluate_results.py \
  --results_dir results/v2_no_memory \
  --results_dir results/v2_raw \
  --results_dir results/v2_reflection \
  --levels levels/v2_pilot.json \
  --output results/evaluation_summary.json \
  --fail_on_validation_error
```

Before any real API run, first run local evaluation sanity checks:

```bash
python3 run_experiment.py \
  --agent rule_based \
  --levels levels/v2_pilot.json \
  --level_split eval \
  --episodes 3 \
  --max_steps 50 \
  --results_dir results/eval_sanity_rule_based

python3 evaluate_results.py \
  --results_dir results/eval_sanity_rule_based \
  --levels levels/v2_pilot.json \
  --output results/eval_sanity_rule_based/evaluation_summary.json \
  --fail_on_validation_error
```

Only move to live LLM smoke tests after `pytest` and the local evaluation report pass.

## Verified Results

Verified locally on April 29, 2026:

- `rule_based` baseline over 10 episodes:
  - solve rate: `0.5`
  - deadlock rate: `0.3`
  - timeout rate: `0.2`
  - average steps: `25.8`
  - invalid move rate: `0.0`
- `llm` smoke test over 1 completed episode:
  - status: `success`
  - step count: `1`
  - model action: `Right`
  - parsed/executed action: `Right`
  - invalid moves: `0`

The LLM smoke test confirms that the API key loading, OpenAI client setup, model call, action parsing, environment step, and result logging all work end to end.

Current V2 tests use fake LLM clients, so they do not spend API credits.

## Board Symbols

```text
# wall
. target
$ box
* box on target
@ player
+ player on target
  empty floor
```

## Tests

```bash
pytest
```

## Next Milestones

- Run a small V2 pilot with the budget guard and response cache enabled.
- Expand the level set beyond the starter pilot levels.
- Add comparison scripts for summarizing multiple result directories.
- Add optional `short_horizon` and `full_plan` modes after the one-step experiments are stable.
