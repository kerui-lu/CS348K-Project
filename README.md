# CS348K Project: Sokoban Memory Prototype

Minimal Python prototype for comparing memory conditions in LLM-style Sokoban agents.

The first version of the agent stack is complete: the repo now has a deterministic Sokoban environment, rule-based and LLM agents, action parsing/fallback handling, trajectory logging, summary metrics, and a small baseline experiment path.

## Current Status

Completed in v1:

- Deterministic grid-based Sokoban environment with movement, pushing, wall collision, two-box blocking, solved-state detection, and simple corner deadlock detection.
- Agent interface plus working `rule_based`, `no_memory`, `raw_trajectory_memory`, `reflection_heuristic`, and `llm` agent types.
- OpenAI-backed LLM action selection through the Responses API.
- Robust action parsing for exact and short natural-language model outputs.
- Per-episode JSON trajectory logs and aggregate `summary.json` metrics.
- Unit tests for the environment, parser, logging, CLI, and LLM-agent wrapper.

Memory conditions currently supported by the CLI:

- `no_memory`
- `raw_trajectory_memory`
- `reflection_heuristic`

The raw trajectory and reflection memory paths are scaffolded for experiments; the current v1 decision policy is intentionally simple so that later memory ablations remain easy to inspect.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run_experiment.py --agent rule_based --episodes 10 --max_steps 100
python3 run_experiment.py --agent no_memory --levels levels/simple.json
python3 run_experiment.py --agent reflection --episodes 10 --max_steps 100
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

- Add a `--max_llm_calls` or equivalent budget guard before running larger LLM experiments.
- Expand the level set beyond the two simple starter levels.
- Make `raw_trajectory_memory` and `reflection_heuristic` actively influence LLM prompts.
- Add comparison scripts for summarizing multiple result directories.
