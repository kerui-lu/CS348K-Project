# CS348K Project: Sokoban Memory Prototype

Minimal Python prototype for comparing memory conditions in LLM-style Sokoban agents:

- `no_memory`
- `raw_trajectory_memory`
- `reflection_heuristic`

The v1 backend is a small, deterministic grid-based Sokoban environment. It is intentionally simple so trajectory logging, invalid action handling, deadlock detection, and later memory ablations stay transparent.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 run_experiment.py --agent rule_based --episodes 10
python3 run_experiment.py --agent no_memory --levels levels/simple.json
python3 run_experiment.py --agent reflection --episodes 10 --max_steps 100
```

Episode JSON files and `summary.json` are written to `results/` by default.

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
