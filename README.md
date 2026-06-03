# CS348K Project: LLM Sokoban Memory

This repository contains the final CS348K project code and report for studying whether memory from previous Sokoban gameplay can improve LLM full-path planning.

## Final Report

Read the final report first: [docs/final_report.md](docs/final_report.md).

The report summarizes the system architecture, v3 Boxoban benchmark, four memory strategies, same-level retry results, train-to-eval heuristic results, and the main lessons from the project.

## What Is Included

- `sokoban_memory/`: deterministic Sokoban environment, full-path LLM agents, local executor/verifier, memory rendering, and metrics.
- `run_experiment.py`: experiment runner for baseline, memory, same-level retry, and train-to-eval conditions.
- `build_memory_bank.py`: builds raw failure memories and reflection/heuristic memories from training trajectories.
- `evaluate_results.py`: validates episode logs and computes aggregate metrics.
- `levels/v3_boxoban_balanced.json`: main 24-train + 24-eval benchmark used in the final report.
- `memory_banks/`: checked-in memory banks used by the v3 experiments.
- `docs/`: final report, figures, v3 evaluation notes, and earlier checkpoint documents.

Older checkpoint markdown files are kept for project history, but `docs/final_report.md` is the canonical final writeup.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the deterministic sanity baseline:

```bash
python3 run_experiment.py \
  --agent rule_based \
  --levels levels/v3_boxoban_balanced.json \
  --episodes 3 \
  --max_steps 120
```

For LLM runs, create a local `.env` file with:

```bash
OPENAI_API_KEY=your_api_key_here
```

Example no-memory v3 eval run:

```bash
python3 run_experiment.py \
  --agent no_memory \
  --levels levels/v3_boxoban_balanced.json \
  --level_split eval \
  --episodes 24 \
  --max_steps 120 \
  --max_llm_calls 24 \
  --temperature 0
```

## Main Conditions

The final code supports four memory strategies:

- `no_memory`: full-path LLM planning with no trajectory memory.
- `raw_trajectory_memory`: replay-style memory from prior failed trajectories.
- `verifier_summary_retry`: compact trajectory summaries produced from verifier evidence.
- `reflection_heuristic`: abstracted Sokoban heuristics distilled from training failures.

The same-level retry pipeline tests whether memory from previous failed attempts on the same level improves later attempts. The train-to-eval pipeline tests whether heuristics learned from training failures transfer to held-out evaluation levels.

## Useful Commands

Build memory banks from training failures:

```bash
python3 build_memory_bank.py \
  --levels levels/v3_boxoban_balanced.json \
  --episodes 24 \
  --raw_memory_path memory_banks/v3_train_one_shot_raw_failures.json \
  --heuristic_memory_path memory_banks/v3_train_one_shot_global_heuristics.json
```

Rebuild global heuristics from the v3 train memory bank:

```bash
python3 scripts/rebuild_v3_global_heuristics.py
```

Run tests:

```bash
python3 -m pytest
```
