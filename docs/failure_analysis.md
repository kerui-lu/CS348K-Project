# Failure Analysis (Post-Guardrail, `aditri-full-path`)

## Scope

- Dataset: `results/post_guardrail_eval_20260527_clean/*/*.json` (36 episodes)
- Aggregate report: `docs/post_guardrail_eval_20260527_summary.json`
- Runtime path: `sokoban_memory/prompts.py`, `sokoban_memory/agents.py`, `sokoban_memory/full_path.py`, `sokoban_memory/experiment.py`
- Visualization path: `scripts/render_episode_gifs.py`

## Execution Model Relevant to Failures

The planner outputs semantic push intents (`box_id` + `push`), then executor expands them into primitive movement:

1. Resolve intent target box (`parse_push_plan`, `_resolve_intent_box`).
2. Compute required player standing cell and push destination.
3. Enforce legal-candidate gate (`_legal_push_candidates`, `_matches_legal_push_candidate`).
4. Compute walk path to standing cell via BFS (`shortest_player_path`).
5. Execute BFS walk then one push (`execute_push_plan`).

Failure occurs when any intent violates standing-cell, destination-cell, or reachability constraints under the **current** board.

## Logged Artifacts Used for Root-Cause Analysis

### Episode-level (`EpisodeResult`)

- `status`: terminal class (`invalid_plan`, `deadlock`, `plan_exhausted`, `success`, ...)
- `metadata.failure_reason`: terminal reason for final attempt
- `metadata.first_attempt_failure_reason`: first-attempt reason (root signal before repair/guardrail effects)
- `metadata.repair_attempts`: full attempt history

### Attempt-level (`metadata.repair_attempts[*]`)

- `planned_pushes`: semantic push sequence proposed by model
- `push_execution_log`: per-push verifier/executor audit trail
- `failure_push_index`: index of failed semantic push
- `repair_alternative_pushes`: legal alternatives at fail state

### Primitive-step level (`trajectory[*]`)

- `executed_action`: primitive move taken
- `info.push_index`: parent semantic push index
- `info.semantic_phase`: `walk_to_push` vs `push`
- `info.planned_push`: semantic intent tied to this primitive step
- `response_text`: full raw LLM output for that call

## Quantitative Failure Breakdown

### Terminal status (36 episodes)

| Status | Count | Rate |
|---|---:|---:|
| `invalid_plan` | 29 | 80.56% |
| `deadlock` | 4 | 11.11% |
| `plan_exhausted` | 1 | 2.78% |
| `success` | 2 | 5.56% |

### Final terminal failure reasons (`metadata.failure_reason`)

| Final reason | Count | Share of episodes |
|---|---:|---:|
| `repeated_failed_push_guardrail` | 13 | 36.11% |
| `required_push_position_blocked_by_wall_or_boundary` | 6 | 16.67% |
| `box_destination_blocked_by_box` | 4 | 11.11% |
| `box_destination_blocked_by_wall_or_boundary` | 3 | 8.33% |
| `required_push_position_unreachable` | 2 | 5.56% |
| `required_push_position_blocked_by_box` | 1 | 2.78% |
| `plan_exhausted` | 1 | 2.78% |

Terminal reasons are partially policy-layer outcomes (guardrail), not purely planner-origin errors.

### First-attempt failure reasons (`metadata.first_attempt_failure_reason`)

| First-attempt reason | Count | Share of episodes |
|---|---:|---:|
| `plan_exhausted` | 9 | 25.00% |
| `required_push_position_blocked_by_wall_or_boundary` | 6 | 16.67% |
| `required_push_position_unreachable` | 4 | 11.11% |
| `box_destination_blocked_by_box` | 2 | 5.56% |
| `box_destination_blocked_by_wall_or_boundary` | 1 | 2.78% |

Root planner behavior is mainly:
- local standing/destination infeasibility
- under-complete plans (`plan_exhausted`) before target completion

### Attempt-level failures (`repair_attempts[*]`)

Total attempts across episodes: 105

| Attempt status | Count | Rate |
|---|---:|---:|
| `invalid_plan` | 69 | 65.71% |
| `deadlock` | 21 | 20.00% |
| `plan_exhausted` | 13 | 12.38% |
| `success` | 2 | 1.90% |

| Attempt failure reason | Count | Share of attempts |
|---|---:|---:|
| `repeated_failed_push_guardrail` | 21 | 20.00% |
| `required_push_position_blocked_by_wall_or_boundary` | 14 | 13.33% |
| `plan_exhausted` | 13 | 12.38% |
| `required_push_position_unreachable` | 13 | 12.38% |
| `box_destination_blocked_by_box` | 8 | 7.62% |
| `box_destination_blocked_by_wall_or_boundary` | 7 | 6.67% |
| `required_push_position_blocked_by_box` | 6 | 5.71% |

### Failure locality by semantic push index

| Failed push index | Count | Share of indexed failures |
|---|---:|---:|
| `1` | 22 | 31.43% |
| `2` | 23 | 32.86% |
| `0` | 8 | 11.43% |
| `3` | 5 | 7.14% |
| `4` | 5 | 7.14% |
| `>=5` | 7 | 10.00% |

Most indexed failures occur at semantic push indices `1-2` (64.29% combined).

### Final failure reason by agent

| Agent | Top final reasons |
|---|---|
| `no_memory` | mixed: `required_push_position_unreachable` (2), `required_push_position_blocked_by_wall_or_boundary` (2), `box_destination_blocked_by_wall_or_boundary` (2), `repeated_failed_push_guardrail` (2) |
| `raw_trajectory_memory` | `repeated_failed_push_guardrail` (4), `required_push_position_blocked_by_wall_or_boundary` (3), `box_destination_blocked_by_box` (2) |
| `reflection_heuristic` | dominated by `repeated_failed_push_guardrail` (7), then `box_destination_blocked_by_box` (1), `required_push_position_blocked_by_wall_or_boundary` (1) |

## Worked Trace (Root Cause Chain)

Episode:
- `results/post_guardrail_eval_20260527_clean/no_memory/20260528T002958Z_no_memory_boxoban_eval_000_000_seed43_invalid_plan.json`

Observed chain:
1. First push executes: `{"box_id":3,"push":"Up"}`.
2. Next planned push fails at `push_index=1`: `{"box_id":0,"push":"Right"}`.
3. `push_execution_log` records:
   - `failure_reason = "required_push_position_unreachable"`
   - legal candidates at that state do not include the proposed push.
4. Later repair attempts may terminate with `repeated_failed_push_guardrail`, which is a guardrail enforcement artifact, not the first planner error.

## GIF-to-Log Alignment

Failure GIFs are derived from episode metadata:

- marker frame: `metadata.failure_push_index`
- arrow direction: `metadata.planned_pushes[failure_push_index].push`
- subtitle: `metadata.failure_reason` / deadlock reason path

Cross-check protocol:
1. identify failed push in `push_execution_log`
2. inspect corresponding GIF marker/arrow
3. verify board transition against `trajectory` and `metadata.final_board`

This ties visual failure frames to concrete executor failure reasons rather than coarse terminal status labels.
