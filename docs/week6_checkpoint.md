## Our project proposal

We are going to build a Sokoban gameplay framework for LLM agents and compare two ways of using past experience: raw trajectory memory versus failure reflection plus heuristic extraction. We will demonstrate success using trajectory visualizations, extracted failure rules, and plots comparing solve rate, number of deadlocks, and solution efficiency across the two memory schemes. Our approach is to let the agent play repeated Sokoban levels, collect trajectories of both successful and failed attempts, and test whether abstracted strategic knowledge is more useful than directly replaying raw past trajectories.

## Our evaluation pipeline

We first run a no_memory agent on the train split to collect failed trajectories, then use those same failures to build two memory banks: raw trajectory memories and reflection-generated heuristics. We then use these memories and evaluate three agents: no_memory, raw_trajectory_memory, and reflection_heuristic on the held-out eval split using the same levels, seed, model, temperature, output cap, horizon, and memory budget. Each episode is automatically labeled as success, deadlock, timeout, budget_exhausted, api_error, or invalid_failure, and we generate both per-episode trajectory logs and aggregate summaries. For this checkpoint, the main goal is to show that this evaluation protocol runs cleanly end-to-end: our current run produced 60 valid episode logs with no validation errors, API errors, budget exhaustion, or invalid-failure contamination. Since all agents currently have zero solve rate, we interpret these results as an evaluation milestone rather than a final performance result; the pipeline is now in place to study whether raw trajectories or reflection heuristics lead to fewer deadlocks, fewer timeouts, and eventually higher solve rates.

## Checkpoint artifacts:
The level split used for this checkpoint is saved in levels/v2_pilot.json, with train levels used for memory construction and eval levels used for comparison. The generated memory artifacts are saved in memory_banks/raw_failures.json and memory_banks/reflection_heuristics.json. The three eval result directories are results/v2_task3_large_no_memory, results/v2_task3_large_raw, and results/v2_task3_large_reflection, and the consolidated comparison summary is saved in results/v2_task3_large_evaluation_summary.json. The relevant code is in the Sokoban gameplay/evaluation pipeline, including the agent implementations for no_memory, raw_trajectory_memory, and reflection_heuristic, the episode outcome labeling logic, and the summary-generation scripts.


## What kinds of heuristics do we think are important?

- Do not push boxes into non-goal corners
- Avoid repeating previously seen board states
- Do not push a box into a position where the player cannot access the side needed for the next push.
- Avoid moves that trap the player or make important push positions unreachable.
- Only push a box if the push moves it closer to a goal or creates a better future position.

## What do we define as success?

- At the episode level, success means the Sokoban puzzle is solved: all boxes are on target squares. In the logged trajectory, a successful episode must end with info.solved == true.
- At the research level, success means showing whether memory improves LLM agent performance under controlled conditions, especially by comparing no_memory, raw_trajectory_memory, and reflection_heuristic agents on the same held-out eval levels.

## How could we evaluate success?

We evaluate success using automatic episode outcomes and aggregate metrics.

**Primary metric:**

- solve_rate = success_count / episodes

**Secondary metrics:**

- deadlock_rate
- timeout_rate
- average_success_steps
- solution_efficiency = optimal_steps / actual_steps
- average_solution_efficiency
- steps_over_optimal_average
- invalid_move_rate
- api_error_rate
- Budget_exhausted_rate

A clean evaluation should first check that validation_error_count == 0, then compare agents under identical eval levels, seeds, model, prompt version, memory budget, temperature, and max output tokens.

## What is our hypothesis?

Reflection-based memory will help the LLM agent avoid repeated Sokoban mistakes better than raw trajectory memory or no memory. Our current checkpoint results do not yet support a final claim about which memory representation is best because all agents have zero solve rate; instead, they show that the evaluation pipeline is clean and can already detect differences in failure modes such as deadlock versus timeout.

## When could we define the situation as a deadlock?

**Current:** prefers false negatives over false positives. We only label an episode as deadlocked when the detected pattern is highly likely to be irreversible. The current implementation covers non-target corner deadlocks. Future extensions will add static dead-square detection, conservative wall deadlocks, and 2x2 freeze patterns.

**Future:**

1. corner deadlock
2. static dead squares
3. 2x2 wall/box freeze pattern
4. wall deadlock with no target and no exit
5. tunnel trap
6. two-box freeze
7. multi-box recursive freeze

## Do we have to create novel data? Was this boxoban included in pre-training

Current results show no evidence that memorization is helping performance, but public Boxoban pretraining contamination remains a possible limitation.

**Primary evaluation:**

Use current Boxoban-derived eval suite to test the pipeline and compare memory conditions.

**Robustness / final evaluation:**

Add a small novel level suite generated by us after the project starts.

## Should we move multiple steps per call if its a trivial move?

We should not simply repeat the same direction blindly. Instead, we should introduce macro-actions for trivial navigation: let the LLM choose meaningful push-level decisions, while the environment deterministically executes the shortest non-pushing path to the push position.

**Example:**

- Push box at (3,4) Right
- Push box at (5,2) Up
- Move to reachable position near box, then push Left

We need to change metrics too.

- primitive_step_count
- decision_count
- push_count
- llm_call_count
- Macro_action_count

This is future work rather than part of the current checkpoint comparison, because changing from primitive actions to push-level macro-actions would also change the meaning of step count, decision count, and solution efficiency.

## What representation of NoMemoryAgent training failures helps the reflection LLM generate more useful Sokoban heuristics?

**Logic:** We first run a NoMemoryAgent on train levels and collect failed episodes. Then we compare different ways of converting those failures into heuristic-generation data.

**Raw trajectory baseline:**

Keep the final few primitive steps of each failed episode. This is simple, but it may mostly show timeout symptoms such as repeated back-and-forth moves.

**Push-level macro trajectory:**

Compress each failure into box-pushing transitions. Each macro step records the grid before a push, the push action, the grid after the push, and whether the push caused progress or deadlock.

**Example:**

```json
{
  "macro_step": 2,
  "grid_before": "...",
  "push_action": "Down",
  "box_from": [7, 6],
  "box_to": [8, 6],
  "grid_after": "...",
  "local_result": "deadlock",
  "failure_reason": "box pushed into a non-target corner"
}
```

**Critical-transition trajectory:**

Extract the most failure-relevant state-action-state examples, such as the first deadlock-causing push or the loop segment before a timeout. This gives the reflection LLM more direct evidence about what went wrong.

For each data format, we use the same reflection prompt and model to generate a heuristic memory bank. Then we evaluate the resulting ReflectionHeuristicAgent on the same held-out eval levels. The data creation method that produces higher solve rate, lower deadlock/timeout rate, and more specific non-redundant heuristics is considered better.

## How did other researchers evaluate performance for Sokoban? [aditri]

Background links:

- [OpenReview paper](https://openreview.net/pdf?id=qeziG97WUZ)
- [arXiv version](http://arxiv.org/html/2505.14552v1)
- [LangGraph Sokoban blog post](https://blog.gopenai.com/using-llms-and-langgraph-to-tackle-sokoban-puzzles-5f50b43b9515)

Sokoban is used in various benchmarks for LLM gameplay because it exposes long-horizon planning failures, sparse feedback, and irreversible mistakes. So by using memory + heuristics we hope that an LLM agent can improve performance and make fewer mistakes.

**LLMGame:** This paper is closest to our project – a benchmark that evaluates LLMs on Sokoban among other games and extra agent scaffolds that help them play games like perception and memory and reasoning modules. They also have a procedural progress report that gives partial credit like number of boxes placed on the last level. Their memory module has transient and reflection which is similar to our raw trajectories and explicit lessons learned. Without this harness support they say over three fourths of models don't score any points on Sokoban. They realized that their perception module mattered more than the memory. This means they convert the game backend state into a symbolic/text representation, such as object coordinates like "Box at (2,3)". The point is to remove visual perception errors so the model can focus on planning. This means we should make sure to represent the prompt as an ascii grid with a coordinate summary. The reflection aspect asks the model to analyze previous game states and actions, focusing on how the state changed, whether the action was effective, what patterns/issues appeared, and strategic insights for future actions. So we should aim for our heuristic memory to not be vague but state-conditioned and specific to guide action selection.

**KORGym:** This paper evaluates LLM agents performance against Sokoban. They generate the game environment, ask the LLM for an action, verify the action and repeat until solved or reached a round limit. The verification step is to make sure the move was legal/if the game is done. This was a zero-shot prompting setup (no task-specific training examples). They only evaluated based on success of solving, where the O3-mini scored 0.78 or normalized benchmarks.

This blogpost uses a LangGraph-based Sokoban agent to separate the system into nodes that call the LLM, check moves, and continue until success or failure. Here too they use a verification step, indicating that we cannot trust the LLM evaluation on success/valid moves. One technique used was removing repeated/cyclic states from the trajectory which aligns with a heuristic of repeating previously seen board states. They evaluated based on response time, success rate, solution quality and conciseness of solutions. They found gpt-oss-20b and gemini-2.5-flash performed better on their example in terms of response speed and solution quality, but even those models were inconsistent.

These prior systems motivate our project, but they do not fully answer our question. KORGym and LMGame-Bench show that Sokoban is difficult for LLM agents and that scaffolding such as symbolic perception, memory, reflection, and verification can improve or clarify performance. The LangGraph blog post similarly shows that an external executor is necessary because LLMs often generate invalid or inconsistent solutions. Our project builds on these ideas by isolating memory representation as the main experimental variable. Rather than only asking whether an LLM can solve Sokoban, we ask whether train-derived experience should be reused as raw failed trajectories or distilled into explicit heuristics. By evaluating these memory conditions under matched levels, model, seed, prompt budget, and verifier, we can measure whether heuristic memory reduces deadlocks, timeouts, and repeated mistakes more effectively than raw memory or no memory.
