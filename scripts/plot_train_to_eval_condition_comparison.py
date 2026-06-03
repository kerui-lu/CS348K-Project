#!/usr/bin/env python3
"""Plot train-to-eval global heuristic comparison figures (one-shot eval)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# Reuse failure plotting helpers
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_failure_subtypes_condition_comparison import (  # noqa: E402
    STACK_GROUP_ORDER,
    STATUS_ORDER,
    collapse_to_stack_groups,
    failure_category_counts,
    failure_status_counts,
    invalid_plan_subtype_counts,
    load_episodes,
    save_grouped_bar_plot,
    save_stacked_failure_ratio_plot,
)

FIGSIZE = (10, 6.5)
TITLE_SIZE = 23
LABEL_SIZE = 20
TICK_SIZE = 17
Y_TICK_SIZE = 18
ANNOTATION_SIZE = 14
LEGEND_SIZE = 15
FONT_WEIGHT = "normal"
ANNOTATION_BOLD = False
BAR_COLORS = ["#728B96", "#969DA8", "#B5C49E"]
X_LABEL = "Memory Condition"


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    short_label: str
    results_dir: Path


DEFAULT_CONDITIONS = (
    ConditionSpec("No Memory", "Baseline", Path("results/v3_track0_eval_gpt52_low_16384")),
    ConditionSpec("Global Heuristic Top 3", "Global (3)", Path("results/v3_eval_train_one_shot_global_heuristic_gpt52_low_16384")),
    ConditionSpec("Global Heuristic All 16", "Global (16)", Path("results/v3_eval_train_one_shot_global_heuristic_all_gpt52_low_16384")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train-to-eval comparison figures.")
    parser.add_argument("--output_dir", default="docs/figures", help="Directory for output PNGs.")
    return parser.parse_args()


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "font.weight": FONT_WEIGHT,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": FONT_WEIGHT,
            "axes.labelsize": LABEL_SIZE,
            "axes.labelweight": FONT_WEIGHT,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": Y_TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def _bar_annotation_weight() -> str | None:
    return "bold" if ANNOTATION_BOLD else "normal"


def _timestamp_prefix(path: Path) -> str:
    idx = path.name.find("_")
    return path.name[:idx] if idx > 0 else path.name


def load_episodes_flat(results_dir: Path) -> dict[tuple[str, int], dict]:
    episodes: dict[tuple[str, int], tuple[str, dict]] = {}
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation_summary.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) or {}
        level_id = str(payload.get("level_id", ""))
        if not level_id:
            continue
        attempt_index = metadata.get("iteration_attempt_index", 0)
        if not isinstance(attempt_index, int):
            attempt_index = 0
        key = (level_id, attempt_index)
        ts = _timestamp_prefix(path)
        existing = episodes.get(key)
        if existing is None or ts > existing[0]:
            episodes[key] = (ts, payload)
    return {key: payload for key, (_, payload) in episodes.items()}


def _group_by_level(episodes: dict[tuple[str, int], dict]) -> dict[str, list[tuple[int, dict]]]:
    by_level: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for (level_id, attempt_index), payload in episodes.items():
        by_level[level_id].append((attempt_index, payload))
    return by_level


def levels_solved(episodes: dict[tuple[str, int], dict]) -> tuple[int, int]:
    by_level = _group_by_level(episodes)
    solved = sum(
        1
        for recs in by_level.values()
        if any(str(payload.get("status")) == "success" for _, payload in recs)
    )
    return solved, len(by_level)


def avg_best_goal(episodes: dict[tuple[str, int], dict]) -> float:
    by_level = _group_by_level(episodes)
    if not by_level:
        return 0.0
    total = 0.0
    for recs in by_level.values():
        best_values = []
        for _, payload in recs:
            metadata = payload.get("metadata", {}) or {}
            value = metadata.get("best_goal_completion_rate", 0.0)
            best_values.append(float(value) if isinstance(value, (int, float)) else 0.0)
        total += max(best_values) if best_values else 0.0
    return total / len(by_level)


def save_metric_bar_plot(
    labels: list[str],
    values: list[float],
    total_levels: int,
    title: str,
    y_label: str,
    out_path: Path,
    as_percent: bool,
    count_labels: list[int] | None = None,
) -> None:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(x, values, color=BAR_COLORS[: len(labels)], edgecolor="#333333", linewidth=0.6)
    ax.set_title(title, pad=14, fontsize=TITLE_SIZE)
    ax.set_ylabel(y_label, fontsize=LABEL_SIZE)
    ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE)
    if as_percent:
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.3)
    for idx, bar in enumerate(bars):
        if count_labels is not None:
            label = f"{count_labels[idx]}/{total_levels}"
        elif as_percent:
            label = f"{values[idx] * 100:.1f}%"
        else:
            label = f"{values[idx]:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            values[idx] + (0.02 if as_percent else max(values[idx] * 0.02, 0.01)),
            label,
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight=_bar_annotation_weight(),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _apply_style()
    output_dir = Path(args.output_dir)
    prefix = "train_to_eval"

    episode_sets: dict[str, dict[tuple[str, int], dict]] = {}
    total_levels = 0
    print("Computing train-to-eval metrics from raw episode JSONs:")
    for spec in DEFAULT_CONDITIONS:
        episodes = load_episodes_flat(spec.results_dir)
        episode_sets[spec.label] = episodes
        solved, level_count = levels_solved(episodes)
        total_levels = max(total_levels, level_count)
        best = avg_best_goal(episodes)
        print(
            f"  {spec.label:24} total={level_count:2d}  solved={solved:2d}  "
            f"avg_best={best * 100:5.1f}%"
        )

    labels = [spec.short_label for spec in DEFAULT_CONDITIONS]
    label_keys = [spec.label for spec in DEFAULT_CONDITIONS]
    solved_counts = [levels_solved(episode_sets[key])[0] for key in label_keys]
    solve_rates = [count / total_levels if total_levels else 0.0 for count in solved_counts]
    best_goals = [avg_best_goal(episode_sets[key]) for key in label_keys]

    solve_path = output_dir / f"{prefix}_condition_solve_rate.png"
    best_path = output_dir / f"{prefix}_avg_best_goal_completion.png"
    mix_path = output_dir / f"{prefix}_failure_subtype_mix.png"
    status_path = output_dir / f"{prefix}_failure_status_by_condition.png"
    subtype_path = output_dir / f"{prefix}_invalid_plan_subtypes_by_condition.png"

    save_metric_bar_plot(
        labels,
        solve_rates,
        total_levels,
        "Train-to-Eval: Levels Solved (One-Shot Eval)",
        "Solve rate",
        solve_path,
        as_percent=True,
        count_labels=solved_counts,
    )
    save_metric_bar_plot(
        labels,
        best_goals,
        total_levels,
        "Train-to-Eval: Avg Best Goal Completion",
        "Avg best goal completion",
        best_path,
        as_percent=True,
    )

    category_by_condition: dict[str, Counter[str]] = {}
    status_by_condition: dict[str, Counter[str]] = {}
    subtype_by_condition: dict[str, Counter[str]] = {}
    for spec in DEFAULT_CONDITIONS:
        episodes = load_episodes(spec.results_dir, max_k=1)
        category_by_condition[spec.label] = failure_category_counts(episodes)
        status_by_condition[spec.label] = failure_status_counts(episodes)
        subtype_by_condition[spec.label] = invalid_plan_subtype_counts(episodes)

    grouped_by_condition = {
        spec.label: collapse_to_stack_groups(category_by_condition[spec.label])
        for spec in DEFAULT_CONDITIONS
    }
    status_categories = [key for key in STATUS_ORDER if any(status_by_condition[s.label].get(key, 0) for s in DEFAULT_CONDITIONS)]
    subtype_categories = sorted(
        {key for counter in subtype_by_condition.values() for key in counter},
        key=lambda key: -sum(counter.get(key, 0) for counter in subtype_by_condition.values()),
    )

    save_stacked_failure_ratio_plot(
        list(DEFAULT_CONDITIONS),
        grouped_by_condition,
        list(STACK_GROUP_ORDER),
        1,
        mix_path,
        title="Train-to-Eval: Failure Subtype Mix (One-Shot Eval)",
    )

    save_grouped_bar_plot(
        list(DEFAULT_CONDITIONS),
        status_by_condition,
        status_categories,
        "Train-to-Eval: Failure Status by Memory Condition",
        "Failure status",
        status_path,
    )
    save_grouped_bar_plot(
        list(DEFAULT_CONDITIONS),
        subtype_by_condition,
        subtype_categories,
        "Train-to-Eval: Invalid-Plan Subtypes by Memory Condition",
        "Invalid-plan subtype",
        subtype_path,
    )

    print(f"Wrote {solve_path}")
    print(f"Wrote {best_path}")
    print(f"Wrote {mix_path}")
    print(f"Wrote {status_path}")
    print(f"Wrote {subtype_path}")


if __name__ == "__main__":
    main()
