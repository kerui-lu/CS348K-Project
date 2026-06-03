#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# =============================================================================
# STYLE CONFIG — edit these to change look-and-feel across all figures
# =============================================================================
FIGSIZE = (10, 6.5)          # figure width × height (inches)
LINE_FIGSIZE = FIGSIZE         # line plots match bar chart size

TITLE_SIZE = 23                # plot title font size
LABEL_SIZE = 20                # x/y axis title font size ("Memory Condition", etc.)
TICK_SIZE = 17                 # default tick label size
Y_TICK_SIZE = 18               # y-axis tick labels (often larger than x)
ANNOTATION_SIZE = 14           # text above bars (e.g. "8/24")
LEGEND_SIZE = 15               # cumulative curve legend

FONT_WEIGHT = "normal"       # "normal", "semibold", "bold"
ANNOTATION_BOLD = False        # bar labels like "8/24" — keep False for unbolded

# Bar / line colors (one per condition, left to right)
BAR_COLORS = ["#728B96", "#969DA8", "#B5C49E", "#E8BE85"]
LINE_COLORS = BAR_COLORS

X_LABEL = "Memory Condition"   # x-axis title for bar charts

# Condition display names (short labels on x-axis + legend)
# Edit DEFAULT_CONDITIONS below to change labels or result directories.
# =============================================================================


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    short_label: str
    results_dir: Path


DEFAULT_CONDITIONS = (
    ConditionSpec("Single Snapshot", "Baseline (Single)", Path("results/v3_track0_eval_gpt52_low_16384")),
    ConditionSpec("Compact Summary", "Compact", Path("results/v3_eval_verifier_retry_k3_gpt52_low_16384")),
    ConditionSpec("Raw Trajectory", "Raw", Path("results/v3_eval_raw_same_level_k3_gpt52_low_16384_clean")),
    ConditionSpec("Heuristic", "Heuristic", Path("results/v3_eval_heuristic_same_level_k3_gpt52_low_16384")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot same-level retry comparison figures from raw episode JSONs."
    )
    parser.add_argument("--max_k", type=int, default=3, help="Max retry budget (default: 3).")
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


def load_episodes(results_dir: Path) -> dict[tuple[str, int], dict]:
    """Load episodes keyed by (level_id, attempt_index), keeping latest timestamp per key."""
    episodes: dict[tuple[str, int], tuple[str, dict]] = {}
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation_summary.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
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


def levels_solved_at_k(episodes: dict[tuple[str, int], dict], k: int) -> tuple[int, int]:
    """Return (levels_solved, total_levels) by attempt cutoff k."""
    by_level = _group_by_level(episodes)
    if not by_level:
        return 0, 0

    solved = 0
    for recs in by_level.values():
        eligible = [payload for idx, payload in sorted(recs, key=lambda item: item[0]) if idx < k]
        if eligible and any(str(payload.get("status")) == "success" for payload in eligible):
            solved += 1
    return solved, len(by_level)


def avg_best_goal_at_k(episodes: dict[tuple[str, int], dict], k: int) -> float:
    by_level = _group_by_level(episodes)
    if not by_level:
        return 0.0

    best_sum = 0.0
    for recs in by_level.values():
        eligible = [(idx, payload) for idx, payload in sorted(recs, key=lambda item: item[0]) if idx < k]
        if not eligible:
            continue
        best_values = []
        for _, payload in eligible:
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            value = metadata.get("best_goal_completion_rate", 0.0)
            best_values.append(float(value) if isinstance(value, (int, float)) else 0.0)
        best_sum += max(best_values) if best_values else 0.0
    return best_sum / len(by_level)


def cumulative_levels_solved(episodes: dict[tuple[str, int], dict], max_k: int) -> list[int]:
    """Cumulative levels solved after each attempt cutoff 1..max_k."""
    return [levels_solved_at_k(episodes, attempt_cutoff)[0] for attempt_cutoff in range(1, max_k + 1)]


def best_goal_curve(episodes: dict[tuple[str, int], dict], max_k: int) -> list[float]:
    """Avg best goal completion after each attempt cutoff 1..max_k."""
    return [avg_best_goal_at_k(episodes, attempt_cutoff) for attempt_cutoff in range(1, max_k + 1)]


def save_levels_solved_bar_plot(
    labels: list[str],
    counts: list[int],
    total_levels: int,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    """Bar height = solve rate (0–100%); label above bar = count/total (e.g. 8/24)."""
    rates = [count / total_levels if total_levels else 0.0 for count in counts]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(x, rates, color=BAR_COLORS[: len(labels)], edgecolor="#333333", linewidth=0.6)
    ax.set_title(title, pad=14)
    ax.set_ylabel(y_label)
    ax.set_xlabel(X_LABEL)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.3)
    for bar, count, rate in zip(bars, counts, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{count}/{total_levels}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight=_bar_annotation_weight(),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_best_goal_bar_plot(
    labels: list[str],
    values: list[float],
    max_k: int,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    """Single bar per condition at k=max_k; height and labels = avg best goal completion %."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(x, values, color=BAR_COLORS[: len(labels)], edgecolor="#333333", linewidth=0.6)
    ax.set_title(title, pad=14)
    ax.set_ylabel(y_label)
    ax.set_xlabel(X_LABEL)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xticks(x, labels, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight=_bar_annotation_weight(),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _zoomed_percent_ylim(values: list[float], pad: float = 0.04) -> tuple[float, float]:
    return max(0.0, min(values) - pad), min(1.0, max(values) + pad)


def save_cumulative_curve_plot(
    specs: list[ConditionSpec],
    cumulative_by_label: dict[str, list[int]],
    total_levels: int,
    max_k: int,
    out_path: Path,
) -> None:
    """Line plot: solve rate vs attempt cutoff; y-axis zoomed to the observed range."""
    attempt_cutoffs = list(range(1, max_k + 1))
    all_rates: list[float] = []
    series: list[tuple[ConditionSpec, str, list[float]]] = []
    for spec, color in zip(specs, LINE_COLORS):
        counts = cumulative_by_label[spec.label]
        rates = [count / total_levels if total_levels else 0.0 for count in counts]
        all_rates.extend(rates)
        series.append((spec, color, rates))

    y_pad = 0.04
    y_min = max(0.0, min(all_rates) - y_pad)
    y_max = min(1.0, max(all_rates) + y_pad)

    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    for spec, color, rates in series:
        ax.plot(
            attempt_cutoffs,
            rates,
            marker="o",
            markersize=9,
            linewidth=2.8,
            label=spec.short_label,
            color=color,
        )
    ax.set_title("Same-Level Retry: Cumulative Levels Solved by Attempt", pad=14, fontsize=TITLE_SIZE)
    ax.set_xlabel("Attempt cutoff (k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Solve rate", fontsize=LABEL_SIZE)
    ax.set_xticks(attempt_cutoffs)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_best_goal_curve_plot(
    specs: list[ConditionSpec],
    best_goal_by_label: dict[str, list[float]],
    max_k: int,
    out_path: Path,
) -> None:
    """Line plot: avg best goal completion vs attempt cutoff; y-axis zoomed to observed range."""
    attempt_cutoffs = list(range(1, max_k + 1))
    all_values: list[float] = []
    series: list[tuple[ConditionSpec, str, list[float]]] = []
    for spec, color in zip(specs, LINE_COLORS):
        values = best_goal_by_label[spec.label]
        all_values.extend(values)
        series.append((spec, color, values))

    y_min, y_max = _zoomed_percent_ylim(all_values)

    fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
    for spec, color, values in series:
        ax.plot(
            attempt_cutoffs,
            values,
            marker="o",
            markersize=9,
            linewidth=2.8,
            label=spec.short_label,
            color=color,
        )
    ax.set_title(
        "Same-Level Retry: Avg Best Goal Completion by Attempt",
        pad=14,
        fontsize=TITLE_SIZE,
    )
    ax.set_xlabel("Attempt cutoff (k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Avg best goal completion", fontsize=LABEL_SIZE)
    ax.set_xticks(attempt_cutoffs)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _apply_style()
    output_dir = Path(args.output_dir)
    max_k = args.max_k

    episode_sets: dict[str, dict[tuple[str, int], dict]] = {}
    total_levels = 0

    print("Computing metrics from raw episode JSONs:")
    for spec in DEFAULT_CONDITIONS:
        episodes = load_episodes(spec.results_dir)
        episode_sets[spec.label] = episodes
        _, level_count = levels_solved_at_k(episodes, k=1)
        total_levels = max(total_levels, level_count)

        solved_k1, _ = levels_solved_at_k(episodes, k=1)
        solved_k3, _ = levels_solved_at_k(episodes, k=max_k)
        best_k1 = avg_best_goal_at_k(episodes, k=1)
        best_k3 = avg_best_goal_at_k(episodes, k=max_k)
        cumulative = cumulative_levels_solved(episodes, max_k=max_k)
        print(
            f"  {spec.label:18} total={level_count:2d}  "
            f"solved@1={solved_k1:2d}  solved@{max_k}={solved_k3:2d}  "
            f"avg_best@1={best_k1 * 100:5.1f}%  avg_best@{max_k}={best_k3 * 100:5.1f}%  "
            f"cumulative={cumulative}"
        )

    labels = [spec.short_label for spec in DEFAULT_CONDITIONS]
    label_keys = [spec.label for spec in DEFAULT_CONDITIONS]
    solved_k1 = [levels_solved_at_k(episode_sets[key], k=1)[0] for key in label_keys]
    solved_k3 = [levels_solved_at_k(episode_sets[key], k=max_k)[0] for key in label_keys]
    best_k3 = [avg_best_goal_at_k(episode_sets[key], k=max_k) for key in label_keys]
    cumulative_by_label = {
        spec.label: cumulative_levels_solved(episode_sets[spec.label], max_k=max_k)
        for spec in DEFAULT_CONDITIONS
    }
    best_goal_by_label = {
        spec.label: best_goal_curve(episode_sets[spec.label], max_k=max_k)
        for spec in DEFAULT_CONDITIONS
    }

    k3_path = output_dir / "same_level_retry_condition_solve_rate_k3.png"
    k1_path = output_dir / "same_level_retry_condition_levels_solved_k1.png"
    cumulative_path = output_dir / "same_level_retry_cumulative_solved_by_attempt.png"
    best_curve_path = output_dir / "same_level_retry_avg_best_goal_completion_by_attempt.png"
    best_path = output_dir / "same_level_retry_condition_avg_best_goal_completion_k3.png"

    save_levels_solved_bar_plot(
        labels,
        solved_k3,
        total_levels,
        f"Same-Level Retry: Levels Solved @ k={max_k}",
        "Solve rate",
        k3_path,
    )
    save_levels_solved_bar_plot(
        labels,
        solved_k1,
        total_levels,
        "Same-Level Retry: Levels Solved @ k=1",
        "Solve rate",
        k1_path,
    )
    save_cumulative_curve_plot(
        list(DEFAULT_CONDITIONS),
        cumulative_by_label,
        total_levels,
        max_k,
        cumulative_path,
    )
    save_best_goal_curve_plot(
        list(DEFAULT_CONDITIONS),
        best_goal_by_label,
        max_k,
        best_curve_path,
    )
    save_best_goal_bar_plot(
        labels,
        best_k3,
        max_k,
        f"Same-Level Retry: Avg Best Goal Completion @ k={max_k}",
        "Avg best goal completion",
        best_path,
    )

    print(f"Wrote {k3_path}")
    print(f"Wrote {k1_path}")
    print(f"Wrote {cumulative_path}")
    print(f"Wrote {best_curve_path}")
    print(f"Wrote {best_path}")


if __name__ == "__main__":
    main()
