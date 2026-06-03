#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# =============================================================================
# STYLE CONFIG — edit these to change look-and-feel (mirrors k3 comparison plots)
# =============================================================================
FIGSIZE = (12, 7.5)

TITLE_SIZE = 23
LINE_TITLE_SIZE = 18
LABEL_SIZE = 20
TICK_SIZE = 17
Y_TICK_SIZE = 18
ANNOTATION_SIZE = 14
LEGEND_SIZE = 15

FONT_WEIGHT = "normal"
ANNOTATION_BOLD = False

BAR_COLORS = ["#728B96", "#969DA8", "#B5C49E", "#E8BE85"]

X_LABEL = "Memory Condition"
STATUS_X_LABEL = "Failure status"
SUBTYPE_X_LABEL = "Invalid-plan subtype"

# Stacked-bar groups (bottom → top)
STACK_GROUP_ORDER = (
    "unreachable_standing_cell",
    "empty_output",
    "deadlock",
    "plan_exhausted",
    "other",
)
STACK_GROUP_LABELS = {
    "unreachable_standing_cell": "Unreachable standing cell",
    "empty_output": "Empty output",
    "deadlock": "Deadlock",
    "plan_exhausted": "Plan exhausted",
    "other": "Other",
}
STACK_GROUP_COLORS = {
    "unreachable_standing_cell": "#5C6B73",
    "empty_output": "#7D8491",
    "deadlock": "#A3B18A",
    "plan_exhausted": "#D4A373",
    "other": "#BC6C25",
}
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

STATUS_LABELS = {
    "invalid_plan": "Invalid plan",
    "deadlock": "Deadlock",
    "plan_exhausted": "Plan exhausted",
    "timeout": "Timeout",
    "budget_exhausted": "Budget exhausted",
    "api_error": "API error",
}

SUBTYPE_LABELS = {
    "unreachable_standing_cell": "Unreachable standing cell",
    "empty_output": "Empty output",
    "blocked_standing_cell": "Blocked standing cell",
    "blocked_destination": "Blocked destination",
    "wrong_box_reference": "Wrong box reference",
    "truncated_output": "Truncated output",
    "json_parse_error": "JSON parse error",
    "schema_error": "Schema error",
    "invalid_plan": "Invalid plan",
    "unknown": "Unknown",
}

CATEGORY_LABELS = {**STATUS_LABELS, **SUBTYPE_LABELS}

STATUS_ORDER = (
    "invalid_plan",
    "deadlock",
    "plan_exhausted",
    "timeout",
    "budget_exhausted",
    "api_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot failure status and invalid-plan subtype counts across memory conditions."
    )
    parser.add_argument("--output_dir", default="docs/figures", help="Directory for output PNGs.")
    parser.add_argument("--max_k", type=int, default=3, help="Include attempts with index < max_k (default: 3).")
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


def load_episodes(results_dir: Path, max_k: int | None = None) -> dict[tuple[str, int], dict]:
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
        if max_k is not None and attempt_index >= max_k:
            continue
        key = (level_id, attempt_index)
        ts = _timestamp_prefix(path)
        existing = episodes.get(key)
        if existing is None or ts > existing[0]:
            episodes[key] = (ts, payload)
    return {key: payload for key, (_, payload) in episodes.items()}


def episode_failure_category(payload: dict) -> str | None:
    """Map a failed episode to a display category (subtype for invalid plans, else status)."""
    status = str(payload.get("status", ""))
    if status == "success":
        return None
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if status == "invalid_plan":
        subtype = metadata.get("failure_subtype")
        if isinstance(subtype, str) and subtype:
            return subtype
        return "invalid_plan"
    return status or "unknown"


def collapse_to_stack_groups(counts: Counter[str]) -> Counter[str]:
    grouped: Counter[str] = Counter()
    for category, count in counts.items():
        if category in STACK_GROUP_ORDER:
            grouped[category] += count
        else:
            grouped["other"] += count
    return grouped


def failure_category_counts(episodes: dict[tuple[str, int], dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for payload in episodes.values():
        category = episode_failure_category(payload)
        if category is not None:
            counts[category] += 1
    return counts


def episode_failure_status(payload: dict) -> str | None:
    status = str(payload.get("status", ""))
    if status == "success":
        return None
    return status or "unknown"


def episode_invalid_plan_subtype(payload: dict) -> str | None:
    if str(payload.get("status", "")) != "invalid_plan":
        return None
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    subtype = metadata.get("failure_subtype")
    if isinstance(subtype, str) and subtype:
        return subtype
    return "unknown"


def failure_status_counts(episodes: dict[tuple[str, int], dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for payload in episodes.values():
        status = episode_failure_status(payload)
        if status is not None:
            counts[status] += 1
    return counts


def invalid_plan_subtype_counts(episodes: dict[tuple[str, int], dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for payload in episodes.values():
        subtype = episode_invalid_plan_subtype(payload)
        if subtype is not None:
            counts[subtype] += 1
    return counts


def _ordered_categories(counts_by_condition: dict[str, Counter[str]], preferred_order: tuple[str, ...]) -> list[str]:
    totals = Counter[str]()
    for counter in counts_by_condition.values():
        totals.update(counter)
    ordered = [key for key in preferred_order if totals.get(key, 0) > 0]
    extras = [key for key, _ in totals.most_common() if key not in ordered]
    return ordered + extras


def _display_label(key: str, label_map: dict[str, str] | None = None) -> str:
    labels = label_map or CATEGORY_LABELS
    return labels.get(key, key.replace("_", " ").title())


def save_stacked_failure_ratio_plot(
    specs: list[ConditionSpec],
    counts_by_condition: dict[str, Counter[str]],
    categories: list[str],
    max_k: int,
    out_path: Path,
    title: str | None = None,
) -> None:
    """One bar per memory condition; stacked segments = share of failed attempts by category."""
    condition_labels = [spec.short_label for spec in specs]
    x = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=FIGSIZE)

    bottoms = np.zeros(len(specs))
    for category in categories:
        values = np.array(
            [counts_by_condition[spec.label].get(category, 0) for spec in specs],
            dtype=float,
        )
        totals = np.array(
            [sum(counts_by_condition[spec.label].values()) for spec in specs],
            dtype=float,
        )
        ratios = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        color = STACK_GROUP_COLORS.get(category, "#ADB5BD")
        bars = ax.bar(
            x,
            ratios,
            bottom=bottoms,
            label=STACK_GROUP_LABELS.get(category, category),
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
            width=0.62,
        )
        for bar, ratio in zip(bars, ratios):
            if ratio < 0.06:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + ratio / 2,
                f"{ratio * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=ANNOTATION_SIZE,
                fontweight=_bar_annotation_weight(),
                color="white" if ratio >= 0.12 else "#333333",
            )
        bottoms += ratios

    ax.set_title(
        title or f"Same-Level Retry: Failure Subtype Mix @ k={max_k}",
        pad=14,
        fontsize=TITLE_SIZE,
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Share of failed attempts")
    ax.set_xticks(x, condition_labels)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.95,
        fontsize=LEGEND_SIZE - 1,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_grouped_bar_plot(
    specs: list[ConditionSpec],
    counts_by_condition: dict[str, Counter[str]],
    categories: list[str],
    title: str,
    x_label: str,
    out_path: Path,
) -> None:
    n_categories = len(categories)
    n_conditions = len(specs)
    x = np.arange(n_categories)
    width = 0.18
    offsets = (np.arange(n_conditions) - (n_conditions - 1) / 2) * width

    fig, ax = plt.subplots(figsize=FIGSIZE)
    max_count = 0
    for cond_idx, spec in enumerate(specs):
        counts = [counts_by_condition[spec.label].get(category, 0) for category in categories]
        max_count = max(max_count, max(counts, default=0))
        bars = ax.bar(
            x + offsets[cond_idx],
            counts,
            width,
            label=spec.short_label,
            color=BAR_COLORS[cond_idx],
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, count in zip(bars, counts):
            if count <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                count + max(max_count * 0.02, 0.3),
                str(count),
                ha="center",
                va="bottom",
                fontsize=ANNOTATION_SIZE,
                fontweight=_bar_annotation_weight(),
            )

    ax.set_title(title, pad=14, fontsize=TITLE_SIZE)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Failed attempts")
    label_map = STATUS_LABELS if x_label == STATUS_X_LABEL else SUBTYPE_LABELS
    ax.set_xticks(x, [_display_label(category, label_map) for category in categories])
    ax.set_ylim(0, max_count * 1.12 if max_count else 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _apply_style()
    output_dir = Path(args.output_dir)
    max_k = args.max_k
    mix_path = output_dir / f"same_level_retry_failure_subtype_mix_k{max_k}.png"
    status_path = output_dir / "same_level_retry_failure_status_by_condition.png"
    subtype_path = output_dir / "same_level_retry_invalid_plan_subtypes_by_condition.png"

    category_by_condition: dict[str, Counter[str]] = {}
    status_by_condition: dict[str, Counter[str]] = {}
    subtype_by_condition: dict[str, Counter[str]] = {}

    print(f"Failure counts from raw episode JSONs (@ k={max_k}):")
    for spec in DEFAULT_CONDITIONS:
        episodes = load_episodes(spec.results_dir, max_k=max_k)
        category_counts = failure_category_counts(episodes)
        status_counts = failure_status_counts(episodes)
        subtype_counts = invalid_plan_subtype_counts(episodes)
        category_by_condition[spec.label] = category_counts
        status_by_condition[spec.label] = status_counts
        subtype_by_condition[spec.label] = subtype_counts
        print(
            f"  {spec.label:18} failed={sum(category_counts.values()):2d}  "
            f"categories={dict(category_counts)}"
        )

    category_list = list(STACK_GROUP_ORDER)
    grouped_by_condition = {
        spec.label: collapse_to_stack_groups(category_by_condition[spec.label])
        for spec in DEFAULT_CONDITIONS
    }
    status_categories = _ordered_categories(status_by_condition, STATUS_ORDER)
    subtype_categories = _ordered_categories(subtype_by_condition, tuple(SUBTYPE_LABELS.keys()))

    print("Stacked groups (@ k={}):".format(max_k))
    for spec in DEFAULT_CONDITIONS:
        print(f"  {spec.short_label:18} {dict(grouped_by_condition[spec.label])}")

    save_stacked_failure_ratio_plot(
        list(DEFAULT_CONDITIONS),
        grouped_by_condition,
        category_list,
        max_k,
        mix_path,
    )
    save_grouped_bar_plot(
        list(DEFAULT_CONDITIONS),
        status_by_condition,
        status_categories,
        "Same-Level Retry: Failure Status by Memory Condition",
        STATUS_X_LABEL,
        status_path,
    )
    save_grouped_bar_plot(
        list(DEFAULT_CONDITIONS),
        subtype_by_condition,
        subtype_categories,
        "Same-Level Retry: Invalid-Plan Subtypes by Memory Condition",
        SUBTYPE_X_LABEL,
        subtype_path,
    )

    print(f"Wrote {mix_path}")
    print(f"Wrote {status_path}")
    print(f"Wrote {subtype_path}")


if __name__ == "__main__":
    main()
