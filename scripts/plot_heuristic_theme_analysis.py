#!/usr/bin/env python3
"""Analyze same-level heuristic themes vs failure modes + level-set redundancy."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGSIZE = (14, 8)
HEATMAP_FIGSIZE = (12, 10)
TITLE_SIZE = 23
LABEL_SIZE = 20
TICK_SIZE = 14
LEGEND_SIZE = 13
FONT_WEIGHT = "normal"

FAILURE_GROUP_ORDER = (
    "unreachable_standing_cell",
    "empty_output",
    "deadlock",
    "plan_exhausted",
    "other",
)
FAILURE_GROUP_LABELS = {
    "unreachable_standing_cell": "Unreachable\nstanding cell",
    "empty_output": "Empty output",
    "deadlock": "Deadlock",
    "plan_exhausted": "Plan exhausted",
    "other": "Other",
}

THEME_ORDER = (
    "standing_cell_reachability",
    "corridor_chokepoint",
    "connectivity",
    "completeness",
    "ordering",
    "deadlock_avoidance",
    "caution_avoidance",
)
THEME_LABELS = {
    "standing_cell_reachability": "Standing-cell\nreachability",
    "corridor_chokepoint": "Corridor /\nchokepoint",
    "connectivity": "Connectivity",
    "completeness": "Completeness /\nanti-stall",
    "ordering": "Push ordering",
    "deadlock_avoidance": "Deadlock\navoidance",
    "caution_avoidance": "Caution /\navoidance",
}

THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("standing_cell_reachability", ("standing cell", "reachable", "reachability")),
    ("corridor_chokepoint", ("corridor", "chokepoint", "narrow", "1-tile", "hallway", "doorway", "passage", "choke")),
    ("connectivity", ("connectivity", "disconnected", "connected component", "circulation loop")),
    (
        "completeness",
        (
            "empty push",
            "never output",
            "complete plan",
            "every box",
            "plan exhaustion",
            "goal-directed macro",
            "do not stop",
        ),
    ),
    ("ordering", ("ordering", "deepest-first", "constrained areas first", "prioritize", "first")),
    ("deadlock_avoidance", ("deadlock", "non-goal corner", "2x2")),
    ("caution_avoidance", ("avoid pushing", "do not push", "avoid push", "reject that push")),
)

RENDERED_HEURISTIC_COUNT = 3
RECONSTRUCTION_PATH = Path("analysis/heuristic_same_level_reconstruction.json")
HEURISTIC_RESULTS_DIR = Path("results/v3_eval_heuristic_same_level_k3_gpt52_low_16384")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot heuristic theme and redundancy analyses.")
    parser.add_argument("--reconstruction", type=Path, default=RECONSTRUCTION_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "font.weight": FONT_WEIGHT,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": FONT_WEIGHT,
            "axes.labelsize": LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )


def classify_themes(text: str) -> set[str]:
    lowered = text.lower()
    themes: set[str] = set()
    for theme, keywords in THEME_RULES:
        if any(keyword in lowered for keyword in keywords):
            themes.add(theme)
    return themes or {"caution_avoidance"}


def collapse_failure_group(category: str) -> str:
    if category in FAILURE_GROUP_ORDER:
        return category
    return "other"


def episode_failure_category(payload: dict) -> str:
    status = str(payload.get("status", ""))
    if status == "success":
        return "success"
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if status == "invalid_plan":
        subtype = metadata.get("failure_subtype")
        if isinstance(subtype, str) and subtype:
            return subtype
        return "invalid_plan"
    return status or "unknown"


def load_failure_group(result_file: str) -> str:
    path = Path(result_file)
    if not path.is_file():
        return "other"
    payload = json.loads(path.read_text(encoding="utf-8"))
    category = episode_failure_category(payload)
    if category == "success":
        return "other"
    return collapse_failure_group(category)


def short_level_label(level_id: str) -> str:
    match = re.search(r"(\d{3})$", level_id)
    return match.group(1) if match else level_id[-8:]


def build_failure_theme_matrix(records: list[dict]) -> np.ndarray:
    counts: dict[str, Counter[str]] = {
        group: Counter() for group in FAILURE_GROUP_ORDER
    }
    for record in records:
        failure_group = load_failure_group(record["result_file"])
        heuristics = record.get("heuristics", [])[:RENDERED_HEURISTIC_COUNT]
        for heuristic in heuristics:
            for theme in classify_themes(str(heuristic)):
                counts[failure_group][theme] += 1
    matrix = np.array(
        [[counts[group].get(theme, 0) for theme in THEME_ORDER] for group in FAILURE_GROUP_ORDER],
        dtype=float,
    )
    return matrix


def token_set(heuristics: list[str]) -> set[str]:
    tokens: set[str] = set()
    for heuristic in heuristics:
        for word in re.findall(r"[a-z0-9]+", heuristic.lower()):
            if len(word) > 3:
                tokens.add(word)
    return tokens


def build_level_heuristic_sets(records: list[dict]) -> tuple[dict[str, set[str]], dict[str, str]]:
    """One heuristic token set + exact set hash per level (rendered top-3)."""
    by_level: dict[str, dict] = {}
    for record in records:
        level_id = record["level_id"]
        attempt = record.get("after_attempt_index", 0)
        existing = by_level.get(level_id)
        if existing is None or attempt < existing.get("after_attempt_index", 999):
            by_level[level_id] = record
    level_token_sets: dict[str, set[str]] = {}
    level_hashes: dict[str, str] = {}
    for level_id, record in by_level.items():
        rendered = [str(item).strip() for item in record.get("heuristics", [])[:RENDERED_HEURISTIC_COUNT]]
        level_token_sets[level_id] = token_set(rendered)
        level_hashes[level_id] = str(record.get("heuristic_set_hash", ""))
    return level_token_sets, level_hashes


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def save_failure_theme_heatmap(matrix: np.ndarray, out_path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.1, 1.0]})

    im0 = axes[0].imshow(matrix, aspect="auto", cmap="YlGnBu")
    axes[0].set_title("Heuristic Themes After Failure", pad=12, fontsize=TITLE_SIZE - 2)
    axes[0].set_xticks(range(len(THEME_ORDER)), [THEME_LABELS[t] for t in THEME_ORDER], rotation=35, ha="right")
    axes[0].set_yticks(range(len(FAILURE_GROUP_ORDER)), [FAILURE_GROUP_LABELS[g] for g in FAILURE_GROUP_ORDER])
    axes[0].set_xlabel("Heuristic theme (rendered top-3)")
    axes[0].set_ylabel("Preceding failure mode")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value <= 0:
                continue
            axes[0].text(j, i, str(value), ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Theme tag count")

    im1 = axes[1].imshow(normalized, aspect="auto", cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title("Theme Mix Within Failure Mode", pad=12, fontsize=TITLE_SIZE - 2)
    axes[1].set_xticks(range(len(THEME_ORDER)), [THEME_LABELS[t] for t in THEME_ORDER], rotation=35, ha="right")
    axes[1].set_yticks(range(len(FAILURE_GROUP_ORDER)), [FAILURE_GROUP_LABELS[g] for g in FAILURE_GROUP_ORDER])
    axes[1].set_xlabel("Heuristic theme (rendered top-3)")
    axes[1].set_ylabel("Preceding failure mode")
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            value = normalized[i, j]
            if value <= 0.03:
                continue
            axes[1].text(
                j,
                i,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color="white" if value >= 0.35 else "black",
                fontsize=11,
            )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Share within failure mode")

    fig.suptitle(
        "Same-Level Heuristic: Failure Modes × Generated Themes",
        fontsize=TITLE_SIZE,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_jaccard_heatmap(
    level_sets: dict[str, set[str]],
    level_hashes: dict[str, str],
    out_path: Path,
) -> float:
    levels = sorted(level_sets)
    n = len(levels)
    matrix = np.eye(n)
    for i, level_a in enumerate(levels):
        for j in range(i + 1, n):
            level_b = levels[j]
            score = jaccard(level_sets[level_a], level_sets[level_b])
            matrix[i, j] = score
            matrix[j, i] = score

    off_diag = matrix[np.triu_indices(n, k=1)]
    mean_sim = float(off_diag.mean()) if off_diag.size else 0.0

    hash_counts = Counter(level_hashes.values())
    unique_hashes = len(hash_counts)
    most_common_hash, most_common_count = hash_counts.most_common(1)[0] if hash_counts else ("", 0)

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma")
    short_labels = [short_level_label(level) for level in levels]
    ax.set_xticks(range(n), short_labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n), short_labels, fontsize=9)
    ax.set_title(
        "Same-Level Heuristic Redundancy Across Levels",
        pad=14,
        fontsize=TITLE_SIZE,
    )
    ax.set_xlabel("Eval level")
    ax.set_ylabel("Eval level")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Token Jaccard (rendered top-3)")
    ax.text(
        0.02,
        1.01,
        (
            f"Mean token Jaccard = {mean_sim:.2f}; "
            f"{unique_hashes} unique exact sets across {n} levels "
            f"(largest duplicate group = {most_common_count} levels)"
        ),
        transform=ax.transAxes,
        fontsize=13,
        va="bottom",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return mean_sim


def save_condition_comparison_panel(out_path: Path) -> None:
    """Side panel: failure mix (4 conditions) vs heuristic theme totals."""
    from plot_failure_subtypes_condition_comparison import (  # type: ignore import-not-found
        DEFAULT_CONDITIONS,
        STACK_GROUP_COLORS,
        STACK_GROUP_LABELS,
        STACK_GROUP_ORDER,
        collapse_to_stack_groups,
        failure_category_counts,
        load_episodes,
    )

    grouped_by_condition: dict[str, Counter] = {}
    for spec in DEFAULT_CONDITIONS:
        episodes = load_episodes(spec.results_dir, max_k=3)
        grouped_by_condition[spec.label] = collapse_to_stack_groups(failure_category_counts(episodes))

    recon = json.loads(RECONSTRUCTION_PATH.read_text(encoding="utf-8"))
    theme_totals = Counter()
    for record in recon["records"]:
        for heuristic in record.get("heuristics", [])[:RENDERED_HEURISTIC_COUNT]:
            theme_totals.update(classify_themes(str(heuristic)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Failure mix across conditions
    specs = list(DEFAULT_CONDITIONS)
    x = np.arange(len(specs))
    bottoms = np.zeros(len(specs))
    for group in STACK_GROUP_ORDER:
        values = np.array(
            [grouped_by_condition[spec.label].get(group, 0) for spec in specs],
            dtype=float,
        )
        totals = np.array([sum(grouped_by_condition[spec.label].values()) for spec in specs], dtype=float)
        ratios = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
        axes[0].bar(
            x,
            ratios,
            bottom=bottoms,
            label=STACK_GROUP_LABELS[group],
            color=STACK_GROUP_COLORS[group],
            edgecolor="#333333",
            linewidth=0.5,
            width=0.62,
        )
        bottoms += ratios
    axes[0].set_title("Failure Subtype Mix @ k=3", fontsize=TITLE_SIZE - 2)
    axes[0].set_xticks(x, [spec.short_label for spec in specs])
    axes[0].set_ylabel("Share of failed attempts")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=LEGEND_SIZE - 2, framealpha=0.95)

    # Heuristic theme totals
    theme_counts = [theme_totals.get(theme, 0) for theme in THEME_ORDER]
    theme_total = sum(theme_counts) or 1
    theme_ratios = [count / theme_total for count in theme_counts]
    axes[1].bar(
        range(len(THEME_ORDER)),
        theme_ratios,
        color="#728B96",
        edgecolor="#333333",
        linewidth=0.6,
    )
    axes[1].set_title("Heuristic Theme Mix (rendered top-3 tags)", fontsize=TITLE_SIZE - 2)
    axes[1].set_xticks(range(len(THEME_ORDER)), [THEME_LABELS[t] for t in THEME_ORDER], rotation=35, ha="right")
    axes[1].set_ylabel("Share of theme tags")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Failure Modes vs Heuristic Themes", fontsize=TITLE_SIZE, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _apply_style()
    output_dir = args.output_dir

    project_root = Path(__file__).resolve().parents[1]
    scripts_dir = Path(__file__).resolve().parent
    for path in (project_root, scripts_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    recon = json.loads(args.reconstruction.read_text(encoding="utf-8"))
    records = recon["records"]

    matrix = build_failure_theme_matrix(records)
    level_sets, level_hashes = build_level_heuristic_sets(records)

    cross_path = output_dir / "heuristic_failure_mode_theme_cross.png"
    heatmap_path = output_dir / "heuristic_level_jaccard_heatmap.png"
    compare_path = output_dir / "heuristic_failure_vs_theme_comparison.png"

    save_failure_theme_heatmap(matrix, cross_path)
    mean_jaccard = save_jaccard_heatmap(level_sets, level_hashes, heatmap_path)

    save_condition_comparison_panel(compare_path)

    print("Failure mode × theme counts (rendered top-3 tags):")
    for i, group in enumerate(FAILURE_GROUP_ORDER):
        row = {THEME_ORDER[j]: int(matrix[i, j]) for j in range(len(THEME_ORDER)) if matrix[i, j] > 0}
        print(f"  {group}: {row}")
    print(f"Mean off-diagonal Jaccard similarity: {mean_jaccard:.3f}")
    print(f"Wrote {cross_path}")
    print(f"Wrote {heatmap_path}")
    print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()
