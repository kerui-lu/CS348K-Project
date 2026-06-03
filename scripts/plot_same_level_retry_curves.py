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


@dataclass(frozen=True)
class AttemptRecord:
    version: str
    level_id: str
    attempt_index: int
    status: str
    best_goal_completion_rate: float
    timestamp_prefix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot solve rate@k and avg best goal completion@k for same-level retry runs."
    )
    parser.add_argument(
        "--run_dir",
        default="results/v3_eval_heuristic_hyp_minimal_statusline",
        help="Directory containing per-version same-level retry outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default="docs/figures",
        help="Directory where plots are written.",
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=None,
        help="Optional cap for k on x-axis (default: inferred from data).",
    )
    return parser.parse_args()


def _timestamp_prefix(path: Path) -> str:
    name = path.name
    idx = name.find("_")
    return name[:idx] if idx > 0 else name


def load_attempts(run_dir: Path) -> list[AttemptRecord]:
    attempts_by_key: dict[tuple[str, str, int], AttemptRecord] = {}
    for version_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        version = version_dir.name
        for level_dir in sorted(p for p in version_dir.iterdir() if p.is_dir()):
            for episode_path in sorted(level_dir.glob("*.json")):
                if episode_path.name in {"summary.json", "evaluation_summary.json"}:
                    continue
                payload = json.loads(episode_path.read_text(encoding="utf-8"))
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                attempt_index = metadata.get("iteration_attempt_index")
                if not isinstance(attempt_index, int):
                    continue
                level_id = str(payload.get("level_id", level_dir.name))
                status = str(payload.get("status", "unknown"))
                best_goal = metadata.get("best_goal_completion_rate", 0.0)
                if not isinstance(best_goal, (int, float)):
                    best_goal = 0.0
                record = AttemptRecord(
                    version=version,
                    level_id=level_id,
                    attempt_index=attempt_index,
                    status=status,
                    best_goal_completion_rate=float(best_goal),
                    timestamp_prefix=_timestamp_prefix(episode_path),
                )
                key = (version, level_id, attempt_index)
                existing = attempts_by_key.get(key)
                if existing is None or record.timestamp_prefix > existing.timestamp_prefix:
                    attempts_by_key[key] = record
    return sorted(
        attempts_by_key.values(),
        key=lambda r: (r.version, r.level_id, r.attempt_index),
    )


def compute_curves(
    attempts: list[AttemptRecord], max_k_override: int | None
) -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[AttemptRecord]]] = defaultdict(lambda: defaultdict(list))
    max_attempt_index = 0
    for rec in attempts:
        grouped[rec.version][rec.level_id].append(rec)
        max_attempt_index = max(max_attempt_index, rec.attempt_index)
    if not grouped:
        raise ValueError("No iterative attempt records found.")

    inferred_max_k = max_attempt_index + 1
    max_k = min(max_k_override, inferred_max_k) if max_k_override is not None else inferred_max_k
    ks = list(range(1, max_k + 1))

    solve_rate_by_version: dict[str, list[float]] = {}
    best_goal_by_version: dict[str, list[float]] = {}

    for version in sorted(grouped.keys()):
        level_map = grouped[version]
        level_count = len(level_map)
        solve_curve: list[float] = []
        best_curve: list[float] = []
        for k in ks:
            solved_levels = 0
            best_sum = 0.0
            for recs in level_map.values():
                eligible = [r for r in recs if r.attempt_index < k]
                if not eligible:
                    continue
                if any(r.status == "success" for r in eligible):
                    solved_levels += 1
                best_sum += max(r.best_goal_completion_rate for r in eligible)
            solve_curve.append(solved_levels / level_count if level_count else 0.0)
            best_curve.append(best_sum / level_count if level_count else 0.0)
        solve_rate_by_version[version] = solve_curve
        best_goal_by_version[version] = best_curve
    return ks, solve_rate_by_version, best_goal_by_version


def _label(version: str) -> str:
    label_map = {
        "baseline": "Baseline",
        "v1_specific": "V1 Specific",
        "v2_complete_plan": "V2 Complete Plan",
        "v3_hybrid_verifier": "V3 Hybrid Verifier",
    }
    return label_map.get(version, version.replace("_", " ").title())


def save_plot(
    x: list[int],
    y_by_version: dict[str, list[float]],
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for version in sorted(y_by_version.keys()):
        plt.plot(x, y_by_version[version], marker="o", linewidth=2, label=_label(version))
    plt.title(title)
    plt.xlabel("Retry budget k (attempt cutoff)")
    plt.ylabel(y_label)
    plt.xticks(x)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    attempts = load_attempts(run_dir)
    ks, solve_rate, best_goal = compute_curves(attempts, args.max_k)

    solve_plot = output_dir / "same_level_retry_solve_rate_at_k.png"
    best_plot = output_dir / "same_level_retry_avg_best_goal_completion_at_k.png"

    save_plot(
        ks,
        solve_rate,
        "Same-Level Retry: Solve Rate@k (4 options)",
        "Solve rate@k",
        solve_plot,
    )
    save_plot(
        ks,
        best_goal,
        "Same-Level Retry: Avg Best Goal Completion@k (4 options)",
        "Avg best goal completion@k",
        best_plot,
    )

    print(f"Wrote {solve_plot}")
    print(f"Wrote {best_plot}")


if __name__ == "__main__":
    main()
