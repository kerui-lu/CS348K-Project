from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from sokoban_memory.levels import load_levels
from sokoban_memory.metrics import STATUSES, summarize_results
from sokoban_memory.types import EpisodeResult

SUMMARY_FILENAMES = {"summary.json", "evaluation_summary.json"}
REQUIRED_EPISODE_FIELDS = {
    "level_id",
    "agent_type",
    "seed",
    "status",
    "step_count",
    "invalid_move_count",
    "total_reward",
    "llm_call_count",
    "token_cost",
    "trajectory",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Sokoban experiment result directories.")
    parser.add_argument("--results_dir", action="append", required=True)
    parser.add_argument("--levels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail_on_validation_error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_result_dirs(
        [Path(path) for path in args.results_dir],
        levels_path=Path(args.levels) if args.levels else None,
    )
    output_text = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    if args.fail_on_validation_error and report["validation_errors"]:
        raise SystemExit(1)


def evaluate_result_dirs(results_dirs: list[Path], levels_path: Path | None = None) -> dict[str, Any]:
    level_metadata = _load_level_metadata(levels_path)
    episodes: list[EpisodeResult] = []
    validation_errors: list[dict[str, str]] = []
    files = _episode_files(results_dirs)

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            validation_errors.append(_error(path, f"invalid json: {exc}"))
            continue

        errors = validate_episode_dict(data, path)
        validation_errors.extend(errors)
        if errors:
            continue

        episodes.append(_episode_from_dict(data, level_metadata))

    by_agent: dict[str, Any] = {}
    for agent_type in sorted({episode.agent_type for episode in episodes}):
        agent_results = [episode for episode in episodes if episode.agent_type == agent_type]
        by_agent[agent_type] = summarize_results(agent_results)
    by_condition: dict[str, Any] = {}
    for condition in sorted({_episode_condition(episode) for episode in episodes}):
        condition_results = [episode for episode in episodes if _episode_condition(episode) == condition]
        by_condition[condition] = summarize_results(condition_results)

    return {
        "results_dirs": [str(path) for path in results_dirs],
        "episode_file_count": len(files),
        "valid_episode_count": len(episodes),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "overall": summarize_results(episodes),
        "by_agent": by_agent,
        "by_condition": by_condition,
        "by_stratum": summarize_by_stratum(episodes, level_metadata),
        "by_condition_stratum": summarize_by_condition_stratum(episodes, level_metadata),
        "per_level": summarize_results(episodes)["per_level"],
    }


def validate_episode_dict(data: Any, path: Path) -> list[dict[str, str]]:
    errors = []
    if not isinstance(data, dict):
        return [_error(path, "episode file must contain a JSON object")]

    missing = sorted(REQUIRED_EPISODE_FIELDS - set(data))
    if missing:
        errors.append(_error(path, f"missing required fields: {', '.join(missing)}"))
        return errors

    status = data.get("status")
    if status not in STATUSES:
        errors.append(_error(path, f"unknown status: {status}"))

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list):
        errors.append(_error(path, "trajectory must be a list"))
        return errors

    if status == "success":
        if not trajectory:
            errors.append(_error(path, "success episode has empty trajectory"))
        elif not trajectory[-1].get("info", {}).get("solved"):
            errors.append(_error(path, "success episode final step is not marked solved"))

    if status == "deadlock":
        if not trajectory:
            errors.append(_error(path, "deadlock episode has empty trajectory"))
        elif not trajectory[-1].get("info", {}).get("deadlocked"):
            errors.append(_error(path, "deadlock episode final step is not marked deadlocked"))

    return errors


def _episode_files(results_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for results_dir in results_dirs:
        files.extend(
            path
            for path in sorted(results_dir.glob("*.json"))
            if path.name not in SUMMARY_FILENAMES
        )
    return files


def _episode_from_dict(data: dict[str, Any], level_metadata: dict[str, dict[str, Any]]) -> EpisodeResult:
    level_id = str(data["level_id"])
    metadata = level_metadata.get(level_id, {})
    enriched = dict(data)
    if enriched.get("level_split") is None:
        enriched["level_split"] = metadata.get("split", "unspecified")
    else:
        enriched.setdefault("level_split", metadata.get("split", "unspecified"))
    if not enriched.get("level_tags"):
        enriched["level_tags"] = metadata.get("tags", [])
    if enriched.get("optimal_steps") is None:
        enriched["optimal_steps"] = metadata.get("optimal_steps")
    field_names = {field.name for field in fields(EpisodeResult)}
    return EpisodeResult(**{key: value for key, value in enriched.items() if key in field_names})


def _load_level_metadata(levels_path: Path | None) -> dict[str, dict[str, Any]]:
    if levels_path is None:
        return {}
    levels = load_levels(levels_path)
    with levels_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)
    raw_levels = raw_data["levels"] if isinstance(raw_data, dict) and "levels" in raw_data else raw_data
    raw_by_id = {
        str(item.get("level_id")): item
        for item in raw_levels
        if isinstance(item, dict) and item.get("level_id") is not None
    }

    metadata = {}
    for level in levels:
        raw = raw_by_id.get(level.level_id, {})
        metadata[level.level_id] = {
            "split": level.split,
            "tags": level.tags,
            "optimal_steps": level.optimal_steps,
            "source_family": raw.get("source_family"),
            "source_split": raw.get("source_split"),
            "difficulty_bucket": raw.get("difficulty_bucket"),
            "wall_density": raw.get("wall_density"),
            "player_reachable_ratio": raw.get("player_reachable_ratio"),
            "initial_legal_push_count": raw.get("initial_legal_push_count"),
            "solver_status": raw.get("solver_status"),
            "solver_min_pushes": raw.get("solver_min_pushes"),
            "solver_min_steps": raw.get("solver_min_steps"),
        }
    return metadata


def summarize_by_stratum(
    episodes: list[EpisodeResult],
    level_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        metadata = level_metadata.get(episode.level_id, {})
        source_family = metadata.get("source_family")
        difficulty_bucket = metadata.get("difficulty_bucket")
        if not source_family or not difficulty_bucket:
            continue
        key = f"{episode.level_split}:{source_family}:{difficulty_bucket}"
        grouped.setdefault(key, []).append(episode)
    return {key: summarize_results(group) for key, group in sorted(grouped.items())}


def summarize_by_condition_stratum(
    episodes: list[EpisodeResult],
    level_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        metadata = level_metadata.get(episode.level_id, {})
        source_family = metadata.get("source_family")
        difficulty_bucket = metadata.get("difficulty_bucket")
        if not source_family or not difficulty_bucket:
            continue
        stratum = f"{episode.level_split}:{source_family}:{difficulty_bucket}"
        key = f"{_episode_condition(episode)}:{stratum}"
        grouped.setdefault(key, []).append(episode)
    return {key: summarize_results(group) for key, group in sorted(grouped.items())}


def _episode_condition(episode: EpisodeResult) -> str:
    condition = episode.metadata.get("condition")
    return str(condition) if condition else episode.agent_type


def _error(path: Path, message: str) -> dict[str, str]:
    return {"path": str(path), "message": message}


if __name__ == "__main__":
    main()
