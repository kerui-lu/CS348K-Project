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

    return {
        "results_dirs": [str(path) for path in results_dirs],
        "episode_file_count": len(files),
        "valid_episode_count": len(episodes),
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors,
        "overall": summarize_results(episodes),
        "by_agent": by_agent,
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
    return {
        level.level_id: {
            "split": level.split,
            "tags": level.tags,
            "optimal_steps": level.optimal_steps,
        }
        for level in levels
    }


def _error(path: Path, message: str) -> dict[str, str]:
    return {"path": str(path), "message": message}


if __name__ == "__main__":
    main()
