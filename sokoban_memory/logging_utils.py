from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sokoban_memory.types import EpisodeResult


def save_episode(result: EpisodeResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"{timestamp}_{result.agent_type}_{result.level_id}_"
        f"seed{result.seed}_{result.status}.json"
    )
    path = results_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def save_summary(summary: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path

