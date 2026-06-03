from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sokoban_memory.boxoban import (
    canonical_grid_hash,
    canonical_grid_key,
    grid_to_level,
    is_standard_boxoban_grid,
    level_entry,
    lightweight_difficulty_score,
    load_boxoban_text_file,
    structural_features,
)

REPO_RAW_BASE = "https://raw.githubusercontent.com/google-deepmind/boxoban-levels/master"
DEFAULT_FILES = ("000.txt",)
SEED = 348


@dataclass(frozen=True)
class SourceSpec:
    family: str
    split: str
    files: tuple[str, ...] = DEFAULT_FILES

    @property
    def source_dir(self) -> str:
        if self.family == "hard":
            return "hard"
        return f"{self.family}/{self.split}"


@dataclass
class Candidate:
    spec: SourceSpec
    file_name: str
    source_index: int
    grid: list[str]
    features: dict[str, Any]
    draft_score: float

    @property
    def source_file(self) -> str:
        return f"{self.spec.source_dir}/{self.file_name}"

    @property
    def source(self) -> str:
        return f"google-deepmind/boxoban-levels {self.source_file}#{self.source_index}"

    @property
    def level_id_suffix(self) -> str:
        file_stem = Path(self.file_name).stem
        return f"{file_stem}_{self.source_index:03d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic Boxoban benchmark level suites.")
    parser.add_argument("--cache_dir", default="data/boxoban-levels")
    parser.add_argument("--balanced_output", default="levels/v3_boxoban_balanced.json")
    parser.add_argument("--ood_output", default="levels/v3_boxoban_ood.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--candidate_limit", type=int, default=300)
    parser.add_argument("--solver_max_states", type=int, default=50_000)
    parser.add_argument("--no_download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    cache_dir = Path(args.cache_dir)

    balanced_specs = (
        SourceSpec("unfiltered", "train"),
        SourceSpec("unfiltered", "valid"),
        SourceSpec("medium", "train"),
        SourceSpec("medium", "valid"),
    )
    hard_spec = SourceSpec("hard", "hard", files=("000.txt",))

    pools = {
        (spec.family, spec.split): collect_candidates(
            spec,
            cache_dir=cache_dir,
            candidate_limit=args.candidate_limit,
            no_download=args.no_download,
        )
        for spec in balanced_specs
    }
    hard_candidates = collect_candidates(
        hard_spec,
        cache_dir=cache_dir,
        candidate_limit=args.candidate_limit,
        no_download=args.no_download,
    )

    used_hashes: set[str] = set()
    balanced_levels: list[dict[str, Any]] = []
    balanced_eval_medium_hashes: set[str] = set()
    for family in ("unfiltered", "medium"):
        for split in ("train", "valid"):
            bucketed = bucket_candidates(pools[(family, split)])
            selected = select_bucketed(bucketed, count_per_bucket=4, rng=rng, used_hashes=used_hashes)
            for candidate, bucket in selected:
                split_name = "train" if split == "train" else "eval"
                entry = candidate_to_entry(
                    candidate,
                    split=split_name,
                    bucket=bucket,
                    solver_max_states=args.solver_max_states,
                    ood=False,
                )
                balanced_levels.append(entry)
                if family == "medium" and split == "valid":
                    balanced_eval_medium_hashes.add(entry["canonical_grid_hash"])

    ood_levels: list[dict[str, Any]] = []
    medium_valid_heldout = [
        candidate
        for candidate in pools[("medium", "valid")]
        if canonical_grid_hash(candidate.grid) not in balanced_eval_medium_hashes
        and canonical_grid_hash(candidate.grid) not in used_hashes
    ]
    for candidate in sorted(medium_valid_heldout, key=_hardest_sort_key)[:8]:
        used_hashes.add(canonical_grid_hash(candidate.grid))
        ood_levels.append(
            candidate_to_entry(
                candidate,
                split="eval",
                bucket="ood_medium_valid_hardest",
                solver_max_states=args.solver_max_states,
                ood=True,
            )
        )

    for candidate in sorted(hard_candidates, key=_hardest_sort_key)[:8]:
        if canonical_grid_hash(candidate.grid) in used_hashes:
            continue
        used_hashes.add(canonical_grid_hash(candidate.grid))
        ood_levels.append(
            candidate_to_entry(
                candidate,
                split="eval",
                bucket="ood_hard",
                solver_max_states=args.solver_max_states,
                ood=True,
            )
        )
        if len([level for level in ood_levels if level["source_family"] == "hard"]) == 8:
            break

    balanced_report = {
        "suite_id": "v3_boxoban_balanced",
        "description": "Balanced Boxoban train/eval benchmark for Sokoban memory generalization.",
        "selection_seed": args.seed,
        "source": "google-deepmind/boxoban-levels",
        "levels": sorted(balanced_levels, key=lambda item: (item["split"], item["source_family"], item["difficulty_bucket"], item["level_id"])),
    }
    ood_report = {
        "suite_id": "v3_boxoban_ood",
        "description": "Optional harder OOD Boxoban evaluation suite. Hard levels are marked OOD because the official hard set has no train/eval split.",
        "selection_seed": args.seed,
        "source": "google-deepmind/boxoban-levels",
        "levels": sorted(ood_levels, key=lambda item: (item["source_family"], item["difficulty_bucket"], item["level_id"])),
    }

    write_json(Path(args.balanced_output), balanced_report)
    write_json(Path(args.ood_output), ood_report)
    print(json.dumps({"balanced": suite_summary(balanced_levels), "ood": suite_summary(ood_levels)}, indent=2))


def collect_candidates(
    spec: SourceSpec,
    *,
    cache_dir: Path,
    candidate_limit: int,
    no_download: bool,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_hashes: set[str] = set()
    for file_name in spec.files:
        path = cached_file(spec, file_name, cache_dir=cache_dir)
        if not path.exists():
            if no_download:
                raise FileNotFoundError(path)
            download_file(spec, file_name, path)
        for puzzle in load_boxoban_text_file(path):
            if len(candidates) >= candidate_limit:
                break
            if not is_standard_boxoban_grid(puzzle.grid):
                continue
            grid_hash = canonical_grid_hash(puzzle.grid)
            if grid_hash in seen_hashes:
                continue
            level = grid_to_level(
                level_id="candidate",
                grid=puzzle.grid,
                split="unspecified",
                tags=[],
                source=f"{spec.source_dir}/{file_name}#{puzzle.source_index}",
            )
            features = structural_features(level)
            candidates.append(
                Candidate(
                    spec=spec,
                    file_name=file_name,
                    source_index=puzzle.source_index,
                    grid=puzzle.grid,
                    features=features,
                    draft_score=lightweight_difficulty_score(features),
                )
            )
            seen_hashes.add(grid_hash)
    if len(candidates) < 12 and spec.family != "hard":
        raise ValueError(f"Not enough candidates for {spec.source_dir}: {len(candidates)}")
    return candidates


def bucket_candidates(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    ordered = sorted(candidates, key=lambda candidate: (candidate.draft_score, candidate.source_index))
    buckets = {"open": [], "middle": [], "constrained": []}
    if not ordered:
        return buckets
    for idx, candidate in enumerate(ordered):
        fraction = idx / len(ordered)
        if fraction < 1 / 3:
            buckets["open"].append(candidate)
        elif fraction < 2 / 3:
            buckets["middle"].append(candidate)
        else:
            buckets["constrained"].append(candidate)
    return buckets


def select_bucketed(
    bucketed: dict[str, list[Candidate]],
    *,
    count_per_bucket: int,
    rng: random.Random,
    used_hashes: set[str],
) -> list[tuple[Candidate, str]]:
    selected: list[tuple[Candidate, str]] = []
    for bucket in ("open", "middle", "constrained"):
        candidates = list(bucketed[bucket])
        rng.shuffle(candidates)
        picked = []
        for candidate in candidates:
            grid_hash = canonical_grid_hash(candidate.grid)
            if grid_hash in used_hashes:
                continue
            picked.append(candidate)
            used_hashes.add(grid_hash)
            if len(picked) == count_per_bucket:
                break
        if len(picked) != count_per_bucket:
            raise ValueError(f"Bucket {bucket} only has {len(picked)} selectable candidates")
        selected.extend((candidate, bucket) for candidate in picked)
    return selected


def candidate_to_entry(
    candidate: Candidate,
    *,
    split: str,
    bucket: str,
    solver_max_states: int,
    ood: bool,
) -> dict[str, Any]:
    family = candidate.spec.family
    source_split = candidate.spec.split
    split_label = source_split if source_split != "hard" else "hard"
    level_id = f"boxoban_{family}_{split_label}_{candidate.level_id_suffix}"
    tags = ["boxoban", f"boxoban_{family}", "multi_box"]
    if ood:
        tags.append("ood")
    else:
        tags.append(f"difficulty_{bucket}")
    return level_entry(
        level_id=level_id,
        grid=candidate.grid,
        split=split,
        tags=tags,
        source=candidate.source,
        source_family=family,
        source_split=source_split,
        source_file=candidate.source_file,
        source_index=candidate.source_index,
        difficulty_bucket=bucket,
        solver_max_states=solver_max_states,
    )


def download_file(spec: SourceSpec, file_name: str, output_path: Path) -> None:
    url = f"{REPO_RAW_BASE}/{spec.source_dir}/{file_name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        output_path.write_bytes(response.read())


def cached_file(spec: SourceSpec, file_name: str, *, cache_dir: Path) -> Path:
    return cache_dir / spec.source_dir / file_name


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def suite_summary(levels: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    by_source_bucket: dict[str, int] = {}
    for level in levels:
        by_split[level["split"]] = by_split.get(level["split"], 0) + 1
        key = f"{level['split']}:{level['source_family']}:{level['difficulty_bucket']}"
        by_source_bucket[key] = by_source_bucket.get(key, 0) + 1
    return {"level_count": len(levels), "by_split": by_split, "by_source_bucket": by_source_bucket}


def _hardest_sort_key(candidate: Candidate) -> tuple[float, int]:
    return (-candidate.draft_score, candidate.source_index)


if __name__ == "__main__":
    main()
