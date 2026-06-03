#!/usr/bin/env python3
"""Render player trajectories overlaid on Sokoban boards from episode JSON logs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from sokoban_memory.env import DIRECTIONS

PLAYER_CHARS = {"@", "+"}
BOX_CHARS = {"$", "*"}
TARGET_CHARS = {".", "*", "+"}

PUSH_COLORS = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725", "#e85d04", "#9b2226"]
WALK_LANE_SPREAD = 0.11
SUCCESS_SEPARATED_LANE_SPREAD = 0.14
VISIT_LANE = 0.22
# Offsets when revisiting a cell — keeps the polyline connected but untangled.
VISIT_OFFSETS: list[tuple[float, float]] = [
    (0.0, 0.0),
    (0.0, VISIT_LANE),
    (VISIT_LANE, 0.0),
    (0.0, -VISIT_LANE),
    (-VISIT_LANE, 0.0),
    (VISIT_LANE, VISIT_LANE),
    (-VISIT_LANE, VISIT_LANE),
    (VISIT_LANE, -VISIT_LANE),
    (-VISIT_LANE, -VISIT_LANE),
]


@dataclass(frozen=True)
class BoardSnapshot:
    width: int
    height: int
    walls: set[tuple[int, int]]
    targets: set[tuple[int, int]]
    boxes: set[tuple[int, int]]
    player: tuple[int, int] | None


@dataclass(frozen=True)
class PathSegment:
    push_index: int
    points: list[tuple[int, int]]


@dataclass(frozen=True)
class BoxPushArrow:
    push_index: int
    box_label: str
    direction: str
    box_from: tuple[int, int]
    box_to: tuple[int, int]


@dataclass(frozen=True)
class PathEdge:
    start: tuple[int, int]
    end: tuple[int, int]
    is_push: bool
    push_index: int


@dataclass(frozen=True)
class LegacyTrajectoryOverlay:
    level_id: str
    status: str
    player_path: list[tuple[int, int]]
    push_indices: set[int]
    failure_push_index: int | None
    subtitle: str


@dataclass(frozen=True)
class SeparatedOriginalOverlay:
    level_id: str
    status: str
    player_path: list[tuple[int, int]]
    edges: list[PathEdge]
    push_indices: set[int]
    failure_push_index: int | None
    subtitle: str


@dataclass(frozen=True)
class TrajectoryOverlay:
    level_id: str
    status: str
    player_path: list[tuple[int, int]]
    segments: list[PathSegment]
    box_pushes: list[BoxPushArrow]
    failure_push_index: int | None
    subtitle: str
    step_push_index: list[int] = field(default_factory=list)


def parse_board(state_text: str) -> BoardSnapshot:
    rows = state_text.strip().split("\n")
    height = len(rows)
    width = max(len(row) for row in rows) if rows else 0
    walls: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None
    for r, row in enumerate(rows):
        for c, char in enumerate(row):
            pos = (r, c)
            if char == "#":
                walls.add(pos)
            if char in TARGET_CHARS:
                targets.add(pos)
            if char in BOX_CHARS:
                boxes.add(pos)
            if char in PLAYER_CHARS:
                player = pos
    return BoardSnapshot(width=width, height=height, walls=walls, targets=targets, boxes=boxes, player=player)


def player_from_state(state_text: str) -> tuple[int, int] | None:
    return parse_board(state_text).player


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, dict) and "row" in value and "col" in value:
        return int(value["row"]), int(value["col"])
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _offset(row: float, col: float, drow: float, dcol: float, amount: float) -> tuple[float, float]:
    length = math.hypot(drow, dcol)
    if length == 0:
        return row, col
    perp_r, perp_c = -dcol / length, drow / length
    return row + amount * perp_r, col + amount * perp_c


def _lane_offset(push_index: int, spread: float = WALK_LANE_SPREAD) -> float:
    center = 2
    return (push_index % 5 - center) * spread


def connected_nonoverlap_positions(path: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """Assign a slightly different lane each time the player re-enters a cell."""
    visit_count: dict[tuple[int, int], int] = {}
    drawn: list[tuple[float, float]] = []
    for row, col in path:
        visit_idx = visit_count.get((row, col), 0)
        visit_count[(row, col)] = visit_idx + 1
        dr, dc = VISIT_OFFSETS[visit_idx % len(VISIT_OFFSETS)]
        drawn.append((row + dr, col + dc))
    return drawn


def _initial_box_labels(boxes: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    return {pos: f"B{idx}" for idx, pos in enumerate(sorted(boxes))}


def _assign_box_labels_through_pushes(
    start_boxes: set[tuple[int, int]],
    pushes: list[tuple[int, str, tuple[int, int], tuple[int, int]]],
) -> list[BoxPushArrow]:
    positions = {label: pos for pos, label in _initial_box_labels(start_boxes).items()}
    arrows: list[BoxPushArrow] = []
    for push_index, direction, box_from, box_to in pushes:
        label = None
        for box_label, pos in positions.items():
            if pos == box_from:
                label = box_label
                break
        if label is None:
            label = min(positions, key=lambda lb: abs(positions[lb][0] - box_from[0]) + abs(positions[lb][1] - box_from[1]))
        arrows.append(
            BoxPushArrow(
                push_index=push_index,
                box_label=label,
                direction=direction,
                box_from=box_from,
                box_to=box_to,
            )
        )
        positions[label] = box_to
    return arrows


def extract_player_path(trajectory: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], set[int]]:
    """Original single-path extraction (plasma overlay, player push arrows)."""
    if not trajectory:
        return [], set()
    path: list[tuple[int, int]] = []
    push_indices: set[int] = set()
    first = player_from_state(str(trajectory[0].get("state", "")))
    if first is not None:
        path.append(first)
    for step in trajectory:
        info = step.get("info") if isinstance(step.get("info"), dict) else {}
        if info.get("pushed_box") or info.get("semantic_phase") == "push":
            push_indices.add(len(path) - 1 if path else 0)
        pos = player_from_state(str(step.get("next_state", "")))
        if pos is not None:
            if not path or path[-1] != pos:
                path.append(pos)
            if info.get("pushed_box") or info.get("semantic_phase") == "push":
                push_indices.add(len(path) - 1)
    return path, push_indices


def extract_path_edges(trajectory: list[dict[str, Any]]) -> list[PathEdge]:
    if not trajectory:
        return []
    edges: list[PathEdge] = []
    previous = player_from_state(str(trajectory[0].get("state", "")))
    for step in trajectory:
        info = step.get("info") if isinstance(step.get("info"), dict) else {}
        current = player_from_state(str(step.get("next_state", "")))
        if current is None or previous is None:
            if current is not None:
                previous = current
            continue
        if current != previous:
            edges.append(
                PathEdge(
                    start=previous,
                    end=current,
                    is_push=bool(info.get("pushed_box") or info.get("semantic_phase") == "push"),
                    push_index=int(info.get("push_index") or 0),
                )
            )
        previous = current
    return edges


def extract_path_and_segments(trajectory: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], list[PathSegment], list[int]]:
    if not trajectory:
        return [], [], []
    path: list[tuple[int, int]] = []
    step_push_index: list[int] = []
    segments: list[PathSegment] = []
    current_index = 0
    current_points: list[tuple[int, int]] = []

    first = player_from_state(str(trajectory[0].get("state", "")))
    if first is not None:
        path.append(first)
        current_points.append(first)

    for step in trajectory:
        info = step.get("info") if isinstance(step.get("info"), dict) else {}
        push_index = int(info.get("push_index") or 0)
        pos = player_from_state(str(step.get("next_state", "")))
        if pos is None:
            continue

        if push_index != current_index and current_points:
            segments.append(PathSegment(push_index=current_index, points=list(current_points)))
            current_index = push_index
            current_points = [path[-1]] if path else []

        if not path or path[-1] != pos:
            path.append(pos)
            current_points.append(pos)
            step_push_index.append(push_index)

    if current_points:
        segments.append(PathSegment(push_index=current_index, points=current_points))
    return path, segments, step_push_index


def box_pushes_from_execution_log(
    log: list[dict[str, Any]],
    start_boxes: set[tuple[int, int]],
) -> list[BoxPushArrow]:
    raw: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        push_index = int(entry.get("push_index", len(raw)))
        intent = entry.get("intent") if isinstance(entry.get("intent"), dict) else {}
        direction = str(intent.get("push") or "")
        if direction not in DIRECTIONS:
            continue
        box_from = _cell_tuple(entry.get("resolved_box_before_push") or entry.get("resolved_box"))
        box_to = _cell_tuple(entry.get("destination_cell") or entry.get("box_destination"))
        if box_from is None or box_to is None:
            continue
        raw.append((push_index, direction, box_from, box_to))
    return _assign_box_labels_through_pushes(start_boxes, raw)


def box_pushes_from_trajectory(
    trajectory: list[dict[str, Any]],
    start_boxes: set[tuple[int, int]],
) -> list[BoxPushArrow]:
    raw: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
    for step in trajectory:
        info = step.get("info") if isinstance(step.get("info"), dict) else {}
        if info.get("semantic_phase") != "push" and not info.get("pushed_box"):
            continue
        planned = info.get("planned_push") if isinstance(info.get("planned_push"), dict) else {}
        direction = str(planned.get("push") or step.get("executed_action") or "")
        if direction not in DIRECTIONS:
            continue
        box = planned.get("box")
        if not isinstance(box, list) or len(box) != 2:
            continue
        box_from = (int(box[0]), int(box[1]))
        dr, dc = DIRECTIONS[direction]
        box_to = (box_from[0] + dr, box_from[1] + dc)
        raw.append((int(info.get("push_index") or len(raw)), direction, box_from, box_to))
    return _assign_box_labels_through_pushes(start_boxes, raw)


def failure_push_index_from_episode(episode: dict[str, Any]) -> int | None:
    metadata = episode.get("metadata")
    if isinstance(metadata, dict):
        idx = metadata.get("failure_push_index")
        if isinstance(idx, int):
            return idx
    diagnostics = episode.get("diagnostics")
    if isinstance(diagnostics, dict):
        idx = diagnostics.get("failure_push_index")
        if isinstance(idx, int):
            return idx
    return None


def _episode_subtitle(episode: dict[str, Any], failure_idx: int | None) -> str:
    metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
    status = str(episode.get("status", "unknown"))
    reason = ""
    if metadata.get("failure_reason"):
        reason = str(metadata["failure_reason"])
    elif episode.get("first_attempt_failure_reason"):
        reason = str(episode["first_attempt_failure_reason"])
    subtitle_parts = [status]
    if reason:
        subtitle_parts.append(reason.replace("_", " "))
    if failure_idx is not None:
        subtitle_parts.append(f"fail@push {failure_idx}")
    return " · ".join(subtitle_parts)


def separated_original_from_episode(episode_path: Path) -> SeparatedOriginalOverlay:
    episode = json.loads(episode_path.read_text())
    trajectory = episode.get("trajectory", [])
    path, push_indices = extract_player_path(trajectory)
    failure_idx = failure_push_index_from_episode(episode)
    return SeparatedOriginalOverlay(
        level_id=str(episode.get("level_id", episode_path.stem)),
        status=str(episode.get("status", "unknown")),
        player_path=path,
        edges=extract_path_edges(trajectory),
        push_indices=push_indices,
        failure_push_index=failure_idx,
        subtitle=_episode_subtitle(episode, failure_idx),
    )


def legacy_overlay_from_episode(episode_path: Path) -> LegacyTrajectoryOverlay:
    episode = json.loads(episode_path.read_text())
    trajectory = episode.get("trajectory", [])
    path, push_indices = extract_player_path(trajectory)
    failure_idx = failure_push_index_from_episode(episode)
    return LegacyTrajectoryOverlay(
        level_id=str(episode.get("level_id", episode_path.stem)),
        status=str(episode.get("status", "unknown")),
        player_path=path,
        push_indices=push_indices,
        failure_push_index=failure_idx,
        subtitle=_episode_subtitle(episode, failure_idx),
    )


def overlay_from_episode(episode_path: Path) -> TrajectoryOverlay:
    episode = json.loads(episode_path.read_text())
    trajectory = episode.get("trajectory", [])
    start_board = parse_board(str(trajectory[0].get("state", ""))) if trajectory else parse_board("")
    path, segments, _ = extract_path_and_segments(trajectory)

    metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
    log = metadata.get("push_execution_log")
    if isinstance(log, list) and log:
        box_pushes = box_pushes_from_execution_log(log, start_board.boxes)
    else:
        box_pushes = box_pushes_from_trajectory(trajectory, start_board.boxes)

    failure_idx = failure_push_index_from_episode(episode)
    return TrajectoryOverlay(
        level_id=str(episode.get("level_id", episode_path.stem)),
        status=str(episode.get("status", "unknown")),
        player_path=path,
        segments=segments,
        box_pushes=box_pushes,
        failure_push_index=failure_idx,
        subtitle=_episode_subtitle(episode, failure_idx),
    )


def _push_color(push_index: int) -> str:
    return PUSH_COLORS[push_index % len(PUSH_COLORS)]


def _path_index_for_push(
    path: list[tuple[int, int]],
    push_indices: set[int],
    failure_push_index: int,
) -> tuple[int, int] | None:
    if not path:
        return None
    ordered_push_positions = sorted(push_indices)
    if failure_push_index < len(ordered_push_positions):
        return path[ordered_push_positions[failure_push_index]]
    if failure_push_index < len(path):
        return path[min(failure_push_index, len(path) - 1)]
    return path[-1]


def _draw_board_base(ax: plt.Axes, board: BoardSnapshot) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, board.width - 0.5)
    ax.set_ylim(board.height - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#1a1a1a")

    for r in range(board.height):
        for c in range(board.width):
            pos = (r, c)
            if pos in board.walls:
                color = "#3d3d3d"
            else:
                color = "#ececec"
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor="#bdbdbd",
                    linewidth=0.4,
                )
            )

    for r, c in board.targets:
        ax.plot(c, r, marker="o", markersize=5, color="#f4b942", alpha=0.9, zorder=2)

    for r, c in board.boxes:
        ax.add_patch(
            Rectangle(
                (c - 0.35, r - 0.35),
                0.7,
                0.7,
                facecolor="#8b5a2b",
                edgecolor="#5a3a1a",
                linewidth=1.0,
                zorder=3,
            )
        )


def draw_panel_original(ax: plt.Axes, board: BoardSnapshot, overlay: LegacyTrajectoryOverlay) -> None:
    _draw_board_base(ax, board)
    path = overlay.player_path
    if len(path) >= 2:
        colors = plt.cm.plasma([i / max(len(path) - 1, 1) for i in range(len(path))])
        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            is_push = i in overlay.push_indices or (i + 1) in overlay.push_indices
            ax.plot(
                [c0, c1],
                [r0, r1],
                color=colors[i],
                linewidth=3.2 if is_push else 1.8,
                alpha=0.95 if is_push else 0.65,
                solid_capstyle="round",
                zorder=4,
            )
            if is_push:
                ax.add_patch(
                    FancyArrowPatch(
                        (c0, r0),
                        (c1, r1),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        linewidth=0,
                        color="#e85d04",
                        alpha=0.95,
                        zorder=5,
                    )
                )

    if path:
        sr, sc = path[0]
        ax.plot(sc, sr, marker="o", markersize=10, color="#2a9d8f", zorder=6)
        er, ec = path[-1]
        end_color = "#d62828" if overlay.status != "success" else "#264653"
        ax.plot(ec, er, marker="X", markersize=10, color=end_color, zorder=6)

    if overlay.failure_push_index is not None and path:
        fail_pos = _path_index_for_push(path, overlay.push_indices, overlay.failure_push_index)
        if fail_pos is not None:
            fr, fc = fail_pos
            ax.plot(fc, fr, marker="*", markersize=16, color="#d62828", zorder=7)

    ax.set_title(f"{overlay.level_id}\n{overlay.subtitle}", fontsize=9, pad=6)


def draw_panel_original_separated(
    ax: plt.Axes,
    board: BoardSnapshot,
    overlay: SeparatedOriginalOverlay,
    *,
    lane_spread: float = SUCCESS_SEPARATED_LANE_SPREAD,
) -> None:
    """Original plasma + orange push arrows, with per-push lane offsets to untangle overlaps."""
    _draw_board_base(ax, board)
    edges = overlay.edges
    if edges:
        colors = plt.cm.plasma([i / max(len(edges) - 1, 1) for i in range(len(edges))])
        for i, edge in enumerate(edges):
            r0, c0 = edge.start
            r1, c1 = edge.end
            dr, dc = r1 - r0, c1 - c0
            lane = _lane_offset(edge.push_index, spread=lane_spread)
            or0, oc0 = _offset(r0, c0, dr, dc, lane)
            or1, oc1 = _offset(r1, c1, dr, dc, lane)
            ax.plot(
                [oc0, oc1],
                [or0, or1],
                color=colors[i],
                linewidth=3.2 if edge.is_push else 1.8,
                alpha=0.95 if edge.is_push else 0.65,
                solid_capstyle="round",
                zorder=4,
            )
            if edge.is_push:
                ax.add_patch(
                    FancyArrowPatch(
                        (oc0, or0),
                        (oc1, or1),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        linewidth=0,
                        color="#e85d04",
                        alpha=0.95,
                        zorder=5,
                    )
                )

    path = overlay.player_path
    if path:
        sr, sc = path[0]
        ax.plot(sc, sr, marker="o", markersize=10, color="#2a9d8f", zorder=6)
        er, ec = path[-1]
        end_color = "#d62828" if overlay.status != "success" else "#264653"
        ax.plot(ec, er, marker="X", markersize=10, color=end_color, zorder=6)

    if overlay.failure_push_index is not None and path:
        fail_pos = _path_index_for_push(path, overlay.push_indices, overlay.failure_push_index)
        if fail_pos is not None:
            fr, fc = fail_pos
            ax.plot(fc, fr, marker="*", markersize=16, color="#d62828", zorder=7)

    ax.set_title(f"{overlay.level_id}\n{overlay.subtitle}", fontsize=9, pad=6)


def draw_panel_labeled(ax: plt.Axes, board: BoardSnapshot, overlay: TrajectoryOverlay) -> None:
    _draw_board_base(ax, board)

    for segment in overlay.segments:
        if len(segment.points) < 2:
            continue
        color = _push_color(segment.push_index)
        lane = _lane_offset(segment.push_index)
        for i in range(len(segment.points) - 1):
            r0, c0 = segment.points[i]
            r1, c1 = segment.points[i + 1]
            dr, dc = r1 - r0, c1 - c0
            or0, oc0 = _offset(r0, c0, dr, dc, lane)
            or1, oc1 = _offset(r1, c1, dr, dc, lane)
            ax.plot(
                [oc0, oc1],
                [or0, or1],
                color=color,
                linewidth=2.0,
                alpha=0.8,
                solid_capstyle="round",
                zorder=4,
            )

    for push in overlay.box_pushes:
        fr, fc = push.box_from
        tr, tc = push.box_to
        color = _push_color(push.push_index)
        ax.add_patch(
            FancyArrowPatch(
                (fc, fr),
                (tc, tr),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=2.8,
                color=color,
                zorder=6,
            )
        )
        mr, mc = (fr + tr) / 2, (fc + tc) / 2
        ax.text(
            mc,
            mr,
            f"{push.push_index + 1}:{push.box_label}",
            fontsize=6.5,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.9),
            zorder=7,
        )

    path = overlay.player_path
    if path:
        sr, sc = path[0]
        ax.plot(sc, sr, marker="o", markersize=10, color="#2a9d8f", zorder=8)
        er, ec = path[-1]
        end_color = "#d62828" if overlay.status != "success" else "#264653"
        ax.plot(ec, er, marker="X", markersize=10, color=end_color, zorder=8)

    if overlay.failure_push_index is not None:
        for push in overlay.box_pushes:
            if push.push_index == overlay.failure_push_index:
                fr, fc = push.box_from
                ax.plot(fc, fr, marker="*", markersize=16, color="#d62828", zorder=9)
                break

    ax.set_title(f"{overlay.level_id}\n{overlay.subtitle}", fontsize=9, pad=6)


def draw_panel_connected_path(ax: plt.Axes, board: BoardSnapshot, overlay: LegacyTrajectoryOverlay) -> None:
    """One connected trajectory; revisits use visit-lanes; arrows show movement only."""
    _draw_board_base(ax, board)
    path = overlay.player_path
    if len(path) < 2:
        return

    drawn = connected_nonoverlap_positions(path)
    edge_count = len(drawn) - 1
    colors = plt.cm.plasma([i / max(edge_count, 1) for i in range(edge_count + 1)])

    for i in range(edge_count):
        r0, c0 = drawn[i]
        r1, c1 = drawn[i + 1]
        color = colors[i]
        ax.add_patch(
            FancyArrowPatch(
                (c0, r0),
                (c1, r1),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=2.0,
                color=color,
                alpha=0.9,
                shrinkA=0,
                shrinkB=0,
                zorder=4,
            )
        )

    sr, sc = drawn[0]
    ax.plot(sc, sr, marker="o", markersize=10, color="#2a9d8f", zorder=6)
    er, ec = drawn[-1]
    end_color = "#d62828" if overlay.status != "success" else "#264653"
    ax.plot(ec, er, marker="X", markersize=10, color=end_color, zorder=6)

    if overlay.failure_push_index is not None:
        fail_pos = _path_index_for_push(path, overlay.push_indices, overlay.failure_push_index)
        if fail_pos is not None:
            visit_count: dict[tuple[int, int], int] = {}
            fail_drawn = None
            for row, col in path:
                visit_idx = visit_count.get((row, col), 0)
                visit_count[(row, col)] = visit_idx + 1
                if (row, col) == fail_pos:
                    dr, dc = VISIT_OFFSETS[visit_idx % len(VISIT_OFFSETS)]
                    fail_drawn = (row + dr, col + dc)
                    break
            if fail_drawn is not None:
                ax.plot(fail_drawn[1], fail_drawn[0], marker="*", markersize=16, color="#d62828", zorder=7)

    ax.set_title(f"{overlay.level_id}\n{overlay.subtitle}", fontsize=9, pad=6)


def _load_boards(episode_paths: list[Path]) -> list[BoardSnapshot]:
    return [
        parse_board(str(json.loads(path.read_text())["trajectory"][0]["state"]))
        for path in episode_paths
    ]


def render_original_figure(episode_paths: list[Path], output_path: Path) -> None:
    overlays = [legacy_overlay_from_episode(path) for path in episode_paths]
    boards = _load_boards(episode_paths)

    fig, axes = plt.subplots(1, len(episode_paths), figsize=(4.2 * len(episode_paths), 4.8))
    if len(episode_paths) == 1:
        axes = [axes]

    for ax, board, overlay in zip(axes, boards, overlays):
        draw_panel_original(ax, board, overlay)

    fig.suptitle("Player trajectories on Sokoban boards", fontsize=12, y=0.98)
    fig.text(
        0.5,
        0.02,
        "Teal = start · Dark end = solved · Red X/★ = failed run · Orange arrows = player push steps · Color = time",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def render_labeled_figure(episode_paths: list[Path], output_path: Path) -> None:
    overlays = [overlay_from_episode(path) for path in episode_paths]
    boards = _load_boards(episode_paths)

    fig, axes = plt.subplots(1, len(episode_paths), figsize=(4.2 * len(episode_paths), 4.8))
    if len(episode_paths) == 1:
        axes = [axes]

    for ax, board, overlay in zip(axes, boards, overlays):
        draw_panel_labeled(ax, board, overlay)

    fig.suptitle("Player trajectories on Sokoban boards (labeled pushes)", fontsize=12, y=0.98)
    fig.text(
        0.5,
        0.02,
        "Each color = one push (walk path offset slightly so overlaps separate) · "
        "Labelled arrow = that push on that box (e.g. 3:B1) · Teal = start · Red ★ = failed push",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def default_success_episode() -> Path:
    """Multi-box success (4 boxes, 7 pushes); solver-optimal is 7 pushes / 15 steps."""
    candidates = [
        Path("results")
        / "v3_eval_heuristic_same_level_k3_gpt52_low_16384_v3loop_baseline"
        / "20260601T051354Z_heuristic_same_level_iterative_boxoban_unfiltered_valid_000_127_seed93_success.json",
        Path("results")
        / "v2_baseline_heuristic_same_level"
        / "20260601T051330Z_heuristic_same_level_iterative_boxoban_unfiltered_valid_000_127_seed93_success.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No default success episode found; pass --episodes explicitly.")


def render_success_moderate_figure(episode_path: Path, output_path: Path) -> None:
    overlay = legacy_overlay_from_episode(episode_path)
    board = parse_board(str(json.loads(episode_path.read_text())["trajectory"][0]["state"]))

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.5))
    draw_panel_connected_path(ax, board, overlay)

    fig.suptitle("Success trajectory (multi-box level)", fontsize=12, y=0.98)
    fig.text(
        0.5,
        0.02,
        f"{overlay.level_id} · Connected path, no overlap on revisits · Plasma = time · Arrows = movement",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def render_success_separated_figure(episode_path: Path, output_path: Path) -> None:
    """Dense success boards: original style with lane offsets."""
    overlay = separated_original_from_episode(episode_path)
    board = parse_board(str(json.loads(episode_path.read_text())["trajectory"][0]["state"]))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    draw_panel_original_separated(ax, board, overlay)

    fig.suptitle("Success trajectory (separated overlapping paths)", fontsize=12, y=0.98)
    fig.text(
        0.5,
        0.02,
        "Same as original style (plasma = time, orange = push) · Paths offset per push so overlaps stay visible",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def default_episodes() -> list[Path]:
    root = Path("results")
    return [
        root
        / "v3_eval_heuristic_same_level_min2_baseline_gpt52_16384"
        / "20260601T003449Z_heuristic_same_level_iterative_boxoban_medium_valid_000_155_seed44_invalid_plan.json",
        root
        / "v3_eval_heuristic_same_level_min2_v1_specific_gpt52_16384"
        / "20260601T052727Z_heuristic_same_level_iterative_boxoban_medium_valid_000_192_seed46_deadlock.json",
        root
        / "smoke_v2_hybrid"
        / "20260531T233716Z_heuristic_same_level_iterative_boxoban_medium_valid_000_082_seed42_success.json",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=Path,
        help="Episode JSON paths (default: three curated examples)",
    )
    parser.add_argument(
        "--style",
        choices=("original", "labeled", "both", "success-moderate", "success-separated"),
        default="both",
        help="Which visualization to render (default: both)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (only with --style original or labeled; not used for both)",
    )
    parser.add_argument(
        "--output-original",
        type=Path,
        default=Path("docs/figures/trajectory_examples_original.png"),
    )
    parser.add_argument(
        "--output-labeled",
        type=Path,
        default=Path("docs/figures/trajectory_examples_labeled.png"),
    )
    parser.add_argument(
        "--output-success-moderate",
        type=Path,
        default=Path("docs/figures/trajectory_examples_success_moderate.png"),
    )
    parser.add_argument(
        "--output-success-separated",
        type=Path,
        default=Path("docs/figures/trajectory_examples_success_separated.png"),
    )
    args = parser.parse_args()

    if args.style == "success-moderate":
        episode = args.episodes[0] if args.episodes else default_success_episode()
        if not episode.exists():
            raise FileNotFoundError(episode)
        out = args.output or args.output_success_moderate
        render_success_moderate_figure(episode, out)
        print(f"Wrote {out}")
        return

    if args.style == "success-separated":
        episode = args.episodes[0] if args.episodes else (
            Path("results")
            / "smoke_v2_hybrid"
            / "20260531T233716Z_heuristic_same_level_iterative_boxoban_medium_valid_000_082_seed42_success.json"
        )
        if not episode.exists():
            raise FileNotFoundError(episode)
        out = args.output or args.output_success_separated
        render_success_separated_figure(episode, out)
        print(f"Wrote {out}")
        return

    episodes = args.episodes or default_episodes()
    for path in episodes:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.style in ("original", "both"):
        out = args.output if args.style == "original" and args.output else args.output_original
        render_original_figure(episodes, out)
        print(f"Wrote {out}")
    if args.style in ("labeled", "both"):
        out = args.output if args.style == "labeled" and args.output else args.output_labeled
        render_labeled_figure(episodes, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
