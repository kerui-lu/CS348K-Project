#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CELL = 36
PADDING = 12
TITLE_H = 52
FOOTER_H = 64
BG = (29, 34, 46)
FLOOR = (232, 232, 228)
WALL = (40, 46, 60)
TARGET = (244, 208, 109)
BOX = (191, 124, 75)
BOX_ON_TARGET = (230, 180, 76)
PLAYER = (57, 133, 224)
ALERT = (220, 80, 80)
ALERT_SOFT = (255, 210, 190)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Sokoban episode JSON files as GIFs.")
    p.add_argument("--results_dir", required=True, help="Directory containing episode .json files.")
    p.add_argument("--output_dir", required=True, help="Directory to write rendered gifs.")
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--allow_success", action="store_true")
    return p.parse_args()


def read_episode(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RenderFrame:
    board: str
    push_index: int | None
    semantic_phase: str | None
    action: str | None


@dataclass
class FailureContext:
    failure_push_index: int | None
    marker_frame_index: int | None
    failed_push_direction: str | None
    instruction_text: str


def board_frames(episode: dict[str, Any]) -> list[RenderFrame]:
    traj = episode.get("trajectory") or []
    frames: list[RenderFrame] = []
    if isinstance(traj, list):
        if traj:
            first = traj[0]
            if isinstance(first, dict):
                state = first.get("state")
                info = first.get("info", {}) if isinstance(first.get("info"), dict) else {}
                if isinstance(state, str) and state:
                    frames.append(
                        RenderFrame(
                            board=state,
                            push_index=info.get("push_index") if isinstance(info.get("push_index"), int) else None,
                            semantic_phase=info.get("semantic_phase") if isinstance(info.get("semantic_phase"), str) else None,
                            action=first.get("executed_action") if isinstance(first.get("executed_action"), str) else None,
                        )
                    )
        for step in traj:
            if not isinstance(step, dict):
                continue
            state = step.get("state")
            next_state = step.get("next_state")
            info = step.get("info", {}) if isinstance(step.get("info"), dict) else {}
            if isinstance(state, str) and state:
                frames.append(
                    RenderFrame(
                        board=state,
                        push_index=info.get("push_index") if isinstance(info.get("push_index"), int) else None,
                        semantic_phase=info.get("semantic_phase") if isinstance(info.get("semantic_phase"), str) else None,
                        action=step.get("executed_action") if isinstance(step.get("executed_action"), str) else None,
                    )
                )
            if isinstance(next_state, str) and next_state:
                frames.append(
                    RenderFrame(
                        board=next_state,
                        push_index=info.get("push_index") if isinstance(info.get("push_index"), int) else None,
                        semantic_phase=info.get("semantic_phase") if isinstance(info.get("semantic_phase"), str) else None,
                        action=None,
                    )
                )
    final_board = episode.get("metadata", {}).get("final_board")
    if isinstance(final_board, str) and final_board:
        frames.append(RenderFrame(board=final_board, push_index=None, semantic_phase=None, action=None))
    deduped: list[RenderFrame] = []
    for frame in frames:
        if not deduped or deduped[-1].board != frame.board:
            deduped.append(frame)
    return deduped


def subsample(frames: list[RenderFrame], max_frames: int) -> list[RenderFrame]:
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 2:
        return [frames[0], frames[-1]]
    step = (len(frames) - 1) / (max_frames - 1)
    idxs = [round(i * step) for i in range(max_frames)]
    return [frames[i] for i in idxs]


def failure_context(episode: dict[str, Any], frames: list[RenderFrame]) -> FailureContext:
    md = episode.get("metadata", {}) or {}
    reason = md.get("failure_reason") if isinstance(md.get("failure_reason"), str) else None
    idx = md.get("failure_push_index") if isinstance(md.get("failure_push_index"), int) else None
    failed_push_direction: str | None = None
    marker_frame_index: int | None = None
    if idx is not None:
        for i, frame in enumerate(frames):
            if frame.push_index == idx:
                marker_frame_index = i
                if frame.action:
                    break
        if marker_frame_index is None and frames:
            marker_frame_index = len(frames) - 1
    pushes = md.get("planned_pushes")
    if isinstance(idx, int) and isinstance(pushes, list) and 0 <= idx < len(pushes):
        intent = pushes[idx]
        if isinstance(intent, dict) and isinstance(intent.get("push"), str):
            failed_push_direction = intent["push"]
        return FailureContext(
            failure_push_index=idx,
            marker_frame_index=marker_frame_index,
            failed_push_direction=failed_push_direction,
            instruction_text=f"Failed push#{idx}: {intent} ({reason})",
        )
    deadlock_reason = md.get("deadlock_reason")
    if isinstance(deadlock_reason, str) and deadlock_reason:
        marker_frame_index = len(frames) - 1 if frames else None
        return FailureContext(
            failure_push_index=idx,
            marker_frame_index=marker_frame_index,
            failed_push_direction=failed_push_direction,
            instruction_text=f"Terminal deadlock reason: {deadlock_reason}",
        )
    traj = episode.get("trajectory") or []
    if isinstance(traj, list) and traj:
        info = (traj[-1] or {}).get("info", {})
        if isinstance(info, dict):
            step_deadlock_reason = info.get("deadlock_reason")
            if isinstance(step_deadlock_reason, str) and step_deadlock_reason:
                marker_frame_index = len(frames) - 1 if frames else None
                return FailureContext(
                    failure_push_index=idx,
                    marker_frame_index=marker_frame_index,
                    failed_push_direction=failed_push_direction,
                    instruction_text=f"Terminal deadlock reason: {step_deadlock_reason}",
                )
    if reason:
        marker_frame_index = len(frames) - 1 if frames else None
        return FailureContext(
            failure_push_index=idx,
            marker_frame_index=marker_frame_index,
            failed_push_direction=failed_push_direction,
            instruction_text=f"Failure: {reason}",
        )
    return FailureContext(
        failure_push_index=idx,
        marker_frame_index=marker_frame_index,
        failed_push_direction=failed_push_direction,
        instruction_text="",
    )


def draw_board(
    frame: RenderFrame,
    frame_idx: int,
    frame_count: int,
    title: str,
    subtitle: str,
    arrow_direction: str | None,
    is_failure_marker: bool,
) -> Image.Image:
    board_text = frame.board
    rows = board_text.splitlines()
    h = len(rows)
    w = max(len(r) for r in rows)
    image = Image.new(
        "RGB",
        (PADDING * 2 + w * CELL, TITLE_H + FOOTER_H + PADDING * 2 + h * CELL),
        BG,
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((PADDING, 8), title, fill=(235, 240, 245), font=font)
    draw.text((PADDING, 26), f"frame {frame_idx + 1}/{frame_count}", fill=(182, 196, 210), font=font)
    origin_x = PADDING
    origin_y = TITLE_H + PADDING
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            x0 = origin_x + c * CELL
            y0 = origin_y + r * CELL
            x1 = x0 + CELL
            y1 = y0 + CELL
            color = FLOOR
            if ch == "#":
                color = WALL
            elif ch == ".":
                color = TARGET
            draw.rectangle((x0, y0, x1, y1), fill=color)
            if ch in {"$", "*"}:
                box_color = BOX_ON_TARGET if ch == "*" else BOX
                draw.rectangle((x0 + 8, y0 + 8, x1 - 8, y1 - 8), fill=box_color, outline=(80, 60, 40))
            if ch in {"@", "+"}:
                draw.ellipse((x0 + 8, y0 + 8, x1 - 8, y1 - 8), fill=PLAYER, outline=(30, 72, 128))
                if arrow_direction:
                    draw_direction_arrow(draw, x0, y0, arrow_direction)
    if is_failure_marker:
        draw.rectangle(
            (
                origin_x - 2,
                origin_y - 2,
                origin_x + w * CELL + 2,
                origin_y + h * CELL + 2,
            ),
            outline=ALERT,
            width=4,
        )
    footer_color = ALERT_SOFT if is_failure_marker else (245, 185, 120)
    draw.text((PADDING, TITLE_H + PADDING + h * CELL + 12), subtitle, fill=footer_color, font=font)
    return image


def draw_direction_arrow(draw: ImageDraw.ImageDraw, x0: int, y0: int, direction: str) -> None:
    deltas = {
        "Up": (0, -1),
        "Down": (0, 1),
        "Left": (-1, 0),
        "Right": (1, 0),
    }
    if direction not in deltas:
        return
    dc, dr = deltas[direction]
    cx = x0 + CELL // 2
    cy = y0 + CELL // 2
    end_x = cx + dc * 22
    end_y = cy + dr * 22
    draw.line((cx, cy, end_x, end_y), fill=ALERT, width=4)
    if direction in {"Up", "Down"}:
        head_y = end_y + (6 if direction == "Up" else -6)
        draw.polygon([(end_x, end_y), (end_x - 6, head_y), (end_x + 6, head_y)], fill=ALERT)
    else:
        head_x = end_x + (6 if direction == "Left" else -6)
        draw.polygon([(end_x, end_y), (head_x, end_y - 6), (head_x, end_y + 6)], fill=ALERT)


def render_episode(path: Path, output_dir: Path, max_frames: int, allow_success: bool) -> Path | None:
    episode = read_episode(path)
    status = str(episode.get("status"))
    if status == "success" and not allow_success:
        return None
    frames = board_frames(episode)
    if not frames:
        return None
    frames = subsample(frames, max_frames=max_frames)
    failure = failure_context(episode, frames)
    title = (
        f"{episode.get('agent_type')} | {episode.get('level_id')} | "
        f"status={status} | seed={episode.get('seed')}"
    )
    subtitle = failure.instruction_text
    rendered: list[Image.Image] = []
    for idx, frame in enumerate(frames):
        is_failure_marker = failure.marker_frame_index == idx
        arrow_direction = failure.failed_push_direction if is_failure_marker and failure.failed_push_direction else frame.action
        rendered.append(
            draw_board(
                frame,
                frame_idx=idx,
                frame_count=len(frames),
                title=title,
                subtitle=subtitle,
                arrow_direction=arrow_direction,
                is_failure_marker=is_failure_marker,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{path.stem}.gif"
    rendered[0].save(
        out_path,
        save_all=True,
        append_images=rendered[1:],
        duration=180,
        loop=0,
    )
    return out_path


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    count = 0
    for path in sorted(results_dir.glob("*.json")):
        if path.name in {"summary.json", "evaluation_summary.json"}:
            continue
        out = render_episode(path, output_dir, max_frames=args.max_frames, allow_success=args.allow_success)
        if out is not None:
            count += 1
    print(f"Rendered {count} gifs to {output_dir}")


if __name__ == "__main__":
    main()
