#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Sokoban episode JSON files as GIFs.")
    p.add_argument("--results_dir", required=True, help="Directory containing episode .json files.")
    p.add_argument("--output_dir", required=True, help="Directory to write rendered gifs.")
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--allow_success", action="store_true")
    return p.parse_args()


def read_episode(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def board_frames(episode: dict[str, Any]) -> list[str]:
    traj = episode.get("trajectory") or []
    boards: list[str] = []
    if isinstance(traj, list):
        for step in traj:
            if not isinstance(step, dict):
                continue
            state = step.get("state")
            next_state = step.get("next_state")
            if isinstance(state, str) and state:
                boards.append(state)
            if isinstance(next_state, str) and next_state:
                boards.append(next_state)
    final_board = episode.get("metadata", {}).get("final_board")
    if isinstance(final_board, str) and final_board:
        boards.append(final_board)
    deduped: list[str] = []
    for board in boards:
        if not deduped or deduped[-1] != board:
            deduped.append(board)
    return deduped


def subsample(frames: list[str], max_frames: int) -> list[str]:
    if len(frames) <= max_frames:
        return frames
    if max_frames <= 2:
        return [frames[0], frames[-1]]
    step = (len(frames) - 1) / (max_frames - 1)
    idxs = [round(i * step) for i in range(max_frames)]
    return [frames[i] for i in idxs]


def failure_instruction_text(episode: dict[str, Any]) -> str:
    md = episode.get("metadata", {}) or {}
    reason = md.get("failure_reason")
    idx = md.get("failure_push_index")
    pushes = md.get("planned_pushes")
    if isinstance(idx, int) and isinstance(pushes, list) and 0 <= idx < len(pushes):
        intent = pushes[idx]
        return f"Last failed instruction: push#{idx} {intent} ({reason})"
    deadlock_reason = md.get("deadlock_reason")
    if isinstance(deadlock_reason, str) and deadlock_reason:
        return f"Terminal deadlock reason: {deadlock_reason}"
    traj = episode.get("trajectory") or []
    if isinstance(traj, list) and traj:
        info = (traj[-1] or {}).get("info", {})
        if isinstance(info, dict):
            step_deadlock_reason = info.get("deadlock_reason")
            if isinstance(step_deadlock_reason, str) and step_deadlock_reason:
                return f"Terminal deadlock reason: {step_deadlock_reason}"
    if reason:
        return f"Failure: {reason}"
    return ""


def draw_board(
    board_text: str,
    title: str,
    subtitle: str,
    step_text: str,
) -> Image.Image:
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
    draw.text((PADDING, 26), step_text, fill=(182, 196, 210), font=font)
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
    draw.text((PADDING, TITLE_H + PADDING + h * CELL + 12), subtitle, fill=(245, 185, 120), font=font)
    return image


def render_episode(path: Path, output_dir: Path, max_frames: int, allow_success: bool) -> Path | None:
    episode = read_episode(path)
    status = str(episode.get("status"))
    if status == "success" and not allow_success:
        return None
    frames = board_frames(episode)
    if not frames:
        return None
    frames = subsample(frames, max_frames=max_frames)
    title = (
        f"{episode.get('agent_type')} | {episode.get('level_id')} | "
        f"status={status} | seed={episode.get('seed')}"
    )
    subtitle = failure_instruction_text(episode)
    rendered: list[Image.Image] = []
    for idx, board in enumerate(frames):
        rendered.append(draw_board(board, title=title, subtitle=subtitle, step_text=f"frame {idx+1}/{len(frames)}"))
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
