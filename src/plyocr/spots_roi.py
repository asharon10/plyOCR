#!/usr/bin/env python3
"""Extract a sample frame from a game video and draw ROI bounding boxes on it.

Produces a single PNG contact sheet that shows:
  - The full frame (resized to roi config resolution) with color-coded ROI boxes.
  - Below: each ROI crop enlarged 5x alongside its label.

This is a quick visual sanity-check that the ROIs line up with the scoreboard
overlay before committing to a full OCR run.

Usage
-----
    python -m plyocr.spots_roi \\
        --game data/games/<id>/game.json \\
        --out  data/games/<id>/derived/spots_roi.png

    # Or pick a specific timestamp (seconds):
    python -m plyocr.spots_roi \\
        --game data/games/<id>/game.json \\
        --at 600 \\
        --out data/games/<id>/derived/spots_roi.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from plyocr.paths import DEFAULT_ROI

ROI_COLORS = {
    "clock":      (255, 64,  64),
    "quarter":    (255, 220, 40),
    "home_score": (64,  200, 255),
    "away_score": (128, 255, 128),
}


def ffprobe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def extract_frame(video: Path, t: float, dest: Path) -> bool:
    pre = max(0.0, t - 5.0)
    off = t - pre
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{pre:.3f}", "-i", str(video),
        "-ss", f"{off:.3f}",
        "-frames:v", "1", "-q:v", "2", "-y", str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest.exists() and dest.stat().st_size > 0


def locate_video(game_path: Path) -> Path | None:
    game_dir = game_path.parent
    if game_path.exists():
        try:
            game = json.loads(game_path.read_text())
            local_file = game.get("video", {}).get("localFile")
            if local_file:
                candidate = game_dir / local_file
                if candidate.exists():
                    return candidate
        except (json.JSONDecodeError, OSError):
            pass
    fallback = game_dir / "raw" / "game.mp4"
    return fallback if fallback.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, type=Path, help="Path to game.json")
    ap.add_argument("--out", default=None, type=Path,
                    help="Output PNG path (default: <game_dir>/derived/spots_roi.png)")
    ap.add_argument("--roi", default=None, type=Path,
                    help="ROI config (default: game_dir/roi.json, then the plyocr default ROI)")
    ap.add_argument("--at", type=float, default=None,
                    help="Timestamp in seconds (default: 30%% of duration, typically mid-Q1)")
    args = ap.parse_args()

    game_path = Path(args.game).resolve()
    video = locate_video(game_path)
    if video is None:
        sys.exit(f"Video not found for {game_path}")

    # ROI resolution: prefer game-local, then arg, then default
    game_dir = game_path.parent
    roi_path = args.roi
    if roi_path is None:
        candidate = game_dir / "roi.json"
        roi_path = candidate if candidate.exists() else DEFAULT_ROI
    roi_cfg = json.loads(roi_path.read_text())
    fw = roi_cfg["frame_width"]
    fh = roi_cfg["frame_height"]
    rois = roi_cfg["rois"]

    dur = ffprobe_duration(video)
    t = args.at if args.at is not None else dur * 0.30
    t = max(0.0, min(t, dur - 1))

    out_path = args.out or (game_path.parent / "derived" / "spots_roi.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Extracting frame at t={t:.1f}s from {video.name} ({dur:.0f}s total)")

    with tempfile.TemporaryDirectory(prefix="spots_") as td:
        frame_file = Path(td) / "frame.png"
        if not extract_frame(video, t, frame_file):
            sys.exit(f"Failed to extract frame at t={t:.1f}s")

        frame = Image.open(frame_file).convert("RGB")
        if frame.size != (fw, fh):
            frame = frame.resize((fw, fh), resample=Image.BILINEAR)

    # --- draw ROI boxes on full frame ---
    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    for name, spec in rois.items():
        x, y, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
        color = ROI_COLORS.get(name, (255, 255, 255))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x, y - 16), name, fill=color, font=font)

    # --- crop each ROI enlarged 5x ---
    SCALE = 5
    crops: list[tuple[str, Image.Image]] = []
    for name, spec in rois.items():
        x, y, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
        crop = frame.crop((x, y, x + w, y + h))
        crop_big = crop.resize((w * SCALE, h * SCALE), resample=Image.NEAREST)
        crops.append((name, crop_big))

    # --- compose contact sheet ---
    pad = 16
    crop_row_h = max(c.size[1] for _, c in crops) + 40
    total_crop_w = sum(c.size[0] for _, c in crops) + pad * (len(crops) + 1)
    sheet_w = max(fw, total_crop_w)
    sheet_h = fh + pad + crop_row_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    sheet.paste(annotated, (0, 0))

    # label row
    sdraw = ImageDraw.Draw(sheet)
    x_off = pad
    for name, crop_img in crops:
        color = ROI_COLORS.get(name, (255, 255, 255))
        sdraw.text((x_off, fh + 4), name, fill=color, font=font)
        sheet.paste(crop_img, (x_off, fh + 24))
        x_off += crop_img.size[0] + pad

    # timestamp label
    mm, ss = divmod(int(t), 60)
    sdraw.text((fw - 160, fh + 4), f"t = {mm}:{ss:02d} ({t:.0f}s)", fill=(200, 200, 200), font=font)

    sheet.save(str(out_path))
    print(f"Wrote {out_path} ({sheet.size[0]}x{sheet.size[1]})")


if __name__ == "__main__":
    main()
