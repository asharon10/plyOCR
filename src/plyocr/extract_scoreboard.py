#!/usr/bin/env python3
"""Run the Tesseract-based scoreboard OCR pipeline on a game video.

Delegates to plyocr.run_tesseract_ocr which:
  1. Samples one frame/sec (configurable) from the video (ffmpeg).
  2. Crops each ROI region (from game-local roi.json or the default).
  3. Binarizes and runs Tesseract per crop.
  4. Validates + cleans readings (clock format, quarter range, score range).
  5. Writes ocr_timeline.csv, timeline.json, timeline_raw.json.

Progress is reported as `PROGRESS <done>/<total>` (frames processed / total),
which the API job runner parses into a UI percentage.

Usage (via API pipeline trigger):
    python -m plyocr.extract_scoreboard --game data/games/<id>/game.json
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys



def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scoreboard OCR timeline from game video")
    parser.add_argument("--game", required=True, help="Path to game.json")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second (default: 0.5 = every 2s)")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0 = all CPU cores)")
    parser.add_argument("--decoder", default="seek", choices=["seek", "keyframe"],
                        help="Frame reader (keyframe = stream I-frames, much faster when the "
                             "sample interval ≈ the GOP; auto-falls back to seek if denser)")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m", "plyocr.run_tesseract_ocr",
        "--game", args.game,
        "--fps", str(args.fps),
        "--workers", str(args.workers),
        "--decoder", args.decoder,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    progress_re = re.compile(r"PROGRESS\s+(\d+)/(\d+)")

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        # Forward PROGRESS lines so the API job runner updates the UI.
        if progress_re.search(line):
            print(line, flush=True)
        else:
            print(line, flush=True)

    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"run_tesseract_ocr.py failed with exit code {proc.returncode}")


if __name__ == "__main__":
    main()
