#!/usr/bin/env python3
"""Build a 'change map' image: sample N frames across the video, stack them, and
highlight pixels that CHANGE between frames. The scoreboard numbers (which tick
over time) glow; static graphics/labels/logos and the court stay cool — so the
operator can see exactly where the digit fields are and draw the ROI box.

Usage:
    python -m plyocr.change_map --video <game.mp4> --out change_map.png
    python -m plyocr.change_map --video <game.mp4> --box x,y,w,h --out crop.png
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from plyocr.hls_frame import grab_preview_frame  # noqa: E402

from plyocr.run_tesseract_ocr import read_frame_at, open_capture, get_duration_ffprobe  # noqa: E402


def _grab_remote(url: str, t: float) -> np.ndarray | None:
    """One BGR frame from an HLS URL for analysis: the 720p rendition upscaled to
    full HD when Pixellot has one (5x less to download), else the HD segment."""
    return grab_preview_frame(url, t)


def build_change_map(video: str, n: int, box: tuple[int, int, int, int] | None,
                     is_remote: bool = False) -> np.ndarray | None:
    dur = get_duration_ffprobe(video)
    times = [dur * f for f in np.linspace(0.06, 0.94, n)]
    if is_remote:
        with ThreadPoolExecutor(max_workers=8) as ex:
            frames = [fr for fr in ex.map(lambda t: _grab_remote(video, t), times) if fr is not None]
    else:
        cap = open_capture(Path(video))
        frames = [fr for t in times if (fr := read_frame_at(cap, t)) is not None]
        cap.release()
    if len(frames) < 2:
        return None
    if box is not None:
        x, y, w, h = box
        fh, fw = frames[0].shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        frames = [f[y0:y1, x0:x1] for f in frames]

    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in frames])
    change = (grays.max(axis=0) - grays.min(axis=0)).astype(np.uint8)   # per-pixel range
    base = np.median(np.stack(frames), axis=0).astype(np.uint8)         # stable backdrop
    heat = cv2.applyColorMap(change, cv2.COLORMAP_JET)
    return cv2.addWeighted(base, 0.5, heat, 0.6, 0)


def _upscale(img: np.ndarray, min_width: int) -> np.ndarray:
    """Enlarge (nearest-neighbour, integer factor) so the crop is easy to read."""
    if min_width <= 0 or img.shape[1] >= min_width:
        return img
    f = int(np.ceil(min_width / img.shape[1]))
    return cv2.resize(img, (img.shape[1] * f, img.shape[0] * f), interpolation=cv2.INTER_NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=str, help="game video path, or an HLS URL (cloud)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10, help="frames to sample (default 10)")
    ap.add_argument("--box", type=str, default=None, help="crop 'x,y,w,h' (px); omit for full frame")
    ap.add_argument("--min-width", type=int, default=0, help="upscale the result to at least this width")
    args = ap.parse_args()
    is_remote = args.video.startswith("http://") or args.video.startswith("https://")
    if not is_remote and not Path(args.video).exists():
        sys.exit(f"Video not found: {args.video}")
    box = None
    if args.box:
        bx, by, bw, bh = (int(v) for v in args.box.split(","))
        box = (bx, by, bw, bh)
    img = build_change_map(args.video, args.n, box, is_remote=is_remote)
    if img is None:
        sys.exit("Could not read enough frames")
    img = _upscale(img, args.min_width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(str(args.out))


if __name__ == "__main__":
    main()
