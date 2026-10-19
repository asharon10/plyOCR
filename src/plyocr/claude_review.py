#!/usr/bin/env python3
"""Claude-vision scoreboard reviewer.

Samples a few mid-game frames from a game's video, shows them to Claude (vision),
and asks it to (a) propose the scoreboard ROI bounding boxes and (b) recommend an
OCR method/configuration — with special attention to how the QUARTER and CLOCK
fonts read. Pure report tool: prints a JSON recommendation to stdout and never
writes roi.json or game.json. Mirrors the roi.json schema so the output can be
hand-applied or wired into the pipeline later.

Usage:
    python -m plyocr.claude_review --game data/games/<id>/game.json
    python -m plyocr.claude_review --game <...> --frames 4 --annotate /tmp/review.png

Requires ANTHROPIC_API_KEY in the environment (already present in services/api/.env,
inherited by the python3 the API spawns).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_DEFAULT = "claude-opus-4-8"
OCR_METHODS = ["sevenseg", "tesseract", "easyocr"]
FIELDS = ["clock", "quarter", "home_score", "away_score"]


# --------------------------------------------------------------------------
# Video / frame helpers (mirrors plyocr.run_tesseract_ocr conventions)
# --------------------------------------------------------------------------
def locate_video(game_path: Path) -> Path | None:
    game_dir = game_path.parent
    if game_path.exists():
        try:
            g = json.loads(game_path.read_text())
            lf = (g.get("video") or {}).get("localFile")
            if lf and (game_dir / lf).exists():
                return game_dir / lf
        except Exception:
            pass
    fallback = game_dir / "raw" / "game.mp4"
    return fallback if fallback.exists() else None


def get_duration_ffprobe(video: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            text=True,
        )
        return float(out.strip())
    except Exception:
        return 0.0


def sample_frames(video: Path, n: int) -> list[np.ndarray]:
    """Grab n frames spread across mid-game (20–80% of duration)."""
    duration = get_duration_ffprobe(video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}")
    frames: list[np.ndarray] = []
    try:
        for i in range(n):
            frac = 0.20 + 0.60 * (i / max(1, n - 1))
            t = (duration * frac) if duration > 0 else (i * 30.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def encode_png_b64(frame: np.ndarray, max_edge: int = 0) -> str:
    """BGR ndarray → base64 PNG. Optionally downscale so the long edge ≤ max_edge
    (Claude vision tops out at 2576px on the long edge; 0 disables resizing)."""
    if max_edge and max(frame.shape[:2]) > max_edge:
        scale = max_edge / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


# --------------------------------------------------------------------------
# Claude structured-output schema (mirrors roi.json + method recommendation)
# --------------------------------------------------------------------------
def _roi_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "integer", "description": "left edge, pixels"},
            "y": {"type": "integer", "description": "top edge, pixels"},
            "w": {"type": "integer", "description": "width, pixels"},
            "h": {"type": "integer", "description": "height, pixels"},
            "color": {"type": "string", "description": "digit color, e.g. red/yellow/white"},
            "format": {"type": "string", "description": "value format, e.g. 'M:SS', '1-4 or OT'"},
        },
        "required": ["x", "y", "w", "h", "color", "format"],
    }


def _field_assessment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "legibility": {
                "type": "string",
                "enum": ["clear", "stylised_font", "low_contrast", "small", "obscured"],
            },
            "is_seven_segment": {
                "type": "boolean",
                "description": "true if rendered as a classic 7-segment LED digit",
            },
            "notes": {"type": "string"},
        },
        "required": ["legibility", "is_seven_segment", "notes"],
    }


def output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "frame_width": {"type": "integer"},
            "frame_height": {"type": "integer"},
            "rois": {
                "type": "object",
                "additionalProperties": False,
                "properties": {f: _roi_schema() for f in FIELDS},
                "required": FIELDS,
            },
            "recommended_ocr_method": {"type": "string", "enum": OCR_METHODS},
            "recommended_quarter_method": {
                "type": "string",
                "enum": OCR_METHODS + ["same"],
                "description": "'same' ⇒ quarter uses recommended_ocr_method",
            },
            "field_assessment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "quarter": _field_assessment_schema(),
                    "clock": _field_assessment_schema(),
                },
                "required": ["quarter", "clock"],
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "reasoning": {"type": "string"},
        },
        "required": [
            "frame_width", "frame_height", "rois", "recommended_ocr_method",
            "recommended_quarter_method", "field_assessment", "confidence", "reasoning",
        ],
    }


SYSTEM_PROMPT = """\
You are an expert at reading sports broadcast scoreboards for an OCR pipeline on \
basketball game video (IBBA / Winner League broadcasts).

Every frame you are given is the SAME video at resolution {w}x{h} pixels. The \
game-clock overlay (a graphic strip, usually along the lower third) shows four \
fields you must locate precisely, in pixel coordinates (origin top-left, x → right, \
y → down):
  - clock       : the game clock (red digits; "M:SS"/"MM:SS" when ≥1:00, or "SS.t" under a minute)
  - quarter     : the period indicator (yellow; "1"/"2"/"3"/"4" or an ordinal like "1st", or "OT")
  - home_score  : the home team's score (red digits, 0–3 digits)
  - away_score  : the away team's score (red digits, 0–3 digits)

IMPORTANT:
- Target the BROADCAST GRAPHIC OVERLAY, not the arena's physical scoreboard in the \
crowd (that one is camera-angle dependent — ignore it).
- Give each box TIGHT around just that field's glyphs (a few px of padding is fine).
- Use the frames to judge the QUARTER and CLOCK fonts specifically — these drive the \
OCR method choice.

OCR method guidance:
  - sevenseg  : a geometric 7-segment LED decoder. Best for classic blocky LED digits. Fast.
  - tesseract : general printed/graphic text. Good for crisp printed fonts. Fast.
  - easyocr   : neural; reads stylised/script/low-contrast fonts the above miss. Slow.
Prefer a fast method (sevenseg/tesseract) when the digits are clean; only choose a \
neural method when a field's font genuinely needs it. If the clock+scores read fine \
on a fast method but the QUARTER uses a stylised font, set recommended_ocr_method to \
the fast method and recommended_quarter_method to a neural one (per-field override); \
otherwise set recommended_quarter_method to "same".

Return pixel coordinates in the {w}x{h} space. Be precise and conservative; if you \
cannot see a field clearly, still give your best box and lower the confidence.\
"""


def build_messages(frames_b64: list[str], prior_roi: dict[str, Any] | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for i, b64 in enumerate(frames_b64):
        content.append({"type": "text", "text": f"Frame {i + 1} of {len(frames_b64)}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })
    ask = ("Locate the four scoreboard fields and recommend the OCR method/config. "
           "Pay special attention to the quarter and clock fonts.")
    if prior_roi and prior_roi.get("rois"):
        ask += ("\n\nFor reference, the pipeline currently uses these ROI boxes "
                "(verify and correct them if wrong):\n"
                + json.dumps(prior_roi.get("rois"), indent=2))
    content.append({"type": "text", "text": ask})
    return [{"role": "user", "content": content}]


def main() -> None:
    ap = argparse.ArgumentParser(description="Claude-vision scoreboard ROI + OCR-method reviewer")
    ap.add_argument("--game", required=True, type=Path, help="path to game.json")
    ap.add_argument("--frames", type=int, default=3, help="number of frames to sample (default 3)")
    ap.add_argument("--model", default=MODEL_DEFAULT, help=f"Claude model (default {MODEL_DEFAULT})")
    ap.add_argument("--max-edge", type=int, default=0,
                    help="downscale so long edge ≤ this many px (0 = native; Claude caps at 2576)")
    ap.add_argument("--annotate", type=Path, default=None,
                    help="optional: write a PNG of frame 1 with the proposed boxes drawn, for visual review")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set in the environment.")

    video = locate_video(args.game)
    if video is None:
        print(json.dumps({"error": "video_not_found"}))
        sys.exit(1)

    frames = sample_frames(video, max(1, args.frames))
    if not frames:
        print(json.dumps({"error": "no_frames_decoded"}))
        sys.exit(1)
    fh, fw = frames[0].shape[:2]
    frames_b64 = [encode_png_b64(f, args.max_edge) for f in frames]

    game_dir = args.game.parent
    prior_roi = None
    roi_path = game_dir / "roi.json"
    if roi_path.exists():
        try:
            prior_roi = json.loads(roi_path.read_text())
        except Exception:
            prior_roi = None

    import anthropic  # imported here so --help works without the dep

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=args.model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT.format(w=fw, h=fh),
        messages=build_messages(frames_b64, prior_roi),
        output_config={"format": {"type": "json_schema", "schema": output_schema()}},
    )

    if resp.stop_reason == "refusal":
        print(json.dumps({"error": "refusal", "stop_details": str(getattr(resp, "stop_details", None))}))
        sys.exit(1)

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        print(json.dumps({"error": "no_text_block"}))
        sys.exit(1)
    result = json.loads(text)
    # Stamp what was actually measured (don't trust the model for frame size).
    result["frame_width"], result["frame_height"] = fw, fh
    result["sampled_frames"] = len(frames)
    result["model"] = resp.model

    if args.annotate:
        annotate(frames[0], result.get("rois", {}), args.annotate)
        result["annotated"] = str(args.annotate)

    print(json.dumps(result, indent=2))


_BOX_COLORS = {  # BGR
    "clock": (0, 0, 255), "quarter": (0, 215, 255),
    "home_score": (0, 200, 0), "away_score": (255, 120, 0),
}


def annotate(frame: np.ndarray, rois: dict[str, Any], out: Path) -> None:
    img = frame.copy()
    for field, roi in rois.items():
        try:
            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        except Exception:
            continue
        col = _BOX_COLORS.get(field, (255, 255, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        cv2.putText(img, field, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


if __name__ == "__main__":
    main()
