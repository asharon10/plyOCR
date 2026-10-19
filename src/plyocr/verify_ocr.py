#!/usr/bin/env python3
"""Tesseract OCR spot-check — writes a self-contained HTML contact sheet.

For each sampled frame the report shows:
  - Full frame thumbnail with ROI boxes drawn
  - Each ROI crop (enlarged) + what Tesseract read
  - Green/red indicator for parse success/failure

Usage
-----
    python -m plyocr.verify_ocr \\
        --game data/games/<id>/game.json

    # Custom timestamps (comma-separated seconds):
    python -m plyocr.verify_ocr \\
        --game data/games/<id>/game.json --at 600,1200,1800

    # More spots:
    python -m plyocr.verify_ocr \\
        --game data/games/<id>/game.json --n 20
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

from plyocr import sevenseg

from plyocr.paths import DEFAULT_ROI

# ── ROI colours for the overlay (BGR) ─────────────────────────────────────
FIELD_COLORS = {
    "clock":      (0, 80, 255),   # red
    "quarter":    (0, 220, 255),  # yellow
    "home_score": (0, 200, 0),    # green
    "away_score": (255, 140, 0),  # blue
}

# ── Tesseract configs ──────────────────────────────────────────────────────
_DIGITS = "0123456789"
_QTR_W  = "1234OTstndrdth"
TESS_CONFIG = {
    "clock":      "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:.",
    "quarter":    f"--psm 8 --oem 3 -c tessedit_char_whitelist={_QTR_W}",
    "home_score": f"--psm 8 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
    "away_score": f"--psm 8 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
}
TESS_FALLBACK = {
    "clock":      "--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789:.",
    "quarter":    f"--psm 13 --oem 3 -c tessedit_char_whitelist={_QTR_W}",
    "home_score": f"--psm 13 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
    "away_score": f"--psm 13 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
}

CLOCK_RE  = re.compile(r"^\d{1,2}:\d{2}$|^\d{1,2}\.\d$")
SCORE_RE  = re.compile(r"^\d{1,3}$")
QUARTER_MAP = {"1": 1, "1ST": 1, "2": 2, "2ND": 2, "3": 3, "3RD": 3,
               "4": 4, "4TH": 4, "OT": 5, "0T": 5}


# ── helpers ────────────────────────────────────────────────────────────────

def fix_common(s: str) -> str:
    return s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")

def parse_clock(r: str) -> bool:
    r2 = fix_common(r)
    if not CLOCK_RE.match(r2):
        return False
    m = re.match(r"^(\d{1,2}):", r2)
    return not (m and int(m.group(1)) > 12)

def parse_quarter(r: str) -> bool:
    s = re.sub(r"\s+", "", r).lower()
    if s.endswith(("st", "nd", "rd", "th")):
        return True
    if "ot" in s or "0t" in s:
        return True
    return bool(re.match(r"^\d", s))

def parse_score(r: str) -> bool:
    r2 = fix_common(r)
    if not SCORE_RE.match(r2):
        return False
    return int(r2) <= 200

VALIDATORS = {
    "clock": parse_clock, "quarter": parse_quarter,
    "home_score": parse_score, "away_score": parse_score,
}


def _remove_spurious_components(binary: np.ndarray) -> np.ndarray:
    text_mask = (binary == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    if n <= 1:
        return binary
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(heights) == 0:
        return binary
    max_h = int(heights.max())
    cleaned = np.full_like(binary, 255)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] >= 0.45 * max_h and stats[i, cv2.CC_STAT_AREA] >= 0.04 * areas.max():
            cleaned[labels == i] = 0
    return cleaned


def preprocess_crop(crop: np.ndarray, trim_components: bool = False, grayscale: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    target_h = 100
    if h < target_h:
        scale = int(target_h / h) + 1
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    invert = float(border.mean()) < 128
    if grayscale:
        g = cv2.bitwise_not(gray) if invert else gray
        _, _b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = np.where(_b == 0)
        if ink[0].size > 0:
            y0, y1 = ink[0].min(), ink[0].max()
            x0, x1 = ink[1].min(), ink[1].max()
            g = g[y0:y1 + 1, x0:x1 + 1]
        return cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        binary = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)
    if trim_components:
        binary = _remove_spurious_components(binary)
    ink = np.where(binary == 0)
    if ink[0].size > 0:
        y0, y1 = ink[0].min(), ink[0].max()
        x0, x1 = ink[1].min(), ink[1].max()
        binary = binary[y0:y1 + 1, x0:x1 + 1]
    return cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def locate_video(game_path: Path) -> Path | None:
    if game_path.exists():
        try:
            g = json.loads(game_path.read_text())
            lf = g.get("video", {}).get("localFile")
            if lf:
                cand = game_path.parent / lf
                if cand.exists():
                    return cand
        except Exception:
            pass
    fb = game_path.parent / "raw" / "game.mp4"
    return fb if fb.exists() else None


def load_roi(game_dir: Path, roi_arg: Path | None) -> dict[str, Any]:
    if roi_arg and roi_arg.exists():
        return json.loads(roi_arg.read_text())
    local = game_dir / "roi.json"
    path = local if local.exists() else DEFAULT_ROI
    return json.loads(path.read_text())


def get_duration(video: Path) -> float:
    import subprocess
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)], text=True
    )
    return float(out.strip())


def seek_frame(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    return f if ok else None


# ── per-frame OCR ──────────────────────────────────────────────────────────

def ocr_field_with_fallback(proc: np.ndarray, field: str) -> str:
    raw = re.sub(r"\s+", "", pytesseract.image_to_string(proc, config=TESS_CONFIG.get(field, "--psm 7 --oem 3")).strip())
    if not raw and field in TESS_FALLBACK:
        raw = re.sub(r"\s+", "", pytesseract.image_to_string(proc, config=TESS_FALLBACK[field]).strip())
    return raw


def ocr_frame(frame: np.ndarray, rois: dict[str, Any],
              src_w: int, src_h: int,
              method: str = "tesseract", grayscale: bool = False) -> list[dict[str, Any]]:
    fh, fw = frame.shape[:2]
    sx, sy = fw / src_w, fh / src_h
    results = []
    for field, roi in rois.items():
        x  = max(0, int(roi["x"] * sx));  y  = max(0, int(roi["y"] * sy))
        x2 = min(fw, x + int(roi["w"] * sx));  y2 = min(fh, y + int(roi["h"] * sy))
        if x2 <= x or y2 <= y:
            results.append({"field": field, "raw": "", "ok": False, "crop_b64": "", "proc_b64": ""})
            continue
        crop = frame[y:y2, x:x2]
        if method == "sevenseg":
            # Show the lit-segment mask used by the decoder as the "processed" view.
            proc = sevenseg._lit_mask(crop) * 255
            raw  = sevenseg.decode_crop(crop, is_clock=(field == "clock"))
        else:
            proc = preprocess_crop(crop, trim_components=field in ("home_score", "away_score"), grayscale=grayscale)
            raw  = ocr_field_with_fallback(proc, field)
        ok   = VALIDATORS.get(field, lambda _: False)(raw)
        scale_x = max(1, 160 // crop.shape[1])
        scaled = cv2.resize(crop, (crop.shape[1]*scale_x, crop.shape[0]*scale_x), interpolation=cv2.INTER_NEAREST)
        results.append({
            "field": field, "raw": raw, "ok": ok,
            "crop_b64": png_b64(scaled),
            "proc_b64": png_b64(proc),
        })
    return results


def draw_overlay(frame: np.ndarray, rois: dict[str, Any], src_w: int, src_h: int) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]
    sx, sy = fw / src_w, fh / src_h
    for field, roi in rois.items():
        x  = max(0, int(roi["x"] * sx));  y  = max(0, int(roi["y"] * sy))
        x2 = min(fw, x + int(roi["w"] * sx));  y2 = min(fh, y + int(roi["h"] * sy))
        color = FIELD_COLORS.get(field, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x2, y2), color, 2)
        cv2.putText(out, field, (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # Scale down thumbnail to ~640px wide
    scale = min(1.0, 640 / fw)
    if scale < 1.0:
        out = cv2.resize(out, (int(fw * scale), int(fh * scale)))
    return out


# ── HTML generation ────────────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>OCR Verification</title>
<style>
  body { font-family: monospace; background:#111; color:#ddd; margin:0; padding:1rem; }
  h1 { color:#fff; margin-bottom:0.25rem; }
  .meta { color:#888; font-size:.85rem; margin-bottom:1.5rem; }
  .frame-block { border:1px solid #333; border-radius:6px; padding:1rem; margin-bottom:2rem; background:#1a1a1a; }
  .frame-header { color:#aaa; font-size:.85rem; margin-bottom:.75rem; }
  .frame-row { display:flex; gap:1rem; flex-wrap:wrap; align-items:flex-start; }
  .thumb img { border-radius:4px; border:1px solid #444; max-width:640px; }
  .crops { display:flex; flex-wrap:wrap; gap:.75rem; }
  .crop-card { background:#222; border-radius:4px; padding:.5rem; min-width:120px; text-align:center; }
  .crop-card .label { font-size:.7rem; color:#888; margin-bottom:.3rem; }
  .crop-card img { border-radius:2px; image-rendering:pixelated; max-width:160px; }
  .crop-card .proc img { filter:invert(1); max-width:160px; }
  .read { font-size:1.1rem; font-weight:bold; margin-top:.3rem; }
  .ok   { color:#4caf50; }
  .fail { color:#f44336; }
</style>
</head>
<body>
<h1>OCR Verification Report</h1>
"""

_HTML_FOOT = "</body></html>\n"


def build_html(video_name: str, roi_path_str: str,
               frame_entries: list[dict[str, Any]],
               method: str = "tesseract", grayscale: bool = False) -> str:
    parts = [_HTML_HEAD]
    method_label = "7-segment LED" if method == "sevenseg" else "Standard (Tesseract)"
    gray_label = "grayscale" if grayscale else "binarized"
    config = f'OCR method: <b>{method_label}</b>'
    if method != "sevenseg":
        config += f' &nbsp;|&nbsp; preprocessing: <b>{gray_label}</b>'
    parts.append(
        f'<p class="meta">Video: <b>{video_name}</b> &nbsp;|&nbsp; ROI: <b>{roi_path_str}</b>'
        f' &nbsp;|&nbsp; {config} &nbsp;|&nbsp; {len(frame_entries)} spots</p>\n'
    )

    for entry in frame_entries:
        t = entry["t"]
        thumb_b64 = entry["thumb_b64"]
        crops = entry["crops"]
        all_ok = all(c["ok"] for c in crops)
        status_color = "#4caf50" if all_ok else "#f44336"
        parts.append(f'<div class="frame-block">\n')
        parts.append(f'<div class="frame-header">⏱ t = {t:.1f}s &nbsp; <span style="color:{status_color}">{"✓ all parsed" if all_ok else "✗ parse errors"}</span></div>\n')
        parts.append('<div class="frame-row">\n')
        if thumb_b64:
            parts.append(f'<div class="thumb"><img src="data:image/png;base64,{thumb_b64}"/></div>\n')
        parts.append('<div class="crops">\n')
        for c in crops:
            ok_cls = "ok" if c["ok"] else "fail"
            parts.append(f'<div class="crop-card">\n')
            parts.append(f'  <div class="label">{c["field"]}</div>\n')
            if c["crop_b64"]:
                parts.append(f'  <img src="data:image/png;base64,{c["crop_b64"]}"/>\n')
            parts.append(f'  <div class="read {ok_cls}">{c["raw"] or "(empty)"}</div>\n')
            parts.append(f'</div>\n')
        parts.append('</div>\n</div>\n</div>\n')

    parts.append(_HTML_FOOT)
    return "".join(parts)


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="Output HTML (default: <game_dir>/derived/verify_ocr.html)")
    ap.add_argument("--roi", default=None, type=Path)
    ap.add_argument("--at", default=None,
                    help="Comma-separated timestamps in seconds")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of auto-spaced spots (default: 10)")
    args = ap.parse_args()

    game_path = args.game.resolve()
    # Use the game's configured OCR method + grayscale so the report matches the
    # real pipeline.
    try:
        _gj = json.loads(game_path.read_text())
    except Exception:
        _gj = {}
    method = _gj.get("ocrMethod") or "tesseract"
    grayscale = bool(_gj.get("ocrGrayscale"))

    video = locate_video(game_path)
    if video is None:
        sys.exit(f"Video not found for {game_path}")

    roi_data = load_roi(game_path.parent, args.roi)
    rois = roi_data.get("rois", {})
    if not rois:
        sys.exit("roi.json has no 'rois' key")
    src_w = roi_data.get("frame_width", 1920)
    src_h = roi_data.get("frame_height", 1080)

    duration = get_duration(video)

    if args.at:
        timestamps = [float(x.strip()) for x in args.at.split(",")]
    else:
        step = duration / (args.n + 1)
        # Skip first 2% and last 2% (usually pre-game/post-game)
        start = duration * 0.02
        timestamps = [start + step * i for i in range(args.n)]

    out_path = args.out or (game_path.parent / "derived" / "verify_ocr.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    roi_label = str(args.roi or (game_path.parent / "roi.json"))

    print(f"OCR verify: {video.name} | method={method} grayscale={grayscale} | {len(timestamps)} spots | → {out_path}", flush=True)

    cap = cv2.VideoCapture(str(video))
    frame_entries: list[dict[str, Any]] = []

    for t in timestamps:
        frame = seek_frame(cap, t)
        if frame is None:
            print(f"  t={t:.0f}s  SKIP (seek failed)", flush=True)
            continue
        overlay = draw_overlay(frame, rois, src_w, src_h)
        crops   = ocr_frame(frame, rois, src_w, src_h, method=method, grayscale=grayscale)
        reads   = {c["field"]: c["raw"] for c in crops}
        status  = "OK" if all(c["ok"] for c in crops) else "FAIL"
        print(f"  t={t:.0f}s  {status}  {reads}", flush=True)
        frame_entries.append({
            "t": t,
            "thumb_b64": png_b64(overlay),
            "crops": crops,
        })

    cap.release()

    html = build_html(video.name, roi_label, frame_entries, method=method, grayscale=grayscale)
    out_path.write_text(html)
    print(f"\nWrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
