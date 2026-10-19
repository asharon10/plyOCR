#!/usr/bin/env python3
"""Run Tesseract OCR on a single video frame and print JSON to stdout.

Usage:
    python -m plyocr.read_frame --game data/games/<id>/game.json --at 2674
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

from plyocr.hls_frame import grab_hls_frame  # noqa: E402

from plyocr import sevenseg

from plyocr.paths import DEFAULT_ROI

_DIGITS = "0123456789"
_QTR_W  = "1234OTstndrdth"
TESS_CONFIG = {
    "clock":      "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:.",
    "quarter":    f"--psm 8 --oem 3 -c tessedit_char_whitelist={_QTR_W}",
    "home_score": f"--psm 8 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
    "away_score": f"--psm 8 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
}
TESS_FALLBACK = {k: v.replace("--psm 7", "--psm 13").replace("--psm 8", "--psm 13")
                 for k, v in TESS_CONFIG.items()}

CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}$|^\d{1,2}\.\d$")
SCORE_RE = re.compile(r"^\d{1,3}$")
QUARTER_MAP = {
    "1": 1, "1ST": 1, "2": 2, "2ND": 2,
    "3": 3, "3RD": 3, "4": 4, "4TH": 4,
    "OT": 5, "0T": 5,
}


def fix_common(s: str) -> str:
    return s.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")


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


def _emphasis_gray(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    """Intensity where digits are bright. Red-emphasis (R−max(G,B)) for red-on-dark
    (luminance crushes red); luminance otherwise. Second value → sharpen vs blur."""
    b, g, r = (crop[:, :, i].astype(np.int16) for i in range(3))
    redness = r - np.maximum(g, b)
    if int((redness > 60).sum()) >= 0.02 * crop.shape[0] * crop.shape[1]:
        return np.clip(redness, 0, 255).astype(np.uint8), True
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), False


def preprocess(crop: np.ndarray, trim_components: bool = False, grayscale: bool = False,
               threshold: int | None = None) -> np.ndarray:
    gray, channel_aware = _emphasis_gray(crop)
    h, w = gray.shape
    if h < 100:
        scale = int(100 / h) + 1
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    if channel_aware:
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    invert = float(border.mean()) < 128
    # Grayscale mode: hand Tesseract the contrast-enhanced grayscale (dark text on
    # light bg) and let it binarize itself, skipping our Otsu/dilate.
    if grayscale:
        g = cv2.bitwise_not(gray) if invert else gray
        _, _b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = np.where(_b == 0)
        if ink[0].size > 0:
            y0, y1 = ink[0].min(), ink[0].max()
            x0, x1 = ink[1].min(), ink[1].max()
            g = g[y0:y1 + 1, x0:x1 + 1]
        return cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    # Manual threshold (0-255) overrides Otsu auto — the user's contrast knob.
    if threshold is not None:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
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


def _png_b64(img: np.ndarray) -> str:
    """Encode a (grayscale or BGR) image as a base64 data-URI payload for the UI."""
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _roi_threshold(roi: dict[str, Any]) -> int | None:
    """Per-field manual binarization threshold (0-255); None/absent → Otsu auto."""
    t = roi.get("threshold")
    if t is None or t == "":
        return None
    try:
        return max(0, min(255, int(t)))
    except (TypeError, ValueError):
        return None


def ocr(img: np.ndarray, field: str) -> str:
    raw = re.sub(r"\s+", "", pytesseract.image_to_string(img, config=TESS_CONFIG[field]).strip())
    if not raw:
        raw = re.sub(r"\s+", "", pytesseract.image_to_string(img, config=TESS_FALLBACK[field]).strip())
    return raw


_EASYOCR: Any = None
_EASY_ALLOW = {"clock": "0123456789:.", "quarter": "1234OTE", "home_score": "0123456789", "away_score": "0123456789"}


def ocr_easyocr(crop: np.ndarray, field: str) -> str:
    """Neural read of one ROI crop (lazy-loaded model). Mirrors run_tesseract_ocr.py."""
    global _EASYOCR
    try:
        if _EASYOCR is None:
            import easyocr
            _EASYOCR = easyocr.Reader(["en"], gpu=False, verbose=False)
        up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        res = _EASYOCR.readtext(up, allowlist=_EASY_ALLOW.get(field, "0123456789"), detail=1, paragraph=False)
    except Exception:
        return ""
    parts = [(b[0][0], t.strip()) for (b, t, c) in res if t and t.strip() and c >= 0.4]
    parts.sort(key=lambda p: p[0])
    return re.sub(r"\s+", "", "".join(t for _, t in parts))


def parse_clock(r: str) -> str | None:
    r = fix_common(r)
    if not CLOCK_RE.match(r):
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", r)
    if m:
        if int(m.group(1)) > 12 or int(m.group(2)) > 59:
            return None
        return r
    m = re.match(r"^(\d{1,2})\.(\d)$", r)
    if m and int(m.group(1)) > 59:
        return None
    return r


def parse_quarter(r: str) -> int | None:
    s = r.strip().lower()
    if s.endswith("st"):
        return 1
    if s.endswith("nd"):
        return 2
    if s.endswith("rd"):
        return 3
    if s.endswith("th"):
        return 4
    if "ot" in s or "0t" in s:
        return 5
    m = re.match(r"^(\d)", s)
    return QUARTER_MAP.get(m.group(1)) if m else None


def parse_score(r: str) -> int | None:
    r = fix_common(r)
    if not SCORE_RE.match(r):
        return None
    v = int(r)
    return v if v <= 200 else None


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


def resolve_video_source(game_path: Path) -> tuple[str, bool]:
    """Return (ffmpeg_input, is_remote). Cloud games (game.json cloudMode=true,
    with a video URL) read the HLS URL directly; otherwise the local game.mp4.
    Mirrors run_tesseract_ocr.resolve_video_source."""
    try:
        g = json.loads(game_path.read_text())
    except Exception:
        g = {}
    url = (g.get("video") or {}).get("downloadUrl") or g.get("videoUrl")
    local = locate_video(game_path)
    if bool(g.get("cloudMode")) and url:
        return url, True
    if local is not None:
        return str(local.resolve()), False
    if url:
        return url, True
    return "", False


def grab_frame(video: str, at: float, is_remote: bool) -> np.ndarray | None:
    """Grab a single BGR frame at `at` seconds. Remote (HLS) uses ffmpeg input
    seeking (fetches just the segment, no download); local uses OpenCV. The remote
    path pipes a PNG and decodes it, so no separate dimension probe is needed."""
    if not is_remote:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    return grab_hls_frame(video, at)


def load_roi(game_dir: Path) -> dict[str, Any]:
    local = game_dir / "roi.json"
    path = local if local.exists() else DEFAULT_ROI
    return json.loads(path.read_text())


# Use the pipeline's own OCR (preprocessing, Tesseract config, parsers) so a
# Setup / Courts test reads exactly what a full OCR run will.
from plyocr import run_tesseract_ocr as _pipe  # noqa: E402
preprocess = _pipe.preprocess_crop
ocr = _pipe.ocr_field
parse_clock, parse_quarter, parse_score = _pipe.parse_clock, _pipe.parse_quarter, _pipe.parse_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, type=Path)
    ap.add_argument("--at", required=True, type=float, help="Timestamp in seconds")
    ap.add_argument("--rois-json", default=None,
                    help="Inline JSON {frame_width,frame_height,rois} overriding roi.json")
    ap.add_argument("--grayscale", default=None, choices=["0", "1"],
                    help="Force grayscale OCR mode (overrides game.json ocrGrayscale)")
    ap.add_argument("--method", default=None, choices=["tesseract", "sevenseg", "easyocr"],
                    help="Force OCR method (overrides game.json ocrMethod)")
    ap.add_argument("--emit-crops", action="store_true",
                    help="Include a base64 PNG of the preprocessed crop per field")
    args = ap.parse_args()

    game_path = args.game.resolve()
    try:
        _gj = json.loads(game_path.read_text())
    except Exception:
        _gj = {}
    # Grayscale OCR mode: explicit flag wins, else read game.json's ocrGrayscale.
    if args.grayscale is not None:
        grayscale = args.grayscale == "1"
    else:
        grayscale = bool(_gj.get("ocrGrayscale"))
    # OCR method: explicit flag wins, else game.json's ocrMethod (default tesseract).
    method = args.method or _gj.get("ocrMethod") or "tesseract"
    # Per-field quarter override (empty ⇒ follow method).
    quarter_method = _gj.get("ocrQuarterMethod") or ""
    video, is_remote = resolve_video_source(game_path)
    if not video:
        json.dump({"error": "video_not_found"}, sys.stdout)
        sys.exit(1)

    if args.rois_json:
        roi_data = json.loads(args.rois_json)
    else:
        roi_data = load_roi(game_path.parent)
    roi_data = _pipe.with_quarter_templates(roi_data)
    rois = roi_data.get("rois", {})
    src_w = roi_data.get("frame_width", 1920)
    src_h = roi_data.get("frame_height", 1080)

    frame = grab_frame(video, args.at, is_remote)
    if frame is None:
        json.dump({"error": "seek_failed", "at": args.at}, sys.stdout)
        sys.exit(1)

    fh, fw = frame.shape[:2]
    sx, sy = fw / src_w, fh / src_h

    reads: dict[str, Any] = {"at": args.at}
    for field, roi in rois.items():
        x  = max(0, int(roi["x"] * sx));  y  = max(0, int(roi["y"] * sy))
        x2 = min(fw, x + int(roi["w"] * sx));  y2 = min(fh, y + int(roi["h"] * sy))
        if x2 <= x or y2 <= y:
            reads[field] = {"raw": "", "parsed": None, "valid": False}
            continue
        crop = frame[y:y2, x:x2]
        thr = _roi_threshold(roi)
        proc: np.ndarray | None = None
        # The court's learned quarter digits first (stylised boards), then the
        # per-field quarter override / method chain — mirrors run_tesseract_ocr.py.
        tq = (_pipe.quarter_templates.classify(crop, _pipe._decoded_templates(roi["templates"]))
              if field == "quarter" and roi.get("templates") else None)
        if tq is not None:
            raw = str(tq)
            if args.emit_crops:
                proc = preprocess(crop, trim_components=False, grayscale=grayscale, threshold=thr)
        elif field == "quarter" and quarter_method and quarter_method != method:
            if quarter_method == "easyocr":
                raw = ocr_easyocr(crop, field)
            elif quarter_method == "sevenseg":
                raw = sevenseg.decode_crop(crop, is_clock=False)
            else:
                raw = ocr(preprocess(crop), field)
        elif method == "easyocr":
            raw = ocr_easyocr(crop, field)
        elif method == "sevenseg" and field != "quarter":
            raw = sevenseg.decode_crop(crop, is_clock=(field == "clock"))
            # Still build a preview crop so the threshold slider has visual feedback.
            if args.emit_crops:
                proc = preprocess(crop, trim_components=field in ("home_score", "away_score"),
                                  grayscale=grayscale, threshold=thr)
        else:
            proc = preprocess(crop, trim_components=field in ("home_score", "away_score"),
                              grayscale=grayscale, threshold=thr)
            raw  = ocr(proc, field)
            if field == "quarter" and method == "sevenseg" and parse_quarter(raw or "") is None:
                sv = sevenseg.decode_crop(crop, is_clock=False)
                if sv and parse_quarter(sv) is not None:
                    raw = sv

        if field == "clock":
            parsed: Any = parse_clock(raw)
        elif field == "quarter":
            parsed = parse_quarter(raw)
        else:
            parsed = parse_score(raw)

        entry: dict[str, Any] = {"raw": raw, "parsed": parsed, "valid": parsed is not None}
        if args.emit_crops and proc is not None:
            entry["crop"] = _png_b64(proc)
        reads[field] = entry

    print(json.dumps(reads))


if __name__ == "__main__":
    main()
