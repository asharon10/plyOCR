#!/usr/bin/env python3
"""Tesseract-based scoreboard OCR pipeline.

Reads roi.json (game-local or default), samples frames from the video at
--fps rate, crops each ROI region, binarizes, and runs Tesseract to read:
  - clock    (M:SS or SS.t format)
  - quarter  (1-4 or OT)
  - home_score
  - away_score

Performance: uses cv2.VideoCapture with CAP_PROP_POS_MSEC seeks — one open,
many seeks — avoiding per-frame ffmpeg subprocess spawns.

Outputs:
  derived/ocr_timeline.csv   (video_time_seconds,quarter,game_clock,home_score,away_score,confidence)
  derived/timeline.json      (same rows as dicts, for downstream compatibility)
  derived/timeline_raw.json  (every sampled frame, before cleaning)

Progress is printed as `PROGRESS <done>/<total>` so the API job runner can
update the UI percentage.

Usage:
    python -m plyocr.run_tesseract_ocr --game data/games/<id>/game.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import bisect
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract

from plyocr import sevenseg
from plyocr import quarter_templates
from plyocr import game_format  # noqa: E402

_TPL_CACHE: dict[int, dict] = {}


def _decoded_templates(tpl: dict) -> dict:
    """Decode a court's quarter templates once per process."""
    k = id(tpl)
    if k not in _TPL_CACHE:
        _TPL_CACHE[k] = quarter_templates.decode_templates(tpl)
    return _TPL_CACHE[k]

# Per-game OCR method ("tesseract" default | "sevenseg" for 7-segment LED boards).
# Set from game.json in run(); threaded into each worker via the task tuple.
_METHOD = "tesseract"
# Per-field override for the quarter ("" ⇒ follow _METHOD). Lets a stylised-font
# quarter use EasyOCR while clock+scores stay on the fast decoder.
_QUARTER_METHOD = ""

# Optional in-process Tesseract backend. pytesseract spawns a `tesseract`
# subprocess on every call (~15k spawns for a full game); tesserocr binds
# libtesseract directly, so it reuses one engine per worker process — a large,
# correctness-neutral speedup. We fall back to pytesseract if it isn't built.
try:
    from PIL import Image as _PILImage
    from tesserocr import PyTessBaseAPI, PSM, OEM
    _HAS_TESSEROCR = True
except Exception:  # pragma: no cover - import availability is environment-dependent
    _HAS_TESSEROCR = False

# Keep each Tesseract/OpenCV worker single-threaded so N processes map cleanly
# onto N cores instead of oversubscribing.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
cv2.setNumThreads(0)

from plyocr.paths import DEFAULT_ROI, ROI_TEMPLATES

# Tesseract configs per field.
# PSM 7  = single text line (good for most)
# PSM 8  = single word (better when Tesseract over-segments a line)
# PSM 13 = raw line, bypass Tesseract's internal layout analysis (fallback)
_DIGITS  = "0123456789"
_CLOCK_W = "0123456789:."
_QTR_W   = "1234OTstndrdth"   # include ordinal suffixes so "4th" survives

TESS_CONFIG = {
    "clock":      f"--psm 7 --oem 3 -c tessedit_char_whitelist={_CLOCK_W}",
    "quarter":    f"--psm 8 --oem 3 -c tessedit_char_whitelist={_QTR_W}",
    "home_score": f"--psm 8 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
    "away_score": f"--psm 8 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
}
# Fallback configs tried when primary returns empty
TESS_FALLBACK = {
    "clock":      f"--psm 13 --oem 3 -c tessedit_char_whitelist={_CLOCK_W}",
    "quarter":    f"--psm 13 --oem 3 -c tessedit_char_whitelist={_QTR_W}",
    "home_score": f"--psm 13 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
    "away_score": f"--psm 13 --oem 3 -c tessedit_char_whitelist={_DIGITS}",
}

# tesserocr equivalents of the TESS_CONFIG/TESS_FALLBACK above. Same OEM (3 =
# default LSTM+legacy), same whitelist, same PSM (primary then RAW_LINE fallback).
if _HAS_TESSEROCR:
    _TESS_PSM = {
        "clock":      PSM.SINGLE_LINE,   # --psm 7
        "quarter":    PSM.SINGLE_WORD,   # --psm 8
        "home_score": PSM.SINGLE_WORD,
        "away_score": PSM.SINGLE_WORD,
    }
    _TESS_PSM_FALLBACK = PSM.RAW_LINE    # --psm 13
_TESS_WHITELIST = {
    "clock": _CLOCK_W,
    "quarter": _QTR_W,
    "home_score": _DIGITS,
    "away_score": _DIGITS,
}

# Per-worker-process libtesseract handle (lazily created after fork). A
# PyTessBaseAPI is not picklable and must live inside the worker, so we keep it
# in a module global that each process initialises on first use.
_TESS_API: Any = None


def _tessdata_path() -> str | None:
    """Locate the tessdata directory (PyTessBaseAPI doesn't read the binary's
    built-in default). Honor TESSDATA_PREFIX, else derive it from the resolved
    `tesseract` binary (handles Homebrew's Cellar symlink)."""
    env = os.environ.get("TESSDATA_PREFIX")
    if env and (Path(env) / "eng.traineddata").exists():
        return env
    if env and (Path(env) / "tessdata" / "eng.traineddata").exists():
        return str(Path(env) / "tessdata")
    exe = shutil.which("tesseract")
    if exe:
        real = Path(exe).resolve()
        cand = real.parent.parent / "share" / "tessdata"
        if (cand / "eng.traineddata").exists():
            return str(cand)
    return None


def _get_tess_api() -> Any:
    global _TESS_API
    if _TESS_API is None:
        path = _tessdata_path()
        kwargs: dict[str, Any] = {"psm": PSM.SINGLE_LINE, "oem": OEM.DEFAULT}
        if path:
            kwargs["path"] = path
        _TESS_API = PyTessBaseAPI(**kwargs)
    return _TESS_API


CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}$|^\d{1,2}\.\d$")
SCORE_RE = re.compile(r"^\d{1,3}$")

# Map raw quarter OCR text → integer quarter number
QUARTER_MAP: dict[str, int] = {
    "1": 1, "1ST": 1, "1st": 1,
    "2": 2, "2ND": 2, "2nd": 2,
    "3": 3, "3RD": 3, "3rd": 3,
    "4": 4, "4TH": 4, "4th": 4,
    "OT": 5, "0T": 5,
}


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_duration_ffprobe(video: Path) -> float:
    # HLS: the playlist already lists every segment's length — sum those. ffprobe
    # on a long remote playlist can hang for good when the CDN resets the
    # connection (it has no timeout of its own).
    v = str(video)
    if v.startswith("http") and ".m3u8" in v:
        try:
            from plyocr.hls_frame import playlist_duration
            d = playlist_duration(v)
            if d > 0:
                return d
        except Exception:
            pass
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-rw_timeout", "30000000", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", v],
        text=True, timeout=180,
    )
    return float(out.strip())


def open_capture(video: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}")
    return cap


def read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


# ---------------------------------------------------------------------------
# ROI loading
# ---------------------------------------------------------------------------

def with_quarter_templates(roi_data: dict[str, Any]) -> dict[str, Any]:
    """Attach the court's learned quarter-digit templates (roi.json
    `quarterTemplates`) to the quarter ROI, so every reader that gets the ROIs
    — including the worker processes — can match the digit."""
    tpl = roi_data.get("quarterTemplates")
    q = (roi_data.get("rois") or {}).get("quarter")
    if tpl and isinstance(q, dict):
        q["templates"] = tpl
    return roi_data


def load_roi(game_dir: Path) -> dict[str, Any]:
    """ROI cascade: game-local roi.json → learned per-resolution template →
    hard-coded baseline. ROIs are scaled to the actual frame at OCR time
    (process_frame), so the template's source resolution doesn't need to match."""
    game_local = game_dir / "roi.json"
    if game_local.exists():
        print(f"Using ROI: {game_local}", flush=True)
        return with_quarter_templates(json.loads(game_local.read_text()))

    templates_path = ROI_TEMPLATES
    if templates_path.exists():
        try:
            templates = json.loads(templates_path.read_text())
        except (ValueError, OSError):
            templates = {}
        if templates:
            # Most-sampled template = most reliable (IBBA standard overlay).
            best = max(templates.values(), key=lambda t: t.get("n_samples", 0))
            print(
                f"Using ROI: learned template {best.get('frame_width')}x{best.get('frame_height')} "
                f"(from {best.get('n_samples')} game(s))",
                flush=True,
            )
            return best

    default = DEFAULT_ROI
    print(f"Using ROI: {default} (hard-coded baseline)", flush=True)
    return json.loads(default.read_text())


def get_frame_size(roi_data: dict[str, Any]) -> tuple[int, int]:
    return roi_data.get("frame_width", 1920), roi_data.get("frame_height", 1080)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _remove_spurious_components(binary: np.ndarray) -> np.ndarray:
    """Drop small/thin connected components (edge lines, dots, divider bars).

    `binary` is black text (0) on white (255). We analyse the text pixels and
    keep only components whose height is a reasonable fraction of the tallest
    component. This kills the thin vertical divider artifact that otherwise
    gets read as a trailing/leading '0' (e.g. '52' → '520').
    """
    text_mask = (binary == 0).astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    if n <= 1:
        return binary
    # stats rows: [x, y, w, h, area]; row 0 is background
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(heights) == 0:
        return binary
    max_h = int(heights.max())
    keep = np.ones(n, dtype=bool)
    keep[0] = False  # background
    for i in range(1, n):
        h_i = stats[i, cv2.CC_STAT_HEIGHT]
        a_i = stats[i, cv2.CC_STAT_AREA]
        # Drop components much shorter than the tallest glyph, or tiny specks
        if h_i < 0.45 * max_h or a_i < 0.04 * areas.max():
            keep[i] = False
    cleaned = np.full_like(binary, 255)
    for i in range(1, n):
        if keep[i]:
            cleaned[_labels == i] = 0
    return cleaned


# When True, feed Tesseract a cleaned GRAYSCALE crop (let it binarize itself)
# instead of our Otsu binary mask. Helps scoreboards where our invert/Otsu guess
# mangles the digits (colored/low-contrast overlays). Set per game from game.json.
_GRAYSCALE = False


def _emphasis_gray(crop: np.ndarray) -> tuple[np.ndarray, bool]:
    """Single-channel intensity where the DIGITS are bright.

    Luminance grayscale crushes red (weight 0.30), so a bright red digit on black
    becomes mid-dark — low contrast. When the crop is red-dominant on a dark
    background we instead use a red-emphasis channel (R − max(G,B)), which makes
    red digits bright and the rest black. Returns (gray, channel_aware) where
    channel_aware also signals to sharpen instead of blur. Other colours keep the
    existing luminance path (so working boards don't change)."""
    b, g, r = (crop[:, :, i].astype(np.int16) for i in range(3))
    redness = r - np.maximum(g, b)
    strong = redness > 60
    # red-dominant if a meaningful patch is strongly red AND the field isn't just
    # generally bright (i.e. red text on dark, not a white/colored panel).
    if int(strong.sum()) >= 0.02 * crop.shape[0] * crop.shape[1]:
        return np.clip(redness, 0, 255).astype(np.uint8), True
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), False


def preprocess_crop(crop: np.ndarray, trim_components: bool = False,
                    grayscale: bool | None = None,
                    threshold: int | None = None) -> np.ndarray:
    """Convert a BGR crop to a clean binary image Tesseract can read.

    Key choices:
    - Channel-aware intensity (red-emphasis for red-on-black; luminance otherwise)
    - Target 100px height — gives Tesseract room for thin strokes like '5'
    - CLAHE for contrast; unsharp-sharpen for colored digits (else gentle blur)
    - Dilation after binarize — thickens strokes so '5' doesn't collapse into '3'
    - Optional component trim — removes edge artifacts that hallucinate extra digits
    """
    gray, channel_aware = _emphasis_gray(crop)

    h, w = gray.shape
    # Target at least 100px height so thin strokes survive binarization
    target_h = 100
    if h < target_h:
        scale = int(target_h / h) + 1
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # CLAHE — boosts local contrast without blowing highlights (helps semi-transparent bars)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    if channel_aware:
        # Unsharp mask — sharpen edges of the LED strokes (the channel already has
        # high digit/background contrast, so we sharpen rather than blur).
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    else:
        # Gentle blur to smooth encoder artifacts before thresholding
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect background brightness from border pixels
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    invert = float(border.mean()) < 128  # dark bg → light text → invert after Otsu

    # Grayscale mode: hand Tesseract the contrast-enhanced grayscale (dark text on
    # light bg) and let it do its own binarization, skipping our Otsu/dilate.
    if grayscale if grayscale is not None else _GRAYSCALE:
        g = cv2.bitwise_not(gray) if invert else gray
        _, _b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = np.where(_b == 0)
        if ink[0].size > 0:
            y0, y1 = ink[0].min(), ink[0].max()
            x0, x1 = ink[1].min(), ink[1].max()
            g = g[y0:y1 + 1, x0:x1 + 1]
        return cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)

    # Manual threshold (0-255) overrides Otsu auto — the field's saved contrast knob.
    if threshold is not None:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        binary = cv2.bitwise_not(binary)

    # Dilate to thicken thin strokes — prevents '5' collapsing to '3' or '9'
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Remove spurious components (scores only — avoids dropping ':' or '.' in clock)
    if trim_components:
        binary = _remove_spurious_components(binary)

    # Tight-crop to the ink bounding box. A wide empty margin (common when the
    # ROI is bigger than the text) makes Tesseract hallucinate extra digits
    # (e.g. '52' → '520'). Cropping to the glyphs fixes this for every field.
    ink = np.where(binary == 0)
    if ink[0].size > 0:
        y0, y1 = ink[0].min(), ink[0].max()
        x0, x1 = ink[1].min(), ink[1].max()
        binary = binary[y0:y1 + 1, x0:x1 + 1]

    # White padding so Tesseract doesn't clip edge glyphs
    binary = cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    return binary


# ---------------------------------------------------------------------------
# Tesseract helpers
# ---------------------------------------------------------------------------

def _ocr_field_tesserocr(img: np.ndarray, field: str) -> str:
    """In-process OCR via libtesseract. Mirrors the pytesseract path: primary PSM
    with the field whitelist, then RAW_LINE fallback if it reads nothing."""
    api = _get_tess_api()
    pil = _PILImage.fromarray(img)
    api.SetVariable("tessedit_char_whitelist", _TESS_WHITELIST[field])
    api.SetPageSegMode(_TESS_PSM[field])
    api.SetImage(pil)
    raw = re.sub(r"\s+", "", (api.GetUTF8Text() or "").strip())
    if not raw:
        api.SetPageSegMode(_TESS_PSM_FALLBACK)
        api.SetImage(pil)
        raw = re.sub(r"\s+", "", (api.GetUTF8Text() or "").strip())
    return raw


def _ocr_field_pytesseract(img: np.ndarray, field: str) -> str:
    raw = re.sub(r"\s+", "", pytesseract.image_to_string(img, config=TESS_CONFIG[field]).strip())
    if not raw and field in TESS_FALLBACK:
        # Primary config returned nothing — retry with PSM 13 (raw-line mode)
        raw = re.sub(r"\s+", "", pytesseract.image_to_string(img, config=TESS_FALLBACK[field]).strip())
    return raw


def ocr_field(img: np.ndarray, field: str) -> str:
    if _HAS_TESSEROCR:
        return _ocr_field_tesserocr(img, field)
    return _ocr_field_pytesseract(img, field)


# ---------------------------------------------------------------------------
# EasyOCR helpers (optional neural backend)
# ---------------------------------------------------------------------------
# Neural recogniser that reads stylised/script fonts the geometric 7-seg decoder
# and Tesseract miss (e.g. a cursive white quarter digit). Heavy (PyTorch) and
# slower per frame, so it's opt-in per game (ocrMethod="easyocr") and the model
# is lazy-loaded once per worker process.
_EASYOCR: Any = None
_EASY_ALLOW = {
    "clock": "0123456789:.",
    "quarter": "1234OTE",
    "home_score": "0123456789",
    "away_score": "0123456789",
}


def _get_easyocr() -> Any:
    global _EASYOCR
    if _EASYOCR is None:
        import easyocr  # heavy import — only when this method is actually used
        _EASYOCR = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR


def _ocr_field_easyocr(crop: np.ndarray, field: str) -> str:
    """Read one ROI crop with EasyOCR. Upscales 4× (small scoreboard glyphs),
    restricts the charset per field, drops low-confidence detections, and joins
    the surviving pieces left-to-right."""
    try:
        reader = _get_easyocr()
        up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        allow = _EASY_ALLOW.get(field, "0123456789")
        res = reader.readtext(up, allowlist=allow, detail=1, paragraph=False)
    except Exception:
        return ""
    # res items: (bbox[4 pts], text, conf). Order left-to-right by x, keep conf ≥ 0.4.
    parts = [(b[0][0], t.strip()) for (b, t, c) in res if t and t.strip() and c >= 0.4]
    parts.sort(key=lambda p: p[0])
    return re.sub(r"\s+", "", "".join(t for _, t in parts))



def fix_common(raw: str) -> str:
    """Fix common OCR substitutions: O→0, l/I→1."""
    return raw.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")


def parse_clock(raw: str) -> str | None:
    raw = fix_common(raw)
    # Some boards blink the colon: with it off the read is just digits.
    m = re.match(r"^(\d{1,2})(\d{2})$", raw)
    if m and len(raw) >= 3:
        raw = f"{int(m.group(1))}:{m.group(2)}"
    if not CLOCK_RE.match(raw):
        return None
    # M:SS form — minutes ≤ 12 (quarter clock max), seconds < 60
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw)
    if m:
        mins, secs = int(m.group(1)), int(m.group(2))
        if mins > 12 or secs > 59:
            return None
        return raw
    # SS.t form (under 1 minute) — seconds < 60
    m = re.match(r"^(\d{1,2})\.(\d)$", raw)
    if m and int(m.group(1)) > 59:
        return None
    return raw


def parse_quarter(raw: str) -> int | None:
    """Parse quarter, trusting the ordinal SUFFIX over the leading digit.

    On this broadcast the quarter is always rendered as an ordinal
    ("1st", "2nd", "3rd", "4th"). The suffix glyphs are visually distinct and
    OCR reads them reliably even when it misclassifies the digit (e.g. it may
    return "4st" for "1st" — the suffix "st" still correctly means Q1).
    """
    s = raw.strip().lower()
    # Suffix-first (most reliable)
    if s.endswith("st"):
        return 1
    if s.endswith("nd"):
        return 2
    if s.endswith("rd"):
        return 3
    if s.endswith("th"):
        return 4
    # Overtime
    if "ot" in s or "0t" in s:
        return 5
    # Fallback: leading digit only when no usable suffix was read
    m = re.match(r"^(\d)", s)
    if m:
        return QUARTER_MAP.get(m.group(1))
    return None


def parse_score(raw: str) -> int | None:
    raw = fix_common(raw)
    if not SCORE_RE.match(raw):
        return None
    val = int(raw)
    return val if val <= 200 else None  # scores above 200 are OCR garbage


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

# ── Change-gating: skip Tesseract when an ROI's pixels are unchanged ──────────
# Only the QUARTER is gated. It is static ~99% of frames and only ever changes by
# a whole digit (1→2→3→4), which a coarse thumbnail detects reliably — so we read
# it once and reuse until it changes. The CLOCK (ticks every second) and the
# SCORES (small single-digit increments a coarse thumbnail can miss) are NOT
# gated: gating them risks holding a stale value, which the monotonic score
# smoothing then locks in. A periodic forced re-read guards the quarter.
_GATED_FIELDS = ("quarter",)
_SIG_W, _SIG_H = 16, 8
_SIG_TOL = 6.0          # mean abs grayscale diff below this = "unchanged"
_FORCE_OCR_EVERY = 12   # force a real OCR at least this often per gated field


def _crop_signature(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(gray, (_SIG_W, _SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)


def _sig_unchanged(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and float(np.mean(np.abs(a - b))) <= _SIG_TOL


def process_frame(frame: np.ndarray, rois: dict[str, Any],
                  src_w: int, src_h: int,
                  gate_state: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return (raw_dict, clean_dict|None).

    If ``gate_state`` (a per-field cache, carried across consecutive frames) is
    provided, an ROI whose crop is unchanged since its last OCR is skipped and
    its previous value reused (change-gating). Pass None to always OCR.
    """
    fh, fw = frame.shape[:2]
    sx, sy = fw / src_w, fh / src_h

    raw: dict[str, Any] = {}
    for field, roi in rois.items():
        x = max(0, int(roi["x"] * sx))
        y = max(0, int(roi["y"] * sy))
        x2 = min(fw, x + int(roi["w"] * sx))
        y2 = min(fh, y + int(roi["h"] * sy))
        if x2 <= x or y2 <= y:
            raw[field] = None
            continue
        crop = frame[y:y2, x:x2]

        def _read(c: np.ndarray) -> str | None:
            if field == "quarter" and roi.get("templates"):
                q = quarter_templates.classify(c, _decoded_templates(roi["templates"]))
                if q is not None:
                    return str(q)
            # Per-field quarter override: read JUST the quarter with a different
            # method (e.g. EasyOCR for a stylised font) while clock+scores stay on
            # the fast decoder.
            if field == "quarter" and _QUARTER_METHOD and _QUARTER_METHOD != _METHOD:
                if _QUARTER_METHOD == "easyocr":
                    return _ocr_field_easyocr(c, field) or None
                if _QUARTER_METHOD == "sevenseg":
                    return sevenseg.decode_crop(c, is_clock=False) or None
                # tesseract
                return ocr_field(preprocess_crop(c), field) or None
            # EasyOCR (neural) reads every field directly — used for stylised/script
            # boards the geometric decoder and Tesseract can't handle.
            if _METHOD == "easyocr":
                return _ocr_field_easyocr(c, field) or None
            # Scores/clock use the geometric 7-seg decoder under sevenseg mode.
            if _METHOD == "sevenseg" and field != "quarter":
                return sevenseg.decode_crop(c, is_clock=(field == "clock")) or None
            # A 7-seg board's quarter digit: the decoder first (Tesseract misreads it
            # as an ordinal, e.g. a slanted "2" → "rd" → Q3); ordinal boards fall through.
            if _METHOD == "sevenseg" and field == "quarter":
                q = sevenseg.quarter_digit(c)
                if q:
                    return q
            # The quarter is read with Tesseract first — it handles ordinal boards
            # ("1st"/"2nd") that the 7-seg decoder can't. But many LED boards show a
            # plain 7-seg quarter DIGIT (e.g. a yellow "3") that Tesseract misreads
            # as None; fall back to the 7-seg decoder in that case.
            trim = field in ("home_score", "away_score")
            thr = roi.get("threshold")
            thr = max(0, min(255, int(thr))) if thr not in (None, "") else None
            val = ocr_field(preprocess_crop(c, trim_components=trim, threshold=thr), field) or None
            # Fall back to the 7-seg decoder when Tesseract's quarter read doesn't
            # parse (None or garbage like "t" for a yellow "4"); only adopt it if it
            # itself parses — so ordinal "1st" boards (which Tesseract reads) are safe.
            if field == "quarter" and _METHOD == "sevenseg" and parse_quarter(val or "") is None:
                sv = sevenseg.decode_crop(c, is_clock=False) or None
                if sv and parse_quarter(sv) is not None:
                    val = sv
            return val

        if gate_state is not None and field in _GATED_FIELDS:
            sig = _crop_signature(crop)
            st = gate_state.get(field)
            if st is not None and st["n"] < _FORCE_OCR_EVERY and _sig_unchanged(sig, st["sig"]):
                raw[field] = st["val"]      # unchanged → reuse, skip OCR
                st["n"] += 1
                continue
            val = _read(crop)
            gate_state[field] = {"sig": sig, "val": val, "n": 0}
            raw[field] = val
            continue

        raw[field] = _read(crop)

    clock = parse_clock(raw.get("clock") or "")
    quarter = parse_quarter(raw.get("quarter") or "")
    home = parse_score(raw.get("home_score") or "")
    away = parse_score(raw.get("away_score") or "")

    # Keep any frame whose CLOCK reads — the quarter is derived from the clock
    # sawtooth in smooth_records (_quarters_by_segment), so a hard-to-read quarter
    # (e.g. a white, stylised quarter font) must NOT discard otherwise-good clock +
    # score data. quarter=0 means "unknown here"; the segment pass fills it in.
    clean: dict[str, Any] | None = None
    if clock is not None:
        clean = {
            "clock": clock,
            "quarter": quarter if quarter is not None else 0,
            "home_score": home,
            "away_score": away,
            "confidence": 1.0,
        }
    return raw, clean



# ---------------------------------------------------------------------------
# Quarter from the clock (boards whose period digit Tesseract can't read)
# ---------------------------------------------------------------------------

def _clock_ms_any(raw: str | None) -> int | None:
    c = parse_clock(raw or "")
    if not c:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", c)
    if m:
        return int(m.group(1)) * 60000 + int(m.group(2)) * 1000
    m = re.match(r"^(\d{1,2})\.(\d)$", c)
    return int(m.group(1)) * 1000 + int(m.group(2)) * 100 if m else None


def _choose_quarter_source(raw_records: list[dict[str, Any]], clean_records: list[dict[str, Any]],
                           roi_data: dict[str, Any], game_path: Path, _in_skip) -> list[dict[str, Any]]:
    """The quarter each row belongs to: the scoreboard digit, or — when the
    court says so, or the digit is unreadable / implausible — the clock resets."""
    # Quarter digit unreadable on this board (few clean rows, or quarters missing)?
    # Work the quarter out from the clock resets instead.
    n_clock = sum(1 for r in raw_records if _clock_ms_any(r.get("clock")) is not None)
    read_qs = {r["quarter"] for r in clean_records} & {1, 2, 3, 4}
    q_source = str(roi_data.get("quarterSource") or "auto")  # court setting (roi.json)
    if q_source == "clock" and n_clock:
        inferred = [r for r in infer_quarters_from_clock(raw_records, game_format.load(game_path)) if not _in_skip(r["video_t"])]
        if inferred:
            print(f"  Quarters from clock resets (court setting): {len(inferred)} rows, "
                  f"quarters {sorted({r['quarter'] for r in inferred})}", flush=True)
            clean_records = inferred
    elif q_source != "digit" and n_clock and (len(clean_records) < 0.5 * n_clock or len(read_qs) < 4
                                              or not _quarters_balanced(clean_records)):
        inferred = [r for r in infer_quarters_from_clock(raw_records, game_format.load(game_path)) if not _in_skip(r["video_t"])]
        inf_qs = {r["quarter"] for r in inferred}
        # Prefer the clock resets when they find more quarters, or when the digit
        # reads are lopsided (e.g. a stylised '1st'/'2nd' read as '3': one
        # quarter held 70% of the game) and the resets give four fair quarters.
        if (len(inferred) > len(clean_records) and len(inf_qs & {1, 2, 3, 4}) > len(read_qs)) or (
                not _quarters_balanced(clean_records) and _quarters_balanced(inferred)):
            print(f"  Quarter digit unreliable ({len(clean_records)} clean of {n_clock} clock reads, "
                  f"quarters read {sorted(read_qs)}) — quarters inferred from clock resets: "
                  f"{len(inferred)} rows, quarters {sorted(inf_qs)}", flush=True)
            clean_records = inferred

    return clean_records


def _quarter_source_info(clean_records: list[dict[str, Any]], roi_data: dict[str, Any]) -> dict[str, Any]:
    """Where each row's quarter came from — for the UI ("Quarters: from …")."""
    from_clock = sum(1 for r in clean_records if r.get("quarter_source") == "clock")
    setting = str(roi_data.get("quarterSource") or "auto")
    if clean_records and from_clock >= 0.5 * len(clean_records):
        why = ("the court is set to 'Quarter from: clock resets'" if setting == "clock"
               else "the scoreboard's quarter digit was unreliable")
        return {"quarter_source": "clock", "quarter_source_reason": why}
    return {"quarter_source": "digit", "quarter_source_reason": "read from the scoreboard's quarter digit"}


def _quarters_balanced(recs: list[dict[str, Any]], min_share: float = 0.10) -> bool:
    """Do the four regulation quarters each hold a fair share of the game's
    rows? A real game splits roughly evenly; a misread quarter digit piles most
    rows on one quarter and leaves others with a handful."""
    counts = {q: 0 for q in (1, 2, 3, 4)}
    for r in recs:
        if r.get("quarter") in counts:
            counts[r["quarter"]] += 1
    total = sum(counts.values())
    return total > 0 and all(c >= min_share * total for c in counts.values())


def infer_quarters_from_clock(raw_records: list[dict[str, Any]], fmt: dict | None = None) -> list[dict[str, Any]]:
    """Clean records with the quarter worked out from the clock alone.

    The game clock restarts (jumps back up) at every period, and a real quarter
    runs SLOWER than real time (stoppages), while pre-game / halftime countdowns
    tick 1:1. So: split the clock reads into downward runs, keep the long runs
    that run slower than real time, and number them in order — letting a
    clearly-read quarter digit (a majority within the run) override the count.
    """
    fmt = fmt or game_format.DEFAULT
    q_ms, ot_ms = fmt["quarterMinutes"] * 60000, fmt["otMinutes"] * 60000
    pts = [(r["video_t"], ms, r) for r in raw_records
           if (ms := _clock_ms_any(r.get("clock"))) is not None]
    runs: list[list[tuple]] = []
    cur: list[tuple] = []
    i = 0
    while i < len(pts):
        t, ms, r = pts[i]
        if cur:
            pt, pms = cur[-1][0], cur[-1][1]
            # A new period starts at the full quarter clock (OT: its own length,
            # right after a period ended); any other upward jump is a misread
            # digit (e.g. a 3 read as 5).
            reset = ms >= q_ms - 60000 or (pms <= 30000 and ms >= ot_ms - 30000)
            # (A gap in readable frames with the clock still running down — a
            # timeout, a replay — is the same period: only a reset starts a new one.)
            if ms > pms + 20000 and reset:
                # A reset only counts when the next read agrees (else it's a misread).
                nxt = pts[i + 1] if i + 1 < len(pts) else None
                if nxt and abs(nxt[1] - ms) < 30000:
                    runs.append(cur)
                    cur = []
                else:
                    i += 1
                    continue
            elif ms > pms + 3000:
                i += 1  # upward blip that isn't a period reset → misread
                continue
        cur.append((t, ms, r))
        i += 1
    if cur:
        runs.append(cur)

    def is_quarter(run: list[tuple]) -> bool:
        clock_span = (run[0][1] - run[-1][1]) / 1000
        video_span = run[-1][0] - run[0][0]
        # enough reads to be real play (not a sparse countdown), ≥ 4 min of clock,
        # and slower than real time (stoppages) — countdowns tick 1:1
        # …and a scoreboard showing a score (pre-game countdowns often have none)
        scored = sum(1 for _, _, r in run if parse_score(r.get("home_score") or "") is not None)
        return (len(run) >= 30 and clock_span >= min(240, 0.4 * q_ms / 1000) and video_span >= 1.1 * clock_span
                and scored >= len(run) // 3)

    out: list[dict[str, Any]] = []
    q = 0
    for run in runs:
        if not is_quarter(run):
            continue
        votes = Counter(v for _, _, r in run if (v := parse_quarter(r.get("quarter") or "")) is not None)
        top, n = votes.most_common(1)[0] if votes else (None, 0)
        q = top if top is not None and top > q and n >= max(5, len(run) // 2) else q + 1
        for t, ms, r in run:
            secs = ms // 1000
            out.append({
                "video_t": t,
                "clock": f"{secs // 60}:{secs % 60:02d}" if ms >= 60000 else f"{secs}.{(ms % 1000) // 100}",
                "quarter": q,
                "home_score": parse_score(r.get("home_score") or ""),
                "away_score": parse_score(r.get("away_score") or ""),
                "confidence": 0.8,
                "quarter_source": "clock",
            })
    return out

# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def _median_filter(values: list[int], window: int) -> list[int]:
    """Sliding-window median over a list of ints (odd window)."""
    half = window // 2
    out: list[int] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(int(np.median(values[lo:hi])))
    return out


def _longest_nondecreasing(values: list[int]) -> list[bool]:
    """Mask of the longest non-decreasing subsequence of `values` (O(n log n)).

    A score's true reads form one long non-decreasing run; a misread that sticks
    for a while (e.g. "11" read as "41") is a short detour off it, so it's left
    out — where a running max would hold it until the real score caught up."""
    tails_val: list[int] = []   # smallest tail value of a run of each length
    tails_idx: list[int] = []
    prev = [-1] * len(values)
    for i, v in enumerate(values):
        k = bisect.bisect_right(tails_val, v)
        if k == len(tails_val):
            tails_val.append(v)
            tails_idx.append(i)
        else:
            tails_val[k] = v
            tails_idx[k] = i
        prev[i] = tails_idx[k - 1] if k > 0 else -1
    keep = [False] * len(values)
    i = tails_idx[-1] if tails_idx else -1
    while i >= 0:
        keep[i] = True
        i = prev[i]
    return keep


def _quarters_by_segment(records: list[dict[str, Any]]) -> list[int]:
    """Assign each record's quarter by segmenting the timeline at clock RESETS
    (a jump back to ~10:00 / ~5:00 after counting down) and taking the MODE of the
    per-frame quarter reads within each segment.

    Far more robust than a running-max clamp: a few stray quarter misreads (e.g. a
    lone '4' inside Q3) can't lock every later frame to the wrong quarter. A 5:00
    segment that is followed by a 10:00 quarter is an inter-period intermission
    countdown, not overtime, so it's merged into the previous period."""
    full10, full5 = 600_000, 300_000
    def is_full(c: int | None) -> bool:
        return c is not None and (abs(c - full10) <= 60_000 or abs(c - full5) <= 60_000)
    def which_full(c: int) -> int:
        return full10 if abs(c - full10) <= abs(c - full5) else full5

    n = len(records)
    clk = [_clock_to_ms(r.get("clock")) for r in records]
    seg_of = [0] * n
    seg = 0
    seg_full = [full10]
    minc: int | None = None
    started = False
    for i in range(n):
        c = clk[i]
        if not started:
            if c is not None and c >= 120_000:
                started = True; seg_full = [which_full(c)]; minc = c
            seg_of[i] = 0
            continue
        if is_full(c) and minc is not None and minc < 180_000:
            seg += 1; seg_full.append(which_full(c)); minc = c
        elif c is not None:
            minc = min(minc, c)
        seg_of[i] = seg

    nseg = seg + 1
    # Real periods only: drop a 5:00 segment that has a later 10:00 segment
    # (intermission countdown). Map every segment to a 1-based period number.
    realnum: dict[int, int] = {}
    p = 0
    for s in range(nseg):
        is_real = seg_full[s] == full10 or not any(seg_full[t] == full10 for t in range(s + 1, nseg))
        if is_real:
            p += 1
        realnum[s] = max(1, p)

    from collections import Counter
    votes: dict[int, Counter] = {}
    for i in range(n):
        q = records[i].get("quarter")
        if isinstance(q, int) and q >= 1:   # 0 = quarter unreadable here; don't vote
            votes.setdefault(realnum[seg_of[i]], Counter())[q] += 1
    maxp = max(realnum.values()) if realnum else 1
    period_q: dict[int, int] = {}
    prev = 0
    for per in range(1, maxp + 1):
        q = votes[per].most_common(1)[0][0] if votes.get(per) else prev + 1
        q = max(q, prev)   # non-decreasing across periods (segment level, not frame level)
        period_q[per] = q
        prev = q
    return [period_q[realnum[seg_of[i]]] for i in range(n)]


def smooth_records(records: list[dict[str, Any]], window: int = 7) -> list[dict[str, Any]]:
    """Clean single-frame OCR errors using temporal context.

    Because we sample every ~2 seconds the underlying signals are smooth:
      - quarter is a monotonic step function (1→2→3→4→OT)
      - each team's score is monotonically non-decreasing

    A sliding-window median kills lone spikes (e.g. a single 'Q4' read in the
    middle of Q1). Scores then keep only their longest non-decreasing run: a
    misread that outlasts the median (too high OR too low) is dropped and filled
    from the reads around it, instead of a running max locking it in.
    """
    if len(records) < 3:
        return records

    # 1) Quarter: segment by clock resets + mode per segment (robust to stray
    #    misreads that the old running-max clamp would lock in).
    quarters = _quarters_by_segment(records)

    # 2) Smooth + monotonic-fill each score channel
    def smooth_scores(key: str) -> list[int | None]:
        # Forward-fill Nones with last known value for the median window
        raw_vals = [r.get(key) for r in records]
        filled: list[int] = []
        last = 0
        for v in raw_vals:
            if v is not None:
                last = v
            filled.append(last)
        med = _median_filter(filled, window)
        # Scores never drop within a game: keep the longest non-decreasing run,
        # each dropped read takes the last kept value before it.
        keep = _longest_nondecreasing(med)
        last_kept = med[keep.index(True)]
        for i in range(len(med)):
            if keep[i]:
                last_kept = med[i]
            else:
                med[i] = last_kept
        # Re-mask leading region that was never observed (all None so far)
        out: list[int | None] = []
        seen = False
        for orig, m in zip(raw_vals, med):
            if orig is not None:
                seen = True
            out.append(m if seen else None)
        return out

    home = smooth_scores("home_score")
    away = smooth_scores("away_score")

    for i, r in enumerate(records):
        r["quarter"] = quarters[i]
        r["home_score"] = home[i]
        r["away_score"] = away[i]
    return records


# ---------------------------------------------------------------------------
# Adaptive break-skipping (pre-game + halftime)
# ---------------------------------------------------------------------------
# The scoreboard shows a real-time countdown during the long breaks: a ~30 min
# pre-game warm-up clock and a ~15 min halftime clock, both counting DOWN to
# 0:00, at which point the quarter clock resets UP to 10:00 (the tip-off). A
# real quarter clock is never above 10:00, so:
#   - clock > 10:30  unambiguously marks a long-break countdown (engages skip);
#   - from any countdown frame reading R remaining, the tip is at ~now + R, so we
#     JUMP forward by R (rule 2);
#   - after the jump we check whether the clock RESET up to ~10:00 — i.e. we
#     crossed the tip (rule 1) — and if so BISECT back (halve) to land on the
#     reset edge (rule 3); otherwise we re-predict from the new reading.
# Short inter-quarter breaks (countdown ≤ ~3 min, never above 10:00) and games
# with no readable countdown both fall through to normal dense sampling.
ADAPT_FULL_MS = 600_000        # a real quarter clock tops out here (10:00)
ADAPT_BREAK_MS = 630_000       # clock above this (10:30) ⇒ long-break countdown
ADAPT_RESET_LO_MS = 540_000    # a post-tip reset clock sits in [9:00, 10:30]
ADAPT_LOW_MS = 6_000           # countdown reading ≤6s ⇒ it has bottomed out
ADAPT_CONVERGE_S = 2.0         # stop bisecting once the bracket is this tight
ADAPT_COARSE_STEP_S = 30.0     # play-scan stride + sparse-marker spacing in breaks
ADAPT_MAX_JUMPS = 30           # cap predict-and-jump iterations per break
ADAPT_PLATEAU_MAX_S = 1_200    # walk a bottomed-out countdown at most this long


def _clock_ms_raw(clock_str: str | None) -> int | None:
    """Parse an OCR clock string to milliseconds WITHOUT the 12-minute quarter
    cap (parse_clock rejects break countdowns like '30:33'). Used only by the
    break finder, which needs the raw countdown value."""
    if not clock_str:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", fix_common(clock_str))
    if not m:
        return None
    mins, secs = int(m.group(1)), int(m.group(2))
    if secs > 59:
        return None
    return mins * 60000 + secs * 1000


class _Seeker:
    """Single-capture sequential reader for the adaptive planning pass."""

    def __init__(self, video: Path, rois: dict[str, Any], src_w: int, src_h: int):
        self.cap = open_capture(video)
        self.rois, self.src_w, self.src_h = rois, src_w, src_h
        self.reads: list[dict[str, Any]] = []   # every raw read (sparse markers)
        self.cleans: list[dict[str, Any]] = []

    def read(self, t: float) -> int | None:
        """OCR the frame at t; record raw (+clean if parseable); return clock_ms."""
        frame = read_frame_at(self.cap, t)
        if frame is None:
            return None
        raw, clean = process_frame(frame, self.rois, self.src_w, self.src_h)
        raw["video_t"] = round(t, 3)
        self.reads.append(raw)
        if clean is not None:
            clean["video_t"] = round(t, 3)
            self.cleans.append(clean)
        return _clock_ms_raw(raw.get("clock"))

    def close(self) -> None:
        self.cap.release()


def _find_break_tip(sk: _Seeker, t0: float, c0: int, duration: float) -> float | None:
    """Given a countdown frame (t0, c0=remaining ms), find the quarter tip-off.

    Phase 1 (rules 1+2): jump forward by the remaining countdown; if the clock
    reset up to ~10:00 we've crossed the tip. Some broadcasts park the clock at
    0:00 for minutes after the countdown bottoms out before it resets — Phase 1b
    walks that low plateau at a coarse stride until the reset appears. Phase 2
    (rule 3): bisect the bracket to land exactly on the reset edge. Returns the
    tip video_t, or None if no reset is found (caller then samples normally)."""
    t_before, c_before = t0, c0      # last frame known to be PRE-RESET (in break)
    t_after: float | None = None     # first frame known to be POST-RESET (tip+)

    # Phase 1 — predict-and-jump while the countdown is clearly running.
    for _ in range(ADAPT_MAX_JUMPS):
        if c_before <= ADAPT_LOW_MS:
            break                    # bottomed out → hand off to the plateau walk
        t_try = t_before + c_before / 1000.0     # rule 2: tip ≈ now + remaining
        if t_try >= duration - 1:
            return None
        c = sk.read(t_try)
        if c is not None and c >= ADAPT_RESET_LO_MS:
            t_after = t_try; break   # clock reset up → crossed the tip
        if c is None or c >= c_before:
            c_before = ADAPT_LOW_MS  # transient miss / non-decreasing → walk it
        else:
            c_before = c
        t_before = t_try

    # Phase 1b — walk a bottomed-out / parked countdown until it resets.
    if t_after is None:
        walked = 0.0
        t = t_before
        while walked < ADAPT_PLATEAU_MAX_S:
            t += ADAPT_COARSE_STEP_S
            walked += ADAPT_COARSE_STEP_S
            if t >= duration - 1:
                return None
            c = sk.read(t)
            if c is not None and c >= ADAPT_RESET_LO_MS:
                t_after = t; break   # reset appeared
            if c is None or c <= ADAPT_FULL_MS:
                t_before = t         # still parked / pre-reset
        if t_after is None:
            return None              # never reset → don't skip this break

    # Phase 2 — bisect [t_before, t_after] onto the reset edge (rule 3).
    while t_after - t_before > ADAPT_CONVERGE_S:
        mid = (t_before + t_after) / 2.0
        c = sk.read(mid)
        if c is not None and c >= ADAPT_RESET_LO_MS:
            t_after = mid            # already reset by mid → tip earlier
        else:
            t_before = mid           # still pre-reset at mid → tip later
    return t_after


def _clock_to_ms(s: str | None) -> int | None:
    """Clean OCR clock string ('M:SS' or 'SS.t') → milliseconds."""
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60000 + int(m.group(2)) * 1000
    m = re.match(r"^(\d{1,2})\.(\d)$", s)
    if m:
        return int(m.group(1)) * 1000 + int(m.group(2)) * 100
    return None


def plan_skip_regions(sk: _Seeker, duration: float, progress: bool = False) -> list[tuple[float, float]]:
    """Coarse sequential scan that fast-forwards long-break countdowns.

    Steps through the video at a coarse stride; whenever the clock reads above
    10:30 (a pre-game / halftime countdown) it locates the following tip-off and
    records [break_start, tip-safety] as a region to sample only sparsely."""
    regions: list[tuple[float, float]] = []
    t = 0.0
    guard = 0
    while t < duration and guard < 2000:
        guard += 1
        if progress:
            print(f"PROGRESS {int(min(t, duration))}/{int(duration)}", flush=True)
        c = sk.read(t)
        if c is not None and c > ADAPT_BREAK_MS:
            tip = _find_break_tip(sk, t, c, duration)
            if tip is None:
                # Couldn't locate the tip — DON'T skip (we'd risk dropping play).
                # Scan past the predicted countdown end and keep dense sampling.
                t += c / 1000.0 + ADAPT_COARSE_STEP_S
                continue
            # Skip up to the tip exactly: the tip is the reset edge (clock →
            # 10:00 = true play start), so [start, tip) is all dead break time.
            # Resuming before the tip would expose the countdown's final
            # seconds, which OCR mislabels with the outgoing quarter.
            regions.append((t, max(t, tip)))
            t = tip + ADAPT_COARSE_STEP_S
            continue
        t += ADAPT_COARSE_STEP_S
    return regions


# ---------------------------------------------------------------------------
# Locate video
# ---------------------------------------------------------------------------

def load_marker_bounds(game_dir: Path) -> tuple[tuple[float, float] | None, list[tuple[float, float]]]:
    """Derive OCR scan bounds + hard-skip spans from hand-set markers.json.

    Returns (bounds, hard_skips):
      • bounds = (scan_start, scan_end) — clamp the sample grid to real play so we
        don't OCR the pre-game wait (before the first tip) or the post-game tail
        (after the last quarter ends). A small margin is kept on each side.
      • hard_skips = long-break spans (halftime/pre-game) to jump over entirely —
        no sampling at all inside them.
    Missing/empty markers → (None, []) so the caller falls back to dense + the
    adaptive break-skip pass.
    """
    path = game_dir / "markers.json"
    try:
        markers = json.loads(path.read_text()).get("markers", [])
    except (OSError, ValueError):
        return None, []
    if not markers:
        return None, []
    MARGIN_S = 30.0

    def is_endgame(m: dict) -> bool:
        return str(m.get("note", "")).lower().startswith("end")

    q_starts = [float(m["start_t"]) for m in markers if m.get("type") == "quarter" and m.get("start_t") is not None]
    q_ends = [float(m["end_t"]) for m in markers if m.get("type") == "quarter" and m.get("end_t") is not None]
    # Play ends at the latest quarter 0:00 OR — covering an unmarked overtime — at
    # the start of the "end-game" marker. Scanning up to there means a late OT
    # isn't clipped off even if it wasn't captured as a quarter.
    endgame_starts = [float(m["start_t"]) for m in markers
                      if m.get("type") == "longbreak" and is_endgame(m) and m.get("start_t") is not None]
    # Hard-skip the mid-game long breaks (pre-game / half-time); NOT the end-game
    # span (it sits past scan_end anyway, and skipping it must not clip an OT).
    hard_skips = [(float(m["start_t"]), float(m["end_t"]))
                  for m in markers if m.get("type") == "longbreak" and not is_endgame(m)
                  and m.get("start_t") is not None and m.get("end_t") is not None]
    bounds = None
    end_candidates = q_ends + endgame_starts
    if q_starts or end_candidates:
        lo = max(0.0, (min(q_starts) - MARGIN_S)) if q_starts else 0.0
        hi = (max(end_candidates) + MARGIN_S) if end_candidates else float("inf")
        bounds = (lo, hi)
    return bounds, hard_skips


def locate_video(game_path: Path) -> Path | None:
    game_dir = game_path.parent
    if game_path.exists():
        try:
            g = json.loads(game_path.read_text())
            lf = g.get("video", {}).get("localFile")
            if lf:
                cand = game_dir / lf
                if cand.exists():
                    return cand
        except Exception:
            pass
    fallback = game_dir / "raw" / "game.mp4"
    return fallback if fallback.exists() else None


# ---------------------------------------------------------------------------
# Parallel chunk worker
# ---------------------------------------------------------------------------

def _process_chunk(task: tuple[int, list[float], str, int, int, dict[str, Any], bool, str, str]):
    """Process a contiguous block of timestamps in a worker process.

    Each worker opens its own VideoCapture (forward seeks within the block are
    cheap) and OCRs every frame. Returns (chunk_index, raw_records, clean_records).
    """
    idx, timestamps, video_path, src_w, src_h, rois, grayscale, method, quarter_method = task
    global _GRAYSCALE, _METHOD, _QUARTER_METHOD
    _GRAYSCALE = grayscale
    _METHOD = method
    _QUARTER_METHOD = quarter_method
    cv2.setNumThreads(0)
    cap = cv2.VideoCapture(video_path)
    raws: list[dict[str, Any]] = []
    cleans: list[dict[str, Any]] = []
    # Per-chunk change-gating cache (frames here are processed in time order).
    gate_state: dict[str, Any] = {}
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            raws.append({"video_t": t, "error": "seek_failed"})
            continue
        raw, clean = process_frame(frame, rois, src_w, src_h, gate_state=gate_state)
        raw["video_t"] = t
        raws.append(raw)
        if clean is not None:
            clean["video_t"] = t
            cleans.append(clean)
    cap.release()
    return idx, raws, cleans


# ---------------------------------------------------------------------------
# Keyframe decoder (ffmpeg -skip_frame nokey) — streams only I-frames
# sequentially instead of a per-sample random seek. ~3x faster decode and far
# lower CPU when the sample interval ≈ the GOP (keyframes land on the grid).
# ---------------------------------------------------------------------------

def probe_keyframes(video: str | Path) -> list[float]:
    """Return sorted keyframe timestamps (s) via a demux-only ffprobe (no decode)."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(video)],
        text=True,
    )
    ts: list[float] = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) >= 2 and parts[0] and "K" in parts[1]:
            try:
                ts.append(round(float(parts[0]), 3))
            except ValueError:
                pass
    return sorted(ts)


def probe_video_dims(video: str | Path) -> tuple[int, int]:
    # key=value output is unambiguous — over HLS the csv form can reorder or emit
    # multiple stream rows, so parse by key and take the first width/height.
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "default=noprint_wrappers=1", str(video)],
        text=True,
    )
    w = h = None
    for line in out.splitlines():
        if line.startswith("width=") and w is None:
            w = int(line.split("=", 1)[1])
        elif line.startswith("height=") and h is None:
            h = int(line.split("=", 1)[1])
    if w is None or h is None:
        raise ValueError(f"could not probe video dimensions: {out!r}")
    return w, h


# ---------------------------------------------------------------------------
# Cloud mode — read the game video directly from its HLS URL (no download).
# ---------------------------------------------------------------------------

def _fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "playvizio-ocr"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted source URL)
        return r.read().decode("utf-8", "replace")


def keyframes_from_m3u8(url: str) -> list[float]:
    """Derive keyframe timestamps from an HLS media playlist WITHOUT downloading
    the media: HLS segments are keyframe-aligned, so each segment's start time
    (cumulative #EXTINF durations) is a keyframe timestamp. Follows a master
    playlist to its first variant."""
    txt = _fetch_text(url)
    lines = txt.splitlines()
    if any(l.startswith("#EXT-X-STREAM-INF") for l in lines):
        for i, l in enumerate(lines):
            if l.startswith("#EXT-X-STREAM-INF") and i + 1 < len(lines):
                variant = lines[i + 1].strip()
                return keyframes_from_m3u8(urllib.parse.urljoin(url, variant))
    ts: list[float] = []
    t = 0.0
    for l in lines:
        if l.startswith("#EXTINF:"):
            try:
                dur = float(l[len("#EXTINF:"):].split(",")[0])
            except ValueError:
                continue
            ts.append(round(t, 3))
            t += dur
    return ts


def resolve_video_source(game_path: Path) -> tuple[str, bool]:
    """Return (ffmpeg_input, is_remote). Cloud games (game.json cloudMode=true,
    with a video URL) read the HLS URL directly; otherwise the local game.mp4.
    Falls back to the URL if there's no local file."""
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


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a pipe (pipes may return short reads)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _process_chunk_keyframe(task: tuple) -> tuple[int, list, list]:
    """Keyframe counterpart to _process_chunk: stream the I-frames in a time range
    via one ffmpeg process and OCR each, tagging it with its true keyframe
    timestamp. `kf_range` is the full (unstrided) keyframe-timestamp list for the
    range — emitted frames align positionally with it; `stride`/`gstart` select a
    coarser grid when the sample interval exceeds the GOP."""
    (idx, rstart, rlen, kf_range, gstart, stride,
     video_path, src_w, src_h, rois, grayscale, method, vw, vh, is_remote, quarter_method, fps_step) = task
    global _GRAYSCALE, _METHOD, _QUARTER_METHOD
    _GRAYSCALE = grayscale
    _METHOD = method
    _QUARTER_METHOD = quarter_method
    cv2.setNumThreads(0)
    raws: list[dict[str, Any]] = []
    cleans: list[dict[str, Any]] = []
    gate_state: dict[str, Any] = {}
    # For remote (HLS) input, auto-reconnect on transient network/segment errors.
    reconnect = (["-reconnect", "1", "-reconnect_streamed", "1",
                  "-reconnect_delay_max", "5"] if is_remote else [])
    if fps_step:
        # Stream mode: decode every frame in order, emit one per fps_step s
        # (frame j ≈ rstart + j·fps_step, aligned with kf_range).
        cmd = ["ffmpeg", "-nostdin", "-v", "error", *reconnect, "-rw_timeout", "60000000",
               "-ss", f"{rstart:.3f}", "-i", video_path, "-t", f"{rlen:.3f}", "-an", "-sn",
               "-vf", f"fps=1/{fps_step:g}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    else:
        cmd = ["ffmpeg", "-nostdin", "-v", "error", *reconnect,
               "-ss", f"{rstart:.3f}", "-skip_frame", "nokey",
               "-i", video_path, "-t", f"{rlen:.3f}", "-an", "-sn", "-vsync", "0",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsize = vw * vh * 3
    try:
        j = 0
        while j < len(kf_range):
            buf = _read_exact(proc.stdout, fsize)
            if len(buf) < fsize:
                break
            t = kf_range[j]
            if (gstart + j) % stride == 0:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(vh, vw, 3)
                raw, clean = process_frame(frame, rois, src_w, src_h, gate_state=gate_state)
                raw["video_t"] = t
                raws.append(raw)
                if clean is not None:
                    clean["video_t"] = t
                    cleans.append(clean)
            j += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    return idx, raws, cleans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _cleanup_segments_if_readable(
    video: Path, game_dir: Path, rois: dict[str, Any], src_w: int, src_h: int, duration: float,
) -> None:
    """If a probe frame from game.mp4 OCRs successfully, delete raw/segments/."""
    seg_dir = game_dir / "raw" / "segments"
    if not seg_dir.is_dir():
        return
    cap = open_capture(video)
    try:
        frame = read_frame_at(cap, max(1.0, duration * 0.3))
    finally:
        cap.release()
    if frame is None:
        print("Probe frame unreadable — keeping raw/segments.", flush=True)
        return
    raw, _clean = process_frame(frame, rois, src_w, src_h)
    if not any(raw.get(f) for f in ("clock", "quarter", "home_score", "away_score")):
        print("Probe frame produced no OCR read — keeping raw/segments.", flush=True)
        return
    try:
        shutil.rmtree(seg_dir)
        # The ffmpeg concat list points at the now-deleted .ts files — drop it too.
        (game_dir / "raw" / "concat.txt").unlink(missing_ok=True)
        print(f"OCR read OK from game.mp4 — deleted {seg_dir} (.ts segments)", flush=True)
    except OSError as e:
        print(f"Could not delete {seg_dir}: {e}", flush=True)


def _is_period_tip(ms: int | None) -> bool:
    """A fresh period clock: ~10:00 (quarter) or ~5:00 (overtime), ±1:00."""
    return ms is not None and (abs(ms - 600000) <= 60000 or abs(ms - 300000) <= 60000)


def _probe_next_periods(sk: _Seeker, duration: float) -> None:
    """After the coarse scan, catch periods the stride jumped over (esp. OVERTIME).

    Each period ends at 0:00; the next one (quarter or OT) tips off after a short
    break. From every detected 0:00 read that ISN'T already followed by a captured
    tip, step forward ~1 min looking for a fresh period clock (10:00 or 5:00). Once
    found, sample that period down to 0:00. Reads are recorded into the seeker, so
    they flow into break_plan.json and let the ramp detector see the new period."""
    LOW_MS, STEP = 5000, 60.0
    # A period end = a clean ~0:00 read. (Explicit None check — clock 0 is falsy.)
    ends = sorted({round(c["video_t"]) for c in sk.cleans
                   if (m := _clock_to_ms(c.get("clock"))) is not None and m <= LOW_MS})
    for e in ends:
        # Already have a tip within ~3 min after this 0:00? Then the next period
        # was captured — nothing to probe.
        if any(_is_period_tip(_clock_to_ms(c.get("clock"))) and 0 < c["video_t"] - e < 180 for c in sk.cleans):
            continue
        t = float(e)
        for _ in range(5):                       # probe up to ~5 min forward
            t += STEP
            if t >= duration - 1:
                break
            ms = sk.read(t)
            if ms is not None and ms > 630000:   # a long-break countdown, not a tip
                break
            if _is_period_tip(ms):
                print(f"  Late-period probe: found a tip at {int(t)}s (clock {ms}ms) — sampling it.", flush=True)
                tt = t
                for _ in range(40):              # sample the new period down to 0:00
                    tt += 30
                    if tt >= duration - 1:
                        break
                    m2 = sk.read(tt)
                    if m2 is not None and m2 <= 3000:
                        break
                break


def run(game_path: Path, fps: float, workers: int = 0, skip_breaks: bool = True,
        decoder: str = "seek", derived_dir: Path | None = None,
        use_markers: bool = True, plan_only: bool = False,
        shard: tuple[int, int] | None = None, merge_shards: list[Path] | None = None) -> None:
    """Full-game OCR into <derived>/. To split one game across machines, run it
    once per shard with ``shard=(i, n)`` (each reads every n-th chunk and writes
    ``shard-<i>.json``), then once with ``merge_shards=[...]`` to assemble and
    clean exactly as a single run would. Every call must use the same fps,
    workers and decoder, so all of them cut the video into the same chunks."""
    game_dir = game_path.parent
    # Outputs normally land in <game>/derived; a comparison run can redirect them
    # to a scratch dir so the game's real timeline is never clobbered.
    derived = derived_dir or (game_dir / "derived")
    derived.mkdir(parents=True, exist_ok=True)

    # Resolve the ffmpeg input: a local game.mp4, or (cloud mode) the HLS URL,
    # which ffmpeg reads directly via HTTP — no 6 GB download to disk.
    video, is_remote = resolve_video_source(game_path)
    if not video:
        sys.exit(f"Video not found for {game_dir} (no local file and no URL)")
    video_label = video.rsplit("/", 1)[-1].split("?", 1)[0] or video
    if is_remote:
        print(f"  Cloud mode: reading video from URL (no download) — {video_label}", flush=True)

    roi_data = load_roi(game_dir)
    rois = roi_data.get("rois", {})
    if not rois:
        sys.exit("roi.json has no 'rois' key")
    src_w, src_h = get_frame_size(roi_data)

    # Per-game OCR options (game.json: ocrGrayscale, ocrMethod, ocrQuarterMethod).
    global _GRAYSCALE, _METHOD, _QUARTER_METHOD
    try:
        _gj = json.loads(game_path.read_text())
    except Exception:
        _gj = {}
    _GRAYSCALE = bool(_gj.get("ocrGrayscale"))
    _METHOD = _gj.get("ocrMethod") or "tesseract"
    _QUARTER_METHOD = _gj.get("ocrQuarterMethod") or ""
    # Per-game break-skip override (game.json: ocrSkipBreaks 0/1). Break-skipping can
    # wrongly drop long stretches when the mid-game clock reads poorly — set 0 to
    # sample the full video.
    if "ocrSkipBreaks" in _gj and _gj.get("ocrSkipBreaks") is not None:
        skip_breaks = bool(_gj.get("ocrSkipBreaks"))
    # Remote input: OpenCV per-sample seeking over HTTP is unreliable, so cloud
    # mode always uses the keyframe (ffmpeg) decoder. Break-skip planning is
    # OpenCV-seek based too, so it's disabled remotely (a follow-up can plan from
    # the playlist). Both are no-ops for local games.
    if is_remote:
        if decoder != "keyframe":
            decoder = "keyframe"
        if skip_breaks:
            skip_breaks = False
            print("  Cloud mode: keyframe decoder, break-skipping off", flush=True)
    if _GRAYSCALE:
        print("  Grayscale OCR mode: ON", flush=True)
    if _METHOD != "tesseract":
        print(f"  OCR method: {_METHOD}", flush=True)
    print(f"  Break-skipping: {'ON' if skip_breaks else 'OFF (full video)'}", flush=True)
    if _QUARTER_METHOD and _QUARTER_METHOD != _METHOD:
        print(f"  Quarter OCR override: {_QUARTER_METHOD}", flush=True)

    duration = get_duration_ffprobe(video)

    # ── Plan-only (Step 2a "Find breaks"): coarse pass only. Detects the
    # pre-game/halftime breaks + collects sparse clock samples, writes
    # derived/break_plan.json, and returns WITHOUT the dense OCR. The harness then
    # turns this into quarter + long-break markers to bound the dense pass (2b).
    if plan_only:
        print(f"Break plan (coarse): {video_label} ({duration:.0f}s)", flush=True)
        sk = _Seeker(video, rois, src_w, src_h)
        try:
            regions = plan_skip_regions(sk, duration, progress=True)
            _probe_next_periods(sk, duration)   # catch OT the coarse stride skipped
            # Build samples from RAW reads (clock only) — the ramp detector ignores
            # the quarter label, and an unreadable quarter (overtime shows "E")
            # must NOT drop an otherwise-good clock reading like the OT 5:00 tip.
            by_t: dict[float, list[Any]] = {}
            for r in sk.reads:
                clk = parse_clock(r.get("clock") or "")
                ms = _clock_to_ms(clk) if clk else None
                if ms is None:
                    continue
                by_t[round(float(r["video_t"]), 2)] = [r["video_t"], parse_quarter(r.get("quarter") or "") or 0, ms]
            samples = [by_t[k] for k in sorted(by_t)]
        finally:
            sk.close()
        (derived / "break_plan.json").write_text(json.dumps({
            "skip_regions": [[round(a, 2), round(b, 2)] for a, b in regions],
            "samples": samples,
            "duration": round(duration, 2),
        }, indent=2))
        print(f"Break plan: {len(regions)} long break(s), {len(samples)} sparse samples → derived/break_plan.json", flush=True)
        print("PROGRESS 1/1", flush=True)
        return

    # Once we can read a frame off the concatenated game.mp4, the intermediate
    # HLS segments (raw/segments/*.ts, ~GBs) are redundant — drop them to save
    # disk. Gated on a successful read. Local games only (nothing to clean remotely).
    if not is_remote:
        _cleanup_segments_if_readable(Path(video), game_dir, rois, src_w, src_h, duration)

    step = 1.0 / fps
    all_ts = [round(i * step, 3) for i in range(int(duration / step))]

    # The keyframe decoder can only sample at keyframe boundaries — it can't go
    # denser than the GOP. Probe the keyframe cadence up front; if the requested
    # interval is finer than the GOP, fall back to the seek decoder so we still
    # honour the sample rate.
    kf_all: list[float] = []
    kf_dt = step
    fps_step = 0.0   # >0: stream mode — decode the stream in order, one frame every fps_step s
    if decoder == "keyframe":
        # Remote HLS: derive keyframe times from the playlist (segments are
        # keyframe-aligned) — no demux of the whole stream. Local: ffprobe packets.
        if is_remote and ".m3u8" in video:
            kf_all = keyframes_from_m3u8(video)
        else:
            kf_all = probe_keyframes(video)
        if not kf_all:
            print("  keyframe: no keyframes found — falling back to seek decoder", flush=True)
            decoder = "seek"
        else:
            diffs = [kf_all[i] - kf_all[i - 1] for i in range(1, len(kf_all))]
            kf_dt = sorted(diffs)[len(diffs) // 2] if diffs else step
            if step < kf_dt * 0.9 and is_remote:
                # Remote: never fall back to per-sample OpenCV seeks over HLS
                # (one network reader per worker, each re-fetching segments —
                # they starve and time out on a slow line). Stream each range in
                # order instead and take a frame every `step` seconds.
                print(f"  stream: sample interval {step:.2f}s finer than GOP {kf_dt:.2f}s — "
                      f"decoding the stream in order (no seeks)", flush=True)
                fps_step = step
                kf_all = [round(i * step, 3) for i in range(int(duration / step))]
                kf_dt = step
            elif step < kf_dt * 0.9:
                print(f"  keyframe: sample interval {step:.2f}s finer than GOP "
                      f"{kf_dt:.2f}s — falling back to seek decoder", flush=True)
                decoder = "seek"

    # Adaptive break-skipping: a cheap sequential pass finds long-break countdowns
    # (pre-game, halftime) and the tip-offs that end them. For the seek decoder we
    # sample those dead spans only sparsely; for the keyframe decoder we still read
    # them (cheap) but the shared pass below drops their clean records, so both
    # decoders produce the same break-free clean timeline.
    # Hand-set markers (Sync-Test "OCR Pipeline" Step 1) bound the scan to real
    # play: start at the first tip-off (skip the pre-game wait), stop at the last
    # quarter's end (skip the post-game tail), and jump over long breaks entirely.
    marker_skips: list[tuple[float, float]] = []
    if use_markers:
        bounds, marker_skips = load_marker_bounds(game_dir)
        if bounds is not None:
            lo, hi = bounds
            before = len(all_ts)
            all_ts = [t for t in all_ts if lo <= t <= hi]
            _hms = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"  # noqa: E731
            print(f"  Marker bounds: scan {_hms(lo)}–{_hms(hi) if hi != float('inf') else 'end'} "
                  f"({before - len(all_ts)} pre/post frames skipped)", flush=True)
        if marker_skips:
            print(f"  Marker long-break skips: {[(round(a), round(b)) for a, b in marker_skips]}", flush=True)

    # Adaptive break-skipping: a cheap sequential pass finds long-break
    # countdowns (pre-game, halftime) and the tip-offs that end them, so we
    # sample those dead spans only sparsely instead of densely.
    skip_regions: list[tuple[float, float]] = list(marker_skips)
    seq_raws: list[dict[str, Any]] = []
    seq_cleans: list[dict[str, Any]] = []
    if skip_breaks:
        try:
            sk = _Seeker(video, rois, src_w, src_h)
            skip_regions = sorted(marker_skips + plan_skip_regions(sk, duration))
            seq_raws, seq_cleans = sk.reads, sk.cleans
            sk.close()
        except Exception as e:   # never let planning break the OCR run
            print(f"Break-skip planning failed ({e}) — sampling full video.", flush=True)
            skip_regions = list(marker_skips)
        if skip_regions:
            skipped_s = sum(b - a for a, b in skip_regions)
            print(
                f"  Skipping {len(skip_regions)} break(s), {skipped_s:.0f}s "
                f"(sparse-only): {[(round(a), round(b)) for a, b in skip_regions]}",
                flush=True,
            )

    def _in_skip(t: float) -> bool:
        return any(a <= t < b for a, b in skip_regions)

    n_workers = workers if workers > 0 else (os.cpu_count() or 4)
    video_str = video  # already a local absolute path or an HLS URL (str)

    raw_by_idx: dict[int, list[dict[str, Any]]] = {}
    clean_all: list[dict[str, Any]] = []
    done = 0

    if decoder == "keyframe":
        # kf_all / kf_dt were probed above (feasibility check). `stride` picks a
        # coarser grid when the sample interval is a multiple of the GOP.
        vw, vh = probe_video_dims(video)
        stride = max(1, round(step / kf_dt)) if kf_dt > 0 else 1
        total = len(range(0, len(kf_all), stride))
        n_workers = max(1, min(n_workers, len(kf_all)))
        if is_remote and workers <= 0:
            n_workers = min(n_workers, 4)   # auto: parallel downloads only split the bandwidth
        print(f"OCR (keyframe): {video_label}  {len(kf_all)} keyframes, "
              f"GOP~{kf_dt:.2f}s, stride {stride} → {total} frames  ({duration:.0f}s)", flush=True)
        print(f"  ROI fields: {list(rois.keys())}", flush=True)
        print(f"  Workers: {n_workers}", flush=True)
        chunk_kf = max(20, len(kf_all) // (n_workers * 8) or 1)
        tasks: list[tuple] = []
        for a in range(0, len(kf_all), chunk_kf):
            b = min(a + chunk_kf, len(kf_all))
            kf_range = kf_all[a:b]
            rstart = kf_range[0] if fps_step else max(0.0, kf_range[0] - 0.1)
            rlen = (kf_range[-1] - kf_range[0]) + (fps_step if fps_step else 0.2)
            tasks.append((a, rstart, rlen, kf_range, a, stride, video_str,
                          src_w, src_h, rois, _GRAYSCALE, _METHOD, vw, vh, is_remote, _QUARTER_METHOD, fps_step))
        worker = _process_chunk_keyframe
    else:
        # Dense grid everywhere except inside skip regions (those keep the sparse
        # sequential reads collected during planning).
        timestamps = [t for t in all_ts if not _in_skip(t)]
        total = len(timestamps)
        n_workers = max(1, min(n_workers, total))
        print(f"OCR (Tesseract): {video_label}  {total} frames @ {fps} fps  ({duration:.0f}s)", flush=True)
        print(f"  ROI fields: {list(rois.keys())}", flush=True)
        print(f"  Workers: {n_workers}", flush=True)
        # Split timestamps into contiguous chunks (smaller than per-worker so we get
        # finer-grained progress updates than just 1/worker).
        chunk_size = max(20, total // (n_workers * 8) or 1)
        tasks = []
        for start in range(0, total, chunk_size):
            tasks.append((start, timestamps[start:start + chunk_size], video_str, src_w, src_h, rois, _GRAYSCALE, _METHOD, _QUARTER_METHOD))
        worker = _process_chunk

    run_tasks = tasks
    if shard:
        i, n = shard
        run_tasks = tasks[i::n]
        total_here = (sum(len(range(0, len(t[3]), t[5])) for t in run_tasks) if decoder == "keyframe"
                      else sum(len(t[1]) for t in run_tasks))
        print(f"  Shard {i + 1}/{n}: {len(run_tasks)} of {len(tasks)} chunks", flush=True)
    else:
        total_here = total

    if not merge_shards:
        # Report the size of this run at once, so a caller knows its total before
        # the first chunk finishes (with shards: every shard counts from the start).
        print(f"PROGRESS 0/{total_here}", flush=True)

    if merge_shards:
        for f in merge_shards:
            d = json.loads(Path(f).read_text())
            for k, v in d["raw_by_idx"].items():
                raw_by_idx[int(k)] = v
            clean_all.extend(d["clean"])
        missing = sorted({t[0] for t in tasks} - set(raw_by_idx))
        if missing:
            sys.exit(f"Shards are missing {len(missing)} of {len(tasks)} chunks — were they run with the same fps/workers/decoder?")
        print(f"  Merged {len(merge_shards)} shard(s): {len(raw_by_idx)} chunks", flush=True)
    elif n_workers == 1:
        # Sequential fallback (avoids pool overhead for tiny jobs)
        for task in run_tasks:
            idx, raws, cleans = worker(task)
            raw_by_idx[idx] = raws
            clean_all.extend(cleans)
            done += len(raws)
            print(f"PROGRESS {done}/{total_here}", flush=True)
    else:
        with mp.Pool(n_workers) as pool:
            for idx, raws, cleans in pool.imap_unordered(worker, run_tasks):
                raw_by_idx[idx] = raws
                clean_all.extend(cleans)
                done += len(raws)
                print(f"PROGRESS {done}/{total_here}", flush=True)

    if shard:
        out = derived / f"shard-{shard[0]}.json"
        out.write_text(json.dumps({"raw_by_idx": raw_by_idx, "clean": clean_all}))
        print(f"Wrote {out} ({done} frames) — merge all shards with --merge-shards", flush=True)
        return

    # Reassemble raw records in chronological order, merging the sparse
    # sequential reads taken inside skipped break regions (dedup by video_t,
    # dense grid wins on collision).
    dense_raw: list[dict[str, Any]] = []
    for start in sorted(raw_by_idx):
        dense_raw.extend(raw_by_idx[start])
    raw_by_t: dict[float, dict[str, Any]] = {}
    for r in seq_raws:
        raw_by_t[r["video_t"]] = r
    for r in dense_raw:
        raw_by_t[r["video_t"]] = r
    raw_records = [raw_by_t[t] for t in sorted(raw_by_t)]

    # Clean timeline drives the sync, so it must contain only real play. A few
    # break-countdown frames parse as "clean" (e.g. a halftime 0:13 read as
    # Q2), which would wrongly extend a quarter across the break — so drop any
    # clean record inside a skip region. The sparse markers live on in raw.
    clean_by_t: dict[float, dict[str, Any]] = {}
    for r in seq_cleans:
        if not _in_skip(r["video_t"]):
            clean_by_t[r["video_t"]] = r
    for r in clean_all:
        if not _in_skip(r["video_t"]):
            clean_by_t[r["video_t"]] = r
    clean_records = [clean_by_t[t] for t in sorted(clean_by_t)]

    clean_records = _choose_quarter_source(raw_records, clean_records, roi_data, game_path, _in_skip)

    # Temporal smoothing pass (quarter median + monotonic, score monotonic)
    clean_records = smooth_records(clean_records)

    # Write outputs
    (derived / "timeline_raw.json").write_text(json.dumps(raw_records, indent=2))
    (derived / "timeline.json").write_text(json.dumps(clean_records, indent=2))

    csv_path = derived / "ocr_timeline.csv"
    rows = ["video_time_seconds,quarter,game_clock,home_score,away_score,confidence"]
    for r in clean_records:
        home = r.get("home_score")
        away = r.get("away_score")
        rows.append(
            f"{r['video_t']},{r['quarter']},{r['clock']},"
            f"{'' if home is None else home},{'' if away is None else away},{r['confidence']}"
        )
    csv_path.write_text("\n".join(rows) + "\n")

    hit_rate = len(clean_records) / total * 100 if total else 0
    print(f"Wrote CSV: {csv_path} ({len(clean_records)}/{total} frames, {hit_rate:.1f}% hit rate)", flush=True)

    # Coverage report — which parts of the game were OCR'd vs skipped, so the job
    # result makes clear what was (and wasn't) read.
    coverage = {
        "duration": round(duration, 1),
        "skip_breaks": skip_breaks,
        "skipped_regions": [[round(a, 1), round(b, 1)] for a, b in skip_regions],
        "clean_rows": len(clean_records),
        "clean_from": round(clean_records[0]["video_t"], 1) if clean_records else None,
        "clean_to": round(clean_records[-1]["video_t"], 1) if clean_records else None,
        "hit_rate": round(hit_rate, 1),
        **_quarter_source_info(clean_records, roi_data),
    }
    (derived / "ocr_coverage.json").write_text(json.dumps(coverage, indent=2))
    cf, ct = coverage["clean_from"], coverage["clean_to"]
    print(f"OCR COVERAGE: read {cf}s–{ct}s of {duration:.0f}s "
          f"({len(clean_records)} rows). Skipped {len(skip_regions)} region(s): "
          f"{coverage['skipped_regions']}", flush=True)
    print("OCR pipeline complete.", flush=True)


def requarter(game_path: Path) -> None:
    """Re-decide each row's quarter from the saved raw reads — no video needed.
    Rewrites derived/timeline.json + ocr_timeline.csv (raw reads untouched), so a
    better quarter rule (or a court's new "Quarter from") applies in seconds."""
    derived = game_path.parent / "derived"
    raw_records = json.loads((derived / "timeline_raw.json").read_text())
    clean_records = json.loads((derived / "timeline.json").read_text())
    try:
        skips = json.loads((derived / "ocr_coverage.json").read_text()).get("skipped_regions") or []
    except Exception:
        skips = []
    in_skip = lambda t: any(a <= t < b for a, b in skips)  # noqa: E731
    roi_data = load_roi(game_path.parent)
    before = Counter(r["quarter"] for r in clean_records)
    clean_records = smooth_records(_choose_quarter_source(raw_records, clean_records, roi_data, game_path, in_skip))
    (derived / "timeline.json").write_text(json.dumps(clean_records, indent=2))
    rows = ["video_time_seconds,quarter,game_clock,home_score,away_score,confidence"]
    for r in clean_records:
        home, away = r.get("home_score"), r.get("away_score")
        rows.append(f"{r['video_t']},{r['quarter']},{r['clock']},"
                    f"{'' if home is None else home},{'' if away is None else away},{r['confidence']}")
    (derived / "ocr_timeline.csv").write_text("\n".join(rows) + "\n")
    try:
        cov = json.loads((derived / "ocr_coverage.json").read_text())
    except Exception:
        cov = {}
    cov.update(_quarter_source_info(clean_records, roi_data))
    (derived / "ocr_coverage.json").write_text(json.dumps(cov, indent=2))
    after = Counter(r["quarter"] for r in clean_records)
    print(f"Re-quartered {len(clean_records)} rows: {dict(sorted(before.items()))} → {dict(sorted(after.items()))}", flush=True)


def run_range(
    game_path: Path,
    start: float,
    end: float,
    fps: float,
    method: str | None = None,
    grayscale: bool | None = None,
    out_path: Path | None = None,
    workers: int = 0,
) -> None:
    """OCR only the video window [start, end] at `fps`, with an explicit method /
    grayscale override, and write the cleaned samples to `out_path` (JSON array,
    same row shape as timeline.json). Used by the Timeline page's "manual OCR"
    source — it never touches the game's main timeline.json; the caller decides
    whether to apply the result. Reuses the same chunked worker + smoothing as the
    full run so results are directly comparable. Prints `PROGRESS done/total`."""
    game_dir = game_path.parent
    # Same source as a full run: the local mp4, or (cloud mode) the stream URL.
    video, _is_remote = resolve_video_source(game_path)
    if not video:
        sys.exit(f"Video not found in {game_dir} (no local file and no URL)")

    roi_data = load_roi(game_dir)
    rois = roi_data.get("rois", {})
    if not rois:
        sys.exit("roi.json has no 'rois' key")
    src_w, src_h = get_frame_size(roi_data)

    global _GRAYSCALE, _METHOD, _QUARTER_METHOD
    if grayscale is None or method is None:
        try:
            _gj = json.loads(game_path.read_text())
        except Exception:
            _gj = {}
    else:
        _gj = {}
    _GRAYSCALE = bool(_gj.get("ocrGrayscale")) if grayscale is None else bool(grayscale)
    _METHOD = (method or _gj.get("ocrMethod") or "tesseract")
    _QUARTER_METHOD = _gj.get("ocrQuarterMethod") or ""

    duration = get_duration_ffprobe(video)
    start = max(0.0, float(start))
    end = min(float(end), duration) if end and end > 0 else duration
    if end <= start:
        sys.exit(f"Empty range: start={start} end={end}")

    step = 1.0 / fps
    n = int((end - start) / step)
    timestamps = [round(start + i * step, 3) for i in range(n + 1) if start + i * step <= end]
    total = len(timestamps)
    if total == 0:
        sys.exit("No frames in range")

    n_workers = workers if workers > 0 else (os.cpu_count() or 4)
    n_workers = max(1, min(n_workers, total))

    print(f"OCR range: {Path(video).name}  {total} frames @ {fps} fps  [{start:.0f}s..{end:.0f}s]  method={_METHOD}", flush=True)

    video_str = video  # local absolute path or the stream URL
    chunk_size = max(20, total // (n_workers * 8) or 1)
    tasks: list[tuple[int, list[float], str, int, int, dict[str, Any], bool, str]] = []
    for s in range(0, total, chunk_size):
        tasks.append((s, timestamps[s:s + chunk_size], video_str, src_w, src_h, rois, _GRAYSCALE, _METHOD, _QUARTER_METHOD))

    raw_by_idx: dict[int, list[dict[str, Any]]] = {}
    clean_all: list[dict[str, Any]] = []
    done = 0
    if n_workers == 1:
        for task in tasks:
            idx, raws, cleans = _process_chunk(task)
            raw_by_idx[idx] = raws
            clean_all.extend(cleans)
            done += len(raws)
            print(f"PROGRESS {done}/{total}", flush=True)
    else:
        with mp.Pool(n_workers) as pool:
            for idx, raws, cleans in pool.imap_unordered(_process_chunk, tasks):
                raw_by_idx[idx] = raws
                clean_all.extend(cleans)
                done += len(raws)
                print(f"PROGRESS {done}/{total}", flush=True)

    clean_by_t: dict[float, dict[str, Any]] = {}
    for r in clean_all:
        clean_by_t[r["video_t"]] = r
    clean_records = smooth_records([clean_by_t[t] for t in sorted(clean_by_t)])

    out = out_path or (game_dir / "derived" / "ocr_range.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean_records, indent=2))
    print(f"Wrote {out} ({len(clean_records)}/{total} frames)", flush=True)
    print("OCR range complete.", flush=True)


ALL_METHODS = ("sevenseg", "tesseract", "easyocr")
# Heavy neural backends — slow per frame, so only auto-picked when the fast
# geometric/Tesseract decoders can't read the board confidently.
_NEURAL_METHODS = ("easyocr",)
# Order to prefer on a tie — cheapest/fastest first, so we only pick a heavy
# neural backend when it is genuinely the most accurate, not on a coin-flip.
_METHOD_PREFERENCE = {"sevenseg": 0, "tesseract": 1, "easyocr": 2}


def _build_pbp_score_fn(game_dir: Path):
    """Build a per-quarter score-by-clock step function from PBP.

    Returns a callable lookup(quarter:int, ms_remaining:int) -> (home, away) | None.
    PBP gives the running score at each event; within a quarter the score is a
    step function of the (descending) game clock. For a query clock ``c`` the
    current score is the most recent event that has already happened by ``c`` —
    i.e. the event with the smallest clock_remaining that is still >= c. This is
    independent of the sync stage, so it is non-circular OCR ground truth."""
    path = game_dir / "pbp" / "events.json"
    try:
        events = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(events, list) or not events:
        return None
    # quarter -> ascending list of (ms_remaining, home, away)
    by_q: dict[int, list[tuple[int, int, int]]] = {}
    for e in events:
        q = parse_quarter(str(e.get("quarter", "")))
        ms = _clock_to_ms(str(e.get("clock_remaining", "")))
        home, away = e.get("score_home"), e.get("score_away")
        if q is None or ms is None or home is None or away is None:
            continue
        by_q.setdefault(q, []).append((ms, int(home), int(away)))
    for q in by_q:
        by_q[q].sort(key=lambda r: r[0])
    if not by_q:
        return None
    keys = {q: [r[0] for r in by_q[q]] for q in by_q}

    def lookup(quarter: int, ms: int):
        rows = by_q.get(quarter)
        if not rows:
            return None
        # First event with clock_remaining >= ms (the most recent that happened).
        i = bisect.bisect_left(keys[quarter], ms)
        if i >= len(rows):
            return None  # query clock is before the quarter's first event
        return (rows[i][1], rows[i][2])

    return lookup


def detect_method(game_path: Path, n: int = 10,
                  methods: tuple[str, ...] = ALL_METHODS) -> dict[str, Any]:
    """Grade each OCR method against PBP and recommend the best one.

    Samples ``n`` frames spread across mid-game, decodes each once, then runs
    every method over the same cached frames. Correctness is graded against the
    PBP score-by-clock step function (non-circular — never uses sync output):
    a frame is *comparable* when OCR parses quarter+clock+both scores AND that
    (quarter, clock) falls within PBP coverage; it *matches* when the OCR scores
    equal the PBP score at that clock. Also records per-method timing (the cost
    that matters for the heavy neural backend). Prints + returns a structured
    breakdown with a flat {method: score_match} for back-compat."""
    global _METHOD, _GRAYSCALE
    _GRAYSCALE = False
    game_dir = game_path.parent
    video = locate_video(game_path)
    if video is None:
        out = {"error": "video_not_found"}
        print(json.dumps(out)); return out
    roi_data = load_roi(game_dir)
    rois = roi_data.get("rois", {})
    src_w, src_h = get_frame_size(roi_data)
    duration = get_duration_ffprobe(video)
    lookup = _build_pbp_score_fn(game_dir)

    # Decode the sample frames once (shared across methods so timing reflects OCR
    # cost only, not video seeking).
    cap = open_capture(video)
    frames: list[np.ndarray] = []
    try:
        for i in range(n):
            t = duration * (0.20 + 0.60 * (i / max(1, n - 1)))   # spread across 20–80%
            f = read_frame_at(cap, t)
            if f is not None:
                frames.append(f)
    finally:
        cap.release()

    results: dict[str, Any] = {}
    for method in methods:
        if method not in ALL_METHODS:
            continue
        _METHOD = method
        if method in _NEURAL_METHODS:
            try: _get_easyocr()   # warm up the model OUTSIDE the timed loop
            except Exception as e:
                results[method] = {"error": f"{method} unavailable: {e}"}
                continue
        hit = {"clock": 0, "quarter": 0, "home_score": 0, "away_score": 0}
        comparable = match = 0
        t0 = time.perf_counter()
        for f in frames:
            raw, _clean = process_frame(f, rois, src_w, src_h)
            q = parse_quarter(raw.get("quarter") or "")
            clk_ms = _clock_to_ms(parse_clock(raw.get("clock") or ""))
            home = parse_score(raw.get("home_score") or "")
            away = parse_score(raw.get("away_score") or "")
            if raw.get("clock"): hit["clock"] += 1
            if q is not None: hit["quarter"] += 1
            if home is not None: hit["home_score"] += 1
            if away is not None: hit["away_score"] += 1
            if lookup and q is not None and clk_ms is not None and home is not None and away is not None:
                exp = lookup(q, clk_ms)
                if exp is not None:
                    comparable += 1
                    if exp == (home, away):
                        match += 1
        elapsed = time.perf_counter() - t0
        nf = len(frames)
        results[method] = {
            "frames": nf,
            "hit": hit,
            "comparable": comparable,
            "score_match": match,
            "score_acc": round(match / comparable, 3) if comparable else None,
            "parse_rate": round(comparable / nf, 3) if nf else None,
            "ms_per_frame": round(elapsed / nf * 1000, 1) if nf else None,
            "total_s": round(elapsed, 2),
        }

    # Pick the winner. Primary metric = PBP-confirmed correct reads (score_match);
    # falls back to raw clock hit-rate when PBP is unavailable.
    graded = {m: r for m, r in results.items() if "error" not in r}
    have_pbp = lookup is not None and any(r["comparable"] for r in graded.values())
    nf = len(frames)
    threshold = max(2, nf // 2)   # "confident" = correct on at least half the frames

    def primary(m: str) -> int:
        r = graded[m]
        return r["score_match"] if have_pbp else r["hit"]["clock"]

    def rank(m: str):
        # Higher primary wins; on a tie prefer the cheaper/faster method so we
        # don't pick the heavy neural backend on a coin-flip.
        return (primary(m), -_METHOD_PREFERENCE.get(m, 9))

    recommended = None
    if graded:
        # Neural backends are expensive per frame, so only auto-pick one when the
        # fast methods (7-seg / Tesseract) can't read the board confidently. Prefer
        # the best confident *fast* method first; then any confident method; then
        # the overall best (low confidence) as a last resort.
        fast = [m for m in graded if m not in _NEURAL_METHODS]
        fast_ok = [m for m in fast if primary(m) >= threshold]
        any_ok = [m for m in graded if primary(m) >= threshold]
        pool = fast_ok or any_ok or list(graded)
        recommended = max(pool, key=rank)

    best = primary(recommended) if recommended else 0

    # Per-field quarter recommendation: if the chosen method reads the QUARTER
    # poorly but a neural backend reads it well, suggest a quarter-only neural
    # override (keeps the fast decoder for clock+scores — the c8571d79 stylised-
    # font case). Pick whichever neural backend reads the quarter best.
    q_hit = {m: graded[m]["hit"]["quarter"] for m in graded}
    quarter_recommended = None
    if recommended:
        rec_q = q_hit.get(recommended, 0)
        neural_q = [(q_hit[m], m) for m in _NEURAL_METHODS if m in graded and m != recommended]
        if neural_q:
            best_q, best_m = max(neural_q)
            if rec_q < threshold and best_q >= threshold and best_q > rec_q + 1:
                quarter_recommended = best_m

    out = {
        "frames": nf,
        "methods": results,
        "recommended": recommended,
        "quarter_recommended": quarter_recommended,
        "quarter_hit": q_hit,
        "graded_against_pbp": have_pbp,
        "confident": recommended is not None and best >= threshold,
        # Flat back-compat fields (older UI / probeOcrMethod read scores.<method>).
        **{m: (results[m].get("score_match", 0) if "error" not in results[m] else 0) for m in ALL_METHODS},
    }
    print(json.dumps(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, help="Path to game.json")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample (default: 0.5 = every 2s)")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0 = all CPU cores)")
    parser.add_argument("--no-skip-breaks", dest="skip_breaks", action="store_false",
                        help="Disable adaptive pre-game/halftime skipping (sample the full video).")
    # Manual-range mode (Timeline page): OCR only [--start,--end] with an explicit
    # --method/--grayscale and write to --out, leaving the main timeline untouched.
    parser.add_argument("--start", type=float, default=None, help="Range start (s) — enables range mode")
    parser.add_argument("--end", type=float, default=None, help="Range end (s) — enables range mode")
    parser.add_argument("--method", default=None, choices=["tesseract", "sevenseg", "easyocr"], help="OCR method override")
    parser.add_argument("--grayscale", default=None, choices=["0", "1"], help="Grayscale OCR override")
    parser.add_argument("--out", default=None, help="Output JSON path (range mode)")
    parser.add_argument("--decoder", default="seek", choices=["seek", "keyframe"],
                        help="Frame reader: 'seek' (per-sample random seek, default) or "
                             "'keyframe' (stream I-frames via ffmpeg — faster when interval ≈ GOP)")
    parser.add_argument("--derived", default=None,
                        help="Output directory for timeline files (default: <game>/derived). "
                             "Use a scratch dir for comparison runs.")
    parser.add_argument("--no-marker-bounds", dest="use_markers", action="store_false",
                        help="Ignore markers.json scan bounds (scan the whole video, breaks aside).")
    parser.add_argument("--requarter", action="store_true",
                        help="Re-decide quarters from the saved raw reads (no video) and rewrite the timeline.")
    parser.add_argument("--plan-only", dest="plan_only", action="store_true",
                        help="Coarse break-finding pass only: write derived/break_plan.json, skip dense OCR.")
    parser.add_argument("--detect-method", dest="detect_method", action="store_true",
                        help="Try both OCR methods on sample frames and print the recommended one (JSON).")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="Read only shard I of N (0-based) of the full run into <derived>/shard-I.json. "
                             "Use the same --fps/--workers/--decoder for every shard and the merge.")
    parser.add_argument("--merge-shards", nargs="+", default=None, metavar="SHARD_JSON",
                        help="Assemble shard files from --shard runs and write the normal outputs.")
    parser.set_defaults(skip_breaks=True, use_markers=True, plan_only=False, detect_method=False)
    args = parser.parse_args()
    if args.detect_method:
        detect_method(Path(args.game))
        return
    if args.requarter:
        requarter(Path(args.game).resolve())
        return

    if args.start is not None or args.end is not None or args.out is not None:
        run_range(
            Path(args.game),
            start=args.start or 0.0,
            end=args.end or 0.0,
            fps=args.fps,
            method=args.method,
            grayscale=None if args.grayscale is None else args.grayscale == "1",
            out_path=Path(args.out) if args.out else None,
            workers=args.workers,
        )
    else:
        shard = None
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            if not 0 <= i < n:
                parser.error("--shard must be I/N with 0 <= I < N")
            shard = (i, n)
        run(Path(args.game), args.fps, workers=args.workers, skip_breaks=args.skip_breaks,
            decoder=args.decoder, derived_dir=Path(args.derived) if args.derived else None,
            use_markers=args.use_markers, plan_only=args.plan_only,
            shard=shard, merge_shards=[Path(f) for f in args.merge_shards] if args.merge_shards else None)


if __name__ == "__main__":
    main()
