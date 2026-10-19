#!/usr/bin/env python3
"""Auto-detect scoreboard ROI positions — template-guided, multi-frame, OCR-validated.

The IBBA/Pixellot overlay sits in a consistent bottom-center location and is
static across the game (only the digits change). A pure color search on a single
frame is unreliable (it snaps to jerseys/court lines/ads). Instead we:

  1. Start from a per-resolution learned TEMPLATE (plyocr.paths.ROI_TEMPLATES,
     built from hand-tuned games; falls back to the baseline ROI
     plyocr.paths.DEFAULT_ROI scaled to the video).
  2. Sample several frames spread across the middle of the video.
  3. For each ROI, search only a small WINDOW around the template box, snap to the
     largest colored-text blob (red = clock/scores, yellow = quarter), and take
     the median box across frames.
  4. OCR-VALIDATE each candidate (clock→M:SS/SS.t, quarter→1-5, score→digits);
     accept the refined box only if enough frames validate, else keep the template
     box. Per-ROI + overall confidence is reported.

Usage:
    python -m plyocr.detect_rois --video <game.mp4>
    python -m plyocr.detect_rois --frame <frame.jpg>   # legacy single-frame

Output: JSON (frame_width, frame_height, resolution, rois{name:{x,y,w,h,...,confidence}}, confidence).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Optional

import cv2
import numpy as np

from plyocr.hls_frame import grab_preview_frame  # noqa: E402
from scipy import ndimage

from plyocr.paths import DEFAULT_ROI, ROI_TEMPLATES
TEMPLATES_PATH = ROI_TEMPLATES
BASELINE_PATH = DEFAULT_ROI

from plyocr.run_tesseract_ocr import (  # noqa: E402
    preprocess_crop,
    ocr_field,
    parse_clock,
    QUARTER_MAP,
    read_frame_at,
    open_capture,
    get_duration_ffprobe,
)
from plyocr import sevenseg  # noqa: E402

ROI_NAMES = ("clock", "quarter", "home_score", "away_score")
RED = ("clock", "home_score", "away_score")
N_FRAMES = 10
WINDOW_FRAC = 0.6      # expand template box by ±60% of its size to form the search window
MIN_VALID = 3          # frames that must OCR-validate to accept the refined box
PAD = 3                # px padding added to a snapped blob

# ── video-discovery params (camera-agnostic localization) ──────────────────
DISC_FRAMES = 60       # frames sampled for discovery (more = better saliency)
DISC_W = 640           # downscale width for the cheap saliency pass
LIT_FREQ_MIN = 0.45    # a pixel "persistently lit" if bright+saturated in ≥ this fraction
BAND_FRAC = 0.22       # the scoreboard sits within the top/bottom this-fraction band
DISC_MIN_VALID = 0.30  # a discovered field is accepted if it validates on ≥ this fraction
                       # (7-seg clocks read ~30-65%; below this we keep the template box)


# ── color masks (same thresholds as the original prototype OCR) ──────────
def _red_mask(arr: np.ndarray) -> np.ndarray:
    r, g, b = (arr[..., i].astype(np.int16) for i in range(3))
    return ((r >= 95) & (r - g >= 30) & (r - b >= 25)).astype(np.uint8)


def _yellow_mask(arr: np.ndarray) -> np.ndarray:
    r, g, b = (arr[..., i].astype(np.int16) for i in range(3))
    return ((r >= 150) & (g >= 120) & (b <= 90)).astype(np.uint8)


def _largest_blob_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    rows, cols = np.where(labeled == best)
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


# ── template loading + scaling ─────────────────────────────────────────────
def _scale_box(box: dict, sx: float, sy: float) -> dict:
    out = {"x": int(box["x"] * sx), "y": int(box["y"] * sy),
           "w": int(box["w"] * sx), "h": int(box["h"] * sy)}
    for k in ("color", "font", "format", "chars"):
        if k in box:
            out[k] = box[k]
    return out


def load_template(fw: int, fh: int) -> Optional[dict[str, dict]]:
    """Return template ROIs scaled to (fw, fh), from the learned templates or the
    hard-coded baseline."""
    res = f"{fw}x{fh}"
    src = None
    if TEMPLATES_PATH.exists():
        templates = json.loads(TEMPLATES_PATH.read_text())
        src = templates.get(res)
        if src is None and templates:
            # nearest by area
            src = min(templates.values(),
                      key=lambda t: abs(t["frame_width"] * t["frame_height"] - fw * fh))
    if src is None and BASELINE_PATH.exists():
        src = json.loads(BASELINE_PATH.read_text())
    if not src or "rois" not in src:
        return None
    sx = fw / src.get("frame_width", fw)
    sy = fh / src.get("frame_height", fh)
    return {name: _scale_box(box, sx, sy) for name, box in src["rois"].items() if name in ROI_NAMES}


# ── per-ROI refinement ─────────────────────────────────────────────────────
def _window(box: dict, fw: int, fh: int) -> tuple[int, int, int, int]:
    ex, ey = int(box["w"] * WINDOW_FRAC), int(box["h"] * WINDOW_FRAC)
    x0 = max(0, box["x"] - ex)
    y0 = max(0, box["y"] - ey)
    x1 = min(fw, box["x"] + box["w"] + ex)
    y1 = min(fh, box["y"] + box["h"] + ey)
    return x0, y0, x1, y1


def _snap(frame_rgb: np.ndarray, box: dict, name: str, fw: int, fh: int) -> Optional[dict]:
    x0, y0, x1, y1 = _window(box, fw, fh)
    win = frame_rgb[y0:y1, x0:x1]
    if win.size == 0:
        return None
    mask = _red_mask(win) if name in RED else _yellow_mask(win)
    if mask.sum() < 8:
        return None
    bb = _largest_blob_bbox(mask)
    if bb is None:
        return None
    bx0, by0, bx1, by1 = bb
    # reject blobs that fill the whole window (likely background, not text)
    if (bx1 - bx0 + 1) >= 0.95 * (x1 - x0) and (by1 - by0 + 1) >= 0.95 * (y1 - y0):
        return None
    x = max(0, x0 + bx0 - PAD)
    y = max(0, y0 + by0 - PAD)
    return {
        "x": x, "y": y,
        "w": min(fw, x0 + bx1 + 1 + PAD) - x,
        "h": min(fh, y0 + by1 + 1 + PAD) - y,
    }


def _validate(frame_bgr: np.ndarray, box: dict, name: str) -> bool:
    crop = frame_bgr[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
    if crop.size == 0:
        return False
    try:
        img = preprocess_crop(crop, trim_components=name in ("home_score", "away_score"))
        raw = ocr_field(img, name)
    except Exception:  # noqa: BLE001
        return False
    if not raw:
        return False
    if name == "clock":
        return parse_clock(raw) is not None
    if name == "quarter":
        return raw.strip().upper() in QUARTER_MAP
    return raw.isdigit() and 0 <= int(raw) <= 199  # scores


# ── camera-agnostic video discovery ────────────────────────────────────────
def _lit(bgr: np.ndarray) -> np.ndarray:
    """0/1 mask of bright + saturated (lit overlay/digit) pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 2] > 110) & (hsv[:, :, 1] > 60)).astype(np.uint8)


def _field_reads(frames_bgr: list[np.ndarray], box: dict, name: str) -> list:
    """Parsed value per frame for a box, trying Tesseract then the 7-seg decoder.
    Returns clock as ms-int-ish via parse_clock string, quarter int, score int, or None."""
    out = []
    for fr in frames_bgr:
        crop = fr[box["y"]:box["y"] + box["h"], box["x"]:box["x"] + box["w"]]
        if crop.size == 0:
            out.append(None); continue
        raws = []
        try:
            raws.append(ocr_field(preprocess_crop(crop, trim_components=name in ("home_score", "away_score")), name))
        except Exception:  # noqa: BLE001
            pass
        raws.append(sevenseg.decode_crop(crop, is_clock=(name == "clock")))
        val = None
        for raw in raws:
            if not raw:
                continue
            if name == "clock":
                if parse_clock(raw) is not None:
                    val = raw; break
            elif name == "quarter":
                if raw.strip().upper() in QUARTER_MAP:
                    val = QUARTER_MAP[raw.strip().upper()]; break
            else:
                if raw.isdigit() and 0 <= int(raw) <= 199:
                    val = int(raw); break
        out.append(val)
    return out


def _valid_rate(reads: list) -> float:
    return sum(1 for r in reads if r is not None) / max(1, len(reads))


def _reads_digit(crop: np.ndarray) -> bool:
    """True if a small crop OCRs as a single digit (0-9) — Tesseract or 7-seg."""
    if crop.size == 0:
        return False
    raws = []
    try:
        raws.append(ocr_field(preprocess_crop(crop, trim_components=True), "home_score"))
    except Exception:  # noqa: BLE001
        pass
    raws.append(sevenseg.decode_crop(crop))
    for raw in raws:
        s = (raw or "").strip()
        if len(s) == 1 and s.isdigit():
            return True
    return False


def _discover_box(frames_bgr: list[np.ndarray], fw: int, fh: int) -> tuple[int, int, int, int] | None:
    """Stage 1: find the overall scoreboard area (the box containing ALL fields).

    All fields are NUMBERS, so look for digit-shaped components that actually OCR
    as digits AND recur at the same position across frames (the scoreboard is
    static; jersey numbers / banners move or don't read as clean digits). Cluster
    the stable digit positions by baseline and box the densest row.
    """
    sample = frames_bgr[:: max(1, len(frames_bgr) // 12)][:12]
    if not sample:
        return None
    digits: list[tuple[float, float, int, int, int, int]] = []  # cx,cy,x,y,w,h
    for fr in sample:
        m = sevenseg._lit_mask(fr)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        cand = []
        for i in range(1, n):
            x, y, w, h, area = (int(stats[i, k]) for k in range(5))
            if not (0.012 * fh <= h <= 0.12 * fh):       # digit height
                continue
            if w < 0.15 * h or w > 1.6 * h:               # digit aspect (tall-ish)
                continue
            cand.append((area, x, y, w, h, cent[i][0], cent[i][1]))
        cand.sort(reverse=True)                            # biggest first; cap work
        for area, x, y, w, h, cx, cy in cand[:40]:
            if _reads_digit(fr[y:y + h, x:x + w]):
                digits.append((cx, cy, x, y, w, h))
    if len(digits) < 4:
        return None
    # Keep digit positions that recur across frames (snap centers to a grid).
    gx, gy = max(8, int(0.01 * fw)), max(8, int(0.01 * fh))
    from collections import defaultdict
    cells: dict[tuple, list] = defaultdict(list)
    for d in digits:
        cells[(round(d[0] / gx), round(d[1] / gy))].append(d)
    min_seen = max(2, int(0.3 * len(sample)))
    stable = [v[0] for v in cells.values() if len(v) >= min_seen]
    if len(stable) < 3:
        stable = [d for ds in cells.values() for d in ds]   # fall back to all
    # Densest baseline row among the stable digits.
    tol = 0.03 * fh
    best, best_cnt = None, 0
    for s in stable:
        row = [d for d in stable if abs(d[1] - s[1]) <= tol]
        if len(row) > best_cnt:
            best_cnt, best = len(row), row
    if not best or best_cnt < 3:
        return None
    # Within the baseline row, keep the largest horizontally-contiguous cluster —
    # drops far outliers (stray jersey/banner digits) so the box stays tight.
    best.sort(key=lambda d: d[2])
    max_gap = 0.06 * fw
    runs, cur = [], [best[0]]
    for d in best[1:]:
        if d[2] - (cur[-1][2] + cur[-1][4]) <= max_gap:
            cur.append(d)
        else:
            runs.append(cur); cur = [d]
    runs.append(cur)
    best = max(runs, key=len)
    if len(best) < 3:
        return None
    xs0 = min(d[2] for d in best); xs1 = max(d[2] + d[4] for d in best)
    ys0 = min(d[3] for d in best); ys1 = max(d[3] + d[5] for d in best)
    padx, pady = int(0.06 * (xs1 - xs0)) + 6, int(0.4 * (ys1 - ys0)) + 6
    return (max(0, xs0 - padx), max(0, ys0 - pady),
            min(fw, xs1 + padx), min(fh, ys1 + pady))


def _discover_band(frames_bgr: list[np.ndarray], fw: int, fh: int) -> tuple[int, int, int, int] | None:
    """Find the scoreboard band via persistently-lit pixels in a top/bottom margin."""
    sw = DISC_W
    sh = int(fh * sw / fw)
    acc = np.zeros((sh, sw), np.float32)
    for fr in frames_bgr:
        acc += cv2.resize(_lit(fr), (sw, sh), interpolation=cv2.INTER_AREA)
    litfreq = acc / len(frames_bgr)
    # Toggling pixels = lit in SOME but not ALL frames → changing digits. Excludes
    # always-on logos/graphics (freq≈1) and background (freq≈0).
    persist = ((litfreq > 0.12) & (litfreq < 0.88)).astype(np.uint8)
    band_h = int(sh * BAND_FRAC)
    top, bot = persist[:band_h], persist[sh - band_h:]
    # pick the margin (top/bottom) with more persistent-lit pixels
    use_top = top.sum() >= bot.sum()
    region = top if use_top else bot
    y_off = 0 if use_top else sh - band_h
    if region.sum() < 20:
        return None
    rows = np.where(region.sum(axis=1) > 0.02 * sw)[0]
    cols = np.where(region.sum(axis=0) > 0.02 * band_h)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    # back to full res, with margin
    scale_x, scale_y = fw / sw, fh / sh
    y0 = int((y_off + rows[0]) * scale_y) - 6
    y1 = int((y_off + rows[-1] + 1) * scale_y) + 6
    x0 = int(cols[0] * scale_x) - 8
    x1 = int((cols[-1] + 1) * scale_x) + 8
    return max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)


def _segment_frame(crop: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Single-frame field segmentation: (x0,x1,y0,y1) per field within the crop.
    Uses the per-frame lit mask (NOT aggregated — aggregation fills inter-field
    gaps); fields sit farther apart than a field's own inter-digit gaps."""
    m = sevenseg._lit_mask(crop)
    H = m.shape[0]
    if H == 0 or m.sum() == 0:
        return []
    on = (m.sum(axis=0) > 1).astype(int)
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    gap = max(6, int(0.6 * H))
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])
    out = []
    for gx0, gx1 in merged:
        if (gx1 - gx0) < 0.2 * H:
            continue
        sub = m[:, gx0:gx1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        out.append((gx0, gx1, int(rows[0]), int(rows[-1]) + 1))
    return out


def _groups_by_change(frames_bgr: list[np.ndarray], area: tuple[int, int, int, int]) -> list[dict]:
    """Within the user's box: take ~10 frames, mark pixels that CHANGE across them
    (the digits tick over time), cluster the changed columns into fields, and pad
    each field box by 15% for safety. Static overlay parts don't change → ignored."""
    x0, y0, x1, y1 = area
    H, W = y1 - y0, x1 - x0
    if H <= 0 or W <= 0:
        return []
    pick = frames_bgr[:: max(1, len(frames_bgr) // 10)][:10]
    if len(pick) < 2:
        return []
    grays = np.stack([cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.int16) for fr in pick])
    change = (grays.max(axis=0) - grays.min(axis=0)).astype(np.uint8)   # per-pixel range
    # Digits flip fully on/off (large range); semi-transparent overlay + background
    # motion change only mildly. Threshold high to isolate the digit changes.
    thr = int(min(160, max(90, change.mean() * 2.0)))
    mask = (change > thr).astype(np.uint8)   # no morph-close: it would bridge field gaps
    if mask.sum() < 10:
        return []
    # Column runs of "enough changed pixels"; merge gaps within a field (fields sit
    # farther apart than a field's own digit gaps — works best on a TIGHT box).
    on = (mask.sum(axis=0) > 0.10 * H).astype(int)
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    gap = max(6, int(0.40 * H))
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])
    groups = []
    for gx0, gx1 in merged:
        if (gx1 - gx0) < 0.2 * H:
            continue
        sub = mask[:, gx0:gx1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        gy0, gy1 = int(rows[0]), int(rows[-1]) + 1
        bw, bh = gx1 - gx0, gy1 - gy0
        px, py = int(0.20 * bw) + 1, int(0.20 * bh) + 1   # 20% extra on every side
        groups.append({
            "x": max(0, x0 + gx0 - px), "y": max(0, y0 + gy0 - py),
            "w": bw + 2 * px, "h": bh + 2 * py,
        })
    return groups


def _discover_groups(frames_bgr: list[np.ndarray], band: tuple[int, int, int, int]) -> list[dict]:
    """Find fields inside the scoreboard box by consensus over single-frame
    segmentations (stable field x-positions across frames)."""
    x0, y0, x1, y1 = band
    H = y1 - y0
    items: list[tuple[float, int, int, int, int]] = []  # (cx, gx0,gx1,gy0,gy1)
    for fr in frames_bgr:
        for gx0, gx1, gy0, gy1 in _segment_frame(fr[y0:y1, x0:x1]):
            items.append(((gx0 + gx1) / 2, gx0, gx1, gy0, gy1))
    if not items:
        return []
    items.sort(key=lambda t: t[0])
    # Greedy 1-D cluster of x-centers (fields are spaced ≫ within-field jitter).
    clusters: list[list[tuple]] = [[items[0]]]
    for it in items[1:]:
        if it[0] - clusters[-1][-1][0] <= 0.7 * H:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    min_support = max(2, int(0.25 * len(frames_bgr)))
    groups = []
    for cl in clusters:
        if len(cl) < min_support:
            continue
        gx0 = int(np.median([c[1] for c in cl])); gx1 = int(np.median([c[2] for c in cl]))
        gy0 = int(np.median([c[3] for c in cl])); gy1 = int(np.median([c[4] for c in cl]))
        groups.append({
            "x": x0 + gx0 - 2, "y": y0 + gy0 - 2,
            "w": (gx1 - gx0) + 4, "h": (gy1 - gy0) + 4,
        })
    return groups


_CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}$|^\d{1,2}\.\d$")


def _classify_group(frames_bgr: list[np.ndarray], g: dict) -> dict:
    """OCR a group across frames and summarize, using the known formats:
    clock = 'M:SS'/'SS.t'; numbers tracked with max length + max value so the
    (ambiguous) single-digit quarter can be told from a score (scores grow past 5
    / to 2+ digits over a game; the quarter stays 1-5)."""
    info = {"clock": 0, "num": 0, "max_len": 0, "max_val": -1}
    sample = frames_bgr[:: max(1, len(frames_bgr) // 10)][:10]
    for fr in sample:
        crop = fr[g["y"]:g["y"] + g["h"], g["x"]:g["x"] + g["w"]]
        if crop.size == 0:
            continue
        if _CLOCK_RE.match(sevenseg.decode_crop(crop, is_clock=True) or ""):
            info["clock"] += 1
            continue
        raws = []
        try:
            raws.append(ocr_field(preprocess_crop(crop, trim_components=True), "home_score"))
        except Exception:  # noqa: BLE001
            pass
        raws.append(sevenseg.decode_crop(crop))
        for raw in raws:
            s = (raw or "").strip().upper()
            if s in QUARTER_MAP:
                info["num"] += 1; info["max_len"] = max(info["max_len"], 1)
                info["max_val"] = max(info["max_val"], QUARTER_MAP[s]); break
            if s.isdigit() and 1 <= len(s) <= 3:
                info["num"] += 1; info["max_len"] = max(info["max_len"], len(s))
                info["max_val"] = max(info["max_val"], int(s)); break
    return info


def _assign_fields(groups: list[dict], frames_bgr: list[np.ndarray]) -> dict[str, dict]:
    """Label groups by their OCR'd FORMAT: clock has ':'/'.'; quarter only ever
    reads 1-5 (single digit); scores grow to >5 / 2+ digits. Home=leftmost score,
    away=rightmost."""
    if len(groups) < 2:
        return {}
    groups = sorted(groups, key=lambda g: g["x"])
    info = [_classify_group(frames_bgr, g) for g in groups]
    out: dict[str, dict] = {}

    # clock = the group reading a clock format (':'/'.'); else the widest group.
    ci = max(range(len(groups)), key=lambda i: info[i]["clock"])
    if info[ci]["clock"] == 0:
        ci = max(range(len(groups)), key=lambda i: groups[i]["w"])
    out["clock"] = groups[ci]

    # Layout is home · clock · quarter · away. Assign the rest by POSITION:
    others = [i for i in range(len(groups)) if i != ci]
    if others:
        out["home_score"] = groups[others[0]]                       # leftmost
        if len(others) > 1:
            out["away_score"] = groups[others[-1]]                  # rightmost
        # quarter = a group to the RIGHT of the clock that isn't the away score.
        right = [i for i in others if i > ci and i != others[-1]]
        if right:
            out["quarter"] = groups[right[0]]
        elif len(others) >= 3:                                      # fallback: middle
            out["quarter"] = groups[others[len(others) // 2]]

    # Scores hold up to 3 digits — grow each score box LEFT by ~one digit width so
    # a third digit isn't clipped (scoreboards add the hundreds digit on the left).
    for s in ("home_score", "away_score"):
        if s in out:
            b = out[s]
            dw = int(0.55 * b["h"])
            b["x"] = max(0, b["x"] - dw)
            b["w"] = b["w"] + dw
    return out


def _area_default(area: tuple[int, int, int, int], name: str) -> dict:
    """A placeholder box INSIDE the drawn area for a field we couldn't locate, by
    its usual horizontal position (home | clock | quarter | away). The user nudges it."""
    x0, y0, x1, y1 = area
    W, H = x1 - x0, y1 - y0
    # (left-fraction, width-fraction) per field along the scoreboard row
    frac = {
        "home_score": (0.02, 0.16),
        "clock":      (0.30, 0.26),
        "quarter":    (0.60, 0.12),
        "away_score": (0.80, 0.16),
    }.get(name, (0.40, 0.16))
    fx, fwf = frac
    return {
        "x": int(x0 + fx * W), "y": int(y0 + 0.15 * H),
        "w": max(12, int(fwf * W)), "h": max(12, int(0.7 * H)),
    }


def _load_pbp_scores(game_dir: Path) -> tuple[int, int] | None:
    """Final (home, away) from pbp/events.json, if available."""
    p = game_dir / "pbp" / "events.json"
    if not p.exists():
        return None
    try:
        evs = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    h = a = None
    for e in evs:
        if isinstance(e.get("score_home"), int):
            h = e["score_home"]
        if isinstance(e.get("score_away"), int):
            a = e["score_away"]
    return (h, a) if (h is not None and a is not None) else None


def _assign_home_away(rois: dict, frames_bgr: list[np.ndarray], game_dir: Path) -> str:
    """Decide which score box is home vs away. Uses the PBP final score as oracle;
    falls back to left=home. Returns the source ('pbp' | 'default')."""
    if "home_score" not in rois or "away_score" not in rois:
        return "none"
    left, right = rois["home_score"], rois["away_score"]
    pbp = _load_pbp_scores(game_dir)
    if pbp is not None:
        # max read of each box ≈ its final score; match to the PBP finals.
        lreads = [r for r in _field_reads(frames_bgr, left, "home_score") if r is not None]
        rreads = [r for r in _field_reads(frames_bgr, right, "away_score") if r is not None]
        lmax = max(lreads) if lreads else -1
        rmax = max(rreads) if rreads else -1
        ph, pa = pbp
        # cost of (left=home,right=away) vs (left=away,right=home)
        c_normal = abs(lmax - ph) + abs(rmax - pa)
        c_swap = abs(lmax - pa) + abs(rmax - ph)
        if c_swap < c_normal:
            rois["home_score"], rois["away_score"] = right, left
        return "pbp"
    return "default"  # left = home convention


def _template_box(name: str, template: dict, frames_bgr: list, frames_rgb: list,
                  fw: int, fh: int) -> tuple[dict, float] | None:
    """The legacy template-window-snap + OCR-validate path for one field."""
    tbox = template.get(name)
    if not tbox:
        return None
    cands = [s for fr in frames_rgb if (s := _snap(fr, tbox, name, fw, fh))]
    if cands:
        consensus = {g: int(round(median(c[g] for c in cands))) for g in ("x", "y", "w", "h")}
    else:
        consensus = {g: tbox[g] for g in ("x", "y", "w", "h")}
    cons_valid = sum(_validate(fr, consensus, name) for fr in frames_bgr)
    tmpl_valid = sum(_validate(fr, {g: tbox[g] for g in ("x", "y", "w", "h")}, name) for fr in frames_bgr)
    if cons_valid >= MIN_VALID and cons_valid >= tmpl_valid:
        box, valid = consensus, cons_valid
    else:
        box, valid = {g: tbox[g] for g in ("x", "y", "w", "h")}, tmpl_valid
    return box, valid / max(1, len(frames_bgr))


def _load_prior(path: Path | None, fw: int, fh: int) -> dict[str, dict] | None:
    """Existing roi.json → {name: box} scaled to the frame, if usable."""
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    rois = data.get("rois")
    if not rois:
        return None
    sx = fw / data.get("frame_width", fw)
    sy = fh / data.get("frame_height", fh)
    return {n: _scale_box(b, sx, sy) for n, b in rois.items() if n in ROI_NAMES}


def _grab_remote(url: str, t: float) -> np.ndarray | None:
    """One BGR frame from an HLS URL for analysis: the 720p rendition upscaled to
    full HD when Pixellot has one (5x less to download), else the HD segment."""
    return grab_preview_frame(url, t)


def detect_video(video: str, box: tuple[int, int, int, int] | None = None,
                 prior: Path | None = None, is_remote: bool = False,
                 game_dir: Path | None = None) -> dict:
    dur = get_duration_ffprobe(video)
    # Remote: every frame is a download — 32 still find the board reliably.
    times = [dur * f for f in np.linspace(0.08, 0.92, DISC_FRAMES if not is_remote else 32)]
    if is_remote:
        # Sparse samples across the game → grab each via ffmpeg. I/O-bound, so pull
        # them in parallel instead of 60 sequential network round-trips.
        with ThreadPoolExecutor(max_workers=8) as ex:
            frames_bgr = [fr for fr in ex.map(lambda t: _grab_remote(video, t), times) if fr is not None]
    else:
        cap = open_capture(Path(video))
        frames_bgr = [fr for t in times if (fr := read_frame_at(cap, t)) is not None]
        cap.release()
    if not frames_bgr:
        return {"rois": {}, "confidence": "none", "error": "no frames"}

    fh, fw = frames_bgr[0].shape[:2]
    frames_rgb = [fr[:, :, ::-1] for fr in frames_bgr]
    game_dir = game_dir or Path(video).parent.parent

    # Prefer existing OCR boxes (roi.json) as the prior; else the learned template.
    prior_rois = _load_prior(prior, fw, fh)
    template = prior_rois or load_template(fw, fh) or {}
    using_prior = prior_rois is not None

    rois: dict[str, dict] = {}
    per_conf: dict[str, float] = {}
    source: dict[str, str] = {}

    manual = box is not None
    area = None

    # ── 1) discovery — skip when refining existing boxes (a prior is present) ──
    discovered: dict[str, dict] = {}
    box_source = "none"
    try:
        # Stage 1: the overall scoreboard area — manual override, else auto.
        # If we're refining existing boxes (prior, no manual area), skip discovery.
        area = box if manual else (None if using_prior else _discover_box(frames_bgr, fw, fh))
        if area is not None:
            box_source = "manual" if manual else "auto"
            # Stage 2: locate + assign fields within the area. For a hand-drawn box
            # use changed-pixels (the digits); auto uses the lit-mask consensus.
            groups = _groups_by_change(frames_bgr, area) if manual else _discover_groups(frames_bgr, area)
            cand = _assign_fields(groups, frames_bgr)
            for name, fbox in cand.items():
                rate = _valid_rate(_field_reads(frames_bgr, fbox, name))
                # With a hand-drawn area, trust the in-area boxes even if OCR is
                # shaky (the user will fine-tune); only auto-mode gates on rate.
                if manual or rate >= DISC_MIN_VALID:
                    discovered[name] = {**fbox, "confidence": round(rate, 2)}
                    per_conf[name] = rate
                    source[name] = "discovered"
    except Exception as e:  # noqa: BLE001 — discovery must never crash detection
        print(f"discovery failed ({e})", file=sys.stderr)

    # ── 2) fill missing fields ───────────────────────────────────────────────
    # Manual area: keep every box INSIDE the area (placeholder if not found) and
    # never jump back to the template. Auto: fall back to the template box.
    for name in ROI_NAMES:
        if name in discovered:
            rois[name] = discovered[name]
            continue
        if manual and area is not None:
            box_ = _area_default(area, name)
            box_["confidence"] = 0.0
            rois[name] = box_
            per_conf[name] = 0.0
            source[name] = "area-default"
            continue
        if using_prior and name in template:
            # Respect the user's existing box — keep it as-is, just score it.
            pb = {g: template[name][g] for g in ("x", "y", "w", "h")}
            rate = _valid_rate(_field_reads(frames_bgr, pb, name))
            pb["confidence"] = round(rate, 2)
            rois[name] = pb
            per_conf[name] = rate
            source[name] = "prior"
            continue
        tb = _template_box(name, template, frames_bgr, frames_rgb, fw, fh)
        if tb is None:
            continue
        tbox, rate = tb
        tbox["confidence"] = round(rate, 2)
        rois[name] = tbox
        per_conf[name] = rate
        source[name] = "prior" if using_prior else "template"

    # carry color/font/format/chars metadata from the template onto every field
    for name, box in rois.items():
        tbox = template.get(name, {})
        for k in ("color", "font", "format", "chars"):
            if k in tbox and k not in box:
                box[k] = tbox[k]

    # ── 3) decide home vs away ───────────────────────────────────────────────
    home_away = _assign_home_away(rois, frames_bgr, game_dir)

    n_ok = sum(1 for c in per_conf.values() if c >= DISC_MIN_VALID)
    confidence = "high" if n_ok == 4 else "partial" if n_ok >= 2 else "low"
    return {
        "frame_width": fw, "frame_height": fh, "resolution": f"{fw}x{fh}",
        "rois": rois, "confidence": confidence,
        "field_source": source, "home_away_source": home_away,
        "box_source": box_source,
    }


def detect_frame(frame_path: Path) -> dict:
    """Legacy single-frame path (no OCR validation, template snap only)."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return {"rois": {}, "confidence": "none", "error": "cannot read frame"}
    fh, fw = img.shape[:2]
    template = load_template(fw, fh)
    if template is None:
        return {"frame_width": fw, "frame_height": fh, "rois": {}, "confidence": "none"}
    rgb = img[:, :, ::-1]
    rois: dict[str, dict] = {}
    for name in ROI_NAMES:
        tbox = template.get(name)
        if not tbox:
            continue
        snap = _snap(rgb, tbox, name, fw, fh) or {g: tbox[g] for g in ("x", "y", "w", "h")}
        for k in ("color", "font", "format", "chars"):
            if k in tbox:
                snap[k] = tbox[k]
        rois[name] = snap
    return {"frame_width": fw, "frame_height": fh, "resolution": f"{fw}x{fh}",
            "rois": rois, "confidence": "partial"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str, help="Path to the game video, or an HLS URL (cloud mode)")
    g.add_argument("--frame", type=Path, help="Path to a single frame image (legacy)")
    ap.add_argument("--save", type=Path, help="Write the detected ROI JSON to this path (e.g. roi.json)")
    ap.add_argument("--box", type=str, default=None,
                    help="Manual scoreboard area 'x,y,w,h' (px) — skips auto box-finding")
    ap.add_argument("--prior", type=Path, default=None,
                    help="Existing roi.json to refine (used instead of the template)")
    ap.add_argument("--game-dir", type=Path, default=None,
                    help="Game directory (for the PBP home/away oracle) — required when "
                         "--video is a URL; derived from a local video path otherwise")
    args = ap.parse_args()

    box = None
    if args.box:
        try:
            bx, by, bw, bh = (int(v) for v in args.box.split(","))
            box = (bx, by, bx + bw, by + bh)
        except Exception:
            sys.exit("--box must be 'x,y,w,h' integers")

    if args.video:
        is_remote = args.video.startswith("http://") or args.video.startswith("https://")
        if not is_remote and not Path(args.video).exists():
            sys.exit(f"Video not found: {args.video}")
        game_dir = args.game_dir or (Path(args.video).parent.parent if not is_remote else None)
        if is_remote and game_dir is None:
            sys.exit("--game-dir is required when --video is a URL")
        result = detect_video(args.video, box=box, prior=args.prior,
                              is_remote=is_remote, game_dir=game_dir)
    else:
        if not args.frame.exists():
            sys.exit(f"Frame not found: {args.frame}")
        result = detect_frame(args.frame)

    if args.save and result.get("rois"):
        args.save.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Saved ROI → {args.save}")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
