#!/usr/bin/env python3
"""Geometric 7-segment scoreboard decoder.

A per-game alternative to the Tesseract OCR path for scoreboards rendered in a
7-segment LED font (which Tesseract's normal-typeface models misread badly —
e.g. 3→4, 7→1, and the dim colon dropped during binarization).

How it works (no training, no templates):
  1. Mask the *lit* pixels of the ROI (bright + saturated) — this isolates the
     glowing digits and discards the dim "ghost" unlit segments and background,
     regardless of digit color (green clock/quarter, magenta scores, …).
  2. Segment the mask into per-character columns; a thin lit column in the middle
     of the clock is the ':' separator.
  3. For each digit cell, sample the seven segment regions (a=top, b=top-right,
     c=bottom-right, d=bottom, e=bottom-left, f=top-left, g=middle), threshold
     each on/off, and look the pattern up in the standard 7-segment truth table.

Public API:
  read_fields(frame_bgr, rois, src_w, src_h) -> {field: raw_str}
      Mirrors the per-field raw strings the Tesseract path produces; callers run
      the same parse_clock / parse_quarter / parse_score on the result.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# segment pattern (a, b, c, d, e, f, g) -> digit
_SEG_TABLE: dict[tuple[int, ...], str] = {
    (1, 1, 1, 1, 1, 1, 0): "0",
    (0, 1, 1, 0, 0, 0, 0): "1",
    (1, 1, 0, 1, 1, 0, 1): "2",
    (1, 1, 1, 1, 0, 0, 1): "3",
    (0, 1, 1, 0, 0, 1, 1): "4",
    (1, 0, 1, 1, 0, 1, 1): "5",
    (1, 0, 1, 1, 1, 1, 1): "6",
    (1, 1, 1, 0, 0, 0, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}

_V_MIN = 110   # min HSV value — lit segments are bright; dim ghosts are excluded
_S_MIN = 60    # min HSV saturation — the LED colors are vivid
_SEG_ON = 0.26     # fraction of lit pixels in a vertical/middle segment region → "on"
_SEG_ON_BAR = 0.17 # lower threshold for the thin top/bottom bars (a, d)

# The clock separator (':' or '.') is rendered in a dim RED that the digit mask
# often drops. Detect it with a relaxed red-hue threshold instead.
_SEP_V_MIN = 60
_SEP_S_MIN = 70
_SEP_MIN_PX = 6

# When the saturation mask covers more than this fraction of the crop, a bright
# saturated background (e.g. a glowing red score panel) has bled into it and the
# digits merge into a blob — fall back to the desaturated-core channel below.
_BLOB_FRAC = 0.42
_CORE_V_MIN = 90   # core channel keeps only genuinely bright pixels


def _lit_mask_sat(crop_bgr: np.ndarray) -> np.ndarray:
    """0/1 mask of lit digit pixels — bright + saturated, excluding the red separator.

    Works for any digit color (green/yellow clock+quarter, magenta scores). The
    red colon/period is removed here so it never merges into a digit cell; it is
    detected separately by _separator().
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lit = (v > _V_MIN) & (s > _S_MIN)
    red = ((h < 12) | (h > 168)) & (s > _SEP_S_MIN) & (v > _SEP_V_MIN)
    # Strip red ONLY when it's a minority of the lit pixels — i.e. a red colon on
    # differently-coloured digits (green/yellow clock). If red dominates, the
    # DIGITS are red (red 7-seg clock/score), so keep them.
    litn, redn = int(lit.sum()), int(red.sum())
    if 0 < redn < 0.35 * max(1, litn):
        return (lit & ~red).astype(np.uint8)
    return lit.astype(np.uint8)


def _lit_mask_core(crop_bgr: np.ndarray) -> np.ndarray:
    """Mask the near-white digit CORES — for digits on a bright, saturated
    background (e.g. a glowing red score panel) that the saturation mask blobs
    over. A blindingly-bright LED stroke desaturates toward white (elevated green
    & blue), while the coloured bloom around it keeps low green/blue; Otsu on
    min(G, B) isolates the cores from the bloom."""
    b = crop_bgr[:, :, 0].astype(np.int16)
    g = crop_bgr[:, :, 1].astype(np.int16)
    white = cv2.GaussianBlur(np.minimum(g, b).astype(np.uint8), (3, 3), 0)
    _, binr = cv2.threshold(white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return ((binr > 0) & (v > _CORE_V_MIN)).astype(np.uint8)


def _lit_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Lit-digit mask, robust to a bright coloured background.

    Default to the saturation mask (correct for the common case: bright digits on
    a dark panel, any colour). ONLY when it blobs — covering > _BLOB_FRAC of the
    crop because a saturated background (glowing red score panel) leaked in — do we
    switch to the desaturated-core channel. An empty saturation mask is left as-is
    so frames with no scoreboard never fabricate a read."""
    sat = _lit_mask_sat(crop_bgr)
    if float(sat.mean()) <= _BLOB_FRAC:
        return sat
    core = _lit_mask_core(crop_bgr)
    if 0.04 <= float(core.mean()) <= _BLOB_FRAC:
        return core
    return sat


def _separator(crop_bgr: np.ndarray) -> tuple[str, float] | None:
    """Detect the clock separator from its red pixels.

    Returns (char, x_center_fraction) where char is ':' (two dots, spans the
    upper+lower band) or '.' (one low dot), or None if no red separator found.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((h < 12) | (h > 168)) & (s > _SEP_S_MIN) & (v > _SEP_V_MIN)
    ys, xs = np.where(red)
    if xs.size < _SEP_MIN_PX:
        return None
    H = crop_bgr.shape[0]
    x_frac = float(np.median(xs)) / crop_bgr.shape[1]
    # ':' has dots in the upper half; '.' sits only near the bottom.
    has_upper = bool((ys < 0.45 * H).any())
    return (":" if has_upper else ".", x_frac)


def _seg_frac(cell: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> float:
    h, w = cell.shape
    sub = cell[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return float(sub.mean()) if sub.size else 0.0


def _decode_digit(cell: np.ndarray) -> str:
    """Classify a single vertically-trimmed digit cell via its 7 segment regions."""
    h, w = cell.shape
    if h == 0 or w == 0:
        return "?"
    if w < 0.30 * h:          # VERY narrow → "1" (a "7" is wider, ~0.4·h, so excluded)
        return "1"
    a = _seg_frac(cell, 0.00, 0.18, 0.20, 0.80)
    g = _seg_frac(cell, 0.42, 0.60, 0.20, 0.80)
    d = _seg_frac(cell, 0.82, 1.00, 0.20, 0.80)
    f = _seg_frac(cell, 0.10, 0.48, 0.00, 0.32)
    e = _seg_frac(cell, 0.52, 0.92, 0.00, 0.32)
    b = _seg_frac(cell, 0.10, 0.48, 0.68, 1.00)
    c = _seg_frac(cell, 0.52, 0.92, 0.68, 1.00)
    # The top/bottom horizontal bars (a, d) render faint/thin on some LED fonts
    # (a "9"'s bottom bar measured ~0.21), so use a lower on-threshold for them —
    # otherwise 9→4, 8→0, etc. Vertical/middle segments keep the normal threshold.
    key = (
        1 if a > _SEG_ON_BAR else 0, 1 if b > _SEG_ON else 0, 1 if c > _SEG_ON else 0,
        1 if d > _SEG_ON_BAR else 0, 1 if e > _SEG_ON else 0, 1 if f > _SEG_ON else 0,
        1 if g > _SEG_ON else 0,
    )
    if key in _SEG_TABLE:
        return _SEG_TABLE[key]
    # Fallback: a single mis-thresholded segment shouldn't drop the digit. Among
    # Hamming-distance-1 table patterns, prefer one that's a SUBSET of the read
    # (i.e. we only turned a spuriously-on segment OFF — phantom-on is the common
    # error, e.g. a "7" reading a faint bottom bar). Else require uniqueness.
    cand = [(pat, dig) for pat, dig in _SEG_TABLE.items()
            if sum(p != k for p, k in zip(pat, key)) == 1]
    if not cand:
        return "?"
    subset = [dig for pat, dig in cand if all(p <= k for p, k in zip(pat, key))]
    if len(subset) == 1:
        return subset[0]
    return cand[0][1] if len(cand) == 1 else "?"


def decode_crop(crop_bgr: np.ndarray, is_clock: bool = False) -> str:
    """Decode one ROI crop to its raw string (digits, plus ':' for the clock)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    m = _lit_mask(crop_bgr)
    H, W = m.shape
    if H == 0 or W == 0 or m.sum() == 0:
        return ""

    on = (m.sum(axis=0) > 1).astype(int)
    runs: list[list[int]] = []
    i, n = 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1

    # Merge sub-character column gaps (digit strokes can break into pieces).
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.05 * H:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])

    # Build cells (vertically trimmed) with their height + center, so we can tell a
    # digit from the short colon/period that sits in the digit row.
    cells = []
    for x0, x1 in merged:
        if (x1 - x0) < 0.06 * H:        # stray pixels
            continue
        sub = m[:, x0:x1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if rows.size == 0:
            continue
        cells.append((x0, x1, sub[rows[0]:rows[-1] + 1, :], int(rows[-1] - rows[0] + 1), float((rows[0] + rows[-1]) / 2 / H)))
    if not cells:
        return ""
    digit_h = max(c[3] for c in cells)

    digits: list[tuple[float, str]] = []
    inline_sep: tuple[float, str] | None = None   # a same-colour colon in the mask
    for x0, x1, cell, ch, cy in cells:
        if ch < 0.62 * digit_h:        # much shorter than the digits → separator
            if is_clock and inline_sep is None:
                inline_sep = (":" if cy < 0.60 else ".", (x0 + x1) / 2 / W)
            continue
        dig = _decode_digit(cell)
        if dig != "?":
            digits.append(((x0 + x1) / 2 / W, dig))

    if not is_clock:
        return "".join(d for _, d in digits)

    ds = [d for _, d in digits]
    xs = [x for x, _ in digits]
    if not ds:
        return ""

    # Split at the widest digit gap (where the separator sits) and decide the scheme
    # by how many digits fall AFTER it — NOT by the separator glyph:
    #   2 digits after → M:SS / MM:SS   (e.g. "1:34", "12:34")
    #   1 digit  after → SS.t  (last-minute tenths)  (e.g. "13.4")
    # Glyph-independent on purpose: some boards print the last-minute tenths with a
    # COLON, e.g. "54:8" = 54.8 s (not 5:48). We normalise tenths to '.' so
    # parse_clock reads them. (Position-from-gap beats the colour-based separator
    # detector, which reports garbage on an all-red clock where every digit is red.)
    if len(ds) >= 3:
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        k = max(range(len(gaps)), key=lambda i: gaps[i])
        n_after = len(ds) - (k + 1)
        if n_after in (1, 2):
            return "".join(ds[:k + 1]) + ("." if n_after == 1 else ":") + "".join(ds[k + 1:])

    # Fallback for ≤2 digits: an explicit separator cell tells tenths vs seconds;
    # otherwise (bare "SS") just join.
    sep = inline_sep or _separator(crop_bgr)
    if sep is None:
        return "".join(ds)
    tail = 1 if sep[0] == "." else 2
    if len(ds) > tail:
        return "".join(ds[:-tail]) + sep[0] + "".join(ds[-tail:])
    return "".join(ds)


def read_fields(frame_bgr: np.ndarray, rois: dict[str, Any],
                src_w: int, src_h: int) -> dict[str, str]:
    """Decode every ROI in a frame → {field: raw_string}. Scales ROI px to frame."""
    fh, fw = frame_bgr.shape[:2]
    sx, sy = fw / src_w, fh / src_h
    out: dict[str, str] = {}
    for field, roi in rois.items():
        x = max(0, int(roi["x"] * sx))
        y = max(0, int(roi["y"] * sy))
        x2 = min(fw, x + int(roi["w"] * sx))
        y2 = min(fh, y + int(roi["h"] * sy))
        if x2 <= x or y2 <= y:
            out[field] = ""
            continue
        out[field] = decode_crop(frame_bgr[y:y2, x:x2], is_clock=(field == "clock"))
    return out
