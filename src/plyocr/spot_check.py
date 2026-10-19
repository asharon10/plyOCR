#!/usr/bin/env python3
"""OCR spot-check for one game: read N frames with a (possibly unsaved) ROI +
OCR config and grade each read against what SHOULD be on the scoreboard.

The truth is the PBP's score timeline — which score was on the board in which
quarter between which clock times — so a read is right when its quarter, clock
and score fit together. That doesn't depend on the video sync (a drifting sync
would otherwise blame good reads). Frames are sampled inside the quarters when
the game is synced, else evenly across the video. Uses the pipeline's own OCR (run_tesseract_ocr.py), so the
result predicts a full run. Prints one JSON object:

  {"frames": [{"t", "reads": {field: {raw, parsed, valid, crop?}},
               "expected": {clock, quarter, home_score, away_score} | null,
               "ok": {field: true|false|null}}],
   "summary": {field: {"valid", "correct", "graded", "n"}},
   "suggestions": [str]}

Usage:
    python -m plyocr.spot_check --game data/games/<id>/game.json \\
        [--rois-json '{...}'] [--method tesseract|sevenseg] [--grayscale 0|1] [--n 12] [--emit-crops]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from plyocr import sevenseg  # noqa: E402
from plyocr import game_format  # noqa: E402
from plyocr import run_tesseract_ocr as pipe  # noqa: E402
from plyocr.read_frame import grab_frame, resolve_video_source, load_roi, _png_b64, _roi_threshold  # noqa: E402

FIELDS = ("clock", "quarter", "home_score", "away_score")
CLOCK_TOL_S = 6  # s; the board operator often posts a basket a few seconds late


def clock_secs(c: Any) -> float | None:
    if not isinstance(c, str):
        return None
    if ":" in c:
        m, s = c.split(":", 1)
        return int(m) * 60 + int(s)
    try:
        return float(c)
    except ValueError:
        return None


def sample_times(data: dict | None, duration: float, n: int) -> list[float]:
    qm = (data or {}).get("quarter_markers") or {}
    wins = sorted((float(v["start_t"]), float(v["end_t"])) for v in qm.values()
                  if v and v.get("start_t") is not None and v.get("end_t") is not None)
    if wins:
        per = max(1, round(n / len(wins)))
        return [s + (e - s) * (k + 1) / (per + 1) for s, e in wins for k in range(per)]
    lo, hi = duration * 0.1, duration * 0.9
    return [lo + (hi - lo) * k / max(1, n - 1) for k in range(n)]


def score_states(events: list[dict], fmt: dict | None = None) -> list[tuple[int, float, float, int, int]]:
    """From the PBP alone: (quarter, clock_hi, clock_lo, home, away) = the score
    on the board in that quarter while the clock ran from clock_hi down to
    clock_lo. Independent of the video sync, so it can grade any read."""
    by_q: dict[int, list[dict]] = {}
    for e in events:
        q = str(e.get("quarter") or "")
        if q.isdigit() and clock_secs(e.get("clock") or e.get("clock_remaining")) is not None:
            by_q.setdefault(int(q), []).append(e)
    states, prev = [], (0, 0)
    for q in sorted(by_q):
        hi, cur = game_format.period_seconds(fmt or game_format.DEFAULT, q), prev
        for e in by_q[q]:
            sc = (int(e.get("score_home") or 0), int(e.get("score_away") or 0))
            if sc != cur:
                c = clock_secs(e.get("clock") or e.get("clock_remaining"))
                states.append((q, hi, c, *cur))
                cur, hi = sc, c
        states.append((q, hi, 0, *cur))
        prev = cur
    return states


def read_frame(frame, rois: dict, src_w: int, src_h: int, method: str, grayscale: bool, crops: bool) -> dict:
    fh, fw = frame.shape[:2]
    sx, sy = fw / src_w, fh / src_h
    out: dict[str, Any] = {}
    for field in FIELDS:
        roi = rois.get(field)
        if not roi:
            continue
        x, y = max(0, int(roi["x"] * sx)), max(0, int(roi["y"] * sy))
        x2, y2 = min(fw, x + int(roi["w"] * sx)), min(fh, y + int(roi["h"] * sy))
        if x2 <= x or y2 <= y:
            out[field] = {"raw": "", "parsed": None, "valid": False}
            continue
        crop = frame[y:y2, x:x2]
        proc = pipe.preprocess_crop(crop, trim_components=field.endswith("score"),
                                    grayscale=grayscale, threshold=_roi_threshold(roi))
        tq = (pipe.quarter_templates.classify(crop, pipe._decoded_templates(roi["templates"]))
              if field == "quarter" and roi.get("templates") else None)
        if tq is not None:
            raw = str(tq)
        elif method == "sevenseg" and field != "quarter":
            raw = sevenseg.decode_crop(crop, is_clock=(field == "clock")) or ""
        else:
            raw = pipe.ocr_field(proc, field) or ""
        parsed = (pipe.parse_clock(raw) if field == "clock"
                  else pipe.parse_quarter(raw) if field == "quarter" else pipe.parse_score(raw))
        e: dict[str, Any] = {"raw": raw, "parsed": parsed, "valid": parsed is not None}
        if crops:
            e["crop"] = _png_b64(proc)
        out[field] = e
    return out


def pregame(reads: dict, fmt: dict) -> bool:
    """Before tip-off the board runs a countdown longer than a quarter (e.g.
    15:32) and shows no score yet (0–0 or blank). Such a frame is read right:
    the clock just isn't a game clock and there's nothing to score."""
    raw = pipe.fix_common(str(reads.get("clock", {}).get("raw") or "")).replace(".", ":")
    m = re.match(r"^(\d{1,2}):?(\d{2})$", raw)
    if not m or int(m.group(2)) > 59 or int(m.group(1)) <= fmt["quarterMinutes"]:
        return False
    for f in ("home_score", "away_score"):
        e = reads.get(f) or {}
        if e.get("parsed") not in (0, None) or (e.get("parsed") is None and str(e.get("raw") or "").strip()):
            return False   # a real score (or garbage) on the board — not the pre-game board
    return True


def mark_pregame(reads: dict, ok: dict) -> None:
    """Accept a pre-game frame's clock + blank/0 scores (quarter stays as graded)."""
    for f in ("clock", "home_score", "away_score"):
        if f in reads:
            reads[f]["valid"] = True
            if reads[f].get("parsed") is None:
                reads[f]["parsed"] = reads[f].get("raw") or (0 if f != "clock" else None)
            ok[f] = True


def grade(reads: dict, states: list) -> tuple[dict, dict | None]:
    """Grade a frame's reads by whether quarter + clock + score fit the PBP's
    score timeline together (sync-independent). Returns (ok per field, what the
    PBP says the board should show, for display)."""
    ok: dict[str, bool | None] = {f: None for f in FIELDS}
    if not states:
        return ok, None
    T = CLOCK_TOL_S
    r = {f: reads.get(f, {}).get("parsed") for f in FIELDS}
    q, c, h, a = r["quarter"], clock_secs(r["clock"]), r["home_score"], r["away_score"]
    within = lambda st: c is not None and st[2] - T <= c <= st[1] + T  # noqa: E731
    show = lambda st: {"quarter": st[0], "clock": [st[2], st[1]], "scores": [[st[3], st[4]]]}  # noqa: E731

    # Score + clock consistent somewhere → those are right; the quarter is
    # right iff it's that quarter (the usual failure: a misread period digit).
    hit = [st for st in states if st[3] == h and st[4] == a and within(st)]
    if hit:
        best = next((st for st in hit if st[0] == q), hit[0])
        ok.update(clock=True, home_score=True, away_score=True, quarter=(q == best[0]))
        return ok, show(best)
    # Score exists in this quarter but at another clock → the clock is off.
    in_q = [st for st in states if st[0] == q and st[3] == h and st[4] == a]
    if q is not None and in_q:
        ok.update(quarter=True, home_score=True, away_score=True, clock=False)
        return ok, show(in_q[0])
    # Clock + one score fit somewhere → that's the quarter; the other score (and
    # a different quarter read) are the misreads.
    part = [st for st in states if within(st) and (st[3] == h or st[4] == a)]
    if part:
        best = next((st for st in part if st[0] == q), part[0])
        ok.update(clock=True, quarter=(q == best[0]), home_score=(best[3] == h), away_score=(best[4] == a))
        return ok, show(best)
    # Trust quarter + clock: what was on the board then? Grade each score digit.
    at = [st for st in states if st[0] == q and within(st)]
    if at:
        ok["home_score"] = any(st[3] == h for st in at)
        ok["away_score"] = any(st[4] == a for st in at)
        if ok["home_score"] or ok["away_score"]:
            ok.update(quarter=True, clock=True)
        return ok, show(at[0])
    # Nothing fits: mark the unreadable fields, leave the rest ungraded.
    for f in FIELDS:
        if r[f] is None:
            ok[f] = False
    return ok, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, type=Path)
    ap.add_argument("--rois-json", default=None)
    ap.add_argument("--method", default=None, choices=["tesseract", "sevenseg"])
    ap.add_argument("--grayscale", default=None, choices=["0", "1"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--emit-crops", action="store_true")
    args = ap.parse_args()

    game_path = args.game.resolve()
    gj = json.loads(game_path.read_text())
    method = args.method or gj.get("ocrMethod") or "tesseract"
    grayscale = (args.grayscale == "1") if args.grayscale is not None else bool(gj.get("ocrGrayscale"))
    roi_data = pipe.with_quarter_templates(json.loads(args.rois_json) if args.rois_json else load_roi(game_path.parent))
    rois = roi_data.get("rois") or {}
    src_w, src_h = int(roi_data.get("frame_width") or 1920), int(roi_data.get("frame_height") or 1080)

    video, is_remote = resolve_video_source(game_path)
    if not video:
        print(json.dumps({"error": "video_not_found"}))
        sys.exit(1)
    data_path = game_path.parent / "app" / "data.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else None
    pbp_path = game_path.parent / "pbp" / "events.json"
    events = (data or {}).get("events") or (json.loads(pbp_path.read_text()) if pbp_path.exists() else [])
    fmt = game_format.load(game_path)
    states = score_states(events, fmt)
    try:
        duration = pipe.get_duration_ffprobe(video)
    except Exception:
        duration = 7200.0

    times = [round(t, 1) for t in sample_times(data, duration, max(1, min(40, args.n)))]

    # Fetch frames in parallel (network-bound); OCR them one at a time — the
    # Tesseract API isn't thread-safe.
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(lambda t: grab_frame(video, t, is_remote), times))
    frames = []
    for t, frame in zip(times, images):
        if frame is None:
            frames.append({"t": t, "error": "seek_failed"})
            continue
        reads = read_frame(frame, rois, src_w, src_h, method, grayscale, args.emit_crops)
        ok, exp = grade(reads, states)
        pre = pregame(reads, fmt)
        if pre:
            mark_pregame(reads, ok)
        frames.append({"t": t, "reads": reads, "expected": exp, "ok": ok, **({"pregame": True} if pre else {})})

    summary = {}
    for f in FIELDS:
        got = [fr for fr in frames if "reads" in fr and f in fr["reads"]]
        graded = [fr["ok"][f] for fr in got if fr["ok"][f] is not None]
        summary[f] = {"n": len(got), "valid": sum(1 for fr in got if fr["reads"][f]["valid"]),
                      "graded": len(graded), "correct": sum(1 for g in graded if g)}

    def rate(f: str) -> float | None:
        s = summary[f]
        return s["correct"] / s["graded"] if s["graded"] else (s["valid"] / s["n"] if s["n"] else None)

    tips = []
    rc, rq = rate("clock"), rate("quarter")
    rs = min((x for x in (rate("home_score"), rate("away_score")) if x is not None), default=None)
    if rc is not None and rc < 0.6:
        tips.append("Clock reads poorly: check the clock box covers all digits (with a little margin), "
                    "or try the 7-segment method / grayscale for LED-style clocks.")
    if rq is not None and rq < 0.6 and (rc or 0) >= 0.6:
        tips.append("The quarter digit reads poorly but the clock is fine: set Quarter from = clock.")
    if rs is not None and rs < 0.6:
        tips.append("Scores read poorly: check the score boxes cover all digits, or try grayscale.")
    if not states:
        tips.append("No play-by-play for this game yet, so reads are only checked for parsing, not correctness.")
    print(json.dumps({"frames": frames, "summary": summary, "suggestions": tips, "method": method,
                      "grayscale": grayscale}))


if __name__ == "__main__":
    main()
