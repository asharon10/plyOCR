#!/usr/bin/env python3
"""Compare the two OCR frame decoders on SPEED and (more importantly) OUTPUT.

Runs run_tesseract_ocr.py twice on the same game — once per decoder — writing each
run's timeline to its own scratch dir (the game's real derived/ is never touched),
times each, then diffs the cleaned timelines field-by-field.

Both runs use --no-skip-breaks so the sample grids line up (seek samples the 2s
grid; keyframe streams the 2s-GOP I-frames — the same frames), making the output
diff a true apples-to-apples check that the faster decoder reads the same values.

Usage:
    python -m plyocr.compare_decoders --game data/games/<id>/game.json
    python -m plyocr.compare_decoders --game <...> --fps 0.5 --workers 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

FIELDS = ("quarter", "clock", "home_score", "away_score")
MATCH_TOL_S = 1.2  # a clean row in run A matches run B's nearest row within this


def run_decoder(game: Path, decoder: str, fps: float, workers: int, derived: Path,
                skip_breaks: bool) -> float:
    """Run one decoder, return wall-clock seconds. Streams child output through."""
    derived.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON_BIN, "-m", "plyocr.run_tesseract_ocr", "--game", str(game), "--fps", str(fps),
           "--workers", str(workers),
           "--decoder", decoder, "--derived", str(derived)]
    if not skip_breaks:
        cmd.append("--no-skip-breaks")
    print(f"\n{'='*70}\n▶ {decoder.upper()} decoder\n{'='*70}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    # Surface the tail so a failure is visible; keep it short on success.
    tail = "\n".join(proc.stdout.strip().splitlines()[-4:])
    if tail:
        print(tail, flush=True)
    if proc.returncode != 0:
        print(proc.stderr.strip()[-2000:], flush=True)
        sys.exit(f"{decoder} run failed (exit {proc.returncode})")
    print(f"⏱  {decoder}: {dt:.1f}s wall", flush=True)
    return dt


def load_clean(derived: Path) -> list[dict]:
    p = derived / "timeline.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_raw_count(derived: Path) -> int:
    p = derived / "timeline_raw.json"
    return len(json.loads(p.read_text())) if p.exists() else 0


def nearest(rows_by_t: list[tuple[float, dict]], t: float) -> dict | None:
    """Nearest row within MATCH_TOL_S (rows sorted by t); None if none close."""
    best, bestd = None, MATCH_TOL_S
    # linear is fine for our sizes; could bisect if needed
    for rt, r in rows_by_t:
        d = abs(rt - t)
        if d <= bestd:
            best, bestd = r, d
        elif rt > t + MATCH_TOL_S:
            break
    return best


def diff(seek: list[dict], keyf: list[dict]) -> dict:
    kf_sorted = sorted(((r["video_t"], r) for r in keyf), key=lambda x: x[0])
    matched = 0
    unmatched = 0
    field_mismatch = {f: 0 for f in FIELDS}
    all_agree = 0
    examples: list[str] = []
    for r in sorted(seek, key=lambda r: r["video_t"]):
        m = nearest(kf_sorted, r["video_t"])
        if m is None:
            unmatched += 1
            continue
        matched += 1
        row_ok = True
        for f in FIELDS:
            if r.get(f) != m.get(f):
                field_mismatch[f] += 1
                row_ok = False
        if row_ok:
            all_agree += 1
        elif len(examples) < 12:
            diffs = ", ".join(f"{f}: {r.get(f)!r}→{m.get(f)!r}"
                              for f in FIELDS if r.get(f) != m.get(f))
            examples.append(f"  t≈{r['video_t']:.1f}s  {diffs}")
    only_in_kf = len(keyf) - matched  # keyf rows no seek row matched (approx)
    return {
        "seek_clean": len(seek), "keyframe_clean": len(keyf),
        "matched": matched, "all_agree": all_agree,
        "seek_unmatched": unmatched, "keyframe_extra": max(0, only_in_kf),
        "field_mismatch": field_mismatch, "examples": examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/ocr_compare"))
    ap.add_argument("--skip-breaks", action="store_true",
                    help="Run both with break-skipping ON (production config). Default: "
                         "OFF so the sample grids align for a strict output diff.")
    args = ap.parse_args()

    seek_dir = args.scratch / "seek"
    keyf_dir = args.scratch / "keyframe"

    t_seek = run_decoder(args.game, "seek", args.fps, args.workers, seek_dir, args.skip_breaks)
    t_keyf = run_decoder(args.game, "keyframe", args.fps, args.workers, keyf_dir, args.skip_breaks)

    seek = load_clean(seek_dir)
    keyf = load_clean(keyf_dir)
    d = diff(seek, keyf)

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print("\n── SPEED ──")
    print(f"  seek     : {t_seek:6.1f}s wall  ({load_raw_count(seek_dir)} raw samples)")
    print(f"  keyframe : {t_keyf:6.1f}s wall  ({load_raw_count(keyf_dir)} raw samples)")
    if t_keyf > 0:
        print(f"  speedup  : {t_seek / t_keyf:.2f}x  "
              f"({'keyframe faster' if t_keyf < t_seek else 'seek faster'})")

    print("\n── OUTPUT (clean timeline) ──")
    print(f"  clean rows      seek={d['seek_clean']}  keyframe={d['keyframe_clean']}")
    print(f"  matched rows    {d['matched']}  (within {MATCH_TOL_S}s)")
    if d["matched"]:
        pct = d["all_agree"] / d["matched"] * 100
        print(f"  all-4-fields agree: {d['all_agree']}/{d['matched']}  ({pct:.2f}%)")
    print(f"  seek rows with no keyframe match : {d['seek_unmatched']}")
    print(f"  keyframe rows beyond seek matches: {d['keyframe_extra']}")
    print("  per-field mismatches (over matched rows):")
    for f in FIELDS:
        print(f"      {f:12s}: {d['field_mismatch'][f]}")
    if d["examples"]:
        print("\n  sample disagreements (seek→keyframe):")
        for e in d["examples"]:
            print(e)
    else:
        print("\n  ✓ no field disagreements on matched rows")

    report = args.scratch / "compare_report.json"
    report.write_text(json.dumps(
        {"speed": {"seek_s": round(t_seek, 2), "keyframe_s": round(t_keyf, 2),
                   "speedup": round(t_seek / t_keyf, 3) if t_keyf else None},
         "output": d}, indent=2))
    print(f"\nReport: {report}")


if __name__ == "__main__":
    main()
