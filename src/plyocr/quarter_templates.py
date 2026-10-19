#!/usr/bin/env python3
"""Per-court quarter-digit templates.

Some scoreboards draw the period number in a stylized font Tesseract can't read
(e.g. it reads "1" and "4" but not "2"/"3"). Each court has ONE scoreboard
style, so we learn what its period digits look like from a game that's already
synced — the quarter windows say which period every frame is in — and match the
quarter box against those images. Templates live in the court profile / the
game's roi.json as `quarterTemplates: {"1": <png b64>, ...}`; the OCR uses them
first and falls back to Tesseract when nothing matches well.

Learn (prints JSON {"templates": {...}, "samples": {q: n}}):
    python -m plyocr.quarter_templates --game data/games/<id>/game.json [--rois-json '{...}'] [--per-q 8]
"""
from __future__ import annotations

import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

W, H = 24, 36          # normalized glyph size
MIN_SCORE = 0.55       # correlation needed to accept a match
MIN_MARGIN = 0.08      # ...and how much better than the runner-up


def normalize(crop: np.ndarray) -> np.ndarray | None:
    """Binary glyph (white on black), cropped to its ink and resized to W×H."""
    if crop is None or crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([b[0, :], b[-1, :], b[:, 0], b[:, -1]])
    if border.mean() > 127:          # light background → make the glyph the white part
        b = 255 - b
    pts = cv2.findNonZero(b)
    if pts is None:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 2 or h < 4:
        return None
    return cv2.resize(b[y:y + h, x:x + w], (W, H), interpolation=cv2.INTER_AREA)


def _score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d else 0.0


def decode_templates(tpl: dict[str, str] | None) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for q, b64 in (tpl or {}).items():
        try:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[int(q)] = cv2.resize(img, (W, H))
        except Exception:
            continue
    return out


def classify(crop: np.ndarray, templates: dict[int, np.ndarray]) -> int | None:
    """The period digit in `crop`, or None when no template matches clearly."""
    if not templates:
        return None
    g = normalize(crop)
    if g is None:
        return None
    scored = sorted(((_score(g, t), q) for q, t in templates.items()), reverse=True)
    best, q = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    return q if best >= MIN_SCORE and best - second >= MIN_MARGIN else None


def learn(game_path: Path, rois_json: str | None, per_q: int) -> dict[str, Any]:
    from plyocr.read_frame import grab_frame, resolve_video_source, load_roi  # noqa: E402

    roi_data = json.loads(rois_json) if rois_json else load_roi(game_path.parent)
    box = (roi_data.get("rois") or {}).get("quarter")
    if not box:
        return {"error": "no quarter box"}
    src_w, src_h = int(roi_data.get("frame_width") or 1920), int(roi_data.get("frame_height") or 1080)
    data_path = game_path.parent / "app" / "data.json"
    if not data_path.exists():
        return {"error": "game not synced yet — quarter windows unknown"}
    qm = json.loads(data_path.read_text()).get("quarter_markers") or {}
    video, is_remote = resolve_video_source(game_path)

    jobs: list[tuple[int, float]] = []
    for q, v in qm.items():
        if not str(q).isdigit() or v.get("start_t") is None or v.get("end_t") is None:
            continue
        s, e = float(v["start_t"]), float(v["end_t"])
        jobs += [(int(q), s + (e - s) * (k + 1) / (per_q + 1)) for k in range(per_q)]

    def crop_at(job: tuple[int, float]):
        q, t = job
        f = grab_frame(video, t, is_remote)
        if f is None:
            return q, None
        fh, fw = f.shape[:2]
        sx, sy = fw / src_w, fh / src_h
        x, y = int(box["x"] * sx), int(box["y"] * sy)
        return q, normalize(f[y:y + int(box["h"] * sy), x:x + int(box["w"] * sx)])

    with ThreadPoolExecutor(max_workers=4) as pool:
        got = list(pool.map(crop_at, jobs))

    by_q: dict[int, list[np.ndarray]] = {}
    for q, g in got:
        if g is not None:
            by_q.setdefault(q, []).append(g.astype(np.float32))
    templates: dict[str, str] = {}
    for q, gs in by_q.items():
        # Median image: robust to a few frames where the box shows something else.
        med = np.median(np.stack(gs), axis=0)
        img = (med > 127).astype(np.uint8) * 255
        ok, buf = cv2.imencode(".png", img)
        if ok:
            templates[str(q)] = base64.b64encode(buf.tobytes()).decode("ascii")
    # Self-check: classify every sample with the learned set.
    dec = decode_templates(templates)
    hits = sum(1 for q, g in got if g is not None and classify(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), dec) == q)
    return {"templates": templates, "samples": {str(q): len(v) for q, v in by_q.items()},
            "selfCheck": [hits, sum(1 for _, g in got if g is not None)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, type=Path)
    ap.add_argument("--rois-json", default=None)
    ap.add_argument("--per-q", type=int, default=8)
    a = ap.parse_args()
    print(json.dumps(learn(a.game.resolve(), a.rois_json, max(3, min(20, a.per_q)))))


if __name__ == "__main__":
    main()
