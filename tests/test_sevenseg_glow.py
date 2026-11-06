"""Golden test: a National League board — red LED scores so bright their cores
go white, on a panel that glows red around them, and a yellow clock. A score with
a narrow "1" (17, 41) used to blob into "8", a "4" read as "9", and the clock
"1:51" as "15.1". Read with the full per-frame pipeline in sevenseg mode.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from plyocr import run_tesseract_ocr as ocr

FIXTURES = Path(__file__).parent / "fixtures" / "glow_sevenseg"
ROI = json.loads((FIXTURES / "roi.json").read_text())
EXPECTED = {k: v for k, v in json.loads((FIXTURES / "expected.json").read_text()).items() if not k.startswith("_")}


def scoreboard_frame(sample: str) -> np.ndarray:
    frame = np.zeros((ROI["frame_height"], ROI["frame_width"], 3), np.uint8)
    for field, b in ROI["rois"].items():
        frame[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = cv2.imread(str(FIXTURES / "scoreboards" / sample / f"{field}.png"))
    return frame


@pytest.mark.parametrize("quarter_method", ["", "sevenseg"])   # "" = Tesseract, 7-seg fallback
@pytest.mark.parametrize("sample", sorted(EXPECTED))
def test_sevenseg_reads_glow_board(sample, quarter_method, monkeypatch):
    monkeypatch.setattr(ocr, "_METHOD", "sevenseg")
    monkeypatch.setattr(ocr, "_QUARTER_METHOD", quarter_method)
    _raw, clean = ocr.process_frame(scoreboard_frame(sample), ROI["rois"], ROI["frame_width"], ROI["frame_height"])
    want = EXPECTED[sample]
    assert clean is not None, _raw
    assert (clean["clock"], clean["quarter"], clean["away_score"], clean["home_score"]) == \
        (want["clock"], want["quarter"], want["away_score"], want["home_score"]), _raw
