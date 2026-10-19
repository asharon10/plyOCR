"""Golden test: real broadcast scoreboard crops (digits only) → clock, quarter, scores.

tests/fixtures/scoreboards/<sample>/<field>.png are cut from a 1920x1080 broadcast at
the boxes in tests/fixtures/roi.json; expected.json was read by eye from the crops.
Each sample is pasted onto a blank frame and read with the full per-frame pipeline.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import needs_tesseract
from plyocr import run_tesseract_ocr as ocr

FIXTURES = Path(__file__).parent / "fixtures"
ROI = ocr.with_quarter_templates(json.loads((FIXTURES / "roi.json").read_text()))
EXPECTED = {k: v for k, v in json.loads((FIXTURES / "expected.json").read_text()).items() if not k.startswith("_")}


def scoreboard_frame(sample: str) -> np.ndarray:
    frame = np.zeros((ROI["frame_height"], ROI["frame_width"], 3), np.uint8)
    for field, b in ROI["rois"].items():
        frame[b["y"]:b["y"] + b["h"], b["x"]:b["x"] + b["w"]] = cv2.imread(str(FIXTURES / "scoreboards" / sample / f"{field}.png"))
    return frame


@needs_tesseract
@pytest.mark.parametrize("sample", sorted(EXPECTED))
def test_tesseract_reads_scoreboard(sample, monkeypatch):
    monkeypatch.setattr(ocr, "_METHOD", "tesseract")
    _raw, clean = ocr.process_frame(scoreboard_frame(sample), ROI["rois"], ROI["frame_width"], ROI["frame_height"])
    want = EXPECTED[sample]
    assert clean is not None
    assert clean["clock"] == want["clock"]
    assert clean["home_score"] == want["home_score"]
    assert clean["away_score"] == want["away_score"]
    # An empty quarter box comes back as 0 ("unknown").
    assert clean["quarter"] == (want["quarter"] or 0)
