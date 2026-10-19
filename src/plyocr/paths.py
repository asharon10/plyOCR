"""Where plyocr finds its default scoreboard boxes.

The package ships a baseline ROI and learned per-resolution templates. A host
app that learns its own templates points at them with the environment:
  PLYOCR_DEFAULT_ROI     baseline roi.json   (used when nothing better exists)
  PLYOCR_ROI_TEMPLATES   roi_templates.json  ({"<w>x<h>": roi, ...})
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ROI = Path(os.environ.get("PLYOCR_DEFAULT_ROI") or DATA_DIR / "default_roi.json")
ROI_TEMPLATES = Path(os.environ.get("PLYOCR_ROI_TEMPLATES") or DATA_DIR / "roi_templates.json")
