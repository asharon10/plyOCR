"""Every command-line tool starts (imports resolve, argparse builds)."""
import subprocess
import sys

import pytest

TOOLS = ["run_tesseract_ocr", "detect_rois", "read_frame", "spot_check", "verify_ocr", "change_map",
         "spots_roi", "quarter_templates", "extract_scoreboard", "compare_decoders", "claude_review"]


@pytest.mark.parametrize("tool", TOOLS)
def test_help(tool):
    r = subprocess.run([sys.executable, "-m", f"plyocr.{tool}", "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
