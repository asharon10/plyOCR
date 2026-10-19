"""A full run split into shards and merged writes exactly what one run writes.

Builds a short local video from the scoreboard fixtures (a local file decodes the
same every time, unlike a stream), then compares the outputs byte for byte.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from conftest import needs_tesseract
from test_scoreboard_golden import EXPECTED, ROI, scoreboard_frame

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
OUTPUTS = ["timeline.json", "timeline_raw.json", "ocr_timeline.csv", "ocr_coverage.json"]


@pytest.fixture(scope="module")
def game(tmp_path_factory):
    """A 96 s, 1 fps-sampled game: each scoreboard sample is held for 8 s."""
    d = tmp_path_factory.mktemp("game")
    frames = d / "frames"
    frames.mkdir()
    n = 0
    for sample in sorted(EXPECTED):
        img = scoreboard_frame(sample)
        for _ in range(8 * 5):          # 8 s at 5 fps
            cv2.imwrite(str(frames / f"{n:05d}.png"), img)
            n += 1
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "5", "-i", str(frames / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "5", str(d / "game.mp4")], check=True)
    shutil.rmtree(frames)
    (d / "game.json").write_text(json.dumps({"video": {"localFile": "game.mp4"}, "ocrMethod": "tesseract"}))
    (d / "roi.json").write_text(json.dumps(ROI))
    return d


def ocr(game_dir: Path, out: Path, *extra: str, decoder: str = "seek") -> None:
    subprocess.run([sys.executable, "-m", "plyocr.run_tesseract_ocr", "--game", str(game_dir / "game.json"),
                    "--fps", "1", "--workers", "2", "--no-skip-breaks", "--decoder", decoder,
                    "--derived", str(out), *extra], check=True, capture_output=True, text=True, timeout=600)


@needs_tesseract
@needs_ffmpeg
@pytest.mark.parametrize("decoder", ["seek", "keyframe"])
def test_shards_merge_to_the_single_run(game, tmp_path, decoder):
    single, sharded = tmp_path / "single", tmp_path / "sharded"
    ocr(game, single, decoder=decoder)
    for i in range(3):
        ocr(game, sharded, "--shard", f"{i}/3", decoder=decoder)
    ocr(game, sharded, "--merge-shards", *(str(sharded / f"shard-{i}.json") for i in range(3)), decoder=decoder)

    rows = json.loads((single / "timeline.json").read_text())
    assert len(rows) > 50, "the synthetic game should read cleanly"
    for name in OUTPUTS:
        assert (sharded / name).read_bytes() == (single / name).read_bytes(), name


@needs_tesseract
@needs_ffmpeg
def test_merge_refuses_missing_shards(game, tmp_path):
    ocr(game, tmp_path, "--shard", "0/2")
    r = subprocess.run([sys.executable, "-m", "plyocr.run_tesseract_ocr", "--game", str(game / "game.json"),
                        "--fps", "1", "--workers", "2", "--no-skip-breaks", "--decoder", "seek",
                        "--derived", str(tmp_path), "--merge-shards", str(tmp_path / "shard-0.json")],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode != 0 and "missing" in (r.stderr + r.stdout)
