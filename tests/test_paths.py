import importlib
import json

import plyocr.paths


def test_shipped_defaults_exist():
    paths = importlib.reload(plyocr.paths)
    assert json.loads(paths.DEFAULT_ROI.read_text())["rois"]["clock"]
    assert json.loads(paths.ROI_TEMPLATES.read_text())


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PLYOCR_DEFAULT_ROI", str(tmp_path / "roi.json"))
    monkeypatch.setenv("PLYOCR_ROI_TEMPLATES", str(tmp_path / "t.json"))
    paths = importlib.reload(plyocr.paths)
    assert paths.DEFAULT_ROI == tmp_path / "roi.json"
    assert paths.ROI_TEMPLATES == tmp_path / "t.json"
    monkeypatch.undo()
    importlib.reload(plyocr.paths)
