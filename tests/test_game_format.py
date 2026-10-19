import json

from plyocr import game_format


def test_defaults_without_a_game():
    assert game_format.load(None) == {"quarters": 4, "quarterMinutes": 10, "otMinutes": 5}


def test_reads_game_json_and_ignores_bad_values(tmp_path):
    (tmp_path / "game.json").write_text(json.dumps({"gameFormat": {"quarters": 4, "quarterMinutes": 12, "otMinutes": -1}}))
    fmt = game_format.load(tmp_path)
    assert fmt == {"quarters": 4, "quarterMinutes": 12, "otMinutes": 5}
    assert game_format.period_seconds(fmt, 1) == 720
    assert game_format.period_seconds(fmt, 5) == 300
