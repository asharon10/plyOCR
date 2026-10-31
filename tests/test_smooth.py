"""smooth_records: the score channels after temporal smoothing."""
from __future__ import annotations

from plyocr.run_tesseract_ocr import smooth_records


def _records(home: list, away: list | None = None) -> list[dict]:
    away = away if away is not None else [0] * len(home)
    return [{"quarter": 1, "game_clock": "05:00", "home_score": h, "away_score": a}
            for h, a in zip(home, away)]


def _home(records: list[dict]) -> list:
    return [r["home_score"] for r in smooth_records(records)]


def test_a_sustained_misread_does_not_lock_the_score():
    # "11" read as "41" for a while (seen on a real game): the running max held
    # 41 for ~50 minutes until the real score passed it. The real reads win.
    home = [9] * 20 + [41] * 12 + [11] * 40 + [14] * 40 + [29] * 40 + [41] * 10 + [43] * 10 + [45] * 10
    out = _home(_records(home))
    assert 41 not in out[:152]
    assert out[40] == 11 and out[100] == 14 and out[140] == 29
    assert out[-1] == 45


def test_a_lone_spike_is_dropped():
    out = _home(_records([4] * 10 + [74] + [4] * 10 + [6] * 10))
    assert max(out[:21]) == 4 and out[-1] == 6


def test_scores_never_go_down():
    home = [2, 2, 4, 4, 4, 3, 4, 6, 6, 5, 6, 8, 8, 8, 8]
    out = _home(_records(home))
    assert all(a <= b for a, b in zip(out, out[1:]))
    assert out[-1] == 8


def test_before_the_first_read_stays_unknown():
    out = _home(_records([None] * 5 + [2] * 10))
    assert out[:5] == [None] * 5 and out[5:] == [2] * 10
