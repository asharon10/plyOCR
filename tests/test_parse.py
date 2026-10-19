import pytest

from plyocr.run_tesseract_ocr import parse_clock, parse_quarter, parse_score


@pytest.mark.parametrize("raw,want", [
    ("6:06", "6:06"), ("06:06", "06:06"), ("10:00", "10:00"), ("0:00", "0:00"),
    ("12.3", "12.3"), ("9.8", "9.8"),            # under a minute: seconds + tenths
    ("4 21", None), ("1:5", None), ("abc", None), ("", None),
])
def test_parse_clock(raw, want):
    assert parse_clock(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("1", 1), ("4", 4), ("2.", 2), ("OT", 5), ("0T", 5),
    ("5", None), ("7", None), ("x", None), ("", None),
])
def test_parse_quarter(raw, want):
    assert parse_quarter(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("0", 0), ("86", 86), ("100", 100), ("O5", 5), ("1O", 10),
    ("999", None), ("1234", None), ("abc", None), ("", None),
])
def test_parse_score(raw, want):
    assert parse_score(raw) == want
