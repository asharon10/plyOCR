"""The game's format — quarter count and quarter / overtime length — from its
game.json (`gameFormat`, written by the API from the league setup). Every clock
calculation should use this instead of assuming FIBA's 4 × 10 min, OT 5."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT = {"quarters": 4, "quarterMinutes": 10, "otMinutes": 5}


def load(game_dir_or_json: str | Path | None) -> dict:
    p = Path(game_dir_or_json) if game_dir_or_json else None
    if p and p.is_dir():
        p = p / "game.json"
    try:
        f = json.loads(p.read_text()).get("gameFormat") or {} if p and p.exists() else {}
    except Exception:
        f = {}
    return {k: f.get(k) if isinstance(f.get(k), (int, float)) and f.get(k) > 0 else v for k, v in DEFAULT.items()}


def period_seconds(fmt: dict, q: int) -> int:
    """Length of period q (1-based) in seconds: regulation quarter or overtime."""
    return int((fmt["otMinutes"] if q > fmt["quarters"] else fmt["quarterMinutes"]) * 60)
