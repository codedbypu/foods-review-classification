"""
Rating → hex color for maps and exports.

COMMON ERRORS:
  - ValueError: rating must be 1..5 — clip/round model expected_rating before rating_to_hex().
"""
from __future__ import annotations

from typing import Dict


PALETTE: Dict[int, str] = {
    1: "#e53935",
    2: "#ff9800",
    3: "#fbc02d",
    4: "#4caf50",
    5: "#00bcd4",
}


def rating_to_hex(rating: int) -> str:
    try:
        r = int(rating)
    except Exception as e:
        raise ValueError(f"rating must be 1..5, got {rating!r}") from e
    if r not in PALETTE:
        raise ValueError(f"rating must be 1..5, got {rating!r}")
    return PALETTE[r]

