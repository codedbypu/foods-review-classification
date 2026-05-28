"""
Load/save review tables (.csv / .parquet).

COMMON ERRORS:
  - Missing column text: CSV ไม่มี header หรือเป็นไฟล์ข้อความคอลัมน์เดียว (แถวแรกกลายเป็นชื่อคอลัมน์)
  - Missing column user_rating: ไฟล์ไม่มีดาว 1-5 (preprocess ใช้ require_rating=False ได้)
  - Wongnai HF/Kaggle: ใช้ review_body + stars → map เป็น text + user_rating อัตโนมัติ
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_TEXT_ALIASES = frozenset(
    {"text", "review", "Review", "review_body", "content", "comment", "review_text"}
)
_RATING_ALIASES = frozenset(
    {"user_rating", "rating", "Rating", "score", "stars", "label", "star", "rate"}
)


def _looks_like_review_text_header(columns: list) -> bool:
    """True when pandas used the first review line as the column name (no real header)."""
    if len(columns) != 1:
        return False
    name = str(columns[0])
    return len(name) > 80 or "\n" in name


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Map Wongnai/HF column names (review_body, stars) to text, user_rating."""
    out = df.copy()
    for c in list(out.columns):
        key = str(c).strip()
        if key in _TEXT_ALIASES and "text" not in out.columns:
            out = out.rename(columns={c: "text"})
        elif key in _RATING_ALIASES and "user_rating" not in out.columns:
            out = out.rename(columns={c: "user_rating"})
    return out


def _load_csv_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ERROR FIX: single-column text export — re-read without treating row 1 as header
    if "text" not in df.columns and (_looks_like_review_text_header(list(df.columns)) or len(df.columns) == 1):
        df = pd.read_csv(path, header=None, names=["text"])

    return _rename_aliases(df)


def read_reviews(path: str | Path, *, require_rating: bool = True) -> pd.DataFrame:
    """
    require_rating=False: for preprocess only.
    require_rating=True: train/evaluate/score (needs user_rating 1..5).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
        df = _rename_aliases(df)
    elif path.suffix.lower() in {".csv"}:
        df = _load_csv_reviews(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")

    if "text" not in df.columns:
        raise ValueError(
            "Missing required column: text. "
            f"Found columns: {list(df.columns)}. "
            "Expected a review text column (e.g. text, review) or a single-column CSV of reviews."
        )
    if require_rating and "user_rating" not in df.columns:
        raise ValueError(
            "Missing required column: user_rating (1..5). "
            f"Found columns: {list(df.columns)}. "
            "This file looks like text-only reviews. For Wongnai labeled data, use "
            "review/review_dataset.zip from https://github.com/wongnai/wongnai-corpus "
            "(not a single-column text export)."
        )

    if not require_rating and "user_rating" not in df.columns:
        logger.warning(
            "Input has no user_rating column; preprocess will continue but training scripts require labels."
        )

    return df


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in {".parquet"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")
