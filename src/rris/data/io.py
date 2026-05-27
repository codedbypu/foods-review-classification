from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def read_reviews(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")

    if "text" not in df.columns:
        raise ValueError("Missing required column: text")
    if "user_rating" not in df.columns:
        raise ValueError("Missing required column: user_rating")

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

