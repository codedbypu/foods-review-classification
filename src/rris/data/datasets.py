"""
Pandas → Hugging Face Dataset for XLM-R training.

COMMON ERRORS:
  - ValueError: Missing text/rating column — use labeled parquet from read_reviews.
  - ValueError: user_rating must be 1..5 — filter invalid stars before train_xlmr_sentiment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset

from rris.data.text import normalize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentDatasetConfig:
    text_col: str = "text"
    rating_col: str = "user_rating"  # 1..5
    lang_col: Optional[str] = "lang"


def df_to_hf_sentiment_dataset(
    df: pd.DataFrame, cfg: SentimentDatasetConfig = SentimentDatasetConfig()
) -> Dataset:
    if cfg.text_col not in df.columns:
        raise ValueError(f"Missing text column: {cfg.text_col}")
    if cfg.rating_col not in df.columns:
        raise ValueError(f"Missing rating column: {cfg.rating_col}")

    out = df.copy()
    out[cfg.text_col] = out[cfg.text_col].astype(str).map(normalize_text)
    y = out[cfg.rating_col].astype(int)
    if not np.isin(y.to_numpy(), [1, 2, 3, 4, 5]).all():
        bad = sorted(set(y.to_list()) - {1, 2, 3, 4, 5})
        raise ValueError(f"{cfg.rating_col} must be 1..5. Bad values: {bad[:20]}")

    out = out.rename(columns={cfg.text_col: "text"})
    out["label"] = (y - 1).astype(int)  # 0..4

    keep_cols = ["text", "label"]
    if cfg.lang_col and cfg.lang_col in out.columns:
        keep_cols.append(cfg.lang_col)

    return Dataset.from_pandas(out[keep_cols], preserve_index=False)

