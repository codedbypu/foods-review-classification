from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from rris.data.tokenizers import MultilingualTokenizer, MultilingualTokenizerConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TfidfXgbConfig:
    # Vectorizer
    max_features: int = 80_000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

    # Model
    n_estimators: int = 400
    learning_rate: float = 0.1
    max_depth: int = 8
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    tree_method: str = "hist"
    n_jobs: int = -1
    random_state: int = 42


def build_vectorizer(cfg: TfidfXgbConfig) -> TfidfVectorizer:
    tok = MultilingualTokenizer(MultilingualTokenizerConfig(keep_punct=False))

    def _tokenize(text: str) -> List[str]:
        return tok.tokenize(text)

    return TfidfVectorizer(
        tokenizer=_tokenize,
        token_pattern=None,  # required when tokenizer is provided
        lowercase=False,  # Thai has no case; keep exact for EN
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=True,
    )


def build_xgb(cfg: TfidfXgbConfig) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        tree_method=cfg.tree_method,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        eval_metric="mlogloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def expected_rating_from_proba(proba_5: np.ndarray) -> np.ndarray:
    """
    Convert softmax probabilities to expected rating in [1,5].
    proba_5: (N, 5) with columns for classes 1..5 mapped to 0..4 indices.
    """
    if proba_5.ndim != 2 or proba_5.shape[1] != 5:
        raise ValueError(f"Expected proba shape (N, 5), got {proba_5.shape}")
    classes = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    s = proba_5.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = proba_5 / s
    return (p * classes[None, :]).sum(axis=1)

