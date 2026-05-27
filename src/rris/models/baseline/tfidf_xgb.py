from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

from rris.data.tokenizers import MultilingualTokenizer, MultilingualTokenizerConfig
from rris.models.baseline.xgb_device import DeviceChoice, build_xgb_classifier, fit_xgb_with_device_fallback
from rris.progress import log_stage, tqdm_if

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
    device: DeviceChoice = "auto"


def _make_tokenizer() -> MultilingualTokenizer:
    return MultilingualTokenizer(MultilingualTokenizerConfig(keep_punct=False))


def _tokenize_one(text: str) -> List[str]:
    return _make_tokenizer().tokenize(text)


def _join_tokens(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def _vectorizer_tokenize(text: str) -> List[str]:
    """Module-level tokenizer (pickle-safe) for inference on raw review text."""
    return _make_tokenizer().tokenize(text)


def _split_prejoined(text: str) -> List[str]:
    """Tokenizer for space-joined pre-tokenized docs during parallel fit."""
    return text.split()


# Backward compat for vectorizers saved with the interim analyzer name
_space_split_analyzer = _split_prejoined


def build_vectorizer(cfg: TfidfXgbConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=_vectorizer_tokenize,
        token_pattern=None,
        lowercase=False,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=True,
    )


def tokenize_corpus_parallel(
    texts: Sequence[str],
    *,
    n_jobs: int = -1,
    show_progress: bool = True,
    desc: str = "Tokenize",
) -> List[List[str]]:
    """Tokenize texts with joblib.Parallel and optional tqdm."""
    n = len(texts)
    if n == 0:
        return []

    jobs = n_jobs if n_jobs != 0 else -1
    chunk = max(1, min(256, n // (max(1, (jobs if jobs > 0 else 1) * 4) or 1)))

    if jobs == 1 or n < 32:
        iterator = tqdm_if(texts, show_progress=show_progress, desc=desc, total=n)
        return [_tokenize_one(t) for t in iterator]

    batches: List[Tuple[int, List[str]]] = []
    for start in range(0, n, chunk):
        batches.append((start, list(texts[start : start + chunk])))

    results: List[Optional[List[str]]] = [None] * n

    def _process_batch(start: int, batch: List[str]) -> Tuple[int, List[List[str]]]:
        return start, [_tokenize_one(t) for t in batch]

    parallel = Parallel(n_jobs=jobs, prefer="threads")
    batch_iter = tqdm_if(
        batches,
        show_progress=show_progress,
        desc=desc,
        total=len(batches),
    )
    for start, tokenized in parallel(delayed(_process_batch)(s, b) for s, b in batch_iter):
        for i, toks in enumerate(tokenized):
            results[start + i] = toks

    return [r if r is not None else [] for r in results]  # type: ignore[misc]


def fit_vectorizer_with_progress(
    vec: TfidfVectorizer,
    train_texts: Sequence[str],
    val_texts: Sequence[str],
    *,
    n_jobs: int = -1,
    show_progress: bool = True,
) -> Tuple[spmatrix, spmatrix]:
    """
    Fit TF-IDF on train and transform val using parallel pre-tokenization.

    The saved vectorizer keeps MultilingualTokenizer for inference on raw text.
    """
    with log_stage("TF-IDF: tokenize train"):
        train_toks = tokenize_corpus_parallel(
            train_texts,
            n_jobs=n_jobs,
            show_progress=show_progress,
            desc="Tokenize train",
        )
    train_docs = [_join_tokens(t) for t in train_toks]

    with log_stage("TF-IDF: fit train"):
        tok_fn = vec.tokenizer
        vec.tokenizer = _split_prejoined
        try:
            Xtr = vec.fit_transform(train_docs)
        finally:
            vec.tokenizer = tok_fn

    with log_stage("TF-IDF: tokenize val"):
        val_toks = tokenize_corpus_parallel(
            val_texts,
            n_jobs=n_jobs,
            show_progress=show_progress,
            desc="Tokenize val",
        )
    val_docs = [_join_tokens(t) for t in val_toks]

    with log_stage("TF-IDF: transform val"):
        tok_fn = vec.tokenizer
        vec.tokenizer = _split_prejoined
        try:
            Xva = vec.transform(val_docs)
        finally:
            vec.tokenizer = tok_fn

    return Xtr, Xva


def transform_texts_with_progress(
    vec: TfidfVectorizer,
    texts: Sequence[str],
    *,
    n_jobs: int = -1,
    show_progress: bool = True,
    desc: str = "Vectorize",
) -> spmatrix:
    """Transform raw texts via parallel tokenize + vectorizer (inference-safe vec)."""
    toks = tokenize_corpus_parallel(
        texts,
        n_jobs=n_jobs,
        show_progress=show_progress,
        desc=desc,
    )
    docs = [_join_tokens(t) for t in toks]
    tok_fn = vec.tokenizer
    vec.tokenizer = _split_prejoined
    try:
        return vec.transform(docs)
    finally:
        vec.tokenizer = tok_fn


def _xgb_base_kwargs(cfg: TfidfXgbConfig) -> Dict[str, Any]:
    return {
        "objective": "multi:softprob",
        "num_class": 5,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "n_estimators": cfg.n_estimators,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_lambda": cfg.reg_lambda,
        "eval_metric": "mlogloss",
        "random_state": cfg.random_state,
    }


def build_xgb(cfg: TfidfXgbConfig) -> XGBClassifier:
    base = _xgb_base_kwargs(cfg)
    model, _, _ = build_xgb_classifier(
        base_kwargs=base,
        device=cfg.device,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
    )
    return model


def train_xgb_with_progress(
    cfg: TfidfXgbConfig,
    Xtr: spmatrix,
    y_train: np.ndarray,
    *,
    show_progress: bool = True,
    verbose: bool = True,
) -> Tuple[XGBClassifier, str, DeviceChoice]:
    """Build and fit XGBoost with device auto-detect and CPU fallback."""
    base = _xgb_base_kwargs(cfg)
    model, resolved, requested = build_xgb_classifier(
        base_kwargs=base,
        device=cfg.device,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
    )
    logger.info("XGBoost device resolved: %s (requested=%s)", resolved, requested)
    fitted, used = fit_xgb_with_device_fallback(
        model,
        Xtr,
        y_train,
        device_requested=cfg.device,
        n_jobs=cfg.n_jobs,
        tree_method=cfg.tree_method,
        base_kwargs=base,
        verbose=verbose,
    )
    return fitted, used, requested


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
