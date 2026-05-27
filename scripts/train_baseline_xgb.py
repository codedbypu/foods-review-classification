from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from rris.data.io import read_reviews
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging
from rris.models.baseline.tfidf_xgb import (
    TfidfXgbConfig,
    build_vectorizer,
    build_xgb,
    expected_rating_from_proba,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TF-IDF + XGBoost baseline (1-5).")
    p.add_argument("--input", required=True, help="Input reviews .csv/.parquet with columns text,user_rating")
    p.add_argument("--out_dir", required=True, help="Output directory for artifacts")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_features", type=int, default=80_000)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_reviews(args.input)
    df = df.copy()
    df["text"] = df["text"].astype(str).map(normalize_text)

    y = df["user_rating"].astype(int).to_numpy()
    if not np.isin(y, [1, 2, 3, 4, 5]).all():
        bad = sorted(set(y.tolist()) - {1, 2, 3, 4, 5})
        raise ValueError(f"user_rating must be 1..5. Bad values: {bad[:20]}")
    y_cls = y - 1  # 0..4

    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].to_numpy(),
        y_cls,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_cls,
    )

    cfg = TfidfXgbConfig(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        random_state=args.seed,
    )

    logger.info("Building vectorizer (max_features=%s)", cfg.max_features)
    vec = build_vectorizer(cfg)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    logger.info("Training XGBoost")
    model = build_xgb(cfg)
    model.fit(Xtr, y_train)

    logger.info("Evaluating")
    proba = model.predict_proba(Xva)
    pred = np.argmax(proba, axis=1)
    acc = float(accuracy_score(y_val, pred))
    report = classification_report(y_val, pred, digits=4, output_dict=True)

    metrics = {"accuracy": acc, "report": report}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save artifacts
    joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")
    model.save_model(str(out_dir / "xgb_model.json"))

    logger.info("Saved artifacts to %s", out_dir)


if __name__ == "__main__":
    main()

