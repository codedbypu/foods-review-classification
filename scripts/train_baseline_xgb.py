from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from rris.data.io import read_reviews
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging
from rris.models.baseline.tfidf_xgb import (
    TfidfXgbConfig,
    build_vectorizer,
    fit_vectorizer_with_progress,
    train_xgb_with_progress,
)
from rris.progress import stage

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
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for tokenize / CPU XGBoost")
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap for smoke/dev runs")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))
    show_progress = not args.no_progress

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with stage("Load and prepare data", enabled=show_progress):
        df = read_reviews(args.input)
        if args.max_rows is not None and len(df) > args.max_rows:
            df = df.sample(n=args.max_rows, random_state=args.seed)
        df = df.copy()
        df["text"] = df["text"].astype(str).map(normalize_text)

        y = df["user_rating"].astype(int).to_numpy()
        if not np.isin(y, [1, 2, 3, 4, 5]).all():
            bad = sorted(set(y.tolist()) - {1, 2, 3, 4, 5})
            raise ValueError(f"user_rating must be 1..5. Bad values: {bad[:20]}")
        y_cls = y - 1

        stratify = y_cls
        if np.min(np.bincount(y_cls)) < 2:
            logger.warning("Some rating classes have <2 samples; split without stratify")
            stratify = None
        X_train, X_val, y_train, y_val = train_test_split(
            df["text"].to_numpy(),
            y_cls,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )

    cfg = TfidfXgbConfig(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        random_state=args.seed,
        device=args.device,
        n_jobs=args.n_jobs,
    )

    with stage("Vectorize (tokenize + TF-IDF)", enabled=show_progress):
        vec = build_vectorizer(cfg)
        Xtr, Xva = fit_vectorizer_with_progress(
            vec,
            X_train,
            X_val,
            n_jobs=args.n_jobs,
            show_progress=show_progress,
        )

    with stage("Train XGBoost", enabled=show_progress):
        model, device_used, device_requested = train_xgb_with_progress(
            cfg,
            Xtr,
            y_train,
            show_progress=show_progress,
            verbose=True,
        )

    with stage("Evaluate on validation set", enabled=show_progress):
        proba = model.predict_proba(Xva)
        pred = np.argmax(proba, axis=1)
        acc = float(accuracy_score(y_val, pred))
        report = classification_report(y_val, pred, digits=4, output_dict=True)

    metrics = {
        "accuracy": acc,
        "report": report,
        "xgb_device_requested": device_requested,
        "xgb_device_used": device_used,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with stage("Save artifacts", enabled=show_progress):
        joblib.dump(vec, out_dir / "tfidf_vectorizer.joblib")
        model.save_model(str(out_dir / "xgb_model.json"))

    logger.info("Saved artifacts to %s (device=%s)", out_dir, device_used)


if __name__ == "__main__":
    main()
