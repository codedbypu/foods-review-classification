"""
Evaluate saved baseline or XLM-R on labeled data.

COMMON ERRORS:
  - Artifact not found: run train_* first with same --out_dir / --artifact_dir.
  - Baseline on full parquet after train with --max_rows: metrics reflect mismatch — use same data slice.
  - Kernel crash: eval ~19k rows with --n_jobs -1 (tokenize RAM) — use --n_jobs 4 in notebooks.
  - XLM-R CUDA OOM during batched predict: lower --batch_size (default 32; try 8 on 6GB GPU).
  - joblib/XGB load fail: artifact_dir must contain tfidf_vectorizer.joblib + xgb_model.json.
  - See docs/ERRORS_AND_FIXES.md §9.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import _scratch_init  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xgboost import XGBClassifier

from rris.data.io import read_reviews
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging
from rris.runtime_env import add_scratch_argument, apply_scratch_from_args
from rris.models.baseline.tfidf_xgb import expected_rating_from_proba, transform_texts_with_progress
from rris.models.xlmr.sentiment_trainer import probs_and_expected_rating_from_logits
from rris.progress import stage, tqdm_if

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model on a labeled dataset.")
    p.add_argument("--input", required=True, help="Input reviews with text,user_rating and optional lang")
    p.add_argument("--model_type", required=True, choices=["baseline_xgb", "xlmr"])
    p.add_argument("--artifact_dir", required=True)
    p.add_argument("--out", required=True, help="Output metrics json path")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for XLM-R inference")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs for baseline vectorize")
    add_scratch_argument(p)
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


@torch.no_grad()
def _predict_xlmr_batched(
    df: pd.DataFrame,
    artifact_dir: Path,
    *,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(str(artifact_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(artifact_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df["text"].astype(str).map(normalize_text).tolist()
    all_logits: list[np.ndarray] = []
    batches = range(0, len(texts), batch_size)
    for start in tqdm_if(
        list(batches),
        show_progress=show_progress,
        desc="XLM-R predict",
        total=(len(texts) + batch_size - 1) // batch_size,
    ):
        batch_texts = texts[start : start + batch_size]
        enc = tok(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        all_logits.append(logits)
    return np.vstack(all_logits) if all_logits else np.zeros((0, 5), dtype=np.float32)


def _predict_baseline(
    df: pd.DataFrame,
    artifact_dir: Path,
    *,
    n_jobs: int,
    show_progress: bool,
) -> np.ndarray:
    vec = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
    model = XGBClassifier()
    model.load_model(str(artifact_dir / "xgb_model.json"))
    texts = df["text"].astype(str).map(normalize_text).to_numpy()
    X = transform_texts_with_progress(
        vec,
        texts,
        n_jobs=n_jobs,
        show_progress=show_progress,
        desc="Baseline vectorize",
    )
    proba = model.predict_proba(X)
    eps = 1e-8
    return np.log(np.clip(proba, eps, 1.0))


def main() -> None:
    args = parse_args()
    apply_scratch_from_args(Path(__file__).resolve().parents[1], args)
    setup_logging(LoggingConfig(level=args.log_level))
    show_progress = not args.no_progress

    artifact_dir = Path(args.artifact_dir)
    with stage("Load data", enabled=show_progress):
        # Evaluates ALL rows in --input; if train used --max_rows, metrics compare different populations
        df = read_reviews(args.input).copy()
        y = df["user_rating"].astype(int).to_numpy()
        y_cls = y - 1

    with stage("Predict", enabled=show_progress):
        if args.model_type == "xlmr":
            logits = _predict_xlmr_batched(
                df,
                artifact_dir,
                batch_size=args.batch_size,
                show_progress=show_progress,
            )
            probs, expected = probs_and_expected_rating_from_logits(logits)
        else:
            logits = _predict_baseline(
                df,
                artifact_dir,
                n_jobs=args.n_jobs,
                show_progress=show_progress,
            )
            x = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(x)
            probs = exp / exp.sum(axis=1, keepdims=True)
            expected = expected_rating_from_proba(probs)

    pred_cls = np.argmax(probs, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_cls, pred_cls)),
        "classification_report": classification_report(y_cls, pred_cls, digits=4, output_dict=True),
        "expected_rating_mae": float(np.mean(np.abs(expected - y.astype(float)))),
    }

    if "lang" in df.columns:
        metrics["by_lang"] = {}
        for lang in sorted(df["lang"].dropna().astype(str).unique().tolist()):
            mask = df["lang"].astype(str) == lang
            if mask.sum() < 5:
                continue
            yl = y_cls[mask.to_numpy()]
            pl = pred_cls[mask.to_numpy()]
            metrics["by_lang"][lang] = {"accuracy": float(accuracy_score(yl, pl)), "n": int(mask.sum())}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote metrics: %s", out)


if __name__ == "__main__":
    main()
