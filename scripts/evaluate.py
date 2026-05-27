from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from rris.data.io import read_reviews
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging
from rris.models.baseline.tfidf_xgb import expected_rating_from_proba
from rris.models.xlmr.sentiment_trainer import probs_and_expected_rating_from_logits

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained model on a labeled dataset.")
    p.add_argument("--input", required=True, help="Input reviews with text,user_rating and optional lang")
    p.add_argument("--model_type", required=True, choices=["baseline_xgb", "xlmr"])
    p.add_argument("--artifact_dir", required=True)
    p.add_argument("--out", required=True, help="Output metrics json path")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


@torch.no_grad()
def _predict_xlmr(df: pd.DataFrame, artifact_dir: Path) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(str(artifact_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(artifact_dir))
    model.eval()
    enc = tok(
        df["text"].astype(str).map(normalize_text).tolist(),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    logits = model(**enc).logits.detach().cpu().numpy()
    return logits


def _predict_baseline(df: pd.DataFrame, artifact_dir: Path) -> np.ndarray:
    vec = joblib.load(artifact_dir / "tfidf_vectorizer.joblib")
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(artifact_dir / "xgb_model.json"))
    X = vec.transform(df["text"].astype(str).map(normalize_text).to_numpy())
    proba = model.predict_proba(X)
    # convert to logits-like via log; keep safe
    eps = 1e-8
    return np.log(np.clip(proba, eps, 1.0))


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))

    artifact_dir = Path(args.artifact_dir)
    df = read_reviews(args.input).copy()
    y = df["user_rating"].astype(int).to_numpy()
    y_cls = y - 1

    if args.model_type == "xlmr":
        logits = _predict_xlmr(df, artifact_dir)
        probs, expected = probs_and_expected_rating_from_logits(logits)
    else:
        logits = _predict_baseline(df, artifact_dir)
        # baseline returned log-probs; re-softmax back to probs for expected
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

    # Per-language slices if available
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

