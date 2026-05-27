from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rris.data.io import read_reviews, write_table
from rris.data.text import normalize_text
from rris.integrity.anomaly import anomaly_check
from rris.logging_utils import LoggingConfig, setup_logging
from rris.models.baseline.tfidf_xgb import expected_rating_from_proba
from rris.models.xlmr.aspect_extractor import AspectExtractionConfig, extract_aspect_mentions
from rris.models.xlmr.sentiment_trainer import probs_and_expected_rating_from_logits
from rris.viz.colors import rating_to_hex
from rris.viz.geo_export import to_geojson_points, write_geojson

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score reviews, compute anomalies, attach hex colors, export CSV/GeoJSON.")
    p.add_argument("--input", required=True, help="Input reviews .csv/.parquet (requires text,user_rating)")
    p.add_argument("--out", required=True, help="Output scored table path (.csv/.parquet)")
    p.add_argument(
        "--model_type",
        required=True,
        choices=["baseline_xgb", "xlmr"],
        help="Which model artifacts to use for scoring",
    )
    p.add_argument("--artifact_dir", required=True, help="Model artifact directory")
    p.add_argument("--anomaly_threshold", type=float, default=2.0)
    p.add_argument("--geojson_out", default=None, help="Optional GeoJSON output path (requires lat/lon columns)")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


@torch.no_grad()
def score_xlmr(df: pd.DataFrame, artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    tok = AutoTokenizer.from_pretrained(str(artifact_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(artifact_dir))
    model.eval()

    texts = df["text"].astype(str).map(normalize_text).tolist()
    enc = tok(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    logits = model(**enc).logits.detach().cpu().numpy()
    probs, expected = probs_and_expected_rating_from_logits(logits)
    return probs, expected


def score_baseline_xgb(df: pd.DataFrame, artifact_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    vec_path = artifact_dir / "tfidf_vectorizer.joblib"
    model_path = artifact_dir / "xgb_model.json"
    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError("Baseline artifacts missing: tfidf_vectorizer.joblib and/or xgb_model.json")

    vec = joblib.load(vec_path)
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(model_path))

    X = vec.transform(df["text"].astype(str).map(normalize_text).to_numpy())
    proba = model.predict_proba(X)
    expected = expected_rating_from_proba(proba)
    return proba, expected


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))

    artifact_dir = Path(args.artifact_dir)
    df = read_reviews(args.input).copy()
    df["text"] = df["text"].astype(str).map(normalize_text)
    if "review_id" not in df.columns:
        df["review_id"] = [str(i) for i in range(len(df))]

    if args.model_type == "xlmr":
        probs, expected = score_xlmr(df, artifact_dir)
    else:
        probs, expected = score_baseline_xgb(df, artifact_dir)

    df["ai_expected_rating"] = expected.astype(float)
    df["ai_pred_class"] = (np.argmax(probs, axis=1) + 1).astype(int)

    # Discretize expected rating to 1..5 for palette (round half up via +0.5 then floor)
    df["ai_rating_round"] = np.clip(np.floor(df["ai_expected_rating"] + 0.5), 1, 5).astype(int)
    df["ai_hex_color"] = df["ai_rating_round"].map(rating_to_hex)

    deltas = []
    flags = []
    for ur, ar in zip(df["user_rating"].astype(float).to_numpy(), df["ai_expected_rating"].to_numpy()):
        res = anomaly_check(ur, float(ar), threshold=float(args.anomaly_threshold))
        deltas.append(res.delta)
        flags.append(res.is_anomaly)
    df["delta"] = deltas
    df["is_anomaly"] = flags

    # Aspect mentions (hybrid), producing aspect-sentence pairs for downstream scoring
    aspect_rows = []
    for rid, text in zip(df["review_id"].astype(str).to_list(), df["text"].to_list()):
        mentions = extract_aspect_mentions(review_id=rid, text=text, cfg=AspectExtractionConfig())
        for m in mentions:
            aspect_rows.append(
                {
                    "review_id": m.review_id,
                    "sentence_idx": m.sentence_idx,
                    "sentence_text": m.sentence_text,
                    "aspect": m.aspect,
                    "surface": m.surface,
                }
            )

    out_path = Path(args.out)
    write_table(df, out_path)

    if aspect_rows:
        aspect_df = pd.DataFrame(aspect_rows)
        write_table(aspect_df, out_path.with_name(out_path.stem + "_aspects" + out_path.suffix))

    if args.geojson_out:
        geo = to_geojson_points(df)
        write_geojson(geo, args.geojson_out)

    logger.info("Wrote scored output: %s", out_path)


if __name__ == "__main__":
    main()

