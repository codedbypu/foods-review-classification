import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
import utils

RATING_CLASSES = np.array([1, 2, 3, 4, 5], dtype=np.float32)
HEX_COLORS = {
    1: "#e53935",
    2: "#ff9800",
    3: "#fbc02d",
    4: "#4caf50",
    5: "#00bcd4",
}


def get_hex_color(rating: float) -> str:
    clamped = max(1, min(5, int(np.round(rating))))
    return HEX_COLORS[clamped]


def expected_rating_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.dot(probs, RATING_CLASSES)


def load_baseline_meta() -> dict:
    """Load training metadata; fall back to config if missing."""
    if os.path.isfile(config.BASELINE_META_PATH):
        with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "use_lsa": config.BASELINE_USE_LSA,
        "use_extra_features": config.BASELINE_USE_EXTRA_FEATURES,
    }


def _truncate_for_inference(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    max_chars = meta.get("max_review_chars", config.MAX_REVIEW_CHARS)
    return utils.apply_text_truncation(df, max_chars)


def transform_baseline_features(
    df: pd.DataFrame,
    vectorizer,
    svd,
    meta: dict,
    char_vectorizer=None,
) -> np.ndarray:
    """Match train_baseline feature pipeline (TF-IDF, LSA, extras)."""
    df_model = _truncate_for_inference(df, meta)
    X_vec = vectorizer.transform(df_model["text"])
    if meta.get("use_char_tfidf", config.BASELINE_USE_CHAR_TFIDF) and char_vectorizer:
        X_char = char_vectorizer.transform(df_model["text"])
        X_vec = hstack([X_vec, X_char], format="csr")

    use_lsa = meta.get("use_lsa", config.BASELINE_USE_LSA)
    use_extra = meta.get("use_extra_features", config.BASELINE_USE_EXTRA_FEATURES)

    if use_lsa and svd is not None:
        main = svd.transform(X_vec)
        if use_extra:
            extra = utils.compute_extra_features(df_model["text"])
            return np.hstack([main, extra])
        return main

    if use_extra:
        extra = utils.compute_extra_features(df_model["text"])
        return hstack([X_vec, csr_matrix(extra)], format="csr")
    return X_vec


def _predict_baseline_raw(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (expected_rating, probs_or_none)."""
    vectorizer_path = config.TFIDF_VECTORIZER_PATH
    model_path = config.XGB_MODEL_PATH
    lsa_path = config.LSA_TRANSFORMER_PATH

    if not os.path.isfile(vectorizer_path) or not os.path.isfile(model_path):
        print(
            "Baseline artifacts not found. Run: python train_baseline.py",
            file=sys.stderr,
        )
        sys.exit(1)

    meta = load_baseline_meta()
    use_lsa = meta.get("use_lsa", config.BASELINE_USE_LSA)

    if use_lsa and not os.path.isfile(lsa_path):
        print(
            f"LSA transformer not found at {lsa_path}\n"
            "Re-run: python train_baseline.py",
            file=sys.stderr,
        )
        sys.exit(1)

    vectorizer = joblib.load(vectorizer_path)
    char_vectorizer = None
    if meta.get("use_char_tfidf", config.BASELINE_USE_CHAR_TFIDF):
        char_path = config.CHAR_TFIDF_VECTORIZER_PATH
        if os.path.isfile(char_path):
            char_vectorizer = joblib.load(char_path)
    svd = joblib.load(lsa_path) if use_lsa else None
    bst = xgb.Booster()
    bst.load_model(model_path)

    X_features = transform_baseline_features(
        df, vectorizer, svd, meta, char_vectorizer=char_vectorizer
    )
    dmatrix = xgb.DMatrix(X_features)
    raw = bst.predict(dmatrix)

    if meta.get("use_regression", config.BASELINE_USE_REGRESSION):
        pred = np.clip(raw + 1.0, 1.0, 5.0)
        return pred.astype(np.float64), None

    n_class = 3 if meta.get("use_3class", config.BASELINE_USE_3CLASS) else 5
    probs = raw.reshape(-1, n_class) if raw.ndim == 1 else raw
    if n_class == 3:
        class_idx = np.argmax(probs, axis=1)
        expected = utils.class3_to_expected_star(class_idx)
        return expected, probs
    return expected_rating_from_probs(probs), probs


def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    expected, _ = _predict_baseline_raw(df)
    return expected


def predict_baseline_with_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    return _predict_baseline_raw(df)


def predict_xlmr(df: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
    model_dir = config.XLMR_ARTIFACTS_DIR
    config_json = os.path.join(model_dir, "config.json")
    if not os.path.isdir(model_dir) or not os.path.isfile(config_json):
        print(
            f"XLM-R artifacts not found at {model_dir}\n"
            "Run: python train_xlmr.py",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device(config.TORCH_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()

    texts = df["text"].astype(str).tolist()
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoding = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=config.MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    stacked = np.vstack(all_probs)
    return expected_rating_from_probs(stacked)


def prepare_scoring_dataframe(file_path: str) -> pd.DataFrame:
    """Load, normalize, and clean reviews (same rules as training)."""
    df = utils.load_and_standardize_data(file_path)
    df, stats = utils.clean_review_dataframe(
        df,
        min_text_length=config.MIN_TEXT_LENGTH,
        drop_duplicates=config.DROP_DUPLICATE_TEXT,
        duplicate_keep=config.DUPLICATE_KEEP,
    )
    utils.log_cleaning_stats(stats, label=os.path.basename(file_path))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score reviews and flag anomalies.")
    parser.add_argument(
        "--model",
        choices=("baseline", "xlmr"),
        default="baseline",
        help="Which trained model to use for inference.",
    )
    parser.add_argument(
        "--input",
        default=config.RAW_DATA_PATH,
        help="Input CSV path (default: config.RAW_DATA_PATH).",
    )
    parser.add_argument(
        "--output",
        default=config.DEFAULT_SCORED_OUTPUT,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"--- Running Inference & Integrity Check (model={args.model}) ---")

    df = prepare_scoring_dataframe(args.input)

    if args.model == "baseline":
        df["ai_expected_rating"] = predict_baseline(df)
    else:
        df["ai_expected_rating"] = predict_xlmr(df)

    df["ai_hex_color"] = df["ai_expected_rating"].apply(get_hex_color)
    df["delta"] = np.abs(df["user_rating"] - df["ai_expected_rating"])
    df["is_anomaly"] = df["delta"] >= config.ANOMALY_THRESHOLD

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Finished scoring! Check result at '{args.output}'")


if __name__ == "__main__":
    main()
