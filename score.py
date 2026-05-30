import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
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


def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    vectorizer_path = config.TFIDF_VECTORIZER_PATH
    model_path = config.XGB_MODEL_PATH

    vectorizer = joblib.load(vectorizer_path)
    bst = xgb.Booster()
    bst.load_model(model_path)

    X_vec = vectorizer.transform(df["text"])
    dmatrix = xgb.DMatrix(X_vec)
    probs = bst.predict(dmatrix)
    return expected_rating_from_probs(probs)


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

    df = utils.load_and_standardize_data(args.input)

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
