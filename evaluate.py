import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from score import prepare_scoring_dataframe
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

import config
import utils
from score import predict_baseline, predict_baseline_with_probs, predict_xlmr

DEFAULT_EVAL_INPUT = config.WONGNAI_TEST_PATH
LABELS = [1, 2, 3, 4, 5]


def _baseline_uses_lsa() -> bool:
    if os.path.isfile(config.BASELINE_META_PATH):
        with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        return bool(meta.get("use_lsa", config.BASELINE_USE_LSA))
    return config.BASELINE_USE_LSA


def _baseline_meta() -> dict:
    if os.path.isfile(config.BASELINE_META_PATH):
        with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _baseline_artifact_paths() -> list[str]:
    paths = [config.TFIDF_VECTORIZER_PATH, config.XGB_MODEL_PATH]
    meta = _baseline_meta()
    if meta.get("use_lsa", _baseline_uses_lsa()):
        paths.append(config.LSA_TRANSFORMER_PATH)
    if meta.get("use_char_tfidf", config.BASELINE_USE_CHAR_TFIDF):
        paths.append(config.CHAR_TFIDF_VECTORIZER_PATH)
    return paths


def _xlmr_artifact_dir() -> str:
    return config.XLMR_ARTIFACTS_DIR


def ensure_artifacts(model: str) -> None:
    if model in ("baseline", "both"):
        missing = [p for p in _baseline_artifact_paths() if not os.path.isfile(p)]
        if missing:
            print(
                "Baseline artifacts not found. Run: python train_baseline.py",
                file=sys.stderr,
            )
            for p in missing:
                print(f"  missing: {p}", file=sys.stderr)
            sys.exit(1)

    if model in ("xlmr", "both"):
        model_dir = _xlmr_artifact_dir()
        config_json = os.path.join(model_dir, "config.json")
        if not os.path.isdir(model_dir) or not os.path.isfile(config_json):
            print(
                "XLM-R artifacts not found. Run: python train_xlmr.py",
                file=sys.stderr,
            )
            print(f"  expected: {model_dir}/", file=sys.stderr)
            sys.exit(1)


def rounded_stars(expected: np.ndarray) -> np.ndarray:
    return np.clip(np.round(expected), 1, 5).astype(int)


def majority_baseline_metrics(
    y_true: np.ndarray,
    majority_class: int = config.BASELINE_MAJORITY_CLASS,
) -> dict:
    """Predict constant star rating (default 4★) for every row."""
    expected = np.full(len(y_true), float(majority_class), dtype=np.float64)
    y_pred = np.full(len(y_true), majority_class, dtype=int)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_samples": int(len(y_true)),
        "predicted_class": majority_class,
        "mae": float(mean_absolute_error(y_true, expected)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, expected))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class_recall": utils.per_class_recall(report),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def compute_metrics(y_true: np.ndarray, expected: np.ndarray) -> dict:
    y_pred = rounded_stars(expected)
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "n_samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, expected)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, expected))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class_recall": utils.per_class_recall(report),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def print_metrics(
    name: str,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    print(f"\n=== {name} ===")
    print(f"n_samples:   {metrics['n_samples']}")
    print(f"MAE:         {metrics['mae']:.4f}")
    print(f"RMSE:        {metrics['rmse']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 macro:    {metrics['f1_macro']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    if metrics.get("per_class_recall"):
        print("Per-class recall:")
        for star, rec in sorted(metrics["per_class_recall"].items()):
            print(f"  star {star}: {rec:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        print(" ", row)


def print_comparison(baseline: dict, xlmr: dict) -> None:
    print("\n=== Comparison (baseline vs xlmr) ===")
    print(f"{'metric':<14} {'baseline':>10} {'xlmr':>10}  better")
    print("-" * 44)
    for key in ("mae", "rmse", "accuracy", "f1_macro"):
        b, x = baseline[key], xlmr[key]
        if key in ("mae", "rmse"):
            better = "xlmr" if x < b else "baseline"
        else:
            better = "xlmr" if x > b else "baseline"
        print(f"{key:<14} {b:>10.4f} {x:>10.4f}  {better}")


def evaluate_model(name: str, df: pd.DataFrame, expected: np.ndarray) -> dict:
    y_true = df["user_rating"].values.astype(int)
    metrics = compute_metrics(y_true, expected)
    y_pred = rounded_stars(expected)
    print_metrics(name, metrics, y_true, y_pred)
    return metrics


def save_predictions(
    df: pd.DataFrame,
    expected: np.ndarray,
    model_name: str,
) -> str:
    os.makedirs(config.EVAL_DIR, exist_ok=True)
    out_path = os.path.join(config.EVAL_DIR, f"eval_{model_name}_preds.csv")
    out_df = pd.DataFrame(
        {
            "text": df["text"],
            "user_rating": df["user_rating"],
            "ai_expected_rating": expected,
            "pred_star_rounded": rounded_stars(expected),
        }
    )
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and/or XLM-R on a labeled test CSV.",
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "xlmr", "both"),
        default="both",
        help="Which trained model(s) to evaluate.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_EVAL_INPUT,
        help=f"Labeled test CSV (default: {DEFAULT_EVAL_INPUT}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save metrics JSON (e.g. outputs/eval_report.json).",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write per-model prediction CSVs under outputs/.",
    )
    parser.add_argument(
        "--export-errors",
        action="store_true",
        help="Export error analysis CSVs for baseline under outputs/eval/.",
    )
    parser.add_argument(
        "--no-majority",
        action="store_true",
        help="Skip majority-class (4★) baseline in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(
            f"Input file not found: {args.input}\n"
            "Use a held-out test set (e.g. data/wongnai/holdout.csv) that was not "
            "used for training.",
            file=sys.stderr,
        )
        sys.exit(1)

    ensure_artifacts(args.model)

    print(f"--- Evaluation (model={args.model}) ---")
    print(f"Input: {args.input}")

    df = prepare_scoring_dataframe(args.input)
    utils.compare_rating_distributions(
        df["user_rating"].values,
        label_train="eval input (after clean)",
    )
    y_true = df["user_rating"].values.astype(int)
    results: dict = {"input": args.input, "models": {}}

    if not args.no_majority:
        maj = majority_baseline_metrics(y_true)
        print_metrics(
            f"Majority baseline (predict {maj['predicted_class']} stars)",
            maj,
            y_true,
            np.full(len(y_true), maj["predicted_class"], dtype=int),
        )
        results["majority_baseline"] = maj

    if args.model in ("baseline", "both"):
        expected, probs = predict_baseline_with_probs(df)
        metrics = evaluate_model("Baseline (TF-IDF + XGBoost)", df, expected)
        results["models"]["baseline"] = metrics
        if args.save_predictions:
            save_predictions(df, expected, "baseline")
        if args.export_errors:
            prefix = os.path.splitext(os.path.basename(args.input))[0]
            paths = utils.export_error_analysis(
                df,
                expected,
                probs,
                config.EVAL_DIR,
                min_delta=config.ERROR_EXPORT_MIN_DELTA,
                low_conf_threshold=config.LOW_CONFIDENCE_THRESHOLD,
                prefix=f"errors_{prefix}",
            )
            results["error_exports"] = paths
            print(f"Error analysis exports: {paths}")

    if args.model in ("xlmr", "both"):
        expected = predict_xlmr(df)
        metrics = evaluate_model("XLM-R", df, expected)
        results["models"]["xlmr"] = metrics
        if args.save_predictions:
            save_predictions(df, expected, "xlmr")

    if args.model == "both":
        print_comparison(results["models"]["baseline"], results["models"]["xlmr"])

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics JSON: {args.output}")

    print("\nDone evaluation.")


if __name__ == "__main__":
    main()
