"""Run baseline config candidates: train + eval on wongnai + mock; log to try-log."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import train_baseline
from evaluate import compute_metrics, majority_baseline_metrics
from score import prepare_scoring_dataframe, predict_baseline_with_probs
from try_log_utils import append_try_log, min_accuracy

EVAL_SETS = {
    "wongnai_test": config.WONGNAI_TEST_PATH,
    "wongnai_holdout": config.HOLDOUT_PATH,
    "mock_test": config.MOCK_TEST_PATH,
}

# Try 0 baseline + plan experiments (A/B/C/D)
CANDIDATES: list[dict] = [
    {"name": "try0_current_config", "hypothesis": "Reproduce current config.py winner."},
    {
        "name": "a1_oversample5_no_weight",
        "hypothesis": "Stronger low-star presence without sample weights.",
        "BASELINE_OVERSAMPLE_FACTOR": 5,
        "XGB_USE_SAMPLE_WEIGHT": False,
    },
    {
        "name": "a1_oversample3_boost25",
        "hypothesis": "More low-star copies + higher boost.",
        "BASELINE_OVERSAMPLE_FACTOR": 3,
        "XGB_LOW_STAR_BOOST": 2.5,
    },
    {
        "name": "a2_trunc300",
        "hypothesis": "Shorter text keeps sentiment signal.",
        "MAX_REVIEW_CHARS": 300,
    },
    {
        "name": "a2_trunc800",
        "hypothesis": "Longer context for detailed reviews.",
        "MAX_REVIEW_CHARS": 800,
    },
    {
        "name": "a3_undersample4_70",
        "hypothesis": "Reduce 4-star bias in training.",
        "BASELINE_UNDERSAMPLE_STAR4_FRACTION": 0.7,
        "BASELINE_OVERSAMPLE_FACTOR": 3,
        "XGB_LOW_STAR_BOOST": 2.5,
    },
    {
        "name": "a4_3class",
        "hypothesis": "Merge 1-2 / 3 / 4-5 to reduce confusion.",
        "BASELINE_USE_3CLASS": True,
    },
    {
        "name": "b1_lsa400_8k",
        "hypothesis": "LSA compresses sparse TF-IDF.",
        "BASELINE_USE_LSA": True,
        "LSA_N_COMPONENTS": 400,
        "TFIDF_MAX_FEATURES": 8000,
    },
    {
        "name": "b1_tfidf12k_no_lsa",
        "hypothesis": "Larger vocabulary without LSA.",
        "TFIDF_MAX_FEATURES": 12000,
        "BASELINE_USE_LSA": False,
    },
    {
        "name": "b2_char_tfidf",
        "hypothesis": "Char n-grams capture slang/typos.",
        "BASELINE_USE_CHAR_TFIDF": True,
    },
    {
        "name": "b2_word_char_hybrid",
        "hypothesis": "Word + char TF-IDF stacked.",
        "BASELINE_USE_CHAR_TFIDF": True,
        "TFIDF_MAX_FEATURES": 6000,
        "BASELINE_CHAR_MAX_FEATURES": 3000,
    },
    {
        "name": "b4_min_df1",
        "hypothesis": "Keep rare negative tokens.",
        "TFIDF_MIN_DF": 1,
    },
    {
        "name": "c1_depth3_reg5",
        "hypothesis": "Shallower trees + stronger L2.",
        "XGB_PARAMS": {"max_depth": 3, "reg_lambda": 5.0, "min_child_weight": 8},
    },
    {
        "name": "c1_depth5_eta05",
        "hypothesis": "Deeper model faster learning rate.",
        "XGB_PARAMS": {"max_depth": 5, "eta": 0.05, "reg_lambda": 2.0},
    },
    {
        "name": "c2_regression",
        "hypothesis": "Ordinal regression may improve MAE/rounding.",
        "BASELINE_USE_REGRESSION": True,
        "XGB_USE_SAMPLE_WEIGHT": False,
    },
    {
        "name": "c4_no_weight_boost_off",
        "hypothesis": "Oversample only, no class weights.",
        "XGB_USE_SAMPLE_WEIGHT": False,
        "BASELINE_OVERSAMPLE_FACTOR": 4,
        "XGB_LOW_STAR_BOOST": 1.0,
    },
    {
        "name": "d1_mock_mix10",
        "hypothesis": "10% mock rows help mock test without abandoning wongnai.",
        "BASELINE_MOCK_MIX_FRACTION": 0.1,
    },
    {
        "name": "d1_mock_mix10_char",
        "hypothesis": "Mock mix + char features for cross-domain.",
        "BASELINE_MOCK_MIX_FRACTION": 0.1,
        "BASELINE_USE_CHAR_TFIDF": True,
    },
    {
        "name": "best_combo_v1",
        "hypothesis": "Combine undersample + oversample + char + trunc500.",
        "BASELINE_UNDERSAMPLE_STAR4_FRACTION": 0.7,
        "BASELINE_OVERSAMPLE_FACTOR": 4,
        "XGB_LOW_STAR_BOOST": 2.5,
        "BASELINE_USE_CHAR_TFIDF": True,
        "MAX_REVIEW_CHARS": 500,
        "TFIDF_MIN_DF": 1,
        "XGB_PARAMS": {"max_depth": 4, "reg_lambda": 4.0, "eta": 0.04},
    },
    {
        "name": "best_combo_mock20",
        "hypothesis": "Higher mock mix for mock test accuracy.",
        "BASELINE_MOCK_MIX_FRACTION": 0.2,
        "BASELINE_USE_CHAR_TFIDF": True,
        "BASELINE_OVERSAMPLE_FACTOR": 5,
        "BASELINE_UNDERSAMPLE_STAR4_FRACTION": 0.65,
        "XGB_LOW_STAR_BOOST": 3.0,
    },
]

TARGET_ACCURACY = 0.9


def apply_overrides(overrides: dict) -> dict:
    """Patch config module; return snapshot of changed keys for restore."""
    snapshot: dict = {}
    for key, value in overrides.items():
        if key in ("name", "hypothesis"):
            continue
        if key == "XGB_PARAMS":
            old = copy.deepcopy(config.XGB_PARAMS)
            snapshot["XGB_PARAMS"] = old
            config.XGB_PARAMS = {**config.XGB_PARAMS, **value}
        elif hasattr(config, key):
            snapshot[key] = getattr(config, key)
            setattr(config, key, value)
        else:
            raise KeyError(f"Unknown config key: {key}")
    return snapshot


def restore_config(snapshot: dict) -> None:
    for key, value in snapshot.items():
        setattr(config, key, value)


def eval_all_sets() -> dict:
    results: dict = {}
    for label, path in EVAL_SETS.items():
        if not os.path.isfile(path):
            results[label] = {"error": f"missing: {path}"}
            continue
        df = prepare_scoring_dataframe(path)
        y_true = df["user_rating"].values.astype(int)
        expected, _ = predict_baseline_with_probs(df)
        metrics = compute_metrics(y_true, expected)
        maj = majority_baseline_metrics(y_true)
        results[label] = {
            "n": metrics["n_samples"],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "mae": metrics["mae"],
            "per_class_recall": metrics["per_class_recall"],
            "majority_acc": maj["accuracy"],
            "majority_f1": maj["f1_macro"],
            "majority_mae": maj["mae"],
        }
    return results


def score_candidate(eval_results: dict) -> float:
    """Higher is better: min accuracy across eval sets (primary objective)."""
    m = min_accuracy(eval_results)
    return m if m is not None else 0.0


def avg_f1(eval_results: dict) -> float:
    f1s = [v["f1_macro"] for v in eval_results.values() if "f1_macro" in v]
    return sum(f1s) / len(f1s) if f1s else 0.0


def run_candidate(candidate: dict, *, append_log: bool) -> dict:
    name = candidate["name"]
    hypothesis = candidate.get("hypothesis", name)
    print(f"\n{'=' * 60}\nCandidate: {name}\n{'=' * 60}")
    snapshot = apply_overrides(candidate)
    try:
        train_baseline.main()
        eval_results = eval_all_sets()
        min_acc = min_accuracy(eval_results)
        row = {
            "name": name,
            "hypothesis": hypothesis,
            "config": {k: v for k, v in candidate.items() if k not in ("name", "hypothesis")},
            "val_mlogloss": None,
            "eval": eval_results,
            "min_accuracy": min_acc,
            "avg_f1_macro": avg_f1(eval_results),
            "score": score_candidate(eval_results),
            "passed_target": min_acc is not None and min_acc > TARGET_ACCURACY,
        }
        if os.path.isfile(config.BASELINE_META_PATH):
            with open(config.BASELINE_META_PATH, encoding="utf-8") as f:
                meta = json.load(f)
            row["val_mlogloss"] = meta.get("best_val_metric")

        if append_log:
            analysis = (
                f"min_acc={min_acc:.4f} "
                f"({'PASS' if row['passed_target'] else 'below 0.9'}); "
                f"avg_f1={row['avg_f1_macro']:.4f}."
            )
            append_try_log(
                try_id=name,
                hypothesis=hypothesis,
                config_changes=row["config"],
                eval_results=eval_results,
                analysis=analysis,
                next_step="Continue sweep or apply winner to config.py.",
                val_mlogloss=row.get("val_mlogloss"),
            )
        return row
    finally:
        restore_config(snapshot)


def print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("ITERATION SUMMARY (primary score = min accuracy)")
    print("=" * 100)
    header = (
        f"{'name':<26} {'min_acc':>8} {'hold_acc':>9} {'wn_acc':>8} "
        f"{'mock_acc':>9} {'avg_f1':>8} {'PASS':>5}"
    )
    print(header)
    print("-" * len(header))

    def _acc(key: str, ev: dict) -> str:
        v = ev.get(key, {})
        return f"{v['accuracy']:.4f}" if "accuracy" in v else "n/a"

    for row in rows:
        ev = row["eval"]
        passed = "Y" if row.get("passed_target") else "N"
        print(
            f"{row['name']:<26} {row.get('min_accuracy', 0) or 0:>8.4f} "
            f"{_acc('wongnai_holdout', ev):>9} {_acc('wongnai_test', ev):>8} "
            f"{_acc('mock_test', ev):>9} {row.get('avg_f1_macro', 0):>8.4f} {passed:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune baseline configs.")
    parser.add_argument(
        "--rounds",
        type=str,
        default="all",
        help="Comma-separated candidate names or 'all' (default).",
    )
    parser.add_argument(
        "--output",
        default=config.EXPERIMENT_TUNE_LOG_PATH,
        help="JSON log path (under experiments/).",
    )
    parser.add_argument(
        "--append-try-log",
        action="store_true",
        help=f"Append each candidate to {config.EXPERIMENT_TRY_LOG_PATH}",
    )
    parser.add_argument(
        "--stop-on-target",
        action="store_true",
        help="Stop early when min accuracy > 0.9 on all eval sets.",
    )
    args = parser.parse_args()

    if args.rounds == "all":
        candidates = CANDIDATES
    else:
        names = {n.strip() for n in args.rounds.split(",")}
        candidates = [c for c in CANDIDATES if c["name"] in names]
        if not candidates:
            print(f"No matching candidates for: {args.rounds}", file=sys.stderr)
            sys.exit(1)

    rows: list[dict] = []
    for c in candidates:
        row = run_candidate(c, append_log=args.append_try_log)
        rows.append(row)
        if args.stop_on_target and row.get("passed_target"):
            print(f"\nTarget reached with {c['name']} — stopping.")
            break

    print_table(rows)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved log: {args.output}")

    best = max(rows, key=lambda r: r["score"])
    print(
        f"\nBest by min accuracy: {best['name']} "
        f"(min_acc={best.get('min_accuracy', 0):.4f})"
    )
    if any(r.get("passed_target") for r in rows):
        winners = [r["name"] for r in rows if r.get("passed_target")]
        print(f"Passed target (>0.9): {winners}")


if __name__ == "__main__":
    main()
