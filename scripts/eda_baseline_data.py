"""EDA for baseline tuning: distributions, text stats, token overlap, cross-domain."""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
import utils

THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
LATIN_RE = re.compile(r"[A-Za-z]")

DATASETS = {
    "wongnai_train_reduce": config.WONGNAI_TRAIN_PATH,
    "wongnai_test": config.WONGNAI_TEST_PATH,
    "wongnai_holdout": config.HOLDOUT_PATH,
    "mock_train": config.MOCK_TRAIN_PATH,
    "mock_test": config.MOCK_TEST_PATH,
}


def _load_clean(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    df = utils.load_and_standardize_data(path)
    df, stats = utils.clean_review_dataframe(
        df,
        min_text_length=config.MIN_TEXT_LENGTH,
        drop_duplicates=config.DROP_DUPLICATE_TEXT,
        duplicate_keep=config.DUPLICATE_KEEP,
    )
    return df, stats


def char_ratios(texts: pd.Series) -> dict:
    thai = latin = total = 0
    for t in texts:
        if not t:
            continue
        thai += len(THAI_RE.findall(t))
        latin += len(LATIN_RE.findall(t))
        total += len(t)
    if total == 0:
        return {"thai_ratio": 0.0, "latin_ratio": 0.0}
    return {
        "thai_ratio": round(thai / total, 4),
        "latin_ratio": round(latin / total, 4),
    }


def length_stats_by_star(df: pd.DataFrame) -> dict:
    out: dict = {}
    for star in range(1, 6):
        lens = df.loc[df["user_rating"] == star, "text"].str.len()
        if len(lens) == 0:
            out[str(star)] = {"n": 0, "median": 0, "p90": 0}
        else:
            out[str(star)] = {
                "n": int(len(lens)),
                "median": float(np.median(lens)),
                "p90": float(np.percentile(lens, 90)),
            }
    return out


def top_tokens(df: pd.DataFrame, top_k: int = 15) -> dict:
    by_star: dict = {}
    for star in range(1, 6):
        counter: Counter = Counter()
        subset = df.loc[df["user_rating"] == star, "text"]
        for text in subset.head(500):
            for tok in utils.thai_tokenizer(text):
                if len(tok) > 1:
                    counter[tok] += 1
        by_star[str(star)] = [w for w, _ in counter.most_common(top_k)]
    return by_star


def token_overlap(star_tokens: dict) -> dict:
    sets = {int(k): set(v) for k, v in star_tokens.items()}
    pairs = {}
    for a in (3, 4, 5):
        for b in range(a + 1, 6):
            sa, sb = sets[a], sets[b]
            inter = len(sa & sb)
            union = len(sa | sb) or 1
            pairs[f"{a}_{b}"] = round(inter / union, 4)
    return pairs


def cross_domain_mock_vs_wongnai(
    mock_df: pd.DataFrame,
    wongnai_df: pd.DataFrame,
) -> dict:
    mock_vocab: set[str] = set()
    for text in mock_df["text"].head(800):
        mock_vocab.update(utils.thai_tokenizer(text))

    wongnai_vocab: set[str] = set()
    for text in wongnai_df["text"].head(2000):
        wongnai_vocab.update(utils.thai_tokenizer(text))

    only_mock = sorted(mock_vocab - wongnai_vocab)[:30]
    coverage = len(mock_vocab & wongnai_vocab) / max(len(mock_vocab), 1)
    return {
        "mock_vocab_size_sample": len(mock_vocab),
        "wongnai_vocab_size_sample": len(wongnai_vocab),
        "mock_tokens_in_wongnai_ratio": round(coverage, 4),
        "mock_only_tokens_sample": only_mock,
    }


def sample_reviews(df: pd.DataFrame, per_star: int = 2) -> dict:
    samples: dict = {}
    for star in range(1, 6):
        rows = df.loc[df["user_rating"] == star, "text"].head(per_star).tolist()
        samples[str(star)] = [t[:200] for t in rows]
    return samples


def main() -> None:
    report: dict = {"datasets": {}, "comparisons": {}}
    loaded: dict[str, pd.DataFrame] = {}

    for name, path in DATASETS.items():
        result = _load_clean(path)
        if result is None:
            report["datasets"][name] = {"error": f"missing: {path}"}
            continue
        df, stats = result
        loaded[name] = df
        dist = utils.rating_distribution_dict(df["user_rating"].values)
        n = len(df)
        pct = {str(k): round(100.0 * v / max(n, 1), 2) for k, v in dist.items()}
        report["datasets"][name] = {
            "path": path,
            "n_rows": n,
            "cleaning": stats,
            "star_counts": dist,
            "star_pct": pct,
            "char_ratios": char_ratios(df["text"]),
            "length_by_star": length_stats_by_star(df),
            "top_tokens": top_tokens(df),
            "samples": sample_reviews(df),
        }
        if name.startswith("wongnai") or name == "mock_train":
            tokens = report["datasets"][name]["top_tokens"]
            report["datasets"][name]["token_overlap_345"] = token_overlap(tokens)

    if "mock_test" in loaded and "wongnai_train_reduce" in loaded:
        report["comparisons"]["mock_vs_wongnai_train"] = cross_domain_mock_vs_wongnai(
            loaded["mock_test"],
            loaded["wongnai_train_reduce"],
        )

    if "wongnai_train_reduce" in loaded and "wongnai_test" in loaded:
        train_dist = report["datasets"]["wongnai_train_reduce"]["star_pct"]
        test_dist = report["datasets"]["wongnai_test"]["star_pct"]
        delta = {
            str(s): round(test_dist.get(str(s), 0) - train_dist.get(str(s), 0), 2)
            for s in range(1, 6)
        }
        report["comparisons"]["train_vs_hf_test_pct_delta"] = delta

    out_path = config.EXPERIMENT_EDA_SUMMARY_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote EDA summary: {out_path}")

    for name, info in report["datasets"].items():
        if "error" in info:
            print(f"{name}: {info['error']}")
            continue
        print(f"\n{name}: n={info['n_rows']}")
        for s in range(1, 6):
            print(f"  star {s}: {info['star_pct'].get(str(s), 0):.1f}%")


if __name__ == "__main__":
    main()
