from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from rris.data.datasets import df_to_hf_sentiment_dataset
from rris.data.io import read_reviews
from rris.logging_utils import LoggingConfig, setup_logging
from rris.models.xlmr.sentiment_trainer import XlmrSentimentConfig, train_xlmr_sentiment

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune xlm-roberta-base for 1-5 sentiment (5-class).")
    p.add_argument("--input", required=True, help="Input reviews .csv/.parquet with columns text,user_rating")
    p.add_argument("--out_dir", required=True, help="Output directory for HF model/tokenizer")
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_reviews(args.input)
    # stratify by label (0..4)
    y = df["user_rating"].astype(int).to_numpy() - 1
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = df_to_hf_sentiment_dataset(train_df)
    val_ds = df_to_hf_sentiment_dataset(val_df)

    cfg = XlmrSentimentConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        lr=args.lr,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    trainer = train_xlmr_sentiment(train_ds, val_ds, cfg=cfg, out_dir=out_dir)

    # Save final eval metrics
    metrics = trainer.evaluate()
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Done. Saved to %s", out_dir)


if __name__ == "__main__":
    main()

