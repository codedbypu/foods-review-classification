"""
Fine-tune XLM-RoBERTa (very slow on CPU; hours to days for ~80k rows).

COMMON ERRORS:
  - Windows exit 3221225477 / 0xC0000005 on startup: import torch + sklearn.model_selection BEFORE
    transformers Trainer (this script imports sentiment_trainer first, then sklearn — do not reorder).
  - CUDA OOM: lower --train_batch_size or --max_length.
  - Disk full: use --scratch_dir D:\\path (sets TEMP + HF_HOME under that folder) or env RRIS_SCRATCH_DIR.
  - ValueError stratify: tiny datasets — some star classes have <2 rows (unlike baseline, no auto-fallback yet).
  - CPU + 80k rows: expect hours–days; use a small parquet for smoke tests.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import _scratch_init  # noqa: F401

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from rris.data.datasets import df_to_hf_sentiment_dataset
from rris.data.io import read_reviews
from rris.logging_utils import LoggingConfig, setup_logging
from rris.runtime_env import add_scratch_argument, apply_scratch_from_args
# Import HF Trainer stack before sklearn.model_selection (Windows DLL clash if reversed).
from rris.models.xlmr.sentiment_trainer import XlmrSentimentConfig, train_xlmr_sentiment
from sklearn.model_selection import train_test_split

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
    add_scratch_argument(p)
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--log_level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_scratch_from_args(Path(__file__).resolve().parents[1], args)
    setup_logging(LoggingConfig(level=args.log_level))

    logger.info(
        "Torch CUDA available: %s",
        torch.cuda.is_available(),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_reviews(args.input)
    y = df["user_rating"].astype(int).to_numpy() - 1
    # No auto non-stratify fallback (unlike train_baseline_xgb) — very small inputs may raise sklearn ValueError
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
        disable_tqdm=args.no_progress,
    )

    trainer = train_xlmr_sentiment(train_ds, val_ds, cfg=cfg, out_dir=out_dir)

    metrics = trainer.evaluate()
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Done. Saved to %s", out_dir)


if __name__ == "__main__":
    main()
