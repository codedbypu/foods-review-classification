from __future__ import annotations

import _bootstrap  # noqa: F401  # must run before rris imports

import argparse
import logging
from pathlib import Path

import pandas as pd

from rris.data.io import read_reviews, write_table
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess reviews (normalize text, basic schema check).")
    p.add_argument("--input", required=True, help="Input .csv/.parquet")
    p.add_argument("--out", required=True, help="Output .csv/.parquet")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))

    df = read_reviews(args.input, require_rating=False).copy()
    df["text"] = df["text"].astype(str).map(normalize_text)

    out = Path(args.out)
    write_table(df, out)
    logger.info("Wrote preprocessed dataset: %s", out)


if __name__ == "__main__":
    main()

