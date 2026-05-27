from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import logging
from pathlib import Path

from rris.data.io import read_reviews, write_table
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging
from rris.progress import log_stage, map_with_progress

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess reviews (normalize text, basic schema check).")
    p.add_argument("--input", required=True, help="Input .csv/.parquet")
    p.add_argument("--out", required=True, help="Output .csv/.parquet")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LoggingConfig(level=args.log_level))
    show_progress = not args.no_progress

    with log_stage("Load data"):
        df = read_reviews(args.input, require_rating=False).copy()

    with log_stage("Normalize text"):
        texts = df["text"].astype(str).tolist()
        normalized = map_with_progress(
            normalize_text,
            texts,
            show_progress=show_progress,
            desc="Normalize",
        )
        df["text"] = normalized

    out = Path(args.out)
    write_table(df, out)
    logger.info("Wrote preprocessed dataset: %s (%s rows)", out, len(df))


if __name__ == "__main__":
    main()
