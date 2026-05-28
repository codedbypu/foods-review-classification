"""
Normalize review text; output parquet/csv.

COMMON ERRORS:
  - Missing column text: wrong CSV format (see io.py).
  - Does NOT require user_rating (require_rating=False) — train still needs labels later.
  - See docs/ERRORS_AND_FIXES.md §2.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import _scratch_init  # noqa: F401

import argparse
import logging
from pathlib import Path

from rris.data.io import read_reviews, write_table
from rris.data.text import normalize_text
from rris.logging_utils import LoggingConfig, setup_logging
from rris.runtime_env import add_scratch_argument, apply_scratch_from_args
from rris.progress import log_stage, map_with_progress

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess reviews (normalize text, basic schema check).")
    p.add_argument("--input", required=True, help="Input .csv/.parquet")
    p.add_argument("--out", required=True, help="Output .csv/.parquet")
    add_scratch_argument(p)
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_scratch_from_args(Path(__file__).resolve().parents[1], args)
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
