from __future__ import annotations

import json
import logging
import time
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

_DEBUG_LOG = Path(__file__).resolve().parents[3] / "debug-9aac7e.log"

_TEXT_ALIASES = frozenset(
    {"text", "review", "Review", "review_body", "content", "comment", "review_text"}
)
_RATING_ALIASES = frozenset(
    {"user_rating", "rating", "Rating", "score", "stars", "label", "star", "rate"}
)


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, *, run_id: str = "pre-fix") -> None:
    payload = {
        "sessionId": "9aac7e",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with _DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _looks_like_review_text_header(columns: list) -> bool:
    if len(columns) != 1:
        return False
    name = str(columns[0])
    return len(name) > 80 or "\n" in name


def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col_map = {str(c): c for c in out.columns}
    for c in list(out.columns):
        key = str(c).strip()
        if key in _TEXT_ALIASES and "text" not in out.columns:
            out = out.rename(columns={c: "text"})
        elif key in _RATING_ALIASES and "user_rating" not in out.columns:
            out = out.rename(columns={c: "user_rating"})
    return out


def _load_csv_reviews(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # #region agent log
    _agent_log(
        "A",
        "io.py:_load_csv_reviews",
        "initial pandas read",
        {"columns": [str(c) for c in df.columns], "ncols": len(df.columns), "nrows": len(df)},
    )
    # #endregion

    if "text" not in df.columns and (_looks_like_review_text_header(list(df.columns)) or len(df.columns) == 1):
        df = pd.read_csv(path, header=None, names=["text"])
        # #region agent log
        _agent_log(
            "A",
            "io.py:_load_csv_reviews",
            "re-read single-column CSV as text",
            {"columns": list(df.columns), "nrows": len(df)},
        )
        # #endregion

    return _rename_aliases(df)


def read_reviews(path: str | Path, *, require_rating: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
        df = _rename_aliases(df)
    elif path.suffix.lower() in {".csv"}:
        df = _load_csv_reviews(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")

    # #region agent log
    _agent_log(
        "B",
        "io.py:read_reviews",
        "schema after coerce",
        {
            "columns": [str(c) for c in df.columns],
            "has_text": "text" in df.columns,
            "has_user_rating": "user_rating" in df.columns,
            "require_rating": require_rating,
        },
    )
    # #endregion

    if "text" not in df.columns:
        raise ValueError(
            "Missing required column: text. "
            f"Found columns: {list(df.columns)}. "
            "Expected a review text column (e.g. text, review) or a single-column CSV of reviews."
        )
    if require_rating and "user_rating" not in df.columns:
        raise ValueError(
            "Missing required column: user_rating (1..5). "
            f"Found columns: {list(df.columns)}. "
            "This file looks like text-only reviews. For Wongnai labeled data, use "
            "review/review_dataset.zip from https://github.com/wongnai/wongnai-corpus "
            "(not a single-column text export)."
        )

    if not require_rating and "user_rating" not in df.columns:
        logger.warning(
            "Input has no user_rating column; preprocess will continue but training scripts require labels."
        )

    return df


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in {".parquet"}:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")
