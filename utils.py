import re

import pandas as pd
from pythainlp.tokenize import word_tokenize

TEXT_ALIASES = ("review_body", "text", "review")
RATING_ALIASES = ("stars", "user_rating", "rating", "star")


def normalize_text(text: str) -> str:
    """Light cleanup: strip and collapse repeated whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = str(text).strip()
    return re.sub(r"\s+", " ", cleaned)


def _pick_column(columns: list[str], aliases: tuple[str, ...], kind: str) -> str:
    matches = [c for c in columns if c in aliases]
    if not matches:
        raise ValueError(
            f"Could not find {kind} column. Expected one of {aliases}, got {list(columns)}"
        )
    return matches[0]


def load_and_standardize_data(file_path: str) -> pd.DataFrame:
    """Load CSV and map review/rating columns to text, user_rating."""
    df = pd.read_csv(file_path)
    text_col = _pick_column(list(df.columns), TEXT_ALIASES, "text")
    rating_col = _pick_column(list(df.columns), RATING_ALIASES, "rating")

    standard_df = pd.DataFrame(
        {
            "text": df[text_col].astype(str).map(normalize_text),
            "user_rating": df[rating_col].astype(int),
        }
    )
    invalid = ~standard_df["user_rating"].between(1, 5)
    if invalid.any():
        raise ValueError(
            f"user_rating must be 1..5; found {int(invalid.sum())} invalid row(s)"
        )
    return standard_df


def thai_tokenizer(text: str) -> list[str]:
    """Thai word tokenizer for TfidfVectorizer."""
    return word_tokenize(normalize_text(text), engine="newmm")
