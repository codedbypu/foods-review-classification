import re

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.utils.class_weight import compute_class_weight

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


def compute_class_weights(
    ratings_1_to_5: np.ndarray,
    num_classes: int = 5,
    low_star_boost: float = 1.0,
) -> np.ndarray:
    """Balanced weights per class index 0..4; optional boost for stars 1-2."""
    classes = np.arange(num_classes)
    labels_0_to_4 = ratings_1_to_5.astype(int) - 1
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_0_to_4,
    ).astype(np.float64)
    if low_star_boost != 1.0:
        weights[0] *= low_star_boost
        weights[1] *= low_star_boost
    return weights


def compute_sample_weights_from_ratings(
    ratings_1_to_5: np.ndarray,
    class_weights: np.ndarray,
) -> np.ndarray:
    """Map per-class weights (index 0..4) to per-sample weights."""
    indices = ratings_1_to_5.astype(int) - 1
    return class_weights[indices]


def print_rating_distribution(
    ratings_1_to_5: np.ndarray,
    class_weights: np.ndarray | None = None,
    label: str = "train",
) -> None:
    """Log star counts and optional class weights for debugging imbalance."""
    counts = pd.Series(ratings_1_to_5).value_counts().sort_index()
    print(f"Rating distribution ({label}):")
    for star in range(1, 6):
        count = int(counts.get(star, 0))
        print(f"  star {star}: {count}")
    if class_weights is not None:
        print(f"Class weights (index 0..4): {np.round(class_weights, 4).tolist()}")
