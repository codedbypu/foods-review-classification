import os
import re

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.utils.class_weight import compute_class_weight

TEXT_ALIASES = ("review_body", "text", "review")
RATING_ALIASES = ("stars", "user_rating", "rating", "star")

# Thai negation cues for extra features (order: longer phrases first)
NEGATION_PHRASES = ("ไม่แนะนำ", "ไม่ค่อย", "ไม่")

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "]+",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Light cleanup: strip and collapse repeated whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    cleaned = str(text).strip()
    return re.sub(r"\s+", " ", cleaned)


def extended_normalize_text(text: str) -> str:
    """Normalize for modeling: strip, emoji, digits, whitespace."""
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    cleaned = _EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\d+", "0", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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
            "text": df[text_col].astype(str).map(extended_normalize_text),
            "user_rating": df[rating_col].astype(int),
        }
    )
    invalid = ~standard_df["user_rating"].between(1, 5)
    if invalid.any():
        raise ValueError(
            f"user_rating must be 1..5; found {int(invalid.sum())} invalid row(s)"
        )
    return standard_df


def clean_review_dataframe(
    df: pd.DataFrame,
    min_text_length: int,
    drop_duplicates: bool = True,
    duplicate_keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """Drop empty/short text and duplicate reviews; return stats for logging."""
    stats: dict = {
        "initial_rows": int(len(df)),
        "removed_empty": 0,
        "removed_short": 0,
        "removed_duplicate": 0,
        "final_rows": 0,
    }
    out = df.copy()
    empty_mask = out["text"].str.len() == 0
    stats["removed_empty"] = int(empty_mask.sum())
    out = out.loc[~empty_mask]

    short_mask = out["text"].str.len() < min_text_length
    stats["removed_short"] = int(short_mask.sum())
    out = out.loc[~short_mask]

    if drop_duplicates:
        before = len(out)
        out = out.drop_duplicates(subset=["text"], keep=duplicate_keep)
        stats["removed_duplicate"] = before - len(out)

    stats["final_rows"] = int(len(out))
    return out.reset_index(drop=True), stats


def log_cleaning_stats(stats: dict, label: str = "dataset") -> None:
    """Log rows removed during cleaning."""
    print(f"Cleaning stats ({label}):")
    print(f"  initial rows:      {stats['initial_rows']}")
    print(f"  removed empty:     {stats['removed_empty']}")
    print(f"  removed short:     {stats['removed_short']}")
    print(f"  removed duplicate: {stats['removed_duplicate']}")
    print(f"  final rows:        {stats['final_rows']}")


def rating_distribution_dict(ratings_1_to_5: np.ndarray) -> dict[int, int]:
    counts = pd.Series(ratings_1_to_5).value_counts().sort_index()
    return {star: int(counts.get(star, 0)) for star in range(1, 6)}


def print_rating_distribution(
    ratings_1_to_5: np.ndarray,
    class_weights: np.ndarray | None = None,
    label: str = "train",
) -> None:
    """Log star counts and optional class weights for debugging imbalance."""
    dist = rating_distribution_dict(ratings_1_to_5)
    print(f"Rating distribution ({label}):")
    for star in range(1, 6):
        print(f"  star {star}: {dist[star]}")
    if class_weights is not None:
        print(f"Class weights (index 0..4): {np.round(class_weights, 4).tolist()}")


def compare_rating_distributions(
    train_ratings: np.ndarray,
    test_ratings: np.ndarray | None = None,
    label_train: str = "train split",
    label_test: str = "test",
) -> None:
    """EDA: print train vs test star counts and share differences."""
    train_dist = rating_distribution_dict(train_ratings)
    n_train = max(sum(train_dist.values()), 1)
    print(f"\n--- Rating distribution: {label_train} (n={n_train}) ---")
    for star in range(1, 6):
        c = train_dist[star]
        print(f"  star {star}: {c} ({100.0 * c / n_train:.1f}%)")

    if test_ratings is None:
        return

    test_dist = rating_distribution_dict(test_ratings)
    n_test = max(sum(test_dist.values()), 1)
    print(f"\n--- Rating distribution: {label_test} (n={n_test}) ---")
    for star in range(1, 6):
        c = test_dist[star]
        print(f"  star {star}: {c} ({100.0 * c / n_test:.1f}%)")

    print("\n--- Train vs test share delta (test% - train%) ---")
    for star in range(1, 6):
        train_pct = 100.0 * train_dist[star] / n_train
        test_pct = 100.0 * test_dist[star] / n_test
        print(f"  star {star}: {test_pct - train_pct:+.1f} pp")


def count_negations(text: str, phrases: tuple[str, ...] = NEGATION_PHRASES) -> int:
    """Count negation phrase occurrences (longer phrases matched first)."""
    if not text:
        return 0
    remaining = text
    total = 0
    for phrase in phrases:
        if not phrase:
            continue
        n = remaining.count(phrase)
        total += n
        remaining = remaining.replace(phrase, " ")
    return total


def compute_extra_features(texts: pd.Series | list[str]) -> np.ndarray:
    """Dense extras: char length, word count (newmm), negation count."""
    series = texts if isinstance(texts, pd.Series) else pd.Series(list(texts))
    char_len = series.str.len().astype(np.float64).values.reshape(-1, 1)
    word_counts = np.array(
        [len(word_tokenize(t, engine="newmm")) for t in series],
        dtype=np.float64,
    ).reshape(-1, 1)
    neg_counts = np.array(
        [count_negations(t) for t in series],
        dtype=np.float64,
    ).reshape(-1, 1)
    return np.hstack([char_len, word_counts, neg_counts])


def truncate_text(text: str, max_chars: int) -> str:
    """Keep leading chars (sentiment often in opening sentences)."""
    if not max_chars or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def apply_text_truncation(df: pd.DataFrame, max_chars: int) -> pd.DataFrame:
    """Return copy with truncated text column."""
    if not max_chars or max_chars <= 0:
        return df
    out = df.copy()
    out["text"] = out["text"].map(lambda t: truncate_text(t, max_chars))
    return out


def rating_to_3class(ratings_1_to_5: np.ndarray) -> np.ndarray:
    """Map 1-2 -> 0, 3 -> 1, 4-5 -> 2 (for XGB class indices)."""
    r = ratings_1_to_5.astype(int)
    out = np.ones(len(r), dtype=int)
    out[(r == 1) | (r == 2)] = 0
    out[r == 3] = 1
    out[(r == 4) | (r == 5)] = 2
    return out


def class3_to_expected_star(class_idx: np.ndarray) -> np.ndarray:
    """Map 3-class prediction index to expected 1-5 star (midpoints)."""
    centers = np.array([1.5, 3.0, 4.5], dtype=np.float64)
    idx = np.clip(class_idx.astype(int), 0, 2)
    return centers[idx]


def undersample_star_ratings(
    df: pd.DataFrame,
    star: int = 4,
    keep_fraction: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomly keep a fraction of rows with the given star rating."""
    if keep_fraction >= 1.0:
        return df.reset_index(drop=True)
    mask = df["user_rating"] == star
    keep = df.loc[mask].sample(frac=keep_fraction, random_state=random_state)
    rest = df.loc[~mask]
    return pd.concat([rest, keep], ignore_index=True)


def mix_mock_training_data(
    wongnai_df: pd.DataFrame,
    mock_path: str,
    mock_fraction: float,
    *,
    min_text_length: int = 5,
    drop_duplicates: bool = True,
    duplicate_keep: str = "first",
    random_state: int = 42,
) -> pd.DataFrame:
    """Append a sample of mock_train rows to wongnai training pool."""
    if mock_fraction <= 0 or not os.path.isfile(mock_path):
        return wongnai_df.reset_index(drop=True)
    mock_df = load_and_standardize_data(mock_path)
    mock_df, _ = clean_review_dataframe(
        mock_df,
        min_text_length=min_text_length,
        drop_duplicates=drop_duplicates,
        duplicate_keep=duplicate_keep,
    )
    n_mock = max(1, int(len(wongnai_df) * mock_fraction))
    n_mock = min(n_mock, len(mock_df))
    mock_sample = mock_df.sample(n=n_mock, random_state=random_state)
    combined = pd.concat([wongnai_df, mock_sample], ignore_index=True)
    return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def oversample_low_star_reviews(
    df: pd.DataFrame,
    factor: int = 3,
    low_stars: tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    """Duplicate rows with stars in low_stars up to `factor` times total."""
    if factor <= 1:
        return df.reset_index(drop=True)
    low = df[df["user_rating"].isin(low_stars)]
    if low.empty:
        return df.reset_index(drop=True)
    extra_copies = max(0, factor - 1)
    parts = [df]
    for _ in range(extra_copies):
        parts.append(low.copy())
    return pd.concat(parts, ignore_index=True)


def per_class_recall(classification_report: dict) -> dict[int, float]:
    """Extract recall per star from sklearn classification_report dict."""
    recalls: dict[int, float] = {}
    for star in range(1, 6):
        key = str(star)
        if key in classification_report:
            recalls[star] = float(classification_report[key].get("recall", 0.0))
    return recalls


def export_error_analysis(
    df: pd.DataFrame,
    expected: np.ndarray,
    probs: np.ndarray | None,
    out_dir: str,
    *,
    min_delta: int = 2,
    low_conf_threshold: float = 0.4,
    prefix: str = "errors",
) -> dict[str, str]:
    """Export severe errors and low-confidence correct predictions."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    pred_rounded = np.clip(np.round(expected), 1, 5).astype(int)
    y_true = df["user_rating"].values.astype(int)
    delta = np.abs(pred_rounded - y_true)

    severe_mask = delta >= min_delta
    severe_path = os.path.join(out_dir, f"{prefix}_severe_delta_ge_{min_delta}.csv")
    severe_df = pd.DataFrame(
        {
            "text": df.loc[severe_mask, "text"].values,
            "user_rating": y_true[severe_mask],
            "pred_star_rounded": pred_rounded[severe_mask],
            "ai_expected_rating": expected[severe_mask],
            "abs_error": delta[severe_mask],
        }
    )
    severe_df.to_csv(severe_path, index=False, encoding="utf-8")

    paths = {"severe_errors": severe_path}
    if probs is not None:
        max_prob = probs.max(axis=1)
        correct = pred_rounded == y_true
        low_conf_mask = correct & (max_prob < low_conf_threshold)
        low_path = os.path.join(
            out_dir, f"{prefix}_low_conf_correct_lt_{low_conf_threshold}.csv"
        )
        pd.DataFrame(
            {
                "text": df.loc[low_conf_mask, "text"].values,
                "user_rating": y_true[low_conf_mask],
                "pred_star_rounded": pred_rounded[low_conf_mask],
                "ai_expected_rating": expected[low_conf_mask],
                "max_prob": max_prob[low_conf_mask],
            }
        ).to_csv(low_path, index=False, encoding="utf-8")
        paths["low_confidence_correct"] = low_path

    return paths


def thai_tokenizer(text: str) -> list[str]:
    """Thai word tokenizer for TfidfVectorizer."""
    return word_tokenize(extended_normalize_text(text), engine="newmm")


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
