import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import config
import utils


def train_xgb_native(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict,
) -> tuple[xgb.Booster, dict, float]:
    """Train with xgb.train(); return booster, eval history, best val mlogloss."""
    watchlist = [(dtrain, "train"), (dval, "val")]
    evals_result: dict = {}

    def _run(p: dict) -> tuple[xgb.Booster, dict]:
        er: dict = {}
        bst = xgb.train(
            params=p,
            dtrain=dtrain,
            num_boost_round=config.XGB_ROUNDS,
            evals=watchlist,
            evals_result=er,
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
            verbose_eval=10,
        )
        return bst, er

    try:
        bst, evals_result = _run(params)
    except xgb.core.XGBoostError as exc:
        if params.get("device") == "cuda":
            print(f"GPU training failed ({exc}); retrying on CPU.")
            cpu_params = {**params, "device": "cpu"}
            bst, evals_result = _run(cpu_params)
        else:
            raise

    best_iter = bst.best_iteration
    val_hist = evals_result.get("val", {}).get("mlogloss", [])
    best_val = float(val_hist[best_iter]) if val_hist else float("inf")
    return bst, evals_result, best_val


def _build_word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=utils.thai_tokenizer,
        token_pattern=None,
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
    )


def _build_char_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=config.BASELINE_CHAR_NGRAM_RANGE,
        max_features=config.BASELINE_CHAR_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
    )


def _stack_features(parts: list) -> object:
    if len(parts) == 1:
        return parts[0]
    return hstack(parts, format="csr")


def fit_vectorizer_and_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[
    TfidfVectorizer,
    TfidfVectorizer | None,
    TruncatedSVD | None,
    object,
    object,
    float | None,
]:
    vectorizer = _build_word_tfidf()
    X_train_vec = vectorizer.fit_transform(df_train["text"])
    X_val_vec = vectorizer.transform(df_val["text"])

    char_vectorizer = None
    if config.BASELINE_USE_CHAR_TFIDF:
        char_vectorizer = _build_char_tfidf()
        X_train_char = char_vectorizer.fit_transform(df_train["text"])
        X_val_char = char_vectorizer.transform(df_val["text"])
        X_train_vec = _stack_features([X_train_vec, X_train_char])
        X_val_vec = _stack_features([X_val_vec, X_val_char])

    svd = None
    explained = None
    use_lsa = config.BASELINE_USE_LSA
    use_extra = config.BASELINE_USE_EXTRA_FEATURES

    if use_lsa:
        svd = TruncatedSVD(
            n_components=config.LSA_N_COMPONENTS,
            random_state=config.LSA_RANDOM_STATE,
        )
        X_train_lsa = svd.fit_transform(X_train_vec)
        explained = float(np.sum(svd.explained_variance_ratio_))
        if use_extra:
            extra_train = utils.compute_extra_features(df_train["text"])
            extra_val = utils.compute_extra_features(df_val["text"])
            X_train = np.hstack([X_train_lsa, extra_train])
            X_val = np.hstack([svd.transform(X_val_vec), extra_val])
        else:
            X_train = X_train_lsa
            X_val = svd.transform(X_val_vec)
    elif use_extra:
        extra_train = utils.compute_extra_features(df_train["text"])
        extra_val = utils.compute_extra_features(df_val["text"])
        X_train = hstack(
            [X_train_vec, csr_matrix(extra_train)],
            format="csr",
        )
        X_val = hstack(
            [X_val_vec, csr_matrix(extra_val)],
            format="csr",
        )
    else:
        X_train = X_train_vec
        X_val = X_val_vec

    return vectorizer, char_vectorizer, svd, X_train, X_val, explained


def _training_labels(ratings: np.ndarray) -> np.ndarray:
    if config.BASELINE_USE_3CLASS:
        return utils.rating_to_3class(ratings)
    if config.BASELINE_USE_REGRESSION:
        return ratings.astype(np.float32) - 1.0
    return ratings - 1


def make_dmatrices(
    X_train,
    X_val,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    y_train = _training_labels(df_train["user_rating"].values)
    y_val = _training_labels(df_val["user_rating"].values)

    if config.XGB_USE_SAMPLE_WEIGHT and not config.BASELINE_USE_REGRESSION:
        class_w = utils.compute_class_weights(
            df_train["user_rating"].values,
            low_star_boost=config.XGB_LOW_STAR_BOOST,
        )
        sample_w = utils.compute_sample_weights_from_ratings(
            df_train["user_rating"].values,
            class_w,
        )
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_w)
    else:
        dtrain = xgb.DMatrix(X_train, label=y_train)

    dval = xgb.DMatrix(X_val, label=y_val)
    return dtrain, dval


def save_holdout_csv(df_holdout: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(config.HOLDOUT_PATH), exist_ok=True)
    df_holdout.to_csv(config.HOLDOUT_PATH, index=False)
    print(f"Saved holdout split: {config.HOLDOUT_PATH} (n={len(df_holdout)})")


def save_artifacts(
    vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer | None,
    svd: TruncatedSVD | None,
    bst: xgb.Booster,
    meta: dict,
) -> None:
    os.makedirs(config.BASELINE_ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PATH)
    if char_vectorizer is not None:
        joblib.dump(char_vectorizer, config.CHAR_TFIDF_VECTORIZER_PATH)
    elif os.path.isfile(config.CHAR_TFIDF_VECTORIZER_PATH):
        os.remove(config.CHAR_TFIDF_VECTORIZER_PATH)
    if config.BASELINE_USE_LSA and svd is not None:
        joblib.dump(svd, config.LSA_TRANSFORMER_PATH)
    elif os.path.isfile(config.LSA_TRANSFORMER_PATH):
        os.remove(config.LSA_TRANSFORMER_PATH)
    bst.save_model(config.XGB_MODEL_PATH)
    with open(config.BASELINE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    print("--- Step 1: Loading data ---")
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH)

    df, clean_stats = utils.clean_review_dataframe(
        df,
        min_text_length=config.MIN_TEXT_LENGTH,
        drop_duplicates=config.DROP_DUPLICATE_TEXT,
        duplicate_keep=config.DUPLICATE_KEEP,
    )
    utils.log_cleaning_stats(clean_stats, label="train_reduce")

    df_pool, df_holdout = train_test_split(
        df,
        test_size=config.HOLDOUT_FRACTION,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )
    df_train, df_val = train_test_split(
        df_pool,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df_pool["user_rating"],
    )
    save_holdout_csv(df_holdout)

    test_ratings = None
    if os.path.isfile(config.WONGNAI_TEST_PATH):
        df_test = utils.load_and_standardize_data(config.WONGNAI_TEST_PATH)
        df_test, _ = utils.clean_review_dataframe(
            df_test,
            min_text_length=config.MIN_TEXT_LENGTH,
            drop_duplicates=config.DROP_DUPLICATE_TEXT,
            duplicate_keep=config.DUPLICATE_KEEP,
        )
        test_ratings = df_test["user_rating"].values

    utils.compare_rating_distributions(
        df_train["user_rating"].values,
        test_ratings,
        label_train="train split (after clean)",
        label_test="HF test (after clean)",
    )

    train_df = df_train
    if config.BASELINE_MOCK_MIX_FRACTION > 0:
        before = len(train_df)
        train_df = utils.mix_mock_training_data(
            train_df,
            config.MOCK_TRAIN_PATH,
            config.BASELINE_MOCK_MIX_FRACTION,
            min_text_length=config.MIN_TEXT_LENGTH,
            drop_duplicates=config.DROP_DUPLICATE_TEXT,
            duplicate_keep=config.DUPLICATE_KEEP,
            random_state=config.RANDOM_STATE,
        )
        print(
            f"Mock mix (fraction={config.BASELINE_MOCK_MIX_FRACTION}): "
            f"{before} -> {len(train_df)} rows"
        )

    if config.BASELINE_UNDERSAMPLE_STAR4_FRACTION < 1.0:
        before = len(train_df)
        train_df = utils.undersample_star_ratings(
            train_df,
            star=4,
            keep_fraction=config.BASELINE_UNDERSAMPLE_STAR4_FRACTION,
            random_state=config.RANDOM_STATE,
        )
        print(
            f"Undersample 4-star (keep={config.BASELINE_UNDERSAMPLE_STAR4_FRACTION}): "
            f"{before} -> {len(train_df)} rows"
        )

    if config.BASELINE_OVERSAMPLE_LOW_STARS:
        train_df = utils.oversample_low_star_reviews(
            train_df,
            factor=config.BASELINE_OVERSAMPLE_FACTOR,
        )
        print(
            f"Oversampled low stars (factor={config.BASELINE_OVERSAMPLE_FACTOR}): "
            f"{len(df_train)} -> {len(train_df)} rows"
        )

    train_df = utils.apply_text_truncation(train_df, config.MAX_REVIEW_CHARS)
    df_val = utils.apply_text_truncation(df_val, config.MAX_REVIEW_CHARS)

    print("--- Step 2: TF-IDF + features ---")
    vectorizer, char_vectorizer, svd, X_train, X_val, explained = (
        fit_vectorizer_and_features(train_df, df_val)
    )
    print(
        f"Feature shape (train): {X_train.shape} | (val): {X_val.shape}"
    )
    if explained is not None:
        print(f"LSA explained variance ratio (sum): {explained:.6f}")

    print("--- Step 3: Training XGBoost ---")
    if config.XGB_USE_SAMPLE_WEIGHT:
        class_w = utils.compute_class_weights(
            train_df["user_rating"].values,
            low_star_boost=config.XGB_LOW_STAR_BOOST,
        )
        utils.print_rating_distribution(
            train_df["user_rating"].values,
            class_w,
            label="train",
        )

    dtrain, dval = make_dmatrices(X_train, X_val, train_df, df_val)
    xgb_params = dict(config.XGB_PARAMS)
    if config.BASELINE_USE_REGRESSION:
        xgb_params["objective"] = "reg:squarederror"
        xgb_params.pop("num_class", None)
        xgb_params["eval_metric"] = "rmse"
    elif config.BASELINE_USE_3CLASS:
        xgb_params["num_class"] = 3
    bst, evals_result, best_val = train_xgb_native(dtrain, dval, xgb_params)

    best_iter = bst.best_iteration
    train_hist = evals_result.get("train", {}).get("mlogloss", [])
    val_hist = evals_result.get("val", {}).get("mlogloss", [])
    if train_hist and val_hist:
        idx = min(best_iter, len(train_hist) - 1, len(val_hist) - 1)
        print(
            f"Best iteration: {best_iter} | "
            f"train mlogloss: {train_hist[idx]:.6f} | "
            f"val mlogloss: {val_hist[idx]:.6f}"
        )

    meta = {
        "tfidf_max_features": config.TFIDF_MAX_FEATURES,
        "use_lsa": config.BASELINE_USE_LSA,
        "lsa_n_components": config.LSA_N_COMPONENTS if config.BASELINE_USE_LSA else None,
        "use_extra_features": config.BASELINE_USE_EXTRA_FEATURES,
        "use_char_tfidf": config.BASELINE_USE_CHAR_TFIDF,
        "use_3class": config.BASELINE_USE_3CLASS,
        "use_regression": config.BASELINE_USE_REGRESSION,
        "mock_mix_fraction": config.BASELINE_MOCK_MIX_FRACTION,
        "undersample_star4_fraction": config.BASELINE_UNDERSAMPLE_STAR4_FRACTION,
        "low_star_boost": config.XGB_LOW_STAR_BOOST,
        "max_review_chars": config.MAX_REVIEW_CHARS,
        "oversample_low_stars": config.BASELINE_OVERSAMPLE_LOW_STARS,
        "oversample_factor": config.BASELINE_OVERSAMPLE_FACTOR,
        "use_sample_weight": config.XGB_USE_SAMPLE_WEIGHT,
        "objective": xgb_params.get("objective", config.XGB_PARAMS["objective"]),
        "best_val_metric": best_val,
        "explained_variance": explained,
    }

    print("--- Step 4: Saving artifacts ---")
    save_artifacts(vectorizer, char_vectorizer, svd, bst, meta)
    print("Done baseline training!")


if __name__ == "__main__":
    main()
