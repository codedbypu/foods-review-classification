import os

import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import config
import utils


def train_xgb_native(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict,
) -> xgb.Booster:
    """Train with xgb.train(); fall back to CPU if GPU fails."""
    watchlist = [(dtrain, "train"), (dval, "val")]
    try:
        return xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=config.XGB_ROUNDS,
            evals=watchlist,
            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
            verbose_eval=10,
        )
    except xgb.core.XGBoostError as exc:
        if params.get("device") == "cuda":
            print(f"GPU training failed ({exc}); retrying on CPU.")
            cpu_params = {**params, "device": "cpu"}
            return xgb.train(
                params=cpu_params,
                dtrain=dtrain,
                num_boost_round=config.XGB_ROUNDS,
                evals=watchlist,
                early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=10,
            )
        raise


def main() -> None:
    print("--- Step 1: Loading & Tokenizing Data ---")
    df = utils.load_and_standardize_data(config.RAW_DATA_PATH)

    df_train, df_val = train_test_split(
        df,
        test_size=0.2,
        random_state=config.RANDOM_STATE,
        stratify=df["user_rating"],
    )

    vectorizer = TfidfVectorizer(
        tokenizer=utils.thai_tokenizer,
        token_pattern=None,
        max_features=config.TFIDF_MAX_FEATURES,
    )
    X_train_vec = vectorizer.fit_transform(df_train["text"])
    X_val_vec = vectorizer.transform(df_val["text"])

    y_train_cls = df_train["user_rating"].values - 1
    y_val_cls = df_val["user_rating"].values - 1

    print("--- Step 2: Converting to Native DMatrix ---")
    dtrain = xgb.DMatrix(X_train_vec, label=y_train_cls)
    dval = xgb.DMatrix(X_val_vec, label=y_val_cls)

    print("--- Step 3: Training via Native API Engine (No .fit) ---")
    bst = train_xgb_native(dtrain, dval, config.XGB_PARAMS)

    print("--- Step 4: Saving Trained Artifacts ---")
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(
        vectorizer,
        os.path.join(config.ARTIFACTS_DIR, "tfidf_vectorizer.joblib"),
    )
    bst.save_model(os.path.join(config.ARTIFACTS_DIR, "xgb_model.json"))
    print("Done baseline training!")


if __name__ == "__main__":
    main()
