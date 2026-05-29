"""Download Wongnai restaurant review dataset from Hugging Face to data/wongnai/."""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

HF_DATASET = "iamwarint/wongnai-restaurant-review"
SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

base = f"hf://datasets/{HF_DATASET}/"
token = os.environ.get("HF_TOKEN")
storage_options = {"token": token} if token else None

train_path = Path(config.WONGNAI_TRAIN_PATH)
test_path = Path(config.WONGNAI_TEST_PATH)
train_path.parent.mkdir(parents=True, exist_ok=True)

df_train = pd.read_parquet(base + SPLITS["train"], storage_options=storage_options)
df_test = pd.read_parquet(base + SPLITS["test"], storage_options=storage_options)

print(f"train shape: {df_train.shape}")
print(f"test shape: {df_test.shape}")
print(df_train.head())

df_train[["review_body", "stars"]].to_csv(
    train_path, index=False, encoding="utf-8-sig"
)
df_test[["review_body", "stars"]].to_csv(
    test_path, index=False, encoding="utf-8-sig"
)
print(f"Saved {train_path}")
print(f"Saved {test_path}")
