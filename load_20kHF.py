import pandas as pd

splits = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}

base = "hf://datasets/iamwarint/wongnai-restaurant-review/"

df_train = pd.read_parquet(base + splits["train"], storage_options={"token": "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"})
df_test = pd.read_parquet(base + splits["test"], storage_options={"token": "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"})

print(df_train.shape)   # ประมาณ ~19,958 แถว
print(df_train.columns) # มี review_body, stars ฯลฯ
print(df_train.head())

# เก็บเฉพาะคอลัมน์ที่ pipeline ใช้ (หรือ export ทั้ง df ก็ได้)
# df_train[["review_body", "stars"]].to_csv(
#     "data/wongnai_train.csv",
#     index=False,
#     encoding="utf-8-sig",
# )

# df_test[["review_body", "stars"]].to_csv(
#     "data/wongnai_test.csv",
#     index=False,
#     encoding="utf-8-sig",
# )