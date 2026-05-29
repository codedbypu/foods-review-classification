# Food Review Classification (Minimal / Custom Loop)

Flat pipeline for Thai restaurant review sentiment (1–5 stars) with **native XGBoost training** and a **manual PyTorch loop** for XLM-R — no `XGBClassifier.fit()` and no Hugging Face `Trainer`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download [iamwarint/wongnai-restaurant-review](https://huggingface.co/datasets/iamwarint/wongnai-restaurant-review) and save as:

`data/wongnai_train.csv`

Expected columns: `review_body`, `stars` (aliases like `text` / `user_rating` also work).

## Run

```bash
python train_baseline.py
python train_xlmr.py
python score.py
python score.py --model xlmr
```

Outputs:

- `artifacts/tfidf_vectorizer.joblib`, `artifacts/xgb_model.json`
- `artifacts/xlmr_model/`
- `outputs/scored_output_minimal.csv`

Full documentation: [README2.md](README2.md)
