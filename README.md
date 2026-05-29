# Food Review Classification (Minimal / Custom Loop)

Flat pipeline for Thai restaurant review sentiment (1–5 stars) with **native XGBoost training** and a **manual PyTorch loop** for XLM-R.

## Project layout

```
foods-review-classification/
├── config.py              # paths + hyperparameters
├── utils.py               # load CSV, Thai tokenizer
├── train_baseline.py      # TF-IDF + XGBoost
├── train_xlmr.py          # XLM-R manual loop
├── evaluate.py            # metrics on test set
├── visualize_eval.py      # HTML dashboard from eval JSON
├── score.py               # inference + anomaly flags
├── scripts/
│   ├── download_wongnai.py
│   └── generate_mock_data.py
├── data/
│   ├── wongnai/           # train.csv, test.csv (HF dataset)
│   └── mock/              # train.csv, test.csv (synthetic)
├── artifacts/
│   ├── baseline/          # tfidf_vectorizer.joblib, xgb_model.json
│   └── xlmr/              # Hugging Face save_pretrained
└── outputs/
    ├── scores/            # scored CSV
    ├── eval/              # eval_report.json, prediction CSVs
    └── reports/           # eval_report_viz.html
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Wongnai (recommended):

```powershell
python scripts/download_wongnai.py
```

Or place CSVs manually under `data/wongnai/` (`train.csv`, `test.csv`) with columns `review_body`, `stars`.

Mock data for quick tests:

```powershell
python scripts/generate_mock_data.py
```

Set training file in `config.py` → `RAW_DATA_PATH` (default: `data/mock/train.csv`).

## Run (all-in-one notebook)

```powershell
jupyter notebook run_pipeline.ipynb
```

Or step-by-step scripts:

```powershell
python train_baseline.py
python train_xlmr.py
python evaluate.py --model both --input data/wongnai/test.csv --output outputs/eval/eval_report.json
python visualize_eval.py
python score.py --input data/wongnai/test.csv --output outputs/scores/scored_test.csv
```

Paths can use forward slashes on Windows. Defaults are defined in `config.py`.

Full documentation: [README2.md](README2.md)
