# Food Review Classification (Minimal / Custom Loop)

Flat pipeline for Thai restaurant review sentiment (1–5 stars): **TF-IDF + XGBoost baseline** (native API) and optional **XLM-R** (manual PyTorch loop).

## Project layout

```
foods-review-classification/
├── config.py                 # paths + hyperparameters (single source of truth)
├── utils.py                  # load/clean CSV, Thai tokenizer, extra features
├── train_baseline.py         # word + char TF-IDF → XGBoost
├── train_xlmr.py             # XLM-R manual loop
├── evaluate.py               # metrics, majority baseline, error export
├── visualize_eval.py         # HTML dashboard from eval JSON
├── score.py                  # inference + anomaly flags
├── run_pipeline.ipynb        # all-in-one notebook
├── scripts/
│   ├── download_wongnai.py   # HF → train.csv + test.csv
│   ├── generate_mock_data.py # synthetic smoke-test CSV
│   ├── eda_baseline_data.py  # EDA → experiments/baseline/
│   ├── tune_baseline.py      # config sweep → experiments/baseline/
│   ├── try_log_utils.py
│   └── summarize_tune_ceiling.py
├── experiments/              # ทดลอง/จูน (แยกจาก outputs)
│   └── baseline/             # try_log.md, tune_log.json, eda, errors
├── data/
│   ├── wongnai/              # train_reduce.csv (train), test.csv, holdout.csv
│   └── mock/                 # train.csv, test.csv (synthetic)
├── artifacts/
│   └── baseline/             # vectorizers, xgb_model.json, baseline_meta.json
└── outputs/                  # production pipeline เท่านั้น (gitignored)
    ├── eval/                 # eval_report.json
    ├── reports/              # eval_report_viz.html
    └── scores/
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

**Wongnai (recommended):**

```powershell
python scripts/download_wongnai.py
```

Training uses **`data/wongnai/train_reduce.csv`** (~5k rows) via `config.RAW_DATA_PATH`. Full HF train (`train.csv`, ~79k) is optional — change `WONGNAI_TRAIN_PATH` / `RAW_DATA_PATH` in `config.py` if needed.

**Mock (smoke test):**

```powershell
python scripts/generate_mock_data.py
```

## Quick run

```powershell
python train_baseline.py
python evaluate.py --model baseline --input data/wongnai/holdout.csv --export-errors
python evaluate.py --model baseline --input data/wongnai/test.csv --output outputs/eval/eval_report.json
python score.py --model baseline --input data/wongnai/test.csv
```

Or use the notebook:

```powershell
jupyter notebook run_pipeline.ipynb
```

## Baseline tuning (optional)

```powershell
python scripts/eda_baseline_data.py
python scripts/tune_baseline.py --append-try-log
python scripts/summarize_tune_ceiling.py
```

See [experiments/README.md](experiments/README.md) and [README2.md §18](README2.md#s18).

Full documentation: [README2.md](README2.md)
