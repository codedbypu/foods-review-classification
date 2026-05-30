from pathlib import Path
import os

import torch
# Project root (this file lives at repo root)
ROOT = Path(__file__).resolve().parent


def _p(*parts: str) -> str:
    """Build absolute path string under project root."""
    return str(ROOT.joinpath(*parts))


# --- Data ---
DATA_DIR = _p("data")
WONGNAI_TRAIN_PATH = _p("data", "wongnai", "train_reduce.csv")
RAW_DATA_PATH = WONGNAI_TRAIN_PATH
WONGNAI_TEST_PATH = _p("data", "wongnai", "test.csv")
HOLDOUT_PATH = _p("data", "wongnai", "holdout.csv")
HOLDOUT_FRACTION = 0.2
MOCK_TRAIN_PATH = _p("data", "mock", "train.csv")
MOCK_TEST_PATH = _p("data", "mock", "test.csv")

# --- Data cleaning (baseline) ---
MIN_TEXT_LENGTH = 5
DROP_DUPLICATE_TEXT = True
DUPLICATE_KEEP = "first"

# --- Artifacts (trained models) ---
ARTIFACTS_DIR = _p("artifacts")
BASELINE_ARTIFACTS_DIR = _p("artifacts", "baseline")
XLMR_ARTIFACTS_DIR = _p("artifacts", "xlmr")
TFIDF_VECTORIZER_PATH = _p("artifacts", "baseline", "tfidf_vectorizer.joblib")
CHAR_TFIDF_VECTORIZER_PATH = _p("artifacts", "baseline", "char_tfidf_vectorizer.joblib")
LSA_TRANSFORMER_PATH = _p("artifacts", "baseline", "lsa_transformer.joblib")
XGB_MODEL_PATH = _p("artifacts", "baseline", "xgb_model.json")
BASELINE_META_PATH = _p("artifacts", "baseline", "baseline_meta.json")

# --- Outputs (production pipeline) ---
OUTPUTS_DIR = _p("outputs")
SCORES_DIR = _p("outputs", "scores")
EVAL_DIR = _p("outputs", "eval")
REPORTS_DIR = _p("outputs", "reports")
DEFAULT_SCORED_OUTPUT = _p("outputs", "scores", "scored_output_minimal.csv")
DEFAULT_EVAL_REPORT = _p("outputs", "eval", "eval_report.json")
DEFAULT_EVAL_VIZ = _p("outputs", "reports", "eval_report_viz.html")

# --- Experiments (tune / EDA / try-log — ไม่ปนกับ outputs หลัก) ---
EXPERIMENTS_DIR = _p("experiments")
BASELINE_EXPERIMENTS_DIR = _p("experiments", "baseline")
EXPERIMENT_EDA_SUMMARY_PATH = _p("experiments", "baseline", "eda_summary.json")
EXPERIMENT_TUNE_LOG_PATH = _p("experiments", "baseline", "tune_log.json")
EXPERIMENT_TRY_LOG_PATH = _p("experiments", "baseline", "try_log.md")
EXPERIMENT_ERRORS_DIR = _p("experiments", "baseline", "errors")
EXPERIMENT_EVAL_DIR = _p("experiments", "baseline", "eval")

# Device (PyTorch vs XGBoost kept separate; override via FORCE_TORCH_DEVICE / FORCE_XGB_DEVICE)
TORCH_DEVICE = os.environ.get("FORCE_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)
XGB_DEVICE = os.environ.get("FORCE_XGB_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --- Baseline data strategy (winner from holdout ablation) ---
MAX_REVIEW_CHARS = 500
BASELINE_OVERSAMPLE_LOW_STARS = True
BASELINE_OVERSAMPLE_FACTOR = 5
BASELINE_OVERSAMPLE_USE_WEIGHT = True
BASELINE_MAJORITY_CLASS = 4

# --- Baseline TF-IDF / features ---
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9
BASELINE_USE_LSA = False
LSA_N_COMPONENTS = 400
BASELINE_USE_EXTRA_FEATURES = True
BASELINE_USE_CHAR_TFIDF = True
BASELINE_CHAR_MAX_FEATURES = 4000
BASELINE_CHAR_NGRAM_RANGE = (3, 5)

# --- Baseline training experiments (tune via scripts/tune_baseline.py) ---
BASELINE_UNDERSAMPLE_STAR4_FRACTION = 0.65  # winner best_combo_mock20
BASELINE_MOCK_MIX_FRACTION = 0.2  # 20% mock_train rows (mock test acc 1.0)
BASELINE_USE_3CLASS = False
BASELINE_USE_REGRESSION = False

# Error analysis export (evaluate.py --export-errors)
ERROR_EXPORT_MIN_DELTA = 2
LOW_CONFIDENCE_THRESHOLD = 0.4

# XGBoost Native API
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 5,
    "max_depth": 4,
    "eta": 0.03,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": XGB_DEVICE,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
}
XGB_ROUNDS = 800
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_USE_SAMPLE_WEIGHT = True
XGB_LOW_STAR_BOOST = 3.0

# XLM-RoBERTa (manual PyTorch loop) — 6GB GPU: batch=8, amp+checkpointing on
XLMR_MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 8
XLMR_GRAD_ACCUM_STEPS = 1
XLMR_USE_AMP = True
XLMR_GRADIENT_CHECKPOINTING = True
LEARNING_RATE = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
XLMR_USE_CLASS_WEIGHT = True
XLMR_LOW_STAR_BOOST = 1.5
XLMR_EARLY_STOPPING_PATIENCE = 3
XLMR_USE_LR_SCHEDULER = True
XLMR_LR_SCHEDULER_FACTOR = 0.5
XLMR_LR_SCHEDULER_PATIENCE = 1

# Scoring
ANOMALY_THRESHOLD = 2.0
RANDOM_STATE = 42
LSA_RANDOM_STATE = RANDOM_STATE
