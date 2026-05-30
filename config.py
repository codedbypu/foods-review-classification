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
RAW_DATA_PATH = _p("data", "wongnai", "train_reduce.csv")
WONGNAI_TRAIN_PATH = RAW_DATA_PATH
WONGNAI_TEST_PATH = _p("data", "wongnai", "test.csv")
MOCK_TRAIN_PATH = RAW_DATA_PATH
MOCK_TEST_PATH = _p("data", "mock", "test.csv")

# --- Artifacts (trained models) ---
ARTIFACTS_DIR = _p("artifacts")
BASELINE_ARTIFACTS_DIR = _p("artifacts", "baseline")
XLMR_ARTIFACTS_DIR = _p("artifacts", "xlmr")
TFIDF_VECTORIZER_PATH = _p("artifacts", "baseline", "tfidf_vectorizer.joblib")
XGB_MODEL_PATH = _p("artifacts", "baseline", "xgb_model.json")

# --- Outputs ---
OUTPUTS_DIR = _p("outputs")
SCORES_DIR = _p("outputs", "scores")
EVAL_DIR = _p("outputs", "eval")
REPORTS_DIR = _p("outputs", "reports")
DEFAULT_SCORED_OUTPUT = _p("outputs", "scores", "scored_output_minimal.csv")
DEFAULT_EVAL_REPORT = _p("outputs", "eval", "eval_report.json")
DEFAULT_EVAL_VIZ = _p("outputs", "reports", "eval_report_viz.html")

# Device (PyTorch vs XGBoost kept separate; override via FORCE_TORCH_DEVICE / FORCE_XGB_DEVICE)
TORCH_DEVICE = os.environ.get("FORCE_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)
XGB_DEVICE = os.environ.get("FORCE_XGB_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)
# TF-IDF
TFIDF_MAX_FEATURES = 20000

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
    "reg_lambda": 1.0,
}
XGB_ROUNDS = 400
XGB_EARLY_STOPPING_ROUNDS = 25
XGB_USE_SAMPLE_WEIGHT = True
XGB_LOW_STAR_BOOST = 1.5

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
