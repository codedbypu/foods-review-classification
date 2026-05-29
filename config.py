import torch

# Paths
RAW_DATA_PATH = "data/wongnai_train.csv"
ARTIFACTS_DIR = "artifacts"
OUTPUTS_DIR = "outputs"

# Device (PyTorch vs XGBoost kept separate)
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TF-IDF
TFIDF_MAX_FEATURES = 20_000

# XGBoost Native API
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 5,
    "max_depth": 6,
    "eta": 0.1,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": XGB_DEVICE,
}
XGB_ROUNDS = 100
XGB_EARLY_STOPPING_ROUNDS = 10

# XLM-RoBERTa (manual PyTorch loop)
XLMR_MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

# Scoring
ANOMALY_THRESHOLD = 2.0
RANDOM_STATE = 42
