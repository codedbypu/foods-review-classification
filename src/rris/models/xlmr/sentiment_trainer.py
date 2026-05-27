from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XlmrSentimentConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 256
    lr: float = 2e-5
    epochs: int = 3
    train_batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    seed: int = 42
    disable_tqdm: bool = False


def _tokenize_dataset(
    ds: Dataset,
    tokenizer,
    max_length: int,
    *,
    desc: str = "Tokenizing",
) -> Dataset:
    def _tok(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return ds.map(_tok, batched=True, desc=desc)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float((preds == labels).mean())
    return {"accuracy": acc}


def train_xlmr_sentiment(
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    cfg: XlmrSentimentConfig,
    out_dir: str | Path,
) -> Trainer:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    logger.info(
        "XLM-R training device: %s (fp16=%s)",
        "cuda" if use_cuda else "cpu",
        use_cuda,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=5)

    train_tok = _tokenize_dataset(
        train_ds, tokenizer, cfg.max_length, desc="Tokenize train"
    )
    val_tok = _tokenize_dataset(val_ds, tokenizer, cfg.max_length, desc="Tokenize val")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(out_dir),
        seed=cfg.seed,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
        fp16=use_cuda,
        disable_tqdm=cfg.disable_tqdm,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return trainer


def probs_and_expected_rating_from_logits(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    logits: (N, 5)
    returns: (probs(N,5), expected_rating(N,))
    """
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError(f"Expected logits shape (N,5), got {logits.shape}")
    x = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(x)
    probs = exp / exp.sum(axis=1, keepdims=True)
    classes = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    expected = (probs * classes[None, :]).sum(axis=1)
    return probs, expected
