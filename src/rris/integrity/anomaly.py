from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def expected_rating_from_probs(probs: np.ndarray) -> float:
    """Expected rating from a single probability vector of shape (5,)."""
    probs = np.asarray(probs)
    if probs.shape != (5,):
        raise ValueError(f"Expected probs shape (5,), got {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite probabilities")
    s = float(probs.sum())
    if s <= 0:
        raise ValueError("Sum of probabilities must be > 0")
    p = probs / s
    classes = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    return float((classes * p).sum())


@dataclass(frozen=True)
class AnomalyResult:
    delta: float
    is_anomaly: bool


def anomaly_check(user_rating: float, ai_rating: float, threshold: float = 2.0) -> AnomalyResult:
    if not (1.0 <= float(user_rating) <= 5.0):
        raise ValueError(f"user_rating out of range: {user_rating}")
    if not (1.0 <= float(ai_rating) <= 5.0):
        raise ValueError(f"ai_rating out of range: {ai_rating}")
    delta = float(abs(float(user_rating) - float(ai_rating)))
    return AnomalyResult(delta=delta, is_anomaly=delta >= float(threshold))

