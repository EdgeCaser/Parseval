"""Evaluation metrics for attribution experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe_metric(fn, y_true: np.ndarray, y_score: np.ndarray, fallback: float | None = None):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return fallback


def classification_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict:
    """Return core binary classification metrics for `not_author` detection."""
    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float64)

    if yt.shape[0] == 0:
        raise ValueError("No labels provided.")
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("y_true and y_score must have same length.")

    yp = (ys >= threshold).astype(np.int64)

    tp = int(np.sum((yp == 1) & (yt == 1)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "n": int(yt.shape[0]),
        "positive_rate": float(np.mean(yt)),
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "roc_auc": _safe_metric(roc_auc_score, yt, ys),
        "pr_auc": _safe_metric(average_precision_score, yt, ys),
        "brier": _safe_metric(brier_score_loss, yt, ys),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
