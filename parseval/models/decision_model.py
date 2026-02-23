"""Probabilistic decision model for not-author scoring.

This module provides a small supervised model wrapper intended for v2 scorer
integration and offline benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


@dataclass
class DecisionModel:
    """Calibrated logistic model over engineered scorer signals."""

    model: CalibratedClassifierCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X, y)
        self.model = clf

    def predict_proba_not_author(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("DecisionModel is not fitted.")
        probs = self.model.predict_proba(X)
        # classes are [0,1] for author/not_author
        return probs[:, 1]


def feature_vector(style_similarity: float | None, embedding_similarity: float | None, combined_score: float | None) -> np.ndarray:
    """Build minimal v2 feature vector from existing scorer outputs.

    Missing inputs are mapped to neutral values so that fragments can still be
    processed with uncertainty handling at policy level.
    """
    s = 0.5 if style_similarity is None else float(style_similarity)
    e = 0.5 if embedding_similarity is None else float(embedding_similarity)
    c = 0.5 if combined_score is None else float(combined_score)
    return np.array([s, e, c, abs(s - e)], dtype=np.float64)
