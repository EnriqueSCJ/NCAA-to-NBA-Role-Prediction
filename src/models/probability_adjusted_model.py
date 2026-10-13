"""Prediction-time probability adjustment wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import src.config as config


class ProbabilityAdjustedRoleClassifier:
    """Wrap a role classifier and multiply class probabilities before argmax."""

    def __init__(self, base_model: Any, class_multipliers: dict[str, float]):
        self.base_model = base_model
        self.class_multipliers = {
            role: float(class_multipliers.get(role, 1.0))
            for role in config.ROLE_LABELS
        }
        self.classes_ = np.array(config.ROLE_LABELS)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.base_model.predict_proba(X), dtype=float)
        labels = [str(label) for label in getattr(self.base_model, "classes_", config.ROLE_LABELS)]

        aligned = np.zeros((len(X), len(config.ROLE_LABELS)), dtype=float)
        for source_index, label in enumerate(labels):
            if label in config.ROLE_LABELS and source_index < probabilities.shape[1]:
                aligned[:, config.ROLE_LABELS.index(label)] = probabilities[:, source_index]

        multipliers = np.array([self.class_multipliers[label] for label in config.ROLE_LABELS], dtype=float)
        adjusted = np.clip(aligned * multipliers, 0.0, None)
        row_sums = adjusted.sum(axis=1, keepdims=True)
        return np.divide(
            adjusted,
            row_sums,
            out=np.full_like(adjusted, 1.0 / len(config.ROLE_LABELS)),
            where=row_sums > 0,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
