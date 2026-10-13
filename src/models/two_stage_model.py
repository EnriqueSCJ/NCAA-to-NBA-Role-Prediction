"""Reusable two-stage estimator wrappers.

These classes stay in a normal importable module so persisted models can be
loaded by the API container without relying on ``__main__`` pickle aliases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import src.config as config


GATE_LABELS = ["NON_NBA", "NBA"]
ROLE_CONDITIONED_LABELS = ["BENCH", "ROTATION", "STARTER_PLUS"]


class BinaryXGBoostClassifier:
    """Small wrapper for binary XGBoost probabilities and labels."""

    def __init__(self, booster: Any, threshold: float = 0.5):
        self.booster = booster
        self.threshold = threshold
        self.classes_ = np.array(GATE_LABELS)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        nba_probability = np.asarray(self.booster.predict(dmatrix), dtype=float).ravel()
        nba_probability = np.clip(nba_probability, 0.0, 1.0)
        return np.column_stack([1.0 - nba_probability, nba_probability])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        nba_probability = self.predict_proba(X)[:, 1]
        return np.where(nba_probability >= self.threshold, "NBA", "NON_NBA")


class TwoStageRoleClassifier:
    """Expose the NBA gate plus conditioned role model as a five-class model."""

    def __init__(self, gate_model: Any, role_model: Any, gate_threshold: float):
        self.gate_model = gate_model
        self.role_model = role_model
        self.gate_threshold = gate_threshold
        self.classes_ = np.array(config.ROLE_LABELS)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        gate_probabilities = np.asarray(self.gate_model.predict_proba(X), dtype=float)
        gate_classes = [str(label) for label in getattr(self.gate_model, "classes_", GATE_LABELS)]
        nba_index = gate_classes.index("NBA") if "NBA" in gate_classes else gate_probabilities.shape[1] - 1
        nba_probability = np.clip(gate_probabilities[:, nba_index], 0.0, 1.0)

        role_probabilities = np.asarray(self.role_model.predict_proba(X), dtype=float)
        role_classes = [str(label) for label in getattr(self.role_model, "classes_", ROLE_CONDITIONED_LABELS)]

        output = np.zeros((len(X), len(config.ROLE_LABELS)), dtype=float)
        output[:, config.ROLE_LABELS.index("NON_NBA")] = 1.0 - nba_probability

        for role in ROLE_CONDITIONED_LABELS:
            if role in role_classes:
                role_source_index = role_classes.index(role)
                role_target_index = config.ROLE_LABELS.index(role)
                output[:, role_target_index] = nba_probability * role_probabilities[:, role_source_index]

        row_sums = output.sum(axis=1, keepdims=True)
        row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        return output / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        labels = np.asarray(config.ROLE_LABELS)
        predictions = labels[probabilities.argmax(axis=1)]
        nba_probability = 1.0 - probabilities[:, config.ROLE_LABELS.index("NON_NBA")]
        return np.where(nba_probability >= self.gate_threshold, predictions, "NON_NBA")
