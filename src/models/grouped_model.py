"""Grouped-role models and helpers for robust public-facing predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import src.config as config


GROUPED_LABELS = list(config.GROUPED_ROLE_LABELS)


def group_roles(y: pd.Series | np.ndarray | list[Any]) -> pd.Series:
    """Collapse BENCH and ROTATION into one NBA role-player class."""
    values = pd.Series(y).astype(str)
    return values.replace(config.ROLE_GROUP_MAP)


def _aligned_probabilities(
    model: Any,
    X: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    """Return model probabilities aligned to an explicit label order."""
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    model_classes = [str(label) for label in getattr(model, "classes_", labels)]
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    for source_index, label in enumerate(model_classes):
        if label in labels and source_index < probabilities.shape[1]:
            aligned[:, labels.index(label)] = probabilities[:, source_index]

    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
    return aligned / row_sums


class GroupedXGBoostClassifier:
    """Small XGBoost wrapper for grouped role labels."""

    def __init__(self, booster: Any, classes: list[str]):
        self.booster = booster
        self.classes_ = np.array(classes)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        return np.asarray(self.booster.predict(dmatrix))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class RoleGroupCascadeClassifier:
    """Use a grouped model for coarse role and a fine model for BENCH/ROTATION."""

    def __init__(self, grouped_model: Any, fine_model: Any):
        self.grouped_model = grouped_model
        self.fine_model = fine_model
        self.classes_ = np.array(config.ROLE_LABELS)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        grouped = _aligned_probabilities(self.grouped_model, X, GROUPED_LABELS)
        fine = _aligned_probabilities(self.fine_model, X, config.ROLE_LABELS)

        output = np.zeros((len(X), len(config.ROLE_LABELS)), dtype=float)
        non_index = config.ROLE_LABELS.index("NON_NBA")
        bench_index = config.ROLE_LABELS.index("BENCH")
        rotation_index = config.ROLE_LABELS.index("ROTATION")
        starter_index = config.ROLE_LABELS.index("STARTER_PLUS")

        grouped_non = GROUPED_LABELS.index("NON_NBA")
        grouped_middle = GROUPED_LABELS.index("NBA_ROLE_PLAYER")
        grouped_starter = GROUPED_LABELS.index("STARTER_PLUS")

        fine_middle = fine[:, bench_index] + fine[:, rotation_index]
        fine_middle = np.where(np.isclose(fine_middle, 0.0), 1.0, fine_middle)
        bench_share = fine[:, bench_index] / fine_middle
        rotation_share = fine[:, rotation_index] / fine_middle

        output[:, non_index] = grouped[:, grouped_non]
        output[:, bench_index] = grouped[:, grouped_middle] * bench_share
        output[:, rotation_index] = grouped[:, grouped_middle] * rotation_share
        output[:, starter_index] = grouped[:, grouped_starter]

        row_sums = output.sum(axis=1, keepdims=True)
        row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        return output / row_sums

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]
