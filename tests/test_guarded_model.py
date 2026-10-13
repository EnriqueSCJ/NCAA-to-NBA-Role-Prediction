"""Smoke tests for guarded role-model wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as config
from src.models.guarded_model import StarterPlusGuardedRoleClassifier


class DummyStarterModel:
    """Tiny base model with two STARTER_PLUS predictions."""

    classes_ = np.array(config.ROLE_LABELS)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                [0.02, 0.03, 0.05, 0.90],
                [0.01, 0.01, 0.00, 0.98],
            ]
        )[: len(X)]


class DummyConfirmationModel:
    """Confirmation model that accepts STARTER_PLUS for every row."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["STARTER_PLUS"] * len(X))


def test_starter_plus_probability_threshold_vetoes_low_confidence_prediction() -> None:
    X = pd.DataFrame({"feature": [1.0, 2.0]})
    model = StarterPlusGuardedRoleClassifier(
        base_model=DummyStarterModel(),
        confirmation_model=DummyConfirmationModel(),
        minimum_starter_plus_probability=0.95,
    )

    predictions = model.predict(X).tolist()
    probabilities = model.predict_proba(X)

    assert predictions == ["ROTATION", "STARTER_PLUS"]
    assert probabilities[0, config.ROLE_LABELS.index("STARTER_PLUS")] == 0.0
    assert np.allclose(probabilities.sum(axis=1), 1.0)
