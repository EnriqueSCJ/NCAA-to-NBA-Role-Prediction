"""Smoke tests for experimental model helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as config
from src.models.experiment_lab import _predict_labels_from_probabilities, _sample_weights
from src.models.probability_adjusted_model import ProbabilityAdjustedRoleClassifier


class DummyProbabilityModel:
    """Tiny model with fixed class probabilities."""

    classes_ = np.array(config.ROLE_LABELS)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile(np.array([[0.40, 0.30, 0.20, 0.10]]), (len(X), 1))


def main() -> None:
    X = pd.DataFrame({"x": [1.0, 2.0]})
    model = ProbabilityAdjustedRoleClassifier(
        base_model=DummyProbabilityModel(),
        class_multipliers={
            "NON_NBA": 1.0,
            "BENCH": 1.0,
            "ROTATION": 1.0,
            "STARTER_PLUS": 6.0,
        },
    )
    probabilities = model.predict_proba(X)
    if probabilities.shape != (2, len(config.ROLE_LABELS)):
        raise AssertionError("Adjusted probability matrix has unexpected shape")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise AssertionError("Adjusted probabilities are not normalized")
    if model.predict(X).tolist() != ["STARTER_PLUS", "STARTER_PLUS"]:
        raise AssertionError("Probability adjustment did not affect argmax as expected")

    labels = _predict_labels_from_probabilities(
        np.array(
            [
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.2, 0.6, 0.1],
            ]
        ),
        config.ROLE_LABELS,
    )
    if labels.tolist() != ["BENCH", "ROTATION"]:
        raise AssertionError("Softprob labels were not decoded in class order")

    weights = _sample_weights(
        pd.Series(["NON_NBA", "BENCH", "STARTER_PLUS"]),
        {"NON_NBA": 0.8, "BENCH": 1.5, "ROTATION": 2.0, "STARTER_PLUS": 3.0},
    )
    if not np.allclose(weights, np.array([0.8, 1.5, 3.0])):
        raise AssertionError("Class weights were not mapped to rows correctly")

    print("Experiment helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
