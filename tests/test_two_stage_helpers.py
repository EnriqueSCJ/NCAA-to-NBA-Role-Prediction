"""Smoke tests for two-stage modeling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.train_two_stage import combine_two_stage_predictions, make_gate_target


class FixedGate:
    classes_ = np.array(["NON_NBA", "NBA"])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])[: len(X)]


class FixedRole:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(["BENCH", "ROTATION", "STARTER_PLUS"])[: len(X)]


def main() -> None:
    y = pd.Series(["NON_NBA", "BENCH", "STARTER_PLUS"])
    gate = make_gate_target(y)
    if gate.tolist() != ["NON_NBA", "NBA", "NBA"]:
        raise AssertionError("Gate target was not built correctly")

    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    predictions = combine_two_stage_predictions(FixedGate(), FixedRole(), X, gate_threshold=0.5)
    if predictions.tolist() != ["NON_NBA", "BENCH", "ROTATION"]:
        raise AssertionError("Two-stage combination did not apply the gate correctly")

    print("Two-stage helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
