"""Smoke tests for final-report helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.final_report import _as_label_series, _row_normalized_confusion_matrix, _top_k_accuracy


def main() -> None:
    labels = _as_label_series([0, "BENCH", 2], index=pd.RangeIndex(3))
    if labels.tolist() != ["NON_NBA", "BENCH", "ROTATION"]:
        raise AssertionError("Prediction labels were not normalized correctly")

    y_true = pd.Series(["STARTER_PLUS", "BENCH", "ROTATION"])
    probabilities = pd.DataFrame(
        {
            "NON_NBA": [0.05, 0.10, 0.20],
            "BENCH": [0.15, 0.60, 0.10],
            "ROTATION": [0.25, 0.20, 0.55],
            "STARTER_PLUS": [0.55, 0.05, 0.15],
        }
    )
    if _top_k_accuracy(y_true, probabilities, k=2) != 1.0:
        raise AssertionError("Top-2 accuracy helper returned an unexpected value")

    normalized = _row_normalized_confusion_matrix(np.array([[8, 2], [0, 0]]))
    if not np.allclose(normalized, np.array([[0.8, 0.2], [0.0, 0.0]])):
        raise AssertionError("Row-normalized confusion matrix is incorrect")

    print("Final-report helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
