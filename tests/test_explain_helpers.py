"""Smoke tests for SHAP helper shape handling."""

from __future__ import annotations

import numpy as np

from src.models.explain import mean_abs_shap_by_class, select_shap_values_for_class


def main() -> None:
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    selected = select_shap_values_for_class(values, sample_idx=1, class_idx=2, n_features=3)
    expected = values[1, :, 2]
    if not np.array_equal(selected, expected):
        raise AssertionError(f"Unexpected selected values: {selected} != {expected}")

    mean_values = mean_abs_shap_by_class(values, class_idx=2, n_features=3)
    expected_mean = np.abs(values[:, :, 2]).mean(axis=0)
    if not np.array_equal(mean_values, expected_mean):
        raise AssertionError(f"Unexpected mean values: {mean_values} != {expected_mean}")

    list_values = [np.ones((2, 3)) * idx for idx in range(4)]
    selected_from_list = select_shap_values_for_class(list_values, sample_idx=0, class_idx=3, n_features=3)
    if not np.array_equal(selected_from_list, np.ones(3) * 3):
        raise AssertionError("List SHAP output not handled correctly")

    print("SHAP helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
