"""Smoke tests for baseline training helpers."""

from __future__ import annotations

import src.config as config
from src.models.train import apply_smote, make_synthetic_training_data, select_best_model_name, split_data_temporal


def main() -> None:
    X, y, years = make_synthetic_training_data()
    X_trains, y_trains = split_data_temporal(X, y, years)

    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    if X_train.empty or X_val.empty or X_test.empty:
        raise AssertionError("Temporal split produced an empty block")
    if years.loc[X_test.index, "draft_year"].max() > config.TEST_END_DRAFT_YEAR:
        raise AssertionError("Final test split should exclude exploratory recent cohorts")

    X_smote, y_smote = apply_smote(X_train, y_train)
    if len(y_smote) <= len(y_train):
        raise AssertionError("SMOTE did not increase the training set size")

    for label in y.unique():
        if label not in set(y_smote):
            raise AssertionError(f"Missing class after SMOTE: {label}")

    selected = select_best_model_name(
        {
            "better_test": {"macro_f1": 0.90, "validation_macro_f1": 0.10},
            "better_validation": {"macro_f1": 0.20, "validation_macro_f1": 0.80},
        }
    )
    if selected != "better_validation":
        raise AssertionError("Model selection should use validation Macro F1, not test Macro F1")

    print("Baseline training helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
