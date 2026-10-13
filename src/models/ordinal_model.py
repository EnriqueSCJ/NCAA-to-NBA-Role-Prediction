"""Ordinal classifiers for NCAA to NBA role prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import src.config as config


class OrdinalXGBoostRoleClassifier:
    """Cumulative ordinal XGBoost model over the project role hierarchy."""

    def __init__(self, boosters: list[Any], classes: list[str] | None = None):
        self.boosters = boosters
        self.classes_ = np.array(classes or config.ROLE_LABELS)

    def _cumulative_probabilities(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        cumulative = [
            np.asarray(booster.predict(dmatrix), dtype=float).reshape(-1)
            for booster in self.boosters
        ]
        if not cumulative:
            return np.zeros((len(X), 0), dtype=float)

        probabilities = np.column_stack(cumulative)
        probabilities = np.clip(probabilities, 0.0, 1.0)

        # Enforce P(y >= BENCH) >= P(y >= ROTATION) >= P(y >= STARTER_PLUS).
        for column in range(1, probabilities.shape[1]):
            probabilities[:, column] = np.minimum(probabilities[:, column - 1], probabilities[:, column])
        return probabilities

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cumulative = self._cumulative_probabilities(X)
        if cumulative.shape[1] != len(self.classes_) - 1:
            return np.full((len(X), len(self.classes_)), 1.0 / len(self.classes_), dtype=float)

        output = np.zeros((len(X), len(self.classes_)), dtype=float)
        output[:, 0] = 1.0 - cumulative[:, 0]
        for index in range(1, len(self.classes_) - 1):
            output[:, index] = cumulative[:, index - 1] - cumulative[:, index]
        output[:, -1] = cumulative[:, -1]
        output = np.clip(output, 0.0, 1.0)
        row_sums = output.sum(axis=1, keepdims=True)
        return np.divide(output, row_sums, out=np.full_like(output, 1.0 / len(self.classes_)), where=row_sums > 0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


def train_ordinal_xgboost(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> OrdinalXGBoostRoleClassifier:
    """Train one binary XGBoost model per cumulative role threshold."""
    import xgboost as xgb

    X_train, X_val, _ = X_trains
    y_train, y_val, _ = y_trains
    role_to_index = {role: index for index, role in enumerate(config.ROLE_LABELS)}
    y_train_idx = y_train.map(role_to_index).astype(int).to_numpy()
    y_val_idx = y_val.map(role_to_index).astype(int).to_numpy()

    boosters: list[Any] = []
    for threshold_index in range(1, len(config.ROLE_LABELS)):
        train_binary = (y_train_idx >= threshold_index).astype(int)
        val_binary = (y_val_idx >= threshold_index).astype(int)
        positive = max(int(train_binary.sum()), 1)
        negative = max(int((train_binary == 0).sum()), 1)

        dtrain = xgb.DMatrix(X_train, label=train_binary, feature_names=list(X_train.columns))
        dval = xgb.DMatrix(X_val, label=val_binary, feature_names=list(X_val.columns))
        booster = xgb.train(
            params={
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "eta": 0.06,
                "max_depth": 4,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "scale_pos_weight": negative / positive,
                "seed": config.RANDOM_STATE + threshold_index,
            },
            dtrain=dtrain,
            num_boost_round=350,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        boosters.append(booster)

    return OrdinalXGBoostRoleClassifier(boosters=boosters, classes=config.ROLE_LABELS)
