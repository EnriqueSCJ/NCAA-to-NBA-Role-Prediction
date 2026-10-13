"""Smoke tests for processed real-data artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.config as config
from src.data.build_master import MAX_MODELING_DRAFT_YEAR, MIN_MODELING_DRAFT_YEAR


def main() -> None:
    master_path = config.DATA_PROCESSED / "master_table.csv"
    x_path = config.DATA_PROCESSED / "X.csv"
    y_path = config.DATA_PROCESSED / "y.csv"
    scaler_path = config.DATA_PROCESSED / "feature_scaler.json"

    for path in [master_path, x_path, y_path, scaler_path]:
        if not path.exists():
            raise AssertionError(f"Missing processed artifact: {path}")

    master = pd.read_csv(master_path, low_memory=False)
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns")

    if not (len(master) == len(X) == len(y)):
        raise AssertionError("Processed master, X and y are not row-aligned")

    if list(X.columns) != list(config.MODEL_FEATURE_COLUMNS):
        raise AssertionError("X.csv does not match the configured feature contract")
    if config.NUMERIC_VECTOR_DIM != len(config.MODEL_FEATURE_COLUMNS):
        raise AssertionError("Milvus numeric vector dim should match the pre-draft model feature contract")

    if X.isna().any().any() or np.isinf(X.to_numpy(dtype=float)).any():
        raise AssertionError("X.csv contains invalid numeric values")

    missing_labels = set(y.unique()).difference(config.ROLE_LABELS)
    if missing_labels:
        raise AssertionError(f"Unexpected target labels: {sorted(missing_labels)}")

    if master["draft_year"].min() < MIN_MODELING_DRAFT_YEAR or master["draft_year"].max() > MAX_MODELING_DRAFT_YEAR:
        raise AssertionError("Processed draft-year window is outside the modeled range")

    print("Processed real-data artifacts smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
