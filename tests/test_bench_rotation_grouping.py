"""Smoke tests for BENCH/ROTATION grouping helpers."""

from __future__ import annotations

import pandas as pd

from src.models.bench_rotation_experiment import GROUPED_LABELS, group_roles


def main() -> None:
    grouped = group_roles(pd.Series(["NON_NBA", "BENCH", "ROTATION", "STARTER_PLUS"]))
    if grouped.tolist() != ["NON_NBA", "NBA_ROLE_PLAYER", "NBA_ROLE_PLAYER", "STARTER_PLUS"]:
        raise AssertionError("BENCH/ROTATION grouping is incorrect")
    if GROUPED_LABELS != ["NON_NBA", "NBA_ROLE_PLAYER", "STARTER_PLUS"]:
        raise AssertionError("Unexpected grouped label order")

    print("BENCH/ROTATION grouping smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
