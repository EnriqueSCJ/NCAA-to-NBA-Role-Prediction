"""Smoke tests for temporal backtesting helpers."""

from __future__ import annotations

import pandas as pd

from src.models.temporal_backtest import make_walk_forward_masks


def main() -> None:
    years = pd.Series([2014, 2015, 2016, 2017, 2017, 2018])
    train_mask, validation_mask, test_mask = make_walk_forward_masks(years, test_year=2017)

    if train_mask.tolist() != [True, True, False, False, False, False]:
        raise AssertionError("Unexpected walk-forward train mask")
    if validation_mask.tolist() != [False, False, True, False, False, False]:
        raise AssertionError("Unexpected walk-forward validation mask")
    if test_mask.tolist() != [False, False, False, True, True, False]:
        raise AssertionError("Unexpected walk-forward test mask")

    print("Temporal backtest helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
