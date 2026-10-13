"""Smoke tests for NBA role target rules."""

from __future__ import annotations

import pandas as pd

from src.features.target import define_nba_role


def main() -> None:
    examples = pd.DataFrame(
        {
            "nba_minutes_y1_y3": [499, 600, 1800, 3600, 5200],
            "nba_bpm_avg": [10.0, -3.0, -1.0, 1.5, 4.5],
            "nba_mpg_avg": [35.0, 12.0, 18.0, 26.0, 31.0],
        }
    )
    expected = ["NON_NBA", "BENCH", "ROTATION", "STARTER_PLUS", "STARTER_PLUS"]
    actual = define_nba_role(examples).tolist()

    if actual != expected:
        raise AssertionError(f"Expected {expected}, got {actual}")

    print("Target rule smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
