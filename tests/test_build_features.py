"""Smoke tests for feature engineering builders."""

from __future__ import annotations

import numpy as np

from src.features.build_features import (
    build_context_features,
    build_ncaa_features,
    build_physical_features,
    combine_all_features,
    demo_feature_input,
)


def main() -> None:
    raw = demo_feature_input()
    ncaa = build_ncaa_features(raw)
    physical = build_physical_features(raw)
    context = build_context_features(raw)
    combined = combine_all_features(ncaa, physical, context)

    required_columns = {
        "player_id",
        "pts_per40",
        "pts_per40_z",
        "bmi",
        "bmi_z",
        "age_at_draft",
        "age_at_draft_z",
        "rec_rank",
        "rec_rank_z",
        "is_unranked",
        "is_unranked_z",
        "official_games",
        "official_games_z",
        "official_mpg",
        "official_mpg_z",
        "official_start_rate",
        "official_start_rate_z",
        "strength_of_schedule",
        "strength_of_schedule_z",
    }
    missing = required_columns.difference(combined.columns)
    if missing:
        raise AssertionError(f"Missing expected columns: {sorted(missing)}")

    numeric = combined.select_dtypes(include=[np.number])
    if numeric.isna().any().any():
        raise AssertionError("Feature matrix contains NaNs")
    if np.isinf(numeric.to_numpy()).any():
        raise AssertionError("Feature matrix contains infinite values")

    if combined.shape[0] != raw.shape[0]:
        raise AssertionError(f"Expected {raw.shape[0]} rows, got {combined.shape[0]}")

    print("Feature engineering smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
