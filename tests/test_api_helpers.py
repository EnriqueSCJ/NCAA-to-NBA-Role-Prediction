"""Smoke tests for API feature preparation and response formatting."""

from __future__ import annotations

import numpy as np

import src.config as config
from src.api.main import build_comparable_query_vector, build_prediction_features, player_stats_to_row, predict_from_features
from src.api.schemas import PlayerStats
from src.models.grouped_model import GROUPED_LABELS
from src.models.train import FEATURE_COLUMNS


class DummyRoleModel:
    """Small deterministic model used to test API formatting."""

    classes_ = np.array(GROUPED_LABELS)

    def predict_proba(self, features):
        return np.array([[0.05, 0.25, 0.70]])


def main() -> None:
    stats = PlayerStats(
        player_name="API Test Player",
        position="G",
        college="Test University",
        height_in=76.0,
        weight_lb=198.0,
        pts_per40=21.5,
        ast_per40=5.0,
        reb_per40=5.8,
        ts_pct=59.0,
        usg_pct=24.5,
        team_adj_o=116.0,
        rec_rank=12.0,
        is_unranked=0.0,
        official_games=31.0,
        official_mpg=30.5,
        official_start_rate=0.9,
    )

    raw_row = player_stats_to_row(stats)
    if raw_row["rec_rank"] != 12.0 or raw_row["is_unranked"] != 0.0:
        raise AssertionError("API row did not preserve recruiting rank fields")
    if raw_row["official_games"] != 31.0 or raw_row["official_mpg"] != 30.5:
        raise AssertionError("API row did not preserve official playing-time fields")

    features, source = build_prediction_features(stats)
    if list(features.columns) != FEATURE_COLUMNS:
        raise AssertionError("API feature columns do not match model feature columns")
    if features.shape != (1, len(FEATURE_COLUMNS)):
        raise AssertionError(f"Unexpected API feature shape: {features.shape}")
    if np.isnan(features.to_numpy(dtype=float)).any():
        raise AssertionError("API features contain NaNs")
    if source not in {"demo_reference_table", "processed_master_table"}:
        raise AssertionError(f"Unexpected feature source: {source}")

    query_vector, vector_source = build_comparable_query_vector(stats)
    if len(query_vector) != config.NUMERIC_VECTOR_DIM:
        raise AssertionError("Comparable query vector has unexpected dimension")
    if vector_source not in {"demo_reference_table", "processed_master_table"}:
        raise AssertionError(f"Unexpected vector source: {vector_source}")

    response = predict_from_features(DummyRoleModel(), features, stats.player_name, source)
    if response.predicted_role != "STARTER_PLUS":
        raise AssertionError(f"Unexpected predicted role: {response.predicted_role}")
    if response.probabilities["STARTER_PLUS"] != 0.70:
        raise AssertionError("Probability map is not aligned with labels")

    print("API helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
