"""Smoke tests for graph embedding helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.graph_embedding_experiment import build_graph_embeddings


def main() -> None:
    master = pd.DataFrame(
        {
            "player_name": ["A", "B", "C", "D"],
            "college": ["Team 1", "Team 1", "Team 2", "Team 3"],
            "conf": ["X", "X", "Y", "Z"],
            "season": [2020, 2020, 2021, 2021],
            "draft_year": [2020, 2020, 2021, 2021],
            "position": ["G", "G", "F", "C"],
            "yr": ["Fr", "So", "Fr", "Jr"],
            "rec_rank": [100.0, 20.0, 0.0, 80.0],
            "is_unranked": [0.0, 0.0, 1.0, 0.0],
            "pts_per40": [25.0, 12.0, 18.0, 10.0],
            "usg_pct": [30.0, 15.0, 24.0, 18.0],
            "ts_pct": [60.0, 52.0, 58.0, 55.0],
            "stocks_per40": [2.0, 1.0, 3.0, 2.5],
            "height_in": [78.0, 74.0, 80.0, 84.0],
            "age_at_draft": [19.0, 20.0, 19.5, 21.0],
            "team_power_rating": [0.9, 0.9, 0.7, 0.5],
            "strength_of_schedule": [10.0, 10.0, 2.0, -4.0],
            "official_mpg": [32.0, 20.0, 28.0, 18.0],
        }
    )
    embeddings = build_graph_embeddings(master, n_components=3)
    if embeddings.shape != (4, 3):
        raise AssertionError(f"Unexpected graph embedding shape: {embeddings.shape}")
    if np.isnan(embeddings.to_numpy(dtype=float)).any():
        raise AssertionError("Graph embeddings contain NaNs")

    print("Graph embedding helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
