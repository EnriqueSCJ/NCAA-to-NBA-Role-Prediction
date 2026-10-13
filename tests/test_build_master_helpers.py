"""Smoke tests for real master-table helper functions."""

from __future__ import annotations

import pandas as pd

import src.config as config
from src.data.build_master import (
    _read_basketball_reference_drafts,
    _select_target_active_seasons,
    canonical_player_name,
    expanded_school_key,
    normalize_name,
    parse_height_to_inches,
    schools_compatible,
    simplify_position,
)
from src.data.build_master import _outcome_from_windows
from src.features.target import define_nba_role


def main() -> None:
    if canonical_player_name("Curry, Stephen") != "Stephen Curry":
        raise AssertionError("Comma-form player name was not normalized")

    if normalize_name("Xavier Tillman Sr.") != "xavier tillman":
        raise AssertionError("Name suffix normalization failed")

    if parse_height_to_inches("4-Jun") != 76.0:
        raise AssertionError("Excel-damaged height 4-Jun should parse as 6-4")

    if parse_height_to_inches("Jun-00") != 72.0:
        raise AssertionError("Excel-damaged height Jun-00 should parse as 6-0")

    if simplify_position("Scoring PG") != "G":
        raise AssertionError("Position simplification failed")

    if expanded_school_key("Michigan St.") != "michigan state":
        raise AssertionError("School-name expansion should normalize St. suffixes")

    if not schools_compatible("Saint Mary's", "St. Mary's College"):
        raise AssertionError("Conservative school matching should handle Saint/St aliases")

    previous_horizon = config.NBA_TARGET_ACTIVE_SEASONS
    previous_offset = config.NBA_TARGET_ACTIVE_SEASON_OFFSET
    try:
        config.NBA_TARGET_ACTIVE_SEASONS = 1
        config.NBA_TARGET_ACTIVE_SEASON_OFFSET = 2
        selected = _select_target_active_seasons(pd.Series([2017, 2018, 2019, 2020]))
    finally:
        config.NBA_TARGET_ACTIVE_SEASONS = previous_horizon
        config.NBA_TARGET_ACTIVE_SEASON_OFFSET = previous_offset
    if selected != [2019]:
        raise AssertionError(f"Expected only the third active season, got {selected}")

    basketball_reference_drafts = _read_basketball_reference_drafts()
    recent_drafts = basketball_reference_drafts[
        basketball_reference_drafts["draft_year"].isin([2022, 2023])
    ]
    if recent_drafts.empty:
        raise AssertionError("Basketball Reference draft CSVs were not parsed")

    if "paolo banchero" not in set(recent_drafts["player_key"]):
        raise AssertionError("Expected 2022 Basketball Reference draft player was not normalized")

    embiid_like_raptor = pd.DataFrame(
        {
            "season": [2017, 2018, 2019],
            "mp": [1786.0, 1912.0, 2154.0],
            "raptor_total": [4.7, 5.2, 4.9],
            "war_total": [6.5, 7.1, 7.4],
        }
    )
    embiid_like_box = pd.DataFrame(
        {
            "season": [2017, 2018, 2019],
            "box_games": [31, 63, 64],
            "box_minutes": [786.0, 1912.0, 2154.0],
            "box_impact_proxy": [4.8, 5.3, 5.0],
            "box_pie_avg": [0.19, 0.20, 0.20],
            "box_net_rating_avg": [5.0, 6.0, 6.5],
        }
    )
    embiid_outcome = _outcome_from_windows(
        selected_seasons=[2017, 2018, 2019],
        raptor_window=embiid_like_raptor,
        box_window=embiid_like_box,
        target_source="test",
    )
    embiid_role = define_nba_role(pd.DataFrame([embiid_outcome])).iloc[0]
    if embiid_role != "STARTER_PLUS":
        raise AssertionError(f"Expected active-season elite profile to be STARTER_PLUS, got {embiid_role}")

    tiny_sample = pd.DataFrame(
        {
            "nba_minutes_y1_y3": [10.0],
            "nba_bpm_avg": [18.0],
            "nba_mpg_avg": [3.3],
        }
    )
    if define_nba_role(tiny_sample).iloc[0] != "NON_NBA":
        raise AssertionError("Tiny high-impact samples should remain NON_NBA")

    print("Build-master helper smoke test passed.")


def test_smoke() -> None:
    main()


if __name__ == "__main__":
    main()
