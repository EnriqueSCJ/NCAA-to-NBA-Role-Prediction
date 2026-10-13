"""Build real processed modeling tables from the raw Datasets directory."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.config as config
from src.data.load import BART_TORVIK_PLAYER_COLUMNS
from src.features.build_features import (
    ALL_BASE_FEATURES,
    apply_z_score_stats,
    build_context_features,
    build_ncaa_features,
    build_physical_features,
    combine_all_features,
    fit_z_score_stats,
)
from src.features.target import define_nba_role


MIN_MODELING_DRAFT_YEAR = 2009
MAX_MODELING_DRAFT_YEAR = 2023
MAX_NON_DRAFTED_MODELING_DRAFT_YEAR = 2022
NBA_GAMES_PER_SEASON_ASSUMPTION = 72

MONTH_TO_FEET = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

YEAR_TO_AGE = {
    "fr": 19.0,
    "so": 20.0,
    "jr": 21.0,
    "sr": 22.0,
}


def normalize_name(value: Any) -> str:
    """Normalize person or team names for pragmatic joins."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_player_name(value: Any) -> str:
    """Convert comma-form player names to normal display order."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        if first and last:
            text = f"{first} {last}"

    return re.sub(r"\s+", " ", text).strip()


def parse_height_to_inches(value: Any) -> float:
    """Parse height values including Excel-damaged strings like `4-Jun`."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)) and np.isfinite(value):
        number = float(value)
        if 50 <= number <= 95:
            return number

    text = str(value).strip().lower()
    if not text:
        return np.nan

    text = text.replace("'", "-").replace('"', "")
    match = re.match(r"^(\d+)\s*[-/]\s*(\d+)$", text)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2))
        if 4 <= feet <= 8 and 0 <= inches <= 11:
            return float(feet * 12 + inches)

    match = re.match(r"^(\d+)\s*[-/]\s*([a-z]{3})$", text)
    if match:
        inches = int(match.group(1))
        month = match.group(2)
        feet = MONTH_TO_FEET.get(month)
        if feet is not None:
            return float(feet * 12 + inches)

    match = re.match(r"^([a-z]{3})\s*[-/]\s*(\d+)$", text)
    if match:
        month = match.group(1)
        inches = int(match.group(2))
        feet = MONTH_TO_FEET.get(month)
        if feet is not None and 0 <= inches <= 11:
            return float(feet * 12 + inches)

    number_match = re.search(r"\d+(?:\.\d+)?", text)
    if number_match:
        number = float(number_match.group(0))
        if 50 <= number <= 95:
            return number

    return np.nan


def simplify_position(value: Any) -> str:
    """Map dataset position archetypes to compact basketball positions."""
    text = "" if pd.isna(value) else str(value).strip()
    lowered = text.lower()
    if "pg" in lowered or "combo" in lowered or lowered == "g":
        return "G"
    if "wing g" in lowered:
        return "W"
    if "wing" in lowered:
        return "W"
    if "stretch" in lowered or "pf" in lowered:
        return "F"
    if "c" == lowered or "c" in lowered:
        return "C"
    return text[:10] if text else "NA"


def expanded_school_key(value: Any) -> str:
    """Normalize school names for conservative cross-source identity checks."""
    key = normalize_name(value)
    key = key.replace(" saint ", " st ")
    if key.startswith("saint "):
        key = "st " + key[len("saint ") :]
    tokens = key.split()
    if tokens and tokens[-1] == "st":
        tokens[-1] = "state"
    return " ".join(tokens)


def schools_compatible(left: Any, right: Any) -> bool:
    """Return whether two school names are compatible enough for relabeling."""
    left_key = expanded_school_key(left)
    right_key = expanded_school_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if len(left_key) >= 6 and left_key in right_key:
        return True
    if len(right_key) >= 6 and right_key in left_key:
        return True
    return False


def _read_ncaa_players() -> pd.DataFrame:
    """Read and lightly normalize the NCAA player-season table."""
    path = config.NCAA_NBA_DATASET_DIR / "CollegeBasketballPlayers2009-2021.csv"
    base = pd.read_csv(path, low_memory=False)
    base["ncaa_source"] = "college_basketball_players_2009_2021"
    frames = [base]

    for year in range(2022, MAX_MODELING_DRAFT_YEAR + 1):
        bart_path = config.BART_TORVIK_DATASET_DIR / f"BartTorvik{year}.csv"
        if not bart_path.exists():
            continue
        bart = pd.read_csv(
            bart_path,
            header=None,
            names=BART_TORVIK_PLAYER_COLUMNS,
            low_memory=False,
        )
        bart["Unnamed: 64"] = bart["position_label"]
        bart["ncaa_source"] = f"bart_torvik_{year}"
        frames.append(bart)

    ncaa = pd.concat(frames, ignore_index=True, sort=False)
    ncaa = ncaa.copy()
    ncaa["player_name"] = ncaa["player_name"].map(canonical_player_name)
    ncaa["player_key"] = ncaa["player_name"].map(normalize_name)
    ncaa["season"] = pd.to_numeric(ncaa["year"], errors="coerce").astype("Int64")
    ncaa["ncaa_height_in"] = ncaa["ht"].map(parse_height_to_inches)
    ncaa["position"] = ncaa["Unnamed: 64"].map(simplify_position)
    ncaa["age"] = ncaa["yr"].astype(str).str.lower().str[:2].map(YEAR_TO_AGE).fillna(21.0)
    ncaa["row_minutes"] = pd.to_numeric(ncaa["mp"], errors="coerce").fillna(0.0)
    ncaa["row_games"] = pd.to_numeric(ncaa["GP"], errors="coerce").fillna(0.0)
    return ncaa[ncaa["season"].notna()].copy()


def _read_drafted_players_legacy() -> pd.DataFrame:
    """Read and clean the 2009-2021 drafted players spreadsheet."""
    path = config.NCAA_NBA_DATASET_DIR / "DraftedPlayers2009-2021.xlsx"
    drafted = pd.read_excel(path)
    drafted = drafted[pd.to_numeric(drafted["YEAR"], errors="coerce").notna()].copy()
    drafted["player_name"] = drafted["PLAYER"].map(canonical_player_name)
    drafted["player_key"] = drafted["player_name"].map(normalize_name)
    drafted["draft_year"] = pd.to_numeric(drafted["YEAR"], errors="coerce").astype(int)
    drafted["draft_round"] = pd.to_numeric(drafted["ROUND"], errors="coerce").fillna(0).astype(int)
    drafted["draft_round_pick"] = pd.to_numeric(drafted["ROUND.1"], errors="coerce").fillna(0).astype(int)
    drafted["draft_pick"] = pd.to_numeric(drafted["OVERALL"], errors="coerce").fillna(0).astype(int)
    drafted["draft_team"] = drafted["TEAM"].astype(str).str.strip()
    drafted["affiliation"] = drafted["AFFILIATION"].astype(str).str.strip()
    drafted["draft_source"] = "ncaa_nba_excel_2009_2021"
    return drafted


def _read_basketball_reference_drafts() -> pd.DataFrame:
    """Read Basketball Reference draft CSV exports and normalize their schema."""
    draft_dir = config.BASKETBALL_REFERENCE_DRAFT_DIR
    if not draft_dir.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in sorted(draft_dir.glob("Draft*.csv")):
        match = re.search(r"Draft(\d{4})", path.stem)
        if not match:
            continue

        draft_year = int(match.group(1))
        raw = pd.read_csv(path, header=1, low_memory=False)
        picks = pd.to_numeric(raw.get("Pk"), errors="coerce")
        raw = raw[picks.notna()].copy()
        if raw.empty:
            continue

        raw["player_name"] = raw["Player"].map(canonical_player_name)
        raw["player_key"] = raw["player_name"].map(normalize_name)
        raw["draft_year"] = draft_year
        raw["draft_pick"] = picks[picks.notna()].astype(int).to_numpy()
        raw["draft_round"] = np.where(raw["draft_pick"] <= 30, 1, 2)
        raw["draft_round_pick"] = np.where(
            raw["draft_pick"] <= 30,
            raw["draft_pick"],
            raw["draft_pick"] - 30,
        )
        raw["draft_team"] = raw["Tm"].fillna("").astype(str).str.strip()
        raw["affiliation"] = raw["College"].fillna("").astype(str).str.strip()
        raw["draft_source"] = "basketball_reference_draft"
        frames.append(
            raw[
                [
                    "player_name",
                    "player_key",
                    "draft_year",
                    "draft_round",
                    "draft_round_pick",
                    "draft_pick",
                    "draft_team",
                    "affiliation",
                    "draft_source",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def _read_twenty_years_draft_data() -> pd.DataFrame:
    """Read the 20 Years of NBA Draft Data CSV as a supplemental draft source."""
    path = config.TWENTY_YEARS_DRAFT_FILE
    if not path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(path, low_memory=False)
    picks = pd.to_numeric(raw.get("Pk"), errors="coerce")
    years = pd.to_numeric(raw.get("DraftYear"), errors="coerce").fillna(
        pd.to_numeric(raw.get("DraftYr"), errors="coerce")
    )
    raw = raw[picks.notna() & years.notna()].copy()
    if raw.empty:
        return pd.DataFrame()

    raw["player_name"] = raw["Player"].map(canonical_player_name)
    raw["player_key"] = raw["player_name"].map(normalize_name)
    raw["draft_year"] = years.loc[raw.index].astype(int)
    raw["draft_pick"] = picks.loc[raw.index].astype(int)
    raw["draft_round"] = np.where(raw["draft_pick"] <= 30, 1, 2)
    raw["draft_round_pick"] = np.where(
        raw["draft_pick"] <= 30,
        raw["draft_pick"],
        raw["draft_pick"] - 30,
    )
    raw["draft_team"] = raw["Tm"].fillna("").astype(str).str.strip()
    raw["affiliation"] = raw["College"].fillna("").astype(str).str.strip()
    raw["draft_source"] = "twenty_years_nba_draft_data"

    return raw[
        [
            "player_name",
            "player_key",
            "draft_year",
            "draft_round",
            "draft_round_pick",
            "draft_pick",
            "draft_team",
            "affiliation",
            "draft_source",
        ]
    ]


def _read_drafted_players() -> pd.DataFrame:
    """Read drafted players from the original spreadsheet plus newer CSV exports."""
    frames = [_read_drafted_players_legacy()]
    twenty_years_draft = _read_twenty_years_draft_data()
    if not twenty_years_draft.empty:
        frames.append(twenty_years_draft)
    basketball_reference = _read_basketball_reference_drafts()
    if not basketball_reference.empty:
        frames.append(basketball_reference)

    drafted = pd.concat(frames, ignore_index=True, sort=False)
    drafted = drafted[drafted["player_key"].ne("")].copy()
    drafted = drafted.sort_values(
        ["draft_year", "draft_pick", "draft_source"],
        ascending=[True, True, True],
    )
    return drafted.drop_duplicates(["player_key", "draft_year"], keep="last").reset_index(drop=True)


def _read_raptor() -> pd.DataFrame:
    """Read RAPTOR seasons available in the source bundle."""
    paths = [
        config.NCAA_NBA_DATASET_DIR / "modern_RAPTOR_by_player.csv",
        config.NCAA_NBA_DATASET_DIR / "latest_RAPTOR_by_player.csv",
    ]
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.exists()]
    raptor = pd.concat(frames, ignore_index=True)
    raptor["player_name"] = raptor["player_name"].map(canonical_player_name)
    raptor["player_key"] = raptor["player_name"].map(normalize_name)
    raptor["season"] = pd.to_numeric(raptor["season"], errors="coerce").astype("Int64")
    raptor["mp"] = pd.to_numeric(raptor["mp"], errors="coerce").fillna(0.0)
    raptor["raptor_total"] = pd.to_numeric(raptor["raptor_total"], errors="coerce").fillna(0.0)
    raptor["war_total"] = pd.to_numeric(raptor["war_total"], errors="coerce").fillna(0.0)
    return raptor.drop_duplicates(["player_key", "season"]).copy()


def _weighted_average(values: pd.Series, weights: pd.Series, default: float = 0.0) -> float:
    """Return a finite weighted average."""
    clean_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    clean_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    total_weight = float(clean_weights.sum())
    if total_weight <= 0:
        return default
    return float((clean_values * clean_weights).sum() / total_weight)


def _read_box_score_seasons() -> pd.DataFrame:
    """Read NBA box-score game logs and aggregate them by player-season."""
    path = config.NBA_BOX_SCORE_PLAYER_STATS_EXTENDED_FILE
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "player_key",
                "season",
                "box_games",
                "box_minutes",
                "box_points_per36",
                "box_rebounds_per36",
                "box_assists_per36",
                "box_stocks_per36",
                "box_turnovers_per36",
                "box_pie_avg",
                "box_net_rating_avg",
                "box_impact_proxy",
            ]
        )

    usecols = [
        "firstName",
        "lastName",
        "gameId",
        "gameDateTimeEst",
        "gameType",
        "numMinutes",
        "points",
        "assists",
        "reboundsTotal",
        "steals",
        "blocks",
        "turnovers",
        "playerImpactEstimate",
        "netRating",
    ]
    stats = pd.read_csv(path, usecols=usecols, low_memory=False)
    stats["player_name"] = (
        stats["firstName"].fillna("").astype(str).str.strip()
        + " "
        + stats["lastName"].fillna("").astype(str).str.strip()
    ).str.strip()
    stats["player_key"] = stats["player_name"].map(normalize_name)
    stats["game_date"] = pd.to_datetime(stats["gameDateTimeEst"], errors="coerce")
    stats = stats[stats["game_date"].notna()].copy()
    stats["season"] = stats["game_date"].dt.year + (stats["game_date"].dt.month >= 10).astype(int)
    stats["numMinutes"] = pd.to_numeric(stats["numMinutes"], errors="coerce").fillna(0.0)
    stats = stats[
        stats["gameType"].astype(str).str.lower().eq("regular season")
        & (stats["numMinutes"] > 0)
        & stats["player_key"].ne("")
    ].copy()

    numeric_columns = [
        "points",
        "assists",
        "reboundsTotal",
        "steals",
        "blocks",
        "turnovers",
        "playerImpactEstimate",
        "netRating",
    ]
    for column in numeric_columns:
        stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0.0)

    stats["stocks"] = stats["steals"] + stats["blocks"]
    stats["pie_weighted"] = stats["playerImpactEstimate"] * stats["numMinutes"]
    stats["net_rating_weighted"] = stats["netRating"] * stats["numMinutes"]

    grouped = (
        stats.groupby(["player_key", "season"], as_index=False)
        .agg(
            box_games=("gameId", "nunique"),
            box_minutes=("numMinutes", "sum"),
            box_points=("points", "sum"),
            box_rebounds=("reboundsTotal", "sum"),
            box_assists=("assists", "sum"),
            box_stocks=("stocks", "sum"),
            box_turnovers=("turnovers", "sum"),
            box_pie_weighted=("pie_weighted", "sum"),
            box_net_rating_weighted=("net_rating_weighted", "sum"),
        )
    )

    minutes = grouped["box_minutes"].replace(0, np.nan)
    grouped["box_points_per36"] = grouped["box_points"] * 36.0 / minutes
    grouped["box_rebounds_per36"] = grouped["box_rebounds"] * 36.0 / minutes
    grouped["box_assists_per36"] = grouped["box_assists"] * 36.0 / minutes
    grouped["box_stocks_per36"] = grouped["box_stocks"] * 36.0 / minutes
    grouped["box_turnovers_per36"] = grouped["box_turnovers"] * 36.0 / minutes
    grouped["box_pie_avg"] = grouped["box_pie_weighted"] / minutes
    grouped["box_net_rating_avg"] = grouped["box_net_rating_weighted"] / minutes

    production_score = (
        grouped["box_points_per36"].fillna(0.0)
        + 0.7 * grouped["box_rebounds_per36"].fillna(0.0)
        + 0.7 * grouped["box_assists_per36"].fillna(0.0)
        + 1.3 * grouped["box_stocks_per36"].fillna(0.0)
        - 0.7 * grouped["box_turnovers_per36"].fillna(0.0)
        - 18.0
    ) / 6.0
    pie_score = (grouped["box_pie_avg"].fillna(0.0) - 0.10) * 60.0
    net_score = grouped["box_net_rating_avg"].fillna(0.0) / 12.0
    grouped["box_impact_proxy"] = 0.45 * pie_score + 0.30 * production_score + 0.25 * net_score

    return grouped.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_bigquery_player_game_features() -> pd.DataFrame:
    """Aggregate optional NCAA BigQuery player-game exports by player-season."""
    path = config.NCAA_BIGQUERY_EXPORT_DIR / "bigquery-public-data__ncaa_basketball__mbb_players_games_sr.parquet"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "player_key",
                "season",
                "official_games",
                "official_minutes",
                "official_mpg",
                "official_starts",
                "official_start_rate",
            ]
        )

    usecols = [
        "season",
        "full_name",
        "played",
        "starter",
        "minutes_int64",
        "points",
    ]
    games = pd.read_parquet(path, columns=usecols)
    games["player_key"] = games["full_name"].map(normalize_name)
    games["season"] = pd.to_numeric(games["season"], errors="coerce") + 1
    games["played"] = games["played"].fillna(False).astype(bool)
    games["starter"] = games["starter"].fillna(False).astype(bool)
    games["minutes_int64"] = pd.to_numeric(games["minutes_int64"], errors="coerce").fillna(0.0)
    games["points"] = pd.to_numeric(games["points"], errors="coerce").fillna(0.0)

    active_rows = games[
        games["player_key"].ne("")
        & games["season"].notna()
        & (
            games["played"]
            | games["starter"]
            | (games["minutes_int64"] > 0)
            | (games["points"] > 0)
        )
    ].copy()
    if active_rows.empty:
        return pd.DataFrame()

    grouped = (
        active_rows.groupby(["player_key", "season"], as_index=False)
        .agg(
            official_games=("full_name", "size"),
            official_minutes=("minutes_int64", "sum"),
            official_starts=("starter", "sum"),
        )
    )
    grouped["season"] = grouped["season"].astype(int)
    grouped["official_mpg"] = grouped["official_minutes"] / grouped["official_games"].replace(0, np.nan)
    grouped["official_start_rate"] = grouped["official_starts"] / grouped["official_games"].replace(0, np.nan)
    return grouped.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_context() -> pd.DataFrame:
    """Read team context tables and normalize yearly schema differences."""
    frames: list[pd.DataFrame] = []
    for path in sorted(config.COLLEGE_CONTEXT_DATASET_DIR.glob("cbb*.csv")):
        df = pd.read_csv(path, low_memory=False)
        if "YEAR" not in df.columns:
            match = re.search(r"cbb(\d{2})$", path.stem)
            if match:
                year_suffix = int(match.group(1))
                df["YEAR"] = 2000 + year_suffix
        frames.append(df)

    context = pd.concat(frames, ignore_index=True, sort=False)
    context["team_key"] = context["TEAM"].map(normalize_name)
    context["season"] = pd.to_numeric(context["YEAR"], errors="coerce").astype("Int64")
    keep = [
        "team_key",
        "season",
        "ADJOE",
        "ADJDE",
        "ADJ_T",
        "BARTHAG",
        "WAB",
        "POSTSEASON",
        "SEED",
    ]
    for column in keep:
        if column not in context.columns:
            context[column] = np.nan

    context = context[keep].dropna(subset=["team_key", "season"])
    return context.drop_duplicates(["team_key", "season"], keep="first")


def _read_combine() -> pd.DataFrame:
    """Read NBA Draft Combine measurements."""
    combine = pd.read_csv(config.DRAFT_COMBINE_FILE, low_memory=False)
    combine["player_name"] = combine["PLAYER"].map(canonical_player_name)
    combine["player_key"] = combine["player_name"].map(normalize_name)
    combine["draft_year"] = pd.to_numeric(combine["YEAR"], errors="coerce").astype("Int64")
    return combine


def _read_nba_players() -> pd.DataFrame:
    """Read NBA player database for fallback measurements."""
    players = pd.read_csv(config.NBA_PLAYERS_FILE, low_memory=False)
    players["player_name"] = players["Name"].map(canonical_player_name)
    players["player_key"] = players["player_name"].map(normalize_name)
    players["nba_height_in"] = pd.to_numeric(players["Height"], errors="coerce")
    players["nba_weight_lb"] = pd.to_numeric(players["Weight"], errors="coerce")
    return players[["player_key", "nba_height_in", "nba_weight_lb", "Birthday"]].drop_duplicates("player_key")


def _read_nba_box_players_identity() -> pd.DataFrame:
    """Read NBA box-score player metadata for identity checks only."""
    path = config.NBA_BOX_SCORE_PLAYERS_FILE
    if not path.exists():
        return pd.DataFrame(columns=["player_key", "nba_school_key"])

    players = pd.read_csv(path, low_memory=False)
    if {"firstName", "lastName"}.issubset(players.columns):
        players["player_name"] = (
            players["firstName"].fillna("").astype(str).str.strip()
            + " "
            + players["lastName"].fillna("").astype(str).str.strip()
        ).str.strip()
    elif "name" in players.columns:
        players["player_name"] = players["name"].map(canonical_player_name)
    elif "playerName" in players.columns:
        players["player_name"] = players["playerName"].map(canonical_player_name)
    else:
        players["player_name"] = ""

    players["player_key"] = players["player_name"].map(normalize_name)
    if "school" in players.columns:
        players["nba_school_key"] = players["school"].map(expanded_school_key)
    else:
        players["nba_school_key"] = ""

    identity = players[players["player_key"].ne("") & players["nba_school_key"].ne("")].copy()
    return identity[["player_key", "nba_school_key"]].drop_duplicates().reset_index(drop=True)


def _select_best_ncaa_row(ncaa_rows: pd.DataFrame, draft_year: int) -> pd.Series | None:
    """Select the latest NCAA row before a player's draft year."""
    candidates = ncaa_rows[
        (ncaa_rows["season"] <= draft_year)
        & (ncaa_rows["season"] >= draft_year - 5)
    ]
    if candidates.empty:
        return None

    return candidates.sort_values(
        ["season", "row_games", "row_minutes"],
        ascending=[False, False, False],
    ).iloc[0]


def _empty_outcome(target_source: str) -> dict[str, float | int | str]:
    """Return a complete zero-valued NBA outcome row."""
    return {
        "nba_bpm_y1": 0.0,
        "nba_bpm_y2": 0.0,
        "nba_bpm_y3": 0.0,
        "nba_raptor_avg": 0.0,
        "nba_box_impact_avg": 0.0,
        "nba_pie_avg": 0.0,
        "nba_net_rating_avg": 0.0,
        "nba_games_y1_y3": 0,
        "nba_minutes_y1_y3": 0.0,
        "nba_bpm_avg": 0.0,
        "nba_mpg_avg": 0.0,
        "nba_war_y1_y3": 0.0,
        "raptor_seasons_observed": 0,
        "nba_active_seasons_observed": 0,
        "nba_active_seasons": "",
        "target_source": target_source,
    }


def _impact_by_active_season(
    selected_seasons: list[int],
    raptor_window: pd.DataFrame,
    box_window: pd.DataFrame,
) -> dict[int, float]:
    """Return one impact value for each active-season slot."""
    by_season: dict[int, float] = {}
    for season in selected_seasons[:3]:
        raptor_row = raptor_window[raptor_window["season"] == season]
        box_row = box_window[box_window["season"] == season]
        if not raptor_row.empty:
            by_season[season] = float(raptor_row["raptor_total"].iloc[0])
        elif not box_row.empty:
            by_season[season] = float(box_row["box_impact_proxy"].iloc[0])
        else:
            by_season[season] = 0.0
    return by_season


def _select_target_active_seasons(seasons: pd.Series) -> list[int]:
    """Select the configured active-season slice for target construction."""
    unique_seasons = seasons.astype(int).drop_duplicates()
    start = max(0, int(config.NBA_TARGET_ACTIVE_SEASON_OFFSET))
    horizon = max(1, int(config.NBA_TARGET_ACTIVE_SEASONS))
    return unique_seasons.iloc[start : start + horizon].tolist()


def _outcome_from_windows(
    selected_seasons: list[int],
    raptor_window: pd.DataFrame,
    box_window: pd.DataFrame,
    target_source: str,
) -> dict[str, float | int | str]:
    """Aggregate NBA outcome metrics from selected active seasons."""
    if not selected_seasons:
        return _empty_outcome("no_active_nba_seasons")

    selected_season_set = set(selected_seasons)
    raptor_window = raptor_window[raptor_window["season"].isin(selected_season_set)].copy()
    box_window = box_window[box_window["season"].isin(selected_season_set)].copy()

    box_minutes = float(box_window["box_minutes"].sum()) if not box_window.empty else 0.0
    box_games = int(box_window["box_games"].sum()) if not box_window.empty else 0
    raptor_minutes = float(raptor_window["mp"].sum()) if not raptor_window.empty else 0.0

    total_minutes = box_minutes if box_minutes > 0 else raptor_minutes
    games = box_games if box_games > 0 else int(NBA_GAMES_PER_SEASON_ASSUMPTION * len(selected_seasons))
    mpg = total_minutes / games if games else 0.0

    box_impact = (
        _weighted_average(box_window["box_impact_proxy"], box_window["box_minutes"])
        if not box_window.empty
        else 0.0
    )
    pie_avg = (
        _weighted_average(box_window["box_pie_avg"], box_window["box_minutes"])
        if not box_window.empty
        else 0.0
    )
    net_rating_avg = (
        _weighted_average(box_window["box_net_rating_avg"], box_window["box_minutes"])
        if not box_window.empty
        else 0.0
    )

    raptor_avg = 0.0
    total_war = 0.0
    raptor_season_count = int(raptor_window["season"].nunique()) if not raptor_window.empty else 0
    if not raptor_window.empty and raptor_minutes > 0:
        raptor_avg = _weighted_average(raptor_window["raptor_total"], raptor_window["mp"])
        total_war = float(raptor_window["war_total"].fillna(0.0).sum())
        raptor_proxy = raptor_avg + min(total_war / 12.0, 3.0)
        if raptor_season_count >= 2 and raptor_minutes >= 1000:
            impact_proxy = raptor_proxy
            if not box_window.empty:
                target_source = "active_seasons_raptor_with_box_score"
            else:
                target_source = "active_seasons_raptor"
        elif not box_window.empty:
            impact_proxy = box_impact
            target_source = "active_seasons_box_score"
        else:
            impact_proxy = raptor_proxy
            target_source = "active_seasons_raptor"
    else:
        impact_proxy = box_impact

    impact_by_season = _impact_by_active_season(selected_seasons, raptor_window, box_window)
    slot_values = [impact_by_season.get(season, 0.0) for season in selected_seasons[:3]]
    while len(slot_values) < 3:
        slot_values.append(0.0)

    return {
        "nba_bpm_y1": float(slot_values[0]),
        "nba_bpm_y2": float(slot_values[1]),
        "nba_bpm_y3": float(slot_values[2]),
        "nba_raptor_avg": float(raptor_avg),
        "nba_box_impact_avg": float(box_impact),
        "nba_pie_avg": float(pie_avg),
        "nba_net_rating_avg": float(net_rating_avg),
        "nba_games_y1_y3": games,
        "nba_minutes_y1_y3": float(total_minutes),
        "nba_bpm_avg": float(impact_proxy),
        "nba_mpg_avg": float(mpg),
        "nba_war_y1_y3": float(total_war),
        "raptor_seasons_observed": raptor_season_count,
        "nba_active_seasons_observed": len(selected_seasons),
        "nba_active_seasons": ",".join(str(season) for season in selected_seasons),
        "target_source": target_source,
    }


def _nba_outcome_for_player(
    raptor: pd.DataFrame,
    box_score_seasons: pd.DataFrame,
    player_key: str,
    draft_year: int,
) -> dict[str, float | int | str]:
    """Aggregate NBA impact over the first active post-draft seasons."""
    box_candidates = box_score_seasons[
        (box_score_seasons["player_key"] == player_key)
        & (box_score_seasons["season"] >= draft_year + 1)
        & (box_score_seasons["box_minutes"] > 0)
    ].sort_values("season")
    raptor_candidates = raptor[
        (raptor["player_key"] == player_key)
        & (raptor["season"] >= draft_year + 1)
        & (raptor["mp"] > 0)
    ].sort_values("season")

    if not box_candidates.empty:
        selected_seasons = _select_target_active_seasons(box_candidates["season"])
        return _outcome_from_windows(
            selected_seasons=selected_seasons,
            raptor_window=raptor_candidates,
            box_window=box_candidates,
            target_source="active_seasons_box_score",
        )

    if not raptor_candidates.empty:
        selected_seasons = _select_target_active_seasons(raptor_candidates["season"])
        return _outcome_from_windows(
            selected_seasons=selected_seasons,
            raptor_window=raptor_candidates,
            box_window=box_candidates,
            target_source="active_seasons_raptor",
        )

    return _empty_outcome("no_active_nba_seasons")


def _build_drafted_rows(
    ncaa: pd.DataFrame,
    drafted: pd.DataFrame,
    raptor: pd.DataFrame,
    box_score_seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Build one NCAA pre-draft row per drafted player when an NCAA row exists."""
    drafted = drafted[
        (drafted["draft_year"] >= MIN_MODELING_DRAFT_YEAR)
        & (drafted["draft_year"] <= MAX_MODELING_DRAFT_YEAR)
    ].copy()

    rows: list[dict[str, Any]] = []
    ncaa_by_key = {key: frame for key, frame in ncaa.groupby("player_key", sort=False)}

    for _, draft_row in drafted.iterrows():
        player_key = draft_row["player_key"]
        ncaa_rows = ncaa_by_key.get(player_key)
        if ncaa_rows is None:
            continue

        best_row = _select_best_ncaa_row(ncaa_rows, int(draft_row["draft_year"]))
        if best_row is None:
            continue

        row = best_row.to_dict()
        row.update(
            {
                "player_name": draft_row["player_name"],
                "player_key": player_key,
                "draft_year": int(draft_row["draft_year"]),
                "draft_pick": int(draft_row["draft_pick"]),
                "draft_round": int(draft_row["draft_round"]),
                "draft_round_pick": int(draft_row["draft_round_pick"]),
                "draft_team": draft_row["draft_team"],
                "draft_affiliation": draft_row["affiliation"],
                "draft_source": draft_row.get("draft_source", ""),
                "is_drafted": True,
            }
        )
        row.update(
            _nba_outcome_for_player(
                raptor=raptor,
                box_score_seasons=box_score_seasons,
                player_key=player_key,
                draft_year=int(draft_row["draft_year"]),
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _build_non_drafted_rows(ncaa: pd.DataFrame, drafted: pd.DataFrame) -> pd.DataFrame:
    """Build one final NCAA row per non-drafted player in the modeling years."""
    all_drafted_keys = set(drafted["player_key"].dropna())
    candidates = ncaa[
        (ncaa["season"] >= MIN_MODELING_DRAFT_YEAR)
        & (ncaa["season"] <= MAX_NON_DRAFTED_MODELING_DRAFT_YEAR)
        & (~ncaa["player_key"].isin(all_drafted_keys))
        & (pd.to_numeric(ncaa["pick"], errors="coerce").isna())
    ].copy()

    candidates["group_key"] = candidates["pid"].fillna(candidates["player_key"]).astype(str)
    final_rows = (
        candidates.sort_values(["season", "row_games", "row_minutes"], ascending=[False, False, False])
        .drop_duplicates("group_key", keep="first")
        .copy()
    )

    final_rows["draft_year"] = final_rows["season"].astype(int)
    final_rows["draft_pick"] = 0
    final_rows["draft_round"] = 0
    final_rows["draft_round_pick"] = 0
    final_rows["draft_team"] = ""
    final_rows["draft_affiliation"] = ""
    final_rows["draft_source"] = ""
    final_rows["is_drafted"] = False
    final_rows["nba_bpm_y1"] = 0.0
    final_rows["nba_bpm_y2"] = 0.0
    final_rows["nba_bpm_y3"] = 0.0
    final_rows["nba_raptor_avg"] = 0.0
    final_rows["nba_box_impact_avg"] = 0.0
    final_rows["nba_pie_avg"] = 0.0
    final_rows["nba_net_rating_avg"] = 0.0
    final_rows["nba_games_y1_y3"] = 0
    final_rows["nba_minutes_y1_y3"] = 0.0
    final_rows["nba_bpm_avg"] = 0.0
    final_rows["nba_mpg_avg"] = 0.0
    final_rows["nba_war_y1_y3"] = 0.0
    final_rows["raptor_seasons_observed"] = 0
    final_rows["nba_active_seasons_observed"] = 0
    final_rows["nba_active_seasons"] = ""
    final_rows["target_source"] = "not_drafted_in_source_window"
    return final_rows


def _school_lookup(nba_box_players_identity: pd.DataFrame) -> dict[str, set[str]]:
    """Build a player-key to school-key lookup from NBA metadata."""
    if nba_box_players_identity.empty:
        return {}
    return {
        player_key: set(group["nba_school_key"].dropna().astype(str))
        for player_key, group in nba_box_players_identity.groupby("player_key", sort=False)
    }


def _has_school_identity_match(row: pd.Series, school_lookup: dict[str, set[str]]) -> bool:
    """Check whether a non-drafted NCAA row matches an NBA player school."""
    school_keys = school_lookup.get(str(row.get("player_key", "")), set())
    if not school_keys:
        return False
    source_school = row.get("team", "")
    return any(schools_compatible(source_school, nba_school_key) for nba_school_key in school_keys)


def _relabel_undrafted_nba_outcomes(
    non_drafted_rows: pd.DataFrame,
    raptor: pd.DataFrame,
    box_score_seasons: pd.DataFrame,
    nba_box_players_identity: pd.DataFrame,
) -> pd.DataFrame:
    """Relabel undrafted players with verified post-NCAA NBA outcomes.

    Non-drafted does not mean non-NBA. This step uses NBA data only as target
    evidence and requires a conservative name + school match before replacing
    the default `NON_NBA` outcome.
    """
    relabeled = non_drafted_rows.copy()
    relabeled["undrafted_nba_identity_match"] = ""
    if relabeled.empty:
        return relabeled

    active_player_keys = set(box_score_seasons["player_key"].dropna()) | set(raptor["player_key"].dropna())
    school_lookup = _school_lookup(nba_box_players_identity)
    if not active_player_keys or not school_lookup:
        return relabeled

    relabel_count = 0
    for index, row in relabeled.iterrows():
        player_key = str(row.get("player_key", ""))
        if player_key not in active_player_keys:
            continue
        if not _has_school_identity_match(row, school_lookup):
            continue

        outcome = _nba_outcome_for_player(
            raptor=raptor,
            box_score_seasons=box_score_seasons,
            player_key=player_key,
            draft_year=int(row["draft_year"]),
        )
        if float(outcome.get("nba_minutes_y1_y3", 0.0)) <= 0.0:
            continue

        target_source = str(outcome.get("target_source", "active_seasons"))
        for column, value in outcome.items():
            relabeled.at[index, column] = value
        relabeled.at[index, "target_source"] = f"undrafted_{target_source}_school_match"
        relabeled.at[index, "undrafted_nba_identity_match"] = "school_match"
        relabel_count += 1

    relabeled.attrs["undrafted_nba_relabels"] = relabel_count
    return relabeled


def _merge_context(master: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Attach team context by team and NCAA season."""
    master = master.copy()
    master["team_key"] = master["team"].map(normalize_name)
    merged = master.merge(context, how="left", on=["team_key", "season"])
    merged = merged.rename(
        columns={
            "ADJOE": "team_adj_o",
            "ADJDE": "team_adj_d",
            "ADJ_T": "team_tempo",
            "BARTHAG": "team_power_rating",
            "WAB": "strength_of_schedule",
            "POSTSEASON": "team_postseason",
            "SEED": "team_seed",
        }
    )
    return merged


def _merge_measurements(master: pd.DataFrame, combine: pd.DataFrame, nba_players: pd.DataFrame) -> pd.DataFrame:
    """Attach physical measurements from combine and NBA player database."""
    master = master.copy()
    exact_combine = combine.sort_values("draft_year").drop_duplicates(["player_key", "draft_year"], keep="last")
    exact_combine = exact_combine.rename(
        columns={
            "HGT": "combine_height_in",
            "WGT": "combine_weight_lb",
            "WNGSPN": "combine_wingspan_in",
            "POS": "combine_position",
            "BMI": "combine_bmi",
        }
    )
    master = master.merge(
        exact_combine[
            [
                "player_key",
                "draft_year",
                "combine_height_in",
                "combine_weight_lb",
                "combine_wingspan_in",
                "combine_position",
                "combine_bmi",
            ]
        ],
        how="left",
        on=["player_key", "draft_year"],
    )

    fallback_combine = exact_combine.drop_duplicates("player_key", keep="last")
    fallback_combine = fallback_combine.rename(
        columns={
            "combine_height_in": "fallback_combine_height_in",
            "combine_weight_lb": "fallback_combine_weight_lb",
            "combine_wingspan_in": "fallback_combine_wingspan_in",
            "combine_position": "fallback_combine_position",
        }
    )
    master = master.merge(
        fallback_combine[
            [
                "player_key",
                "fallback_combine_height_in",
                "fallback_combine_weight_lb",
                "fallback_combine_wingspan_in",
                "fallback_combine_position",
            ]
        ],
        how="left",
        on="player_key",
    )
    master = master.merge(nba_players, how="left", on="player_key")

    master["height_in"] = (
        pd.to_numeric(master["combine_height_in"], errors="coerce")
        .fillna(pd.to_numeric(master["fallback_combine_height_in"], errors="coerce"))
        .fillna(pd.to_numeric(master["nba_height_in"], errors="coerce"))
        .fillna(pd.to_numeric(master["ncaa_height_in"], errors="coerce"))
    )
    master["weight_lb"] = (
        pd.to_numeric(master["combine_weight_lb"], errors="coerce")
        .fillna(pd.to_numeric(master["fallback_combine_weight_lb"], errors="coerce"))
        .fillna(pd.to_numeric(master["nba_weight_lb"], errors="coerce"))
    )
    master["wingspan_in"] = (
        pd.to_numeric(master["combine_wingspan_in"], errors="coerce")
        .fillna(pd.to_numeric(master["fallback_combine_wingspan_in"], errors="coerce"))
    )

    master["position"] = master["combine_position"].fillna(master["fallback_combine_position"]).fillna(master["position"])
    master["position"] = master["position"].map(simplify_position)

    return master


def _merge_bigquery_player_features(master: pd.DataFrame, bigquery_player_features: pd.DataFrame) -> pd.DataFrame:
    """Attach official NCAA BigQuery player-game aggregates when exported."""
    master = master.copy()
    if bigquery_player_features.empty:
        for column in ["official_games", "official_minutes", "official_mpg", "official_starts", "official_start_rate"]:
            master[column] = 0.0
        return master

    merged = master.merge(
        bigquery_player_features,
        how="left",
        on=["player_key", "season"],
    )
    for column in ["official_games", "official_minutes", "official_mpg", "official_starts", "official_start_rate"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def _impute_measurements(master: pd.DataFrame) -> pd.DataFrame:
    """Impute missing physical measurements conservatively."""
    master = master.copy()
    for column in ["height_in", "weight_lb"]:
        master[column] = pd.to_numeric(master[column], errors="coerce")
        position_medians = master.groupby("position")[column].transform("median")
        master[column] = master[column].fillna(position_medians).fillna(master[column].median())

    master["wingspan_in"] = pd.to_numeric(master["wingspan_in"], errors="coerce")
    diff = master["wingspan_in"] - master["height_in"]
    diff_median = diff.groupby(master["position"]).transform("median").fillna(diff.median()).fillna(3.0)
    master["wingspan_in"] = master["wingspan_in"].fillna(master["height_in"] + diff_median)
    master["bmi"] = (master["weight_lb"] / (master["height_in"] ** 2)) * 703.0
    master["wingspan_diff"] = master["wingspan_in"] - master["height_in"]
    return master


def _add_modeling_columns(master: pd.DataFrame) -> pd.DataFrame:
    """Create clean modeling columns from the source schema."""
    master = master.copy()
    base_ids = []
    for row_index, row in master.iterrows():
        kind = "drafted" if bool(row["is_drafted"]) else "ncaa"
        name_slug = normalize_name(row["player_name"]).replace(" ", "_")
        source_id = row.get("pid", row_index)
        try:
            source_id_text = str(int(float(source_id)))
        except (TypeError, ValueError):
            source_id_text = str(row_index)
        base_ids.append(f"{kind}_{name_slug}_{int(row['draft_year'])}_{source_id_text}"[:240])

    master["player_id"] = base_ids
    duplicate_number = master.groupby("player_id").cumcount()
    master["player_id"] = [
        player_id if duplicate == 0 else f"{player_id}_{duplicate}"
        for player_id, duplicate in zip(master["player_id"], duplicate_number)
    ]
    master["college"] = master["team"].astype(str).str.strip()
    master["season"] = master["season"].astype(int)
    master["draft_year"] = master["draft_year"].astype(int)
    master["draft_pick"] = pd.to_numeric(master["draft_pick"], errors="coerce").fillna(0).astype(int)

    birthdates = pd.to_datetime(master.get("birthdate"), errors="coerce")
    if "Birthday" in master.columns:
        birthdates = birthdates.fillna(pd.to_datetime(master["Birthday"], errors="coerce"))
    draft_dates = pd.to_datetime(master["draft_year"].astype(str) + "-06-25", errors="coerce")
    age_at_draft = ((draft_dates - birthdates).dt.days / 365.2425).where(lambda series: series.between(16.0, 30.0))
    master["age_at_draft"] = age_at_draft.fillna(pd.to_numeric(master["age"], errors="coerce")).fillna(21.0)

    rec_rank = pd.to_numeric(master.get("Rec Rank"), errors="coerce")
    master["is_unranked"] = rec_rank.isna().astype(int)
    master["rec_rank"] = rec_rank.fillna(0.0).clip(lower=0.0, upper=100.0)

    minutes = pd.to_numeric(master["mp"], errors="coerce").replace(0, np.nan)
    for source, target in {
        "pts": "pts_per40",
        "ast": "ast_per40",
        "treb": "reb_per40",
        "stl": "stl_per40",
        "blk": "blk_per40",
    }.items():
        master[target] = pd.to_numeric(master[source], errors="coerce") * 40.0 / minutes

    master["tov_per40"] = pd.to_numeric(master.get("TO_per"), errors="coerce").fillna(0.0)
    master["ts_pct"] = pd.to_numeric(master["TS_per"], errors="coerce")
    master["efg_pct"] = pd.to_numeric(master["eFG"], errors="coerce")
    master["usg_pct"] = pd.to_numeric(master["usg"], errors="coerce")
    master["three_par"] = (
        pd.to_numeric(master["TPA"], errors="coerce")
        / (
            pd.to_numeric(master["TPA"], errors="coerce")
            + pd.to_numeric(master["twoPA"], errors="coerce")
        ).replace(0, np.nan)
    )
    master["ast_tov_ratio"] = pd.to_numeric(master["ast/tov"], errors="coerce")
    master["stocks_per40"] = master["stl_per40"].fillna(0.0) + master["blk_per40"].fillna(0.0)
    master["ftr"] = pd.to_numeric(master["ftr"], errors="coerce")

    for column in ["team_adj_o", "team_adj_d", "team_tempo", "team_power_rating", "strength_of_schedule"]:
        master[column] = pd.to_numeric(master[column], errors="coerce").fillna(0.0)

    master["target_role"] = define_nba_role(master)
    master["is_training_eligible"] = True
    master["target_horizon_games_assumption"] = NBA_GAMES_PER_SEASON_ASSUMPTION
    master["target_active_seasons_horizon"] = config.NBA_TARGET_ACTIVE_SEASONS
    master["target_active_seasons_start_offset"] = config.NBA_TARGET_ACTIVE_SEASON_OFFSET
    return master


def build_master_table() -> pd.DataFrame:
    """Build the real modeling table from raw project datasets."""
    ncaa = _read_ncaa_players()
    drafted = _read_drafted_players()
    raptor = _read_raptor()
    box_score_seasons = _read_box_score_seasons()
    bigquery_player_features = _read_bigquery_player_game_features()
    context = _read_context()
    combine = _read_combine()
    nba_players = _read_nba_players()
    nba_box_players_identity = _read_nba_box_players_identity()

    drafted_rows = _build_drafted_rows(ncaa, drafted, raptor, box_score_seasons)
    non_drafted_rows = _build_non_drafted_rows(ncaa, drafted)
    non_drafted_rows = _relabel_undrafted_nba_outcomes(
        non_drafted_rows=non_drafted_rows,
        raptor=raptor,
        box_score_seasons=box_score_seasons,
        nba_box_players_identity=nba_box_players_identity,
    )
    master = pd.concat([drafted_rows, non_drafted_rows], ignore_index=True, sort=False)
    master = _merge_context(master, context)
    master = _merge_measurements(master, combine, nba_players)
    master = _merge_bigquery_player_features(master, bigquery_player_features)
    master = _impute_measurements(master)
    master = _add_modeling_columns(master)

    sort_columns = ["draft_year", "is_drafted", "draft_pick", "player_name"]
    master = master.sort_values(sort_columns, ascending=[True, False, True, True]).reset_index(drop=True)
    return master


def build_processed_features(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build and align processed X/y files from a master table."""
    ncaa_features = build_ncaa_features(master)
    physical_features = build_physical_features(master)
    context_features = build_context_features(master)
    combined = combine_all_features(ncaa_features, physical_features, context_features)
    train_mask = pd.to_numeric(master["draft_year"], errors="coerce") <= config.TRAIN_MAX_DRAFT_YEAR
    scaler_stats = fit_z_score_stats(combined, columns=ALL_BASE_FEATURES, fit_mask=train_mask.to_numpy())
    combined = apply_z_score_stats(combined, scaler_stats)
    combined.attrs["z_score_stats"] = scaler_stats

    feature_frame = combined.set_index("player_id").reindex(master["player_id"])
    X = feature_frame.reindex(columns=config.MODEL_FEATURE_COLUMNS, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.reset_index(drop=True)
    X.attrs["z_score_stats"] = scaler_stats
    y = master["target_role"].reset_index(drop=True)
    y.name = "target_role"
    return X, y


def save_processed_outputs(master: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> dict[str, Path]:
    """Persist master table, feature matrix, target and reports."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    master_path = config.DATA_PROCESSED / "master_table.csv"
    x_path = config.DATA_PROCESSED / "X.csv"
    y_path = config.DATA_PROCESSED / "y.csv"
    index_path = config.DATA_PROCESSED / "modeling_index.csv"
    target_report_path = config.REPORTS_DIR / "target_distribution.csv"
    summary_path = config.REPORTS_DIR / "master_table_summary.json"
    scaler_path = config.DATA_PROCESSED / "feature_scaler.json"

    master.to_csv(master_path, index=False)
    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)
    master[["player_id", "player_name", "college", "draft_year", "draft_pick", "target_role", "is_drafted"]].to_csv(
        index_path,
        index=False,
    )
    y.value_counts().reindex(config.ROLE_LABELS, fill_value=0).rename_axis("target_role").reset_index(
        name="count"
    ).to_csv(target_report_path, index=False)
    with scaler_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "fit_rule": f"draft_year <= {config.TRAIN_MAX_DRAFT_YEAR}",
                "stats": X.attrs.get("z_score_stats", {}),
            },
            file,
            indent=2,
            sort_keys=True,
        )

    summary = {
        "master_rows": int(len(master)),
        "feature_rows": int(len(X)),
        "feature_columns": list(X.columns),
        "draft_year_min": int(master["draft_year"].min()),
        "draft_year_max": int(master["draft_year"].max()),
        "drafted_rows": int(master["is_drafted"].sum()),
        "non_drafted_rows": int((~master["is_drafted"]).sum()),
        "target_distribution": y.value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict(),
        "target_source_distribution": master["target_source"].value_counts().to_dict(),
        "undrafted_nba_identity_relabels": int(
            master.get("undrafted_nba_identity_match", pd.Series(index=master.index, dtype=object))
            .fillna("")
            .eq("school_match")
            .sum()
        ),
        "undrafted_nba_relabel_target_distribution": (
            master.loc[
                master.get("undrafted_nba_identity_match", pd.Series(index=master.index, dtype=object))
                .fillna("")
                .eq("school_match"),
                "target_role",
            ]
            .value_counts()
            .reindex(config.ROLE_LABELS, fill_value=0)
            .astype(int)
            .to_dict()
        ),
        "target_active_seasons_horizon": int(config.NBA_TARGET_ACTIVE_SEASONS),
        "target_active_seasons_start_offset": int(config.NBA_TARGET_ACTIVE_SEASON_OFFSET),
        "draft_source_distribution": (
            master["draft_source"].fillna("").replace("", "not_drafted").value_counts().to_dict()
            if "draft_source" in master.columns
            else {}
        ),
        "ncaa_source_distribution": (
            master["ncaa_source"].fillna("unknown").value_counts().to_dict()
            if "ncaa_source" in master.columns
            else {}
        ),
        "nba_games_per_season_assumption": NBA_GAMES_PER_SEASON_ASSUMPTION,
        "z_score_fit_rule": f"draft_year <= {config.TRAIN_MAX_DRAFT_YEAR}",
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)

    return {
        "master": master_path,
        "X": x_path,
        "y": y_path,
        "index": index_path,
        "target_report": target_report_path,
        "summary": summary_path,
        "feature_scaler": scaler_path,
    }


def main() -> None:
    """Build and persist processed real-data modeling artifacts."""
    print("=== Construccion de tabla maestra real ===")
    master = build_master_table()
    X, y = build_processed_features(master)
    paths = save_processed_outputs(master, X, y)

    print(f"Master table shape: {master.shape}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("\nDistribucion target:")
    print(y.value_counts().reindex(config.ROLE_LABELS, fill_value=0).to_string())
    print("\nFilas por fuente de target:")
    print(master["target_source"].value_counts().to_string())
    print("\nArchivos guardados:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nConstruccion completada sin excepciones.")


if __name__ == "__main__":
    main()
