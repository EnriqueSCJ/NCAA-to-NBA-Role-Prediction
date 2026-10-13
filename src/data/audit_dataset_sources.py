"""Audit unused or partially used datasets for modeling opportunities.

This module is intentionally read-only: it inspects the raw ``Datasets`` folder
and the current processed master table, then writes a compact report under
``reports/``. It does not change training artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.config as config
from src.data.build_master import (
    _nba_outcome_for_player,
    _read_box_score_seasons,
    _read_raptor,
    canonical_player_name,
    normalize_name,
    parse_height_to_inches,
)
from src.data.load import BART_TORVIK_PLAYER_COLUMNS
from src.features.target import define_nba_role


ATHLETIC_COMBINE_COLUMNS = [
    "STNDVERT",
    "LPVERT",
    "LANE",
    "SHUTTLE",
    "SPRINT",
    "BENCH",
]

TORVIK_TEAM_COLUMNS = [
    "adjoe",
    "adjde",
    "barthag",
    "sos",
    "ncsos",
    "consos",
    "Opp OE",
    "Opp DE",
    "Con Adj OE",
    "Con Adj DE",
    "WAB",
    "adjt",
]


def _safe_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes, Path)) else False:
        return None
    return value


def _read_master() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "master_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe la tabla procesada: {path}")
    master = pd.read_csv(path, low_memory=False)
    master["player_key"] = master["player_name"].map(normalize_name)
    master["team_key"] = master["college"].map(normalize_name)
    master["draft_year"] = pd.to_numeric(master["draft_year"], errors="coerce").astype(int)
    master["season"] = pd.to_numeric(master["season"], errors="coerce").astype(int)
    return master


def _table_shape(path: Path, **kwargs: Any) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path, low_memory=False, **kwargs)
    return {
        "exists": True,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }


def _normalize_ncaa_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["player_name"] = normalized["player_name"].map(canonical_player_name)
    normalized["player_key"] = normalized["player_name"].map(normalize_name)
    normalized["team_key"] = normalized["team"].map(normalize_name)
    normalized["season"] = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    normalized["row_key"] = (
        normalized["player_key"].astype(str)
        + "|"
        + normalized["team_key"].astype(str)
        + "|"
        + normalized["season"].astype(str)
    )
    return normalized


def audit_2022_ncaa_overlap(master: pd.DataFrame) -> dict[str, Any]:
    base_path = config.NCAA_NBA_DATASET_DIR / "CollegeBasketballPlayers2022.csv"
    torvik_path = config.BART_TORVIK_DATASET_DIR / "BartTorvik2022.csv"
    result: dict[str, Any] = {
        "base_path": str(base_path),
        "torvik_path": str(torvik_path),
    }
    if not base_path.exists() or not torvik_path.exists():
        result["status"] = "missing_source"
        return result

    base = pd.read_csv(base_path, low_memory=False)
    torvik = pd.read_csv(torvik_path, header=None, names=BART_TORVIK_PLAYER_COLUMNS, low_memory=False)
    base = _normalize_ncaa_keys(base)
    torvik = _normalize_ncaa_keys(torvik)

    base_keys = set(base["row_key"].dropna())
    torvik_keys = set(torvik["row_key"].dropna())
    overlap = base_keys & torvik_keys
    base_only = base[base["row_key"].isin(base_keys - torvik_keys)]
    torvik_only = torvik[torvik["row_key"].isin(torvik_keys - base_keys)]
    current_2022 = master[master["season"].eq(2022)]
    current_keys = set(
        current_2022["player_key"].astype(str)
        + "|"
        + current_2022["team_key"].astype(str)
        + "|"
        + current_2022["season"].astype(str)
    )

    return {
        **result,
        "status": "ok",
        "college_basketball_players_2022_rows": int(len(base)),
        "bart_torvik_2022_rows": int(len(torvik)),
        "college_basketball_players_2022_unique_players": int(base["player_key"].nunique()),
        "bart_torvik_2022_unique_players": int(torvik["player_key"].nunique()),
        "row_key_overlap": int(len(overlap)),
        "college_2022_only_rows": int(len(base_only)),
        "bart_torvik_2022_only_rows": int(len(torvik_only)),
        "college_2022_only_not_in_current_master": int(len((base_keys - torvik_keys) - current_keys)),
        "bart_torvik_2022_only_not_in_current_master": int(len((torvik_keys - base_keys) - current_keys)),
        "college_2022_only_examples": base_only[["player_name", "team", "year"]].head(10).to_dict("records"),
        "bart_torvik_2022_only_examples": torvik_only[["player_name", "team", "year"]].head(10).to_dict("records"),
    }


def _read_kaggle_combine() -> pd.DataFrame:
    combine = pd.read_csv(config.DRAFT_COMBINE_FILE, low_memory=False)
    combine["player_name"] = combine["PLAYER"].map(canonical_player_name)
    combine["player_key"] = combine["player_name"].map(normalize_name)
    combine["draft_year"] = pd.to_numeric(combine["YEAR"], errors="coerce").astype("Int64")
    return combine


def _read_nl_combine() -> pd.DataFrame:
    path = config.DATASETS_DIR / "NBA Draft Combine & Player Info + Drafted Data" / "NBA_Draft_Combine_2000_2024_nl.csv"
    if not path.exists():
        return pd.DataFrame()
    combine = pd.read_csv(path, low_memory=False)
    combine["player_name"] = combine["Player Name (Last Name, First Name)"].map(canonical_player_name)
    combine["player_key"] = combine["player_name"].map(normalize_name)
    combine["draft_year"] = pd.to_numeric(combine["Year"], errors="coerce").astype("Int64")
    combine["nl_height_in"] = combine["Height in Ft & In"].map(parse_height_to_inches)
    return combine


def _read_nba_box_players() -> pd.DataFrame:
    path = config.NBA_BOX_SCORE_PLAYERS_FILE
    if not path.exists():
        return pd.DataFrame()

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
    players["school_key"] = players.get("school", "").map(normalize_name) if "school" in players.columns else ""
    return players


def _expanded_school_key(value: Any) -> str:
    key = normalize_name(value)
    key = key.replace(" saint ", " st ")
    if key.startswith("saint "):
        key = "st " + key[len("saint ") :]
    tokens = key.split()
    if tokens and tokens[-1] == "st":
        tokens[-1] = "state"
    return " ".join(tokens)


def _school_keys_compatible(left: Any, right: Any) -> bool:
    left_key = _expanded_school_key(left)
    right_key = _expanded_school_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if len(left_key) >= 6 and left_key in right_key:
        return True
    if len(right_key) >= 6 and right_key in left_key:
        return True
    return False


def _coverage_by_split(master: pd.DataFrame, flag_column: str) -> dict[str, Any]:
    splits = {
        "train_2009_2018": master["draft_year"].le(config.TRAIN_MAX_DRAFT_YEAR),
        "validation_2019_2020": master["draft_year"].between(
            config.VALIDATION_START_DRAFT_YEAR,
            config.VALIDATION_END_DRAFT_YEAR,
        ),
        "test_2021_2022": master["draft_year"].between(
            config.TEST_START_DRAFT_YEAR,
            config.TEST_END_DRAFT_YEAR,
        ),
        "exploratory_2023": master["draft_year"].eq(2023),
    }
    coverage: dict[str, Any] = {}
    for split, mask in splits.items():
        frame = master.loc[mask]
        if frame.empty:
            coverage[split] = {"rows": 0, "covered_rows": 0, "coverage_pct": 0.0}
            continue
        covered = int(frame[flag_column].sum())
        coverage[split] = {
            "rows": int(len(frame)),
            "covered_rows": covered,
            "coverage_pct": covered / len(frame),
        }
    return coverage


def audit_combine_sources(master: pd.DataFrame) -> dict[str, Any]:
    kaggle = _read_kaggle_combine()
    exact_cols = ["player_key", "draft_year"]
    exact_keys = kaggle[exact_cols].dropna().drop_duplicates()
    by_player = kaggle[["player_key"]].dropna().drop_duplicates()

    audited = master.merge(exact_keys.assign(has_kaggle_exact=True), how="left", on=exact_cols)
    audited = audited.merge(by_player.assign(has_kaggle_player=True), how="left", on="player_key")
    audited["has_kaggle_exact"] = audited["has_kaggle_exact"].fillna(False).astype(bool)
    audited["has_kaggle_player"] = audited["has_kaggle_player"].fillna(False).astype(bool)

    athletic_coverage: dict[str, Any] = {}
    master_exact = master[["player_id", "player_key", "draft_year", "target_role", "is_drafted"]].merge(
        kaggle,
        how="left",
        on=exact_cols,
    )
    for column in ATHLETIC_COMBINE_COLUMNS:
        if column not in master_exact.columns:
            continue
        values = pd.to_numeric(master_exact[column], errors="coerce")
        athletic_coverage[column] = {
            "covered_all_rows": int(values.notna().sum()),
            "covered_drafted_rows": int(values[master_exact["is_drafted"].astype(bool)].notna().sum()),
            "covered_starter_plus_rows": int(values[master_exact["target_role"].eq("STARTER_PLUS")].notna().sum()),
        }

    nl = _read_nl_combine()
    nl_summary: dict[str, Any] = {"exists": not nl.empty}
    if not nl.empty:
        nl_keys = nl[exact_cols].dropna().drop_duplicates()
        audited = audited.merge(nl_keys.assign(has_nl_exact=True), how="left", on=exact_cols)
        audited["has_nl_exact"] = audited["has_nl_exact"].fillna(False).astype(bool)
        nl_summary.update(
            {
                "rows": int(len(nl)),
                "unique_players": int(nl["player_key"].nunique()),
                "exact_master_matches": int(audited["has_nl_exact"].sum()),
                "additional_exact_matches_beyond_kaggle": int(
                    (audited["has_nl_exact"] & ~audited["has_kaggle_exact"]).sum()
                ),
            }
        )

    return {
        "kaggle_rows": int(len(kaggle)),
        "kaggle_unique_players": int(kaggle["player_key"].nunique()),
        "pipeline_currently_uses": ["HGT", "WGT", "WNGSPN", "POS", "BMI"],
        "pipeline_ignores_athletic_tests": [column for column in ATHLETIC_COMBINE_COLUMNS if column in kaggle.columns],
        "exact_master_matches": int(audited["has_kaggle_exact"].sum()),
        "player_fallback_master_matches": int(audited["has_kaggle_player"].sum()),
        "exact_coverage_by_split": _coverage_by_split(audited, "has_kaggle_exact"),
        "athletic_test_coverage_on_exact_matches": athletic_coverage,
        "nl_combine_source": nl_summary,
        "methodological_note": (
            "Athletic combine tests are pre-draft, but missingness is not random because combine invitations "
            "are a scouting signal. They should be added only with explicit missing indicators and validated "
            "by temporal validation."
        ),
    }


def _read_torvik_team_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(config.BART_TORVIK_DATASET_DIR.glob("*_team_results.csv")):
        df = pd.read_csv(path, low_memory=False)
        year = int(path.stem.split("_", 1)[0])
        df["season"] = year
        team_col = "team" if "team" in df.columns else df.columns[0]
        df["team_key"] = df[team_col].map(normalize_name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def audit_team_context(master: pd.DataFrame) -> dict[str, Any]:
    cbb_files = sorted(config.COLLEGE_CONTEXT_DATASET_DIR.glob("cbb*.csv"))
    cbb_shapes = {path.name: _table_shape(path)["rows"] for path in cbb_files if path.exists()}
    team_results = _read_torvik_team_results()
    current_nonzero = {
        column: int(pd.to_numeric(master[column], errors="coerce").fillna(0.0).ne(0.0).sum())
        for column in ["team_adj_o", "team_adj_d", "team_tempo", "team_power_rating", "strength_of_schedule"]
        if column in master.columns
    }
    result: dict[str, Any] = {
        "college_basketball_dataset_files": cbb_shapes,
        "current_context_nonzero_counts": current_nonzero,
        "team_results_exists": not team_results.empty,
    }
    if team_results.empty:
        return result

    available_cols = [column for column in TORVIK_TEAM_COLUMNS if column in team_results.columns]
    coverage_by_year: dict[str, Any] = {}
    for year in sorted(team_results["season"].dropna().astype(int).unique()):
        year_master = master[master["season"].eq(year)]
        if year_master.empty:
            coverage_by_year[str(year)] = {
                "master_rows": 0,
                "matched_rows": 0,
                "coverage_pct": 0.0,
            }
            continue
        keys = team_results.loc[team_results["season"].eq(year), ["team_key", "season"]].drop_duplicates()
        joined = year_master.merge(keys.assign(has_torvik_team=True), how="left", on=["team_key", "season"])
        matched = int(joined["has_torvik_team"].fillna(False).sum())
        coverage_by_year[str(year)] = {
            "master_rows": int(len(year_master)),
            "matched_rows": matched,
            "coverage_pct": matched / len(year_master),
        }

    return {
        **result,
        "team_results_rows": int(len(team_results)),
        "team_results_years": sorted(team_results["season"].dropna().astype(int).unique().tolist()),
        "team_results_candidate_columns": available_cols,
        "team_results_coverage_by_year": coverage_by_year,
        "methodological_note": (
            "The team_results files contain richer schedule/context variables, but in this folder they start in "
            "2022. Adding columns that exist only in test/future years would create a cohort-availability artifact "
            "unless equivalent historical columns are added for training years."
        ),
    }


def audit_bigquery_coverage(master: pd.DataFrame) -> dict[str, Any]:
    path = config.NCAA_BIGQUERY_EXPORT_DIR / "bigquery-public-data__ncaa_basketball__mbb_players_games_sr.parquet"
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result

    usecols = ["season", "full_name", "played", "starter", "minutes_int64", "points"]
    games = pd.read_parquet(path, columns=usecols)
    games["model_season"] = pd.to_numeric(games["season"], errors="coerce") + 1
    result["raw_rows"] = int(len(games))
    result["raw_source_seasons"] = sorted(pd.to_numeric(games["season"], errors="coerce").dropna().astype(int).unique().tolist())
    result["model_seasons_after_current_shift"] = sorted(
        games["model_season"].dropna().astype(int).unique().tolist()
    )
    coverage = (
        master.assign(has_official=pd.to_numeric(master.get("official_games", 0), errors="coerce").fillna(0).gt(0))
        .groupby("draft_year")["has_official"]
        .agg(rows="size", covered_rows="sum")
        .reset_index()
    )
    coverage["coverage_pct"] = coverage["covered_rows"] / coverage["rows"].replace(0, np.nan)
    result["current_official_feature_coverage_by_draft_year"] = coverage.to_dict("records")
    result["methodological_note"] = (
        "The current pipeline uses only official games, MPG and start rate. More box-score aggregates could be "
        "derived from this parquet, but only for the exported seasons; uneven year coverage can introduce covariate shift."
    )
    return result


def audit_undrafted_nba_outcomes(master: pd.DataFrame) -> dict[str, Any]:
    non_drafted = master[~master["is_drafted"].astype(bool)].copy()
    raptor = _read_raptor()
    box_score_seasons = _read_box_score_seasons()
    active_keys = set(box_score_seasons["player_key"]) | set(raptor["player_key"])
    candidates = non_drafted[non_drafted["player_key"].isin(active_keys)].copy()
    if candidates.empty:
        return {
            "non_drafted_rows": int(len(non_drafted)),
            "non_drafted_with_any_nba_name_match": 0,
            "non_drafted_with_post_ncaa_nba_activity": 0,
            "would_change_target_counts": {},
            "examples": [],
        }

    outcomes: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        outcome = _nba_outcome_for_player(
            raptor=raptor,
            box_score_seasons=box_score_seasons,
            player_key=str(row["player_key"]),
            draft_year=int(row["draft_year"]),
        )
        if float(outcome.get("nba_minutes_y1_y3", 0.0)) <= 0.0:
            continue
        outcome_row = {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "college": row["college"],
            "draft_year": int(row["draft_year"]),
            **outcome,
        }
        outcomes.append(outcome_row)

    if not outcomes:
        return {
            "non_drafted_rows": int(len(non_drafted)),
            "non_drafted_with_any_nba_name_match": int(len(candidates)),
            "non_drafted_with_post_ncaa_nba_activity": 0,
            "would_change_target_counts": {},
            "examples": [],
        }

    outcome_frame = pd.DataFrame(outcomes)
    outcome_frame["recomputed_target_role"] = define_nba_role(outcome_frame)
    current_targets = non_drafted.set_index("player_id")["target_role"]
    outcome_frame["current_target_role"] = outcome_frame["player_id"].map(current_targets)
    outcome_frame["college_key"] = outcome_frame["college"].map(normalize_name)

    name_counts = non_drafted.groupby("player_key").size()
    outcome_frame["non_drafted_rows_with_same_name"] = outcome_frame["player_id"].map(
        non_drafted.set_index("player_id")["player_key"].map(name_counts)
    )

    nba_players = _read_nba_box_players()
    school_keys_by_player = (
        nba_players[nba_players["player_key"].ne("")]
        .groupby("player_key")["school_key"]
        .apply(lambda values: sorted({str(value) for value in values if str(value)}))
        .to_dict()
        if not nba_players.empty and "school_key" in nba_players.columns
        else {}
    )
    outcome_frame["nba_school_keys"] = outcome_frame["player_id"].map(
        non_drafted.set_index("player_id")["player_key"].map(lambda key: school_keys_by_player.get(key, []))
    )
    outcome_frame["school_match"] = [
        any(_school_keys_compatible(college_key, school_key) for school_key in school_keys)
        for college_key, school_keys in zip(outcome_frame["college_key"], outcome_frame["nba_school_keys"])
    ]
    outcome_frame["unambiguous_name_in_master"] = outcome_frame["non_drafted_rows_with_same_name"].eq(1)
    outcome_frame["confidence"] = np.select(
        [
            outcome_frame["school_match"],
            outcome_frame["unambiguous_name_in_master"],
        ],
        [
            "school_match",
            "unambiguous_name_only",
        ],
        default="name_only_ambiguous",
    )
    changed = outcome_frame[outcome_frame["recomputed_target_role"].ne("NON_NBA")].copy()
    changed_from_current = outcome_frame[
        outcome_frame["recomputed_target_role"].ne(outcome_frame["current_target_role"])
    ].copy()
    already_matches_current = outcome_frame[
        outcome_frame["recomputed_target_role"].eq(outcome_frame["current_target_role"])
    ].copy()
    changed = changed.sort_values(
        ["confidence", "recomputed_target_role", "nba_minutes_y1_y3"],
        ascending=[True, True, False],
    )
    role_counts = outcome_frame["recomputed_target_role"].value_counts().reindex(config.ROLE_LABELS, fill_value=0)
    changed_counts = changed["recomputed_target_role"].value_counts().reindex(config.ROLE_LABELS, fill_value=0)
    confidence_counts = (
        changed.groupby(["confidence", "recomputed_target_role"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["confidence", "recomputed_target_role"])
    )

    example_columns = [
        "player_name",
        "college",
        "draft_year",
        "current_target_role",
        "recomputed_target_role",
        "confidence",
        "nba_school_keys",
        "nba_minutes_y1_y3",
        "nba_mpg_avg",
        "nba_bpm_avg",
        "nba_active_seasons",
    ]
    return {
        "non_drafted_rows": int(len(non_drafted)),
        "non_drafted_with_any_nba_name_match": int(len(candidates)),
        "non_drafted_with_post_ncaa_nba_activity": int(len(outcome_frame)),
        "recomputed_role_counts_among_matched": role_counts.astype(int).to_dict(),
        "would_change_from_non_nba_counts": changed_counts.astype(int).to_dict(),
        "would_change_from_non_nba_total": int(len(changed)),
        "would_change_from_current_target_total": int(len(changed_from_current)),
        "already_matches_current_target_total": int(len(already_matches_current)),
        "would_change_from_non_nba_counts_by_confidence": confidence_counts.to_dict("records"),
        "would_change_from_non_nba_school_match_total": int(changed["school_match"].sum()),
        "would_change_from_non_nba_unambiguous_name_total": int(
            changed["unambiguous_name_in_master"].sum()
        ),
        "examples": changed[example_columns].head(30).to_dict("records"),
        "methodological_note": (
            "Existing NBA outcome files can identify undrafted players who later logged NBA minutes. Production "
            "relabeling should use identity checks such as school or another stable player identifier to avoid "
            "homonym errors; the current pipeline applies the conservative school-match subset."
        ),
    }


def audit_nba_auxiliary_sources(master: pd.DataFrame) -> dict[str, Any]:
    players_path = config.NBA_BOX_SCORE_PLAYERS_FILE
    result: dict[str, Any] = {
        "nba_box_players_path": str(players_path),
        "nba_box_players_exists": players_path.exists(),
    }
    if players_path.exists():
        players = _read_nba_box_players()
        name_columns = [column for column in ["firstName", "lastName", "name", "playerName"] if column in players.columns]
        key_matches = master[["player_key"]].merge(
            players[["player_key"]].drop_duplicates().assign(has_nba_box_player=True),
            how="left",
            on="player_key",
        )
        result.update(
            {
                "rows": int(len(players)),
                "columns": list(players.columns),
                "name_columns_detected": name_columns,
                "master_name_matches": int(key_matches["has_nba_box_player"].fillna(False).sum()),
                "draft_metadata_columns_present": [
                    column for column in ["draftYear", "draftRound", "draftNumber"] if column in players.columns
                ],
                "physical_columns_present": [
                    column
                    for column in ["height", "heightInches", "bodyWeight", "bodyWeightLbs"]
                    if column in players.columns
                ],
            }
        )

    result["methodological_note"] = (
        "NBA player and career/stat files are useful for target construction, identity matching and measurement fallback. "
        "Career NBA stats, draft number and post-draft metadata should not be model input features for a pre-draft predictor."
    )
    return result


def build_audit_report() -> dict[str, Any]:
    master = _read_master()
    report = {
        "current_processed_data": {
            "master_rows": int(len(master)),
            "drafted_rows": int(master["is_drafted"].astype(bool).sum()),
            "non_drafted_rows": int((~master["is_drafted"].astype(bool)).sum()),
            "draft_year_min": int(master["draft_year"].min()),
            "draft_year_max": int(master["draft_year"].max()),
            "target_distribution": master["target_role"].value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict(),
            "ncaa_source_distribution": master["ncaa_source"].fillna("unknown").value_counts().to_dict()
            if "ncaa_source" in master.columns
            else {},
        },
        "ncaa_2022_overlap": audit_2022_ncaa_overlap(master),
        "combine_sources": audit_combine_sources(master),
        "team_context": audit_team_context(master),
        "bigquery_player_games": audit_bigquery_coverage(master),
        "undrafted_nba_outcomes": audit_undrafted_nba_outcomes(master),
        "nba_auxiliary_sources": audit_nba_auxiliary_sources(master),
        "recommendations": [
            {
                "priority": "high",
                "area": "target_labels",
                "summary": "Relabel non-drafted NCAA players who later appear in NBA box-score/RAPTOR outcomes instead of forcing all non-drafted rows to NON_NBA.",
            },
            {
                "priority": "medium",
                "area": "combine",
                "summary": "Evaluate unused pre-draft athletic tests from combine sources with missing indicators and temporal validation.",
            },
            {
                "priority": "medium",
                "area": "ncaa_2022",
                "summary": "Compare CollegeBasketballPlayers2022.csv with BartTorvik2022.csv and add non-overlapping safe rows if schema-quality checks pass.",
            },
            {
                "priority": "medium",
                "area": "identity_matching",
                "summary": "Use NBA Box Scores Players.csv as an auxiliary identity/measurement fallback, while excluding draft number and career stats from features.",
            },
            {
                "priority": "low",
                "area": "team_context",
                "summary": "Do not add Torvik team_results-only columns to the supervised model unless equivalent historical training-year coverage is available.",
            },
        ],
    }
    return _as_jsonable(report)


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_audit_report()
    output_path = config.REPORTS_DIR / "data_source_audit.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=== Auditoria de fuentes en Datasets ===")
    print(f"Reporte guardado en: {output_path}")
    print("\nResumen:")
    print(json.dumps(report["current_processed_data"], indent=2, sort_keys=True))
    print("\nNo drafteados con actividad NBA detectada:")
    print(json.dumps(report["undrafted_nba_outcomes"], indent=2, sort_keys=True)[:4000])


if __name__ == "__main__":
    main()
