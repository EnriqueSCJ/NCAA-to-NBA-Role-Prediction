"""Feature engineering for NCAA to NBA role prediction.

The builders accept both clean modeling-column names and several aliases found
in the raw datasets. They return deterministic, numeric, NaN-free features that
can feed tabular models and Milvus numeric vectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NCAA_BASE_FEATURES = [
    "pts_per40",
    "ast_per40",
    "reb_per40",
    "stl_per40",
    "blk_per40",
    "tov_per40",
    "ts_pct",
    "efg_pct",
    "usg_pct",
    "ast_tov_ratio",
    "stocks_per40",
    "three_par",
    "ftr",
    "rec_rank",
    "is_unranked",
    "official_games",
    "official_mpg",
    "official_start_rate",
]

PHYSICAL_BASE_FEATURES = [
    "height_in",
    "weight_lb",
    "wingspan_in",
    "bmi",
    "wingspan_diff",
    "age_at_draft",
]

CONTEXT_BASE_FEATURES = [
    "team_adj_o",
    "team_adj_d",
    "team_tempo",
    "team_power_rating",
    "strength_of_schedule",
]

ALL_BASE_FEATURES = NCAA_BASE_FEATURES + PHYSICAL_BASE_FEATURES + CONTEXT_BASE_FEATURES

ZScoreStats = dict[str, dict[str, float]]


def _empty_numeric(index: pd.Index, default: float = 0.0) -> pd.Series:
    """Return a numeric fallback series aligned to a DataFrame index."""
    return pd.Series(default, index=index, dtype="float64")


def _first_numeric(df: pd.DataFrame, aliases: list[str], default: float = 0.0) -> pd.Series:
    """Return the first available numeric column among aliases."""
    for column in aliases:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return _empty_numeric(df.index, default)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    """Divide two numeric series and replace invalid values."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _per40_from_total_or_rate(
    df: pd.DataFrame,
    clean_aliases: list[str],
    raw_aliases: list[str],
    minutes: pd.Series,
) -> pd.Series:
    """Use an existing per-40 feature or derive one from per-game volume."""
    existing = _first_numeric(df, clean_aliases, default=np.nan)
    if existing.notna().any():
        return existing.fillna(0.0)

    raw = _first_numeric(df, raw_aliases, default=0.0)
    return _safe_divide(raw * 40.0, minutes)


def _add_z_scores(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Append z-score columns for selected numeric columns."""
    for column in columns:
        values = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        std = values.std(ddof=0)
        if std and not np.isclose(std, 0.0):
            features[f"{column}_z"] = (values - values.mean()) / std
        else:
            features[f"{column}_z"] = 0.0

    return features


def fit_z_score_stats(
    features: pd.DataFrame,
    columns: list[str] | None = None,
    fit_mask: pd.Series | np.ndarray | None = None,
) -> ZScoreStats:
    """Fit z-score means/stds on a selected cohort."""
    target_columns = columns or ALL_BASE_FEATURES
    fit_frame = features.loc[fit_mask].copy() if fit_mask is not None else features
    stats: ZScoreStats = {}

    for column in target_columns:
        if column not in features.columns:
            continue
        values = pd.to_numeric(fit_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        std = float(values.std(ddof=0))
        stats[column] = {
            "mean": float(values.mean()),
            "std": std if std and not np.isclose(std, 0.0) else 0.0,
        }

    return stats


def apply_z_score_stats(features: pd.DataFrame, stats: ZScoreStats) -> pd.DataFrame:
    """Apply pre-fit z-score stats without refitting on validation/test rows."""
    features = features.copy()
    for column, column_stats in stats.items():
        if column not in features.columns:
            continue
        values = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        std = float(column_stats.get("std", 0.0))
        mean = float(column_stats.get("mean", 0.0))
        if std and not np.isclose(std, 0.0):
            features[f"{column}_z"] = (values - mean) / std
        else:
            features[f"{column}_z"] = 0.0

    return _clean_numeric_frame(features)


def _clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities and NaNs in numeric columns."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def build_ncaa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build normalized individual NCAA statistical features.

    Required input:
    - `player_id`

    Supported source aliases include clean columns such as `pts_per40` and raw
    columns such as `pts` + `mp`.
    """
    if "player_id" not in df.columns:
        raise ValueError("build_ncaa_features requiere una columna 'player_id'")

    features = pd.DataFrame(index=df.index)
    features["player_id"] = df["player_id"].astype(str)

    minutes = _first_numeric(df, ["mp", "minutes", "min", "mpg"], default=0.0)

    features["pts_per40"] = _per40_from_total_or_rate(df, ["pts_per40", "ncaa_pts_per40"], ["pts", "ppg"], minutes)
    features["ast_per40"] = _per40_from_total_or_rate(df, ["ast_per40", "ncaa_ast_per40"], ["ast", "apg"], minutes)
    features["reb_per40"] = _per40_from_total_or_rate(df, ["reb_per40", "treb_per40", "ncaa_reb_per40"], ["treb", "reb", "rpg"], minutes)
    features["stl_per40"] = _per40_from_total_or_rate(df, ["stl_per40", "ncaa_stl_per40"], ["stl", "spg"], minutes)
    features["blk_per40"] = _per40_from_total_or_rate(df, ["blk_per40", "ncaa_blk_per40"], ["blk", "bpg"], minutes)
    features["tov_per40"] = _per40_from_total_or_rate(df, ["tov_per40", "ncaa_tov_per40"], ["tov", "to"], minutes)

    features["ts_pct"] = _first_numeric(df, ["ts_pct", "TS_per", "ts", "ncaa_ts"], default=0.0)
    features["efg_pct"] = _first_numeric(df, ["efg_pct", "eFG", "efg", "ncaa_efg"], default=0.0)
    features["usg_pct"] = _first_numeric(df, ["usg_pct", "usg", "USG_per", "ncaa_usg"], default=0.0)

    existing_ast_tov = _first_numeric(df, ["ast_tov_ratio", "ast/tov"], default=np.nan)
    derived_ast_tov = _safe_divide(features["ast_per40"], features["tov_per40"])
    features["ast_tov_ratio"] = existing_ast_tov.fillna(derived_ast_tov).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features["stocks_per40"] = features["stl_per40"] + features["blk_per40"]

    three_attempts = _first_numeric(df, ["TPA", "three_pa", "three_attempts"], default=0.0)
    field_goal_attempts = (
        _first_numeric(df, ["FGA", "fga"], default=np.nan)
        .fillna(_first_numeric(df, ["twoPA"], default=0.0) + three_attempts)
        .replace(0, np.nan)
    )
    existing_three_par = _first_numeric(df, ["three_par", "3par", "ncaa_3par"], default=np.nan)
    features["three_par"] = existing_three_par.fillna(_safe_divide(three_attempts, field_goal_attempts))
    features["ftr"] = _first_numeric(df, ["ftr", "FTR", "ncaa_ftr"], default=0.0)

    raw_rec_rank = _first_numeric(df, ["rec_rank", "Rec Rank", "recruiting_rank"], default=np.nan)
    derived_unranked = raw_rec_rank.isna().astype(float)
    explicit_unranked = _first_numeric(df, ["is_unranked", "unranked"], default=np.nan)
    features["is_unranked"] = explicit_unranked.fillna(derived_unranked).clip(lower=0.0, upper=1.0)
    features["rec_rank"] = raw_rec_rank.fillna(0.0).clip(lower=0.0, upper=100.0)
    features["official_games"] = _first_numeric(df, ["official_games", "bq_games"], default=0.0)
    features["official_mpg"] = _first_numeric(df, ["official_mpg", "bq_mpg"], default=0.0)
    features["official_start_rate"] = _first_numeric(
        df,
        ["official_start_rate", "bq_start_rate"],
        default=0.0,
    ).clip(lower=0.0, upper=1.0)

    features = _add_z_scores(features, NCAA_BASE_FEATURES)
    return _clean_numeric_frame(features)


def build_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build normalized physical features."""
    if "player_id" not in df.columns:
        raise ValueError("build_physical_features requiere una columna 'player_id'")

    features = pd.DataFrame(index=df.index)
    features["player_id"] = df["player_id"].astype(str)
    features["height_in"] = _first_numeric(df, ["height_in", "HGT", "height", "ht_in"], default=0.0)
    features["weight_lb"] = _first_numeric(df, ["weight_lb", "WGT", "weight", "wt"], default=0.0)
    features["wingspan_in"] = _first_numeric(df, ["wingspan_in", "WNGSPN", "wingspan"], default=0.0)

    features["bmi"] = _first_numeric(df, ["bmi", "BMI"], default=np.nan)
    derived_bmi = _safe_divide(features["weight_lb"] * 703.0, features["height_in"] ** 2)
    features["bmi"] = features["bmi"].fillna(derived_bmi).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features["wingspan_diff"] = _first_numeric(df, ["wingspan_diff", "wingspan_minus_height"], default=np.nan)
    derived_wingspan_diff = features["wingspan_in"] - features["height_in"]
    derived_wingspan_diff = derived_wingspan_diff.where(
        (features["wingspan_in"] > 0) & (features["height_in"] > 0),
        0.0,
    )
    features["wingspan_diff"] = features["wingspan_diff"].fillna(derived_wingspan_diff)

    raw_age_at_draft = _first_numeric(df, ["age_at_draft"], default=np.nan)
    fallback_age = _first_numeric(df, ["age", "AGE"], default=21.0)
    features["age_at_draft"] = raw_age_at_draft.fillna(fallback_age).clip(lower=16.0, upper=30.0)

    features = _add_z_scores(features, PHYSICAL_BASE_FEATURES)
    return _clean_numeric_frame(features)


def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build normalized team/context features.

    `strength_of_schedule` is kept as an explicit feature. If a clean SoS
    column is not present, WAB is used as a pragmatic proxy until a true SoS
    source is merged.
    """
    if "player_id" not in df.columns:
        raise ValueError("build_context_features requiere una columna 'player_id'")

    features = pd.DataFrame(index=df.index)
    features["player_id"] = df["player_id"].astype(str)
    features["team_adj_o"] = _first_numeric(df, ["team_adj_o", "ADJOE", "adj_o"], default=0.0)
    features["team_adj_d"] = _first_numeric(df, ["team_adj_d", "ADJDE", "adj_d"], default=0.0)
    features["team_tempo"] = _first_numeric(df, ["team_tempo", "ADJ_T", "tempo"], default=0.0)
    features["team_power_rating"] = _first_numeric(df, ["team_power_rating", "BARTHAG", "power_rating"], default=0.0)
    features["strength_of_schedule"] = _first_numeric(
        df,
        ["strength_of_schedule", "sos", "SOS", "team_sos", "WAB", "wab"],
        default=0.0,
    )

    features = _add_z_scores(features, CONTEXT_BASE_FEATURES)
    return _clean_numeric_frame(features)


def combine_all_features(
    ncaa_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join NCAA, physical, and context feature frames by `player_id`."""
    for name, frame in {
        "ncaa_df": ncaa_df,
        "physical_df": physical_df,
        "context_df": context_df,
    }.items():
        if "player_id" not in frame.columns:
            raise ValueError(f"{name} debe contener 'player_id'")

    combined = (
        ncaa_df.set_index("player_id")
        .join(physical_df.set_index("player_id"), how="inner", rsuffix="_physical")
        .join(context_df.set_index("player_id"), how="inner", rsuffix="_context")
        .reset_index()
    )

    return _clean_numeric_frame(combined)


def demo_feature_input() -> pd.DataFrame:
    """Create a small deterministic dataset for local feature checkpoints."""
    return pd.DataFrame(
        [
            {
                "player_id": "player_guard_star",
                "pts": 24.0,
                "ast": 6.4,
                "treb": 5.1,
                "stl": 2.0,
                "blk": 0.3,
                "tov": 3.2,
                "mp": 34.0,
                "TS_per": 62.0,
                "eFG": 58.0,
                "usg": 29.0,
                "TPA": 9.0,
                "twoPA": 10.0,
                "ftr": 34.0,
                "height_in": 76.0,
                "weight_lb": 195.0,
                "wingspan_in": 79.0,
                "ADJOE": 122.0,
                "ADJDE": 91.0,
                "ADJ_T": 70.0,
                "BARTHAG": 0.96,
                "WAB": 9.5,
                "Rec Rank": 96.0,
                "official_games": 33,
                "official_mpg": 32.0,
                "official_start_rate": 0.95,
                "age_at_draft": 20.8,
            },
            {
                "player_id": "player_wing_rotation",
                "pts": 15.0,
                "ast": 2.2,
                "treb": 7.3,
                "stl": 1.4,
                "blk": 0.8,
                "tov": 1.7,
                "mp": 30.0,
                "TS_per": 57.0,
                "eFG": 54.0,
                "usg": 21.0,
                "TPA": 4.0,
                "twoPA": 7.0,
                "ftr": 28.0,
                "height_in": 79.0,
                "weight_lb": 215.0,
                "wingspan_in": 83.0,
                "ADJOE": 116.0,
                "ADJDE": 94.0,
                "ADJ_T": 66.0,
                "BARTHAG": 0.90,
                "WAB": 5.2,
                "Rec Rank": 82.0,
                "official_games": 34,
                "official_mpg": 30.5,
                "official_start_rate": 0.85,
                "age_at_draft": 21.4,
            },
            {
                "player_id": "player_big_bench",
                "pts": 10.5,
                "ast": 1.0,
                "treb": 8.8,
                "stl": 0.5,
                "blk": 2.4,
                "tov": 2.0,
                "mp": 24.0,
                "TS_per": 53.0,
                "eFG": 50.0,
                "usg": 18.0,
                "TPA": 0.5,
                "twoPA": 8.0,
                "ftr": 42.0,
                "height_in": 82.0,
                "weight_lb": 245.0,
                "wingspan_in": 87.0,
                "ADJOE": 104.0,
                "ADJDE": 101.0,
                "ADJ_T": 63.0,
                "BARTHAG": 0.68,
                "WAB": -1.5,
                "Rec Rank": 61.0,
                "official_games": 31,
                "official_mpg": 24.0,
                "official_start_rate": 0.40,
                "age_at_draft": 22.2,
            },
            {
                "player_id": "player_guard_low",
                "pts": 8.0,
                "ast": 3.3,
                "treb": 2.4,
                "stl": 0.9,
                "blk": 0.1,
                "tov": 2.6,
                "mp": 22.0,
                "TS_per": 49.0,
                "eFG": 46.0,
                "usg": 17.0,
                "TPA": 3.5,
                "twoPA": 5.0,
                "ftr": 20.0,
                "height_in": 73.0,
                "weight_lb": 180.0,
                "wingspan_in": 74.0,
                "ADJOE": 99.0,
                "ADJDE": 108.0,
                "ADJ_T": 72.0,
                "BARTHAG": 0.40,
                "WAB": -8.0,
                "Rec Rank": np.nan,
                "official_games": 28,
                "official_mpg": 20.0,
                "official_start_rate": 0.15,
                "age_at_draft": 21.0,
            },
        ]
    )


def main() -> None:
    """Run a local feature-engineering checkpoint."""
    print("=== Construccion de features ===")
    raw = demo_feature_input()

    ncaa_features = build_ncaa_features(raw)
    physical_features = build_physical_features(raw)
    context_features = build_context_features(raw)
    combined = combine_all_features(ncaa_features, physical_features, context_features)

    numeric = combined.select_dtypes(include=[np.number])
    has_invalid = numeric.isna().any().any() or np.isinf(numeric.to_numpy()).any()
    if has_invalid:
        raise ValueError("La matriz combinada contiene NaNs o infinitos")

    print(f"NCAA features shape: {ncaa_features.shape}")
    print(f"Physical features shape: {physical_features.shape}")
    print(f"Context features shape: {context_features.shape}")
    print(f"Combined features shape: {combined.shape}")

    z_columns = [column for column in combined.columns if column.endswith("_z")]
    print(f"Z-score columns: {len(z_columns)}")
    print(f"Invalid numeric values: {has_invalid}")

    correlations = numeric.corr()["pts_per40_z"].drop(labels=["pts_per40_z"], errors="ignore")
    print("\nCorrelaciones principales con pts_per40_z:")
    print(correlations.abs().sort_values(ascending=False).head(5).to_string())

    print("\nSample combinado:")
    print(combined.head(3).to_string(index=False))
    print("\nCheckpoint completado sin excepciones.")


if __name__ == "__main__":
    main()
