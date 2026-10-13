"""NBA role target construction.

The target is defined from early active NBA impact: total minutes, an impact
proxy, and average minutes per game over the evaluation horizon.
"""

from __future__ import annotations

import pandas as pd

import src.config as config


REQUIRED_TARGET_COLUMNS = (
    "nba_minutes_y1_y3",
    "nba_bpm_avg",
    "nba_mpg_avg",
)


def validate_target_columns(df: pd.DataFrame) -> None:
    """Validate that all columns required for role assignment exist."""
    missing = [column for column in REQUIRED_TARGET_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Faltan columnas para construir el target NBA: "
            + ", ".join(missing)
        )


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric target column with invalid values filled conservatively."""
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def define_nba_role(df: pd.DataFrame) -> pd.Series:
    """Assign an ordinal NBA role from early active NBA production.

    Expected columns:
    - `nba_minutes_y1_y3`: total NBA minutes in the target window.
    - `nba_bpm_avg`: average or minutes-weighted impact proxy in that window.
    - `nba_mpg_avg`: average minutes per game in that window.

    Rules:
    - `NON_NBA`: fewer than `MINUTES_NON_NBA_THRESHOLD` total minutes.
    - `BENCH`: clears the minutes threshold but not rotation thresholds.
    - `ROTATION`: impact and MPG reach stable rotation thresholds.
    - `STARTER_PLUS`: impact and MPG reach starter or star thresholds.

    Injury gaps should be handled upstream by selecting the first active NBA
    seasons, not by promoting labels after the fact.
    """
    validate_target_columns(df)

    total_minutes = _numeric_column(df, "nba_minutes_y1_y3")
    avg_bpm = _numeric_column(df, "nba_bpm_avg")
    avg_mpg = _numeric_column(df, "nba_mpg_avg")

    roles = pd.Series("BENCH", index=df.index, dtype="object")

    non_nba = total_minutes < config.MINUTES_NON_NBA_THRESHOLD
    nba_sample = ~non_nba

    rotation = (
        nba_sample
        & (avg_bpm >= config.BPM_ROTATION_THRESHOLD)
        & (avg_mpg >= config.MPG_ROTATION_THRESHOLD)
    )
    starter_plus = (
        nba_sample
        & (avg_bpm >= config.BPM_STARTER_THRESHOLD)
        & (avg_mpg >= config.MPG_STARTER_THRESHOLD)
    )
    star_like = (
        nba_sample
        & (avg_bpm >= config.BPM_STAR_THRESHOLD)
        & (avg_mpg >= config.MPG_STAR_THRESHOLD)
    )

    roles.loc[non_nba] = "NON_NBA"
    roles.loc[rotation] = "ROTATION"
    roles.loc[starter_plus | star_like] = "STARTER_PLUS"

    return roles


def build_master_table() -> pd.DataFrame:
    """Build the real merged modeling table from raw datasets."""
    from src.data.build_master import build_master_table as build_real_master_table

    master = build_real_master_table()
    print("Distribucion del target:")
    print(master["target_role"].value_counts().reindex(config.ROLE_LABELS, fill_value=0))
    return master


def demo_target_dataframe() -> pd.DataFrame:
    """Create deterministic examples that exercise every target class."""
    return pd.DataFrame(
        [
            {"player": "Low Minute 1", "nba_minutes_y1_y3": 0, "nba_bpm_avg": 0.0, "nba_mpg_avg": 0.0},
            {"player": "Low Minute 2", "nba_minutes_y1_y3": 120, "nba_bpm_avg": 2.5, "nba_mpg_avg": 10.0},
            {"player": "Low Minute 3", "nba_minutes_y1_y3": 240, "nba_bpm_avg": -1.0, "nba_mpg_avg": 8.0},
            {"player": "Low Minute 4", "nba_minutes_y1_y3": 450, "nba_bpm_avg": 5.0, "nba_mpg_avg": 22.0},
            {"player": "Low Minute 5", "nba_minutes_y1_y3": 499, "nba_bpm_avg": 4.5, "nba_mpg_avg": 31.0},
            {"player": "Deep Bench 1", "nba_minutes_y1_y3": 600, "nba_bpm_avg": -3.5, "nba_mpg_avg": 9.0},
            {"player": "Deep Bench 2", "nba_minutes_y1_y3": 900, "nba_bpm_avg": -1.0, "nba_mpg_avg": 12.0},
            {"player": "Rotation Player", "nba_minutes_y1_y3": 1800, "nba_bpm_avg": -1.5, "nba_mpg_avg": 18.0},
            {"player": "Starter Player", "nba_minutes_y1_y3": 3600, "nba_bpm_avg": 1.7, "nba_mpg_avg": 27.0},
            {"player": "Star Player", "nba_minutes_y1_y3": 5200, "nba_bpm_avg": 5.1, "nba_mpg_avg": 33.0},
        ]
    )


def main() -> None:
    """Run a small target-definition checkpoint."""
    print("=== Construccion del target NBA ===")

    examples = demo_target_dataframe()
    examples["target_role"] = define_nba_role(examples)

    print("\nEjemplos clasificados:")
    print(examples.to_string(index=False))

    print("\nDistribucion de roles:")
    counts = examples["target_role"].value_counts().reindex(config.ROLE_LABELS, fill_value=0)
    print(counts.to_string())

    majority_role = counts.idxmax()
    print(f"\nClase mayoritaria en el ejemplo: {majority_role}")
    print("Checkpoint completado sin excepciones.")


if __name__ == "__main__":
    main()
