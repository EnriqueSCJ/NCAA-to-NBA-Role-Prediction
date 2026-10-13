"""Load and inspect the raw basketball datasets.

The loaders keep the original source files untouched and return pandas
DataFrames that later steps can clean, merge, and validate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.config as config


BART_TORVIK_PLAYER_COLUMNS = [
    "player_name",
    "team",
    "conf",
    "GP",
    "Min_per",
    "Ortg",
    "usg",
    "eFG",
    "TS_per",
    "ORB_per",
    "DRB_per",
    "AST_per",
    "TO_per",
    "FTM",
    "FTA",
    "FT_per",
    "twoPM",
    "twoPA",
    "twoP_per",
    "TPM",
    "TPA",
    "TP_per",
    "blk_per",
    "stl_per",
    "ftr",
    "yr",
    "ht",
    "num",
    "porpag",
    "adjoe",
    "pfr",
    "year",
    "pid",
    "hometown",
    "Rec Rank",
    "ast/tov",
    "rimmade",
    "rimmade+rimmiss",
    "midmade",
    "midmade+midmiss",
    "rimmade/(rimmade+rimmiss)",
    "midmade/(midmade+midmiss)",
    "dunksmade",
    "dunksmiss+dunksmade",
    "dunksmade/(dunksmade+dunksmiss)",
    "pick",
    "drtg",
    "adrtg",
    "dporpag",
    "stops",
    "bpm",
    "obpm",
    "dbpm",
    "gbpm",
    "mp",
    "ogbpm",
    "dgbpm",
    "oreb",
    "dreb",
    "treb",
    "ast",
    "stl",
    "blk",
    "pts",
    "position_label",
    "source_score",
    "birthdate",
]


def _require_path(path: Path, kind: str) -> None:
    """Fail early with a useful message if a source path is missing."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontro {kind}: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV files with settings that work well for mixed sports datasets."""
    return pd.read_csv(path, low_memory=False)


def _read_bart_torvik_player_csv(path: Path) -> pd.DataFrame:
    """Read Bart Torvik player files, which are exported without headers."""
    return pd.read_csv(path, header=None, names=BART_TORVIK_PLAYER_COLUMNS, low_memory=False)


def _print_dataframe_overview(name: str, df: pd.DataFrame) -> None:
    """Print the checkpoint view requested by the implementation guide."""
    print(f"\n{name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    print("  Primeras 2 filas:")
    print(df.head(2).to_string(index=False))


def _load_csv_directory(path: Path, pattern: str = "*.csv") -> dict[str, pd.DataFrame]:
    """Load all CSV files matching a pattern from a directory."""
    _require_path(path, "directorio de datos")

    files_found = sorted(path.glob(pattern))
    if not files_found:
        raise FileNotFoundError(f"No se encontraron archivos {pattern} en {path}")

    dataframes: dict[str, pd.DataFrame] = {}
    for file_path in files_found:
        df = _read_csv(file_path)
        dataframes[file_path.stem] = df
        _print_dataframe_overview(file_path.stem, df)

    return dataframes


def load_college_basketball_nba() -> dict[str, pd.DataFrame]:
    """Load the main NCAA player datasets and NBA advanced-stat CSV files."""
    print(f"Ruta: {config.NCAA_NBA_DATASET_DIR}")
    return _load_csv_directory(config.NCAA_NBA_DATASET_DIR)


def load_college_context() -> dict[str, pd.DataFrame]:
    """Load NCAA team context files from the College Basketball Dataset."""
    print(f"Ruta: {config.COLLEGE_CONTEXT_DATASET_DIR}")
    return _load_csv_directory(config.COLLEGE_CONTEXT_DATASET_DIR, pattern="cbb*.csv")


def load_bart_torvik() -> dict[str, pd.DataFrame]:
    """Load Bart Torvik player and team result files."""
    print(f"Ruta: {config.BART_TORVIK_DATASET_DIR}")
    _require_path(config.BART_TORVIK_DATASET_DIR, "directorio Bart Torvik Data")

    files_found = sorted(config.BART_TORVIK_DATASET_DIR.glob("*.csv"))
    if not files_found:
        raise FileNotFoundError(f"No se encontraron CSVs en {config.BART_TORVIK_DATASET_DIR}")

    dataframes: dict[str, pd.DataFrame] = {}
    for file_path in files_found:
        if file_path.name.startswith("BartTorvik"):
            df = _read_bart_torvik_player_csv(file_path)
        else:
            df = _read_csv(file_path)
        dataframes[file_path.stem] = df
        _print_dataframe_overview(file_path.stem, df)

    return dataframes


def load_combine() -> pd.DataFrame:
    """Load NBA Draft Combine measurements and athletic testing data."""
    _require_path(config.DRAFT_COMBINE_FILE, "archivo Draft Combine")
    df = _read_csv(config.DRAFT_COMBINE_FILE)
    _print_dataframe_overview(config.DRAFT_COMBINE_FILE.stem, df)
    return df


def load_nba_players() -> pd.DataFrame:
    """Load the NBA player database used for biographical and career metadata."""
    _require_path(config.NBA_PLAYERS_FILE, "archivo NBA players")
    df = _read_csv(config.NBA_PLAYERS_FILE)
    _print_dataframe_overview(config.NBA_PLAYERS_FILE.stem, df)
    return df


def load_drafted_players() -> pd.DataFrame:
    """Load the drafted players spreadsheet when Excel support is available."""
    path = config.NCAA_NBA_DATASET_DIR / "DraftedPlayers2009-2021.xlsx"
    _require_path(path, "archivo DraftedPlayers2009-2021.xlsx")
    df = pd.read_excel(path)
    _print_dataframe_overview(path.stem, df)
    return df


def load_basketball_reference_drafts() -> dict[str, pd.DataFrame]:
    """Load Basketball Reference draft CSV exports."""
    print(f"Ruta: {config.BASKETBALL_REFERENCE_DRAFT_DIR}")
    _require_path(config.BASKETBALL_REFERENCE_DRAFT_DIR, "directorio Basketball Reference Draft")

    files_found = sorted(config.BASKETBALL_REFERENCE_DRAFT_DIR.glob("Draft*.csv"))
    if not files_found:
        raise FileNotFoundError(
            f"No se encontraron archivos Draft*.csv en {config.BASKETBALL_REFERENCE_DRAFT_DIR}"
        )

    dataframes: dict[str, pd.DataFrame] = {}
    for file_path in files_found:
        df = pd.read_csv(file_path, header=1, low_memory=False)
        dataframes[file_path.stem] = df
        _print_dataframe_overview(file_path.stem, df)

    return dataframes


def load_twenty_years_draft_data() -> pd.DataFrame:
    """Load the 20 Years of NBA Draft Data supplemental CSV."""
    _require_path(config.TWENTY_YEARS_DRAFT_FILE, "archivo 20 Years of NBA Draft Data")
    df = _read_csv(config.TWENTY_YEARS_DRAFT_FILE)
    _print_dataframe_overview(config.TWENTY_YEARS_DRAFT_FILE.stem, df)
    return df


def main() -> None:
    """Run a local data-loading checkpoint with shape, columns, and samples."""
    print("=== Cargando datos originales ===")

    print("\n1. College Basketball 2009-2021 + NBA Advanced Stats")
    ncaa_nba = load_college_basketball_nba()

    print("\n2. College Basketball Dataset")
    college_context = load_college_context()

    print("\n3. NBA Draft Combine")
    combine = load_combine()

    print("\n4. V2 NBA Player Database")
    nba_players = load_nba_players()

    print("\n5. Drafted Players 2009-2021")
    drafted_players = load_drafted_players()

    print("\n6. Bart Torvik Data")
    bart_torvik = load_bart_torvik()

    print("\n7. Basketball Reference Draft")
    basketball_reference_drafts = load_basketball_reference_drafts()

    print("\n8. 20 Years of NBA Draft Data")
    twenty_years_draft = load_twenty_years_draft_data()

    total_tables = (
        len(ncaa_nba)
        + len(college_context)
        + len(bart_torvik)
        + len(basketball_reference_drafts)
        + 4
    )
    total_rows = (
        sum(len(df) for df in ncaa_nba.values())
        + sum(len(df) for df in college_context.values())
        + sum(len(df) for df in bart_torvik.values())
        + sum(len(df) for df in basketball_reference_drafts.values())
        + len(combine)
        + len(nba_players)
        + len(drafted_players)
        + len(twenty_years_draft)
    )

    print("\n=== Resumen de carga ===")
    print(f"Tablas cargadas: {total_tables}")
    print(f"Filas totales inspeccionadas: {total_rows:,}")
    print("Checkpoint completado sin excepciones.")


if __name__ == "__main__":
    main()
