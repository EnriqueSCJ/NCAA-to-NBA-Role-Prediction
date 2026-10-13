"""Export selected NCAA public BigQuery tables into the local Datasets folder.

The script intentionally materializes BigQuery results as local parquet files.
That keeps the modeling pipeline reproducible after the first download and
avoids querying the public dataset during every training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

import src.config as config


PUBLIC_DATASET = "bigquery-public-data.ncaa_basketball"
DEFAULT_TABLES = ["mbb_players_games_sr", "mbb_teams_games_sr"]
SEASON_COLUMNS = ["season", "season_year", "year"]


def _load_bigquery_client(project: str | None):
    """Create a BigQuery client with a helpful dependency error."""
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "Falta google-cloud-bigquery. Reconstruye la imagen con "
            "`docker compose build data-checkpoint ncaa-bigquery-export`."
        ) from exc

    return bigquery.Client(project=project)


def _table_ref(table: str) -> str:
    """Return a fully qualified table reference."""
    if "." in table:
        return table.strip("`")
    return f"{PUBLIC_DATASET}.{table}"


def _safe_filename(table_ref: str) -> str:
    """Convert a table reference to a stable local filename."""
    return table_ref.replace("`", "").replace(".", "__")


def list_tables(client: Any, dataset: str = PUBLIC_DATASET) -> list[str]:
    """List available public NCAA BigQuery tables."""
    tables = sorted(table.table_id for table in client.list_tables(dataset))
    print(f"Tablas disponibles en {dataset}:")
    for table in tables:
        print(f"  {table}")
    return tables


def _season_column(schema: list[Any]) -> str | None:
    """Find the first season-like column in a table schema."""
    names = {field.name for field in schema}
    for column in SEASON_COLUMNS:
        if column in names:
            return column
    return None


def build_query(
    table_ref: str,
    schema: list[Any],
    start_season: int,
    end_season: int,
    limit: int | None,
) -> tuple[str, list[Any]]:
    """Build a parameterized SELECT for one BigQuery table."""
    from google.cloud import bigquery

    where_clauses: list[str] = []
    parameters: list[Any] = []
    season_column = _season_column(schema)
    if season_column is not None:
        where_clauses.append(f"{season_column} BETWEEN @start_season AND @end_season")
        parameters.extend(
            [
                bigquery.ScalarQueryParameter("start_season", "INT64", start_season),
                bigquery.ScalarQueryParameter("end_season", "INT64", end_season),
            ]
        )

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    limit_sql = "LIMIT @limit" if limit is not None else ""
    if limit is not None:
        parameters.append(bigquery.ScalarQueryParameter("limit", "INT64", limit))

    sql = f"""
    SELECT *
    FROM `{table_ref}`
    {where_sql}
    {limit_sql}
    """
    return sql, parameters


def export_table(
    client: Any,
    table: str,
    output_dir: Path,
    start_season: int,
    end_season: int,
    limit: int | None,
    csv: bool,
) -> dict[str, Any]:
    """Export one BigQuery table/query to parquet and optionally CSV."""
    from google.cloud import bigquery

    table_ref = _table_ref(table)
    table_obj = client.get_table(table_ref)
    sql, parameters = build_query(
        table_ref=table_ref,
        schema=table_obj.schema,
        start_season=start_season,
        end_season=end_season,
        limit=limit,
    )
    job_config = bigquery.QueryJobConfig(query_parameters=parameters)

    print(f"\nConsultando {table_ref}")
    print(f"  Temporadas: {start_season}-{end_season}")
    if limit is not None:
        print(f"  Limite: {limit:,}")

    query_job = client.query(sql, job_config=job_config)
    df = query_job.result().to_dataframe(create_bqstorage_client=False)
    df = sanitize_for_local_files(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(table_ref)
    parquet_path = output_dir / f"{stem}.parquet"
    df.to_parquet(parquet_path, index=False)

    csv_path = None
    if csv:
        csv_path = output_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=False)

    summary = {
        "table": table_ref,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path) if csv_path is not None else "",
    }
    print(f"  Filas exportadas: {len(df):,}")
    print(f"  Columnas: {len(df.columns)}")
    print(f"  Guardado: {parquet_path}")
    return summary


def sanitize_for_local_files(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BigQuery extension dtypes to portable pandas values."""
    clean = df.copy()
    for column in clean.columns:
        dtype_name = str(clean[column].dtype).lower()
        if dtype_name.startswith("db") or dtype_name in {"date32[day]", "time64[us]"}:
            clean[column] = clean[column].astype("string")
            continue
        if is_datetime64_any_dtype(clean[column]):
            clean[column] = pd.to_datetime(clean[column], errors="coerce", utc=True).dt.tz_localize(None)
    return clean


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("BIGQUERY_BILLING_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--output-dir", type=Path, default=config.NCAA_BIGQUERY_EXPORT_DIR)
    parser.add_argument("--start-season", type=int, default=2009)
    parser.add_argument("--end-season", type=int, default=2023)
    parser.add_argument("--tables", nargs="+", default=DEFAULT_TABLES)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per table for smoke tests.")
    parser.add_argument("--csv", action="store_true", help="Also write CSV files next to parquet outputs.")
    parser.add_argument("--list-tables", action="store_true", help="List available NCAA BigQuery tables and exit.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    client = _load_bigquery_client(args.project)

    if args.list_tables:
        tables = list_tables(client)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_dir / "tables_manifest.json"
        manifest_path.write_text(json.dumps({"tables": tables}, indent=2), encoding="utf-8")
        print(f"\nManifest guardado: {manifest_path}")
        return

    exports = [
        export_table(
            client=client,
            table=table,
            output_dir=args.output_dir,
            start_season=args.start_season,
            end_season=args.end_season,
            limit=args.limit,
            csv=args.csv,
        )
        for table in args.tables
    ]

    manifest = {
        "source_dataset": PUBLIC_DATASET,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "exports": exports,
    }
    manifest_path = args.output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nManifest guardado: {manifest_path}")


if __name__ == "__main__":
    main()
