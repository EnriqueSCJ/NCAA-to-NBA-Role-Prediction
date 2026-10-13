"""Graph-embedding classification experiment.

The production model remains tabular. This module builds a label-free
heterogeneous player graph and tests whether graph embeddings improve
classification metrics when appended to the current feature matrix.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix

import src.config as config
from src.models.train import (
    apply_smote,
    evaluate_model,
    get_training_data,
    save_model,
    split_data_temporal,
    train_xgboost,
)


GRAPH_EMBEDDING_DIMS = 16


@dataclass
class GraphExperimentResult:
    """Serializable metrics for the graph embedding experiment."""

    model: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    graph_dims: int
    promoted_to_best_model: bool
    baseline_macro_f1: float
    confusion_matrix: list[list[int]]
    notes: str


def _safe_text(value: Any, default: str = "unknown") -> str:
    """Normalize a value for use as a graph node id."""
    text = str(value if value is not None else "").strip().lower()
    text = "_".join(part for part in text.replace("/", "_").replace("|", "_").split())
    return text or default


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column with conservative filling."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _bucket_series(series: pd.Series, prefix: str, bins: int = 5) -> pd.Series:
    """Convert a numeric series into stable quantile bucket node ids."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
        return pd.Series(f"{prefix}:unknown", index=series.index)

    ranks = values.rank(method="first")
    try:
        buckets = pd.qcut(ranks, q=min(bins, values.nunique(dropna=True)), labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(f"{prefix}:unknown", index=series.index)

    return buckets.fillna(-1).astype(int).map(lambda value: f"{prefix}:{value}" if value >= 0 else f"{prefix}:unknown")


def _attribute_nodes(master: pd.DataFrame) -> list[list[str]]:
    """Build label-free graph attribute nodes for each player row."""
    season = _numeric_series(master, "season").astype(int)
    draft_year = _numeric_series(master, "draft_year").astype(int)
    rec_rank = _numeric_series(master, "rec_rank")
    is_unranked = _numeric_series(master, "is_unranked")

    bucket_sources = {
        "pts": _numeric_series(master, "pts_per40"),
        "usage": _numeric_series(master, "usg_pct"),
        "ts": _numeric_series(master, "ts_pct"),
        "stocks": _numeric_series(master, "stocks_per40"),
        "height": _numeric_series(master, "height_in"),
        "age": _numeric_series(master, "age_at_draft"),
        "team_power": _numeric_series(master, "team_power_rating"),
        "sos": _numeric_series(master, "strength_of_schedule"),
        "official_mpg": _numeric_series(master, "official_mpg"),
    }
    buckets = {
        name: _bucket_series(values, name)
        for name, values in bucket_sources.items()
    }

    rows: list[list[str]] = []
    for index, row in master.reset_index(drop=True).iterrows():
        team = _safe_text(row.get("college", row.get("team", "unknown")))
        conf = _safe_text(row.get("conf", "unknown"))
        position = _safe_text(row.get("position", "unknown"))
        year_class = _safe_text(row.get("yr", "unknown"))
        rec_value = float(rec_rank.iloc[index])
        unranked_flag = int(float(is_unranked.iloc[index]) >= 0.5)
        if unranked_flag:
            rec_node = "rec:unranked"
        elif rec_value >= 80:
            rec_node = "rec:elite"
        elif rec_value >= 50:
            rec_node = "rec:high"
        elif rec_value > 0:
            rec_node = "rec:known"
        else:
            rec_node = "rec:unknown"

        rows.append(
            [
                f"team_season:{team}:{int(season.iloc[index])}",
                f"conf_season:{conf}:{int(season.iloc[index])}",
                f"draft_year:{int(draft_year.iloc[index])}",
                f"position:{position}",
                f"class:{year_class}",
                rec_node,
                buckets["pts"].iloc[index],
                buckets["usage"].iloc[index],
                buckets["ts"].iloc[index],
                buckets["stocks"].iloc[index],
                buckets["height"].iloc[index],
                buckets["age"].iloc[index],
                buckets["team_power"].iloc[index],
                buckets["sos"].iloc[index],
                buckets["official_mpg"].iloc[index],
            ]
        )
    return rows


def build_graph_embeddings(master: pd.DataFrame, n_components: int = GRAPH_EMBEDDING_DIMS) -> pd.DataFrame:
    """Build graph embeddings from player-attribute incidence structure."""
    node_rows = _attribute_nodes(master)
    node_to_column: dict[str, int] = {}
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []

    for row_index, nodes in enumerate(node_rows):
        for node in nodes:
            column_index = node_to_column.setdefault(node, len(node_to_column))
            row_indices.append(row_index)
            column_indices.append(column_index)
            data.append(1.0)

    incidence = sparse.csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(node_rows), len(node_to_column)),
        dtype=np.float32,
    )
    row_norms = np.sqrt(incidence.multiply(incidence).sum(axis=1)).A1
    row_norms[row_norms == 0] = 1.0
    incidence = sparse.diags(1.0 / row_norms).dot(incidence)

    max_components = max(1, min(n_components, incidence.shape[0] - 1, incidence.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=config.RANDOM_STATE)
    embedding = svd.fit_transform(incidence)

    if max_components < n_components:
        padding = np.zeros((embedding.shape[0], n_components - max_components), dtype=embedding.dtype)
        embedding = np.hstack([embedding, padding])

    return pd.DataFrame(
        embedding,
        columns=[f"graph_emb_{index:02d}" for index in range(n_components)],
        index=master.index,
    )


def load_master_for_graph() -> pd.DataFrame:
    """Load the processed master table aligned with X/y."""
    path = config.DATA_PROCESSED / "master_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed master table not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _load_baseline_macro_f1() -> float:
    """Return current production XGBoost validation Macro F1 when available."""
    path = config.REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        return 0.0
    with path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    xgboost_metrics = metrics.get("xgboost_model", {})
    return float(xgboost_metrics.get("validation_macro_f1", xgboost_metrics.get("macro_f1", 0.0)))


def _save_graph_results(result: GraphExperimentResult) -> None:
    """Persist graph experiment result artifacts."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "graph_embedding_experiment.json"
    csv_path = config.REPORTS_DIR / "graph_embedding_experiment.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2, sort_keys=True)
    pd.DataFrame([asdict(result)]).to_csv(csv_path, index=False)


def run_graph_embedding_experiment(promote_if_better: bool = True) -> GraphExperimentResult:
    """Train XGBoost with graph embeddings and compare against production metrics."""
    X, y, years, source = get_training_data()
    if source != "processed":
        raise RuntimeError("Graph experiment requires processed training data.")

    master = load_master_for_graph()
    if len(master) != len(X):
        raise ValueError("master_table.csv and X.csv are not row-aligned")

    graph_embeddings = build_graph_embeddings(master, n_components=GRAPH_EMBEDDING_DIMS)
    X_graph = pd.concat([X.reset_index(drop=True), graph_embeddings.reset_index(drop=True)], axis=1)

    X_trains, y_trains = split_data_temporal(X_graph, y, years)
    X_train_smote, y_train_smote = apply_smote(X_trains[0], y_trains[0])
    model, metrics = train_xgboost(
        (X_train_smote, X_trains[1], X_trains[2]),
        (y_train_smote, y_trains[1], y_trains[2]),
    )

    y_pred = model.predict(X_trains[2])
    cm = confusion_matrix(y_trains[2], y_pred, labels=config.ROLE_LABELS)
    baseline_macro = _load_baseline_macro_f1()
    improved = bool(metrics.get("validation_macro_f1", metrics["macro_f1"]) > baseline_macro)

    if promote_if_better and improved:
        save_model(model, "graph_embedding_xgboost_model.pkl")
        save_model(model, "best_model.pkl")
    elif not improved:
        # Keep a non-promoted artifact for auditability, not for serving.
        with (config.MODELS_DIR / "graph_embedding_xgboost_discarded.pkl").open("wb") as file:
            pickle.dump(model, file)

    result = GraphExperimentResult(
        model="xgboost_graph_embeddings",
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        graph_dims=GRAPH_EMBEDDING_DIMS,
        promoted_to_best_model=bool(promote_if_better and improved),
        baseline_macro_f1=baseline_macro,
        confusion_matrix=cm.astype(int).tolist(),
        notes=(
            "Graph embeddings are built from label-free player-context nodes. "
            "The experiment is discarded when validation Macro F1 does not exceed the production XGBoost baseline."
        ),
    )
    _save_graph_results(result)

    print("\n=== Graph embedding experiment ===")
    print(pd.DataFrame([asdict(result)]).to_string(index=False))
    return result


def main() -> None:
    run_graph_embedding_experiment(promote_if_better=True)


if __name__ == "__main__":
    main()
