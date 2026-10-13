"""Experiment that groups BENCH and ROTATION into one middle NBA role."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

import src.config as config
from src.models.explain import load_model
from src.models.grouped_model import (
    GROUPED_LABELS,
    GroupedXGBoostClassifier,
    RoleGroupCascadeClassifier,
    group_roles,
)
from src.models.train import save_model
from src.models.train import get_training_data, split_data_temporal


@dataclass
class GroupedExperimentResult:
    """Metrics for one grouped-label experiment."""

    name: str
    split: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    distribution: dict[str, int]
    confusion_matrix: list[list[int]]
    classification_report: dict[str, Any]
    notes: str


@dataclass
class CascadeExperimentResult:
    """Fine 4-class metrics for the grouped-to-fine cascade."""

    name: str
    split: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    distribution: dict[str, int]
    confusion_matrix: list[list[int]]
    notes: str


def _encode_grouped_roles(y: pd.Series) -> np.ndarray:
    """Encode grouped labels."""
    mapping = {label: index for index, label in enumerate(GROUPED_LABELS)}
    encoded = y.map(mapping)
    if encoded.isna().any():
        unknown = sorted(set(y[encoded.isna()]))
        raise ValueError(f"Etiquetas agrupadas desconocidas: {unknown}")
    return encoded.astype(int).to_numpy()


def _grouped_smote_sampling_strategy(y_train: pd.Series) -> dict[str, int] | str:
    """Build a SMOTE strategy for grouped labels."""
    counts = y_train.value_counts()
    if len(counts) < 2:
        return "auto"

    majority_count = int(counts.max())
    target_count = max(2, int(round(majority_count * config.SMOTE_SAMPLING_STRATEGY)))
    strategy = {
        str(label): target_count
        for label, count in counts.items()
        if count < target_count
    }
    return strategy or "auto"


def apply_grouped_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to grouped labels with readable grouped-class logging."""
    counts = y_train.value_counts()
    min_count = int(counts.min())
    if len(counts) < 2 or min_count < 2:
        print("SMOTE omitido para etiquetas agrupadas: no hay suficientes muestras por clase.")
        return X_train.copy(), y_train.copy()

    smote = SMOTE(
        sampling_strategy=_grouped_smote_sampling_strategy(y_train),
        random_state=config.RANDOM_STATE,
        k_neighbors=min(5, min_count - 1),
    )
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    y_balanced = pd.Series(y_balanced, name=y_train.name or "grouped_target_role")

    print("SMOTE aplicado a etiquetas agrupadas.")
    print("Distribucion agrupada original:")
    print(counts.reindex(GROUPED_LABELS, fill_value=0).to_string())
    print("Distribucion agrupada balanceada:")
    print(y_balanced.value_counts().reindex(GROUPED_LABELS, fill_value=0).to_string())
    return X_balanced, y_balanced


def _evaluate_grouped_predictions(
    name: str,
    split: str,
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    notes: str,
) -> GroupedExperimentResult:
    """Evaluate grouped-label predictions."""
    y_pred_series = pd.Series(y_pred, index=y_true.index, name="predicted_grouped_role").astype(str)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred_series)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_series)),
        "macro_f1": float(f1_score(y_true, y_pred_series, labels=GROUPED_LABELS, average="macro", zero_division=0)),
    }
    cm = confusion_matrix(y_true, y_pred_series, labels=GROUPED_LABELS)
    report = classification_report(
        y_true,
        y_pred_series,
        labels=GROUPED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return GroupedExperimentResult(
        name=name,
        split=split,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        distribution=y_true.value_counts().reindex(GROUPED_LABELS, fill_value=0).astype(int).to_dict(),
        confusion_matrix=cm.astype(int).tolist(),
        classification_report=report,
        notes=notes,
    )


def _evaluate_fine_predictions(
    name: str,
    split: str,
    y_true: pd.Series,
    y_pred: pd.Series | np.ndarray,
    notes: str,
) -> CascadeExperimentResult:
    """Evaluate 4-class predictions for cascade comparison."""
    y_pred_series = pd.Series(y_pred, index=y_true.index, name="predicted_role").astype(str)
    cm = confusion_matrix(y_true, y_pred_series, labels=config.ROLE_LABELS)
    return CascadeExperimentResult(
        name=name,
        split=split,
        accuracy=float(accuracy_score(y_true, y_pred_series)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred_series)),
        macro_f1=float(f1_score(y_true, y_pred_series, labels=config.ROLE_LABELS, average="macro", zero_division=0)),
        distribution=y_true.value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict(),
        confusion_matrix=cm.astype(int).tolist(),
        notes=notes,
    )


def train_grouped_xgboost(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
    *,
    max_depth: int = 4,
    eta: float = 0.05,
    num_boost_round: int = 400,
) -> GroupedXGBoostClassifier:
    """Train the conservative final XGBoost setup on grouped role labels."""
    import xgboost as xgb

    X_train, X_val, _ = X_trains
    y_train, y_val, _ = y_trains
    dtrain = xgb.DMatrix(
        X_train,
        label=_encode_grouped_roles(y_train),
        feature_names=list(X_train.columns),
    )
    dval = xgb.DMatrix(
        X_val,
        label=_encode_grouped_roles(y_val),
        feature_names=list(X_val.columns),
    )
    booster = xgb.train(
        params={
            "objective": "multi:softprob",
            "num_class": len(GROUPED_LABELS),
            "max_depth": max_depth,
            "eta": eta,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "seed": config.RANDOM_STATE,
        },
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    return GroupedXGBoostClassifier(booster=booster, classes=GROUPED_LABELS)


def _save_results(
    grouped_results: list[GroupedExperimentResult],
    cascade_results: list[CascadeExperimentResult],
    promoted_to_best_model: bool,
) -> None:
    """Persist grouped experiment artifacts."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "bench_rotation_grouping_experiment.json"
    csv_path = config.REPORTS_DIR / "bench_rotation_grouping_experiment.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "grouped_labels": GROUPED_LABELS,
                "mapping": {"BENCH": "NBA_ROLE_PLAYER", "ROTATION": "NBA_ROLE_PLAYER"},
                "grouped_results": [result.__dict__ for result in grouped_results],
                "cascade_results": [result.__dict__ for result in cascade_results],
                "promoted_to_best_model": promoted_to_best_model,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    flat_rows = []
    for result in grouped_results:
        row = result.__dict__.copy()
        row.pop("classification_report")
        row["result_type"] = "grouped"
        row["distribution"] = json.dumps(row["distribution"], sort_keys=True)
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        flat_rows.append(row)
    for result in cascade_results:
        row = result.__dict__.copy()
        row["result_type"] = "cascade_fine"
        row["distribution"] = json.dumps(row.pop("distribution"), sort_keys=True)
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"Grouped experiment details saved: {json_path}")
    print(f"Grouped experiment table saved: {csv_path}")


def run_grouping_experiment(promote_if_better: bool = True) -> list[GroupedExperimentResult]:
    """Run grouped-label evaluation and retraining."""
    X, y, years, source = get_training_data()
    if source != "processed":
        raise RuntimeError("Bench/rotation grouping experiment requires processed data artifacts.")

    X_trains, y_trains = split_data_temporal(X, y, years)
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains
    y_val_grouped = group_roles(y_val)
    y_test_grouped = group_roles(y_test)
    y_val_grouped.index = y_val.index
    y_test_grouped.index = y_test.index

    current_model = load_model()
    current_validation_grouped_pred = group_roles(current_model.predict(X_val))
    current_validation_grouped_pred.index = y_val.index
    current_validation_result = _evaluate_grouped_predictions(
        name="current_best_model_mapped_to_grouped_roles",
        split="validation",
        y_true=y_val_grouped,
        y_pred=current_validation_grouped_pred,
        notes="Current best 4-class model predictions mapped after inference.",
    )
    current_grouped_pred = group_roles(current_model.predict(X_test))
    current_grouped_pred.index = y_test.index
    current_result = _evaluate_grouped_predictions(
        name="current_best_model_mapped_to_grouped_roles",
        split="test",
        y_true=y_test_grouped,
        y_pred=current_grouped_pred,
        notes="Best 4-class model predictions mapped after inference: BENCH and ROTATION collapse into NBA_ROLE_PLAYER.",
    )

    y_train_grouped = group_roles(y_train)
    y_train_grouped.index = y_train.index
    grouped_model = train_grouped_xgboost(
        (X_train, X_val, X_test),
        (y_train_grouped, y_val_grouped, y_test_grouped),
    )
    grouped_validation_pred = grouped_model.predict(X_val)
    trained_grouped_validation_result = _evaluate_grouped_predictions(
        name="xgboost_trained_on_grouped_roles",
        split="validation",
        y_true=y_val_grouped,
        y_pred=grouped_validation_pred,
        notes="Conservative XGBoost trained directly on grouped labels without SMOTE.",
    )
    grouped_pred = grouped_model.predict(X_test)
    trained_grouped_result = _evaluate_grouped_predictions(
        name="xgboost_trained_on_grouped_roles",
        split="test",
        y_true=y_test_grouped,
        y_pred=grouped_pred,
        notes="Conservative XGBoost trained directly on grouped labels without SMOTE.",
    )

    cascade_model = RoleGroupCascadeClassifier(grouped_model=grouped_model, fine_model=current_model)
    current_validation = _evaluate_fine_predictions(
        name="current_best_model",
        split="validation",
        y_true=y_val,
        y_pred=current_model.predict(X_val),
        notes="Current selected 4-class model.",
    )
    cascade_validation = _evaluate_fine_predictions(
        name="grouped_to_fine_cascade",
        split="validation",
        y_true=y_val,
        y_pred=cascade_model.predict(X_val),
        notes="Grouped model decides coarse role; fine model splits BENCH vs ROTATION.",
    )
    cascade_test = _evaluate_fine_predictions(
        name="grouped_to_fine_cascade",
        split="test",
        y_true=y_test,
        y_pred=cascade_model.predict(X_test),
        notes="Grouped model decides coarse role; fine model splits BENCH vs ROTATION.",
    )

    promoted = bool(promote_if_better)
    if promoted:
        save_model(grouped_model, "grouped_xgboost_model.pkl")
        save_model(grouped_model, "best_model.pkl")

    results = [
        current_validation_result,
        current_result,
        trained_grouped_validation_result,
        trained_grouped_result,
    ]
    cascade_results = [current_validation, cascade_validation, cascade_test]
    _save_results(results, cascade_results, promoted_to_best_model=promoted)

    print("\n=== BENCH/ROTATION grouping experiment ===")
    for result in results:
        print(
            f"{result.name}: "
            f"split={result.split}, "
            f"accuracy={result.accuracy:.4f}, "
            f"balanced_accuracy={result.balanced_accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}"
        )
        print(pd.DataFrame(result.confusion_matrix, index=GROUPED_LABELS, columns=GROUPED_LABELS).to_string())
    print("\n=== Grouped-to-fine cascade ===")
    for result in cascade_results:
        print(
            f"{result.name} [{result.split}]: "
            f"accuracy={result.accuracy:.4f}, "
            f"balanced_accuracy={result.balanced_accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}"
        )
        print(pd.DataFrame(result.confusion_matrix, index=config.ROLE_LABELS, columns=config.ROLE_LABELS).to_string())
    if promoted:
        print("Promoted direct 3-class grouped model to models/best_model.pkl.")

    return results


def main() -> None:
    """CLI entrypoint."""
    run_grouping_experiment()


if __name__ == "__main__":
    main()
