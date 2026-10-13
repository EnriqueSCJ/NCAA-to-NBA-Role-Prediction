"""Feature ablations for data-quality checks.

The experiment is read-only with respect to production model artifacts: it
trains temporary models and writes a comparison report under ``reports/``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import src.config as config
from src.models.train import (
    apply_smote,
    split_data_temporal,
    train_catboost,
    train_xgboost,
)
from src.models.train_two_stage import (
    _conditioned_role_data,
    combine_two_stage_predictions,
    evaluate_multiclass_predictions,
    make_gate_target,
    train_gate_candidates,
    train_role_conditioned_candidates,
    tune_gate_threshold_for_combined_macro_f1,
)


OFFICIAL_FEATURES = [
    "official_games_z",
    "official_mpg_z",
    "official_start_rate_z",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_processed() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X = pd.read_csv(config.DATA_PROCESSED / "X.csv")
    y = pd.read_csv(config.DATA_PROCESSED / "y.csv").squeeze("columns")
    master = pd.read_csv(config.DATA_PROCESSED / "master_table.csv", low_memory=False)
    return X.select_dtypes(include=[np.number]).copy(), y, master[["draft_year"]].copy()


def _feature_variant(X: pd.DataFrame, variant: str) -> pd.DataFrame:
    X_variant = X.copy()
    if variant == "full_features":
        return X_variant
    if variant == "official_zeroed":
        for column in OFFICIAL_FEATURES:
            if column in X_variant.columns:
                X_variant[column] = 0.0
        return X_variant
    raise ValueError(f"Unknown feature variant: {variant}")


def _class_support(y: pd.Series) -> dict[str, int]:
    return y.value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict()


def _starter_plus_recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.astype(str).eq("STARTER_PLUS")
    if not mask.any():
        return 0.0
    return float(pd.Series(y_pred, index=y_true.index).loc[mask].eq("STARTER_PLUS").mean())


def _train_two_stage_temporary(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
) -> dict[str, Any]:
    X_trains, y_trains = split_data_temporal(X, y, years)
    X_train, X_val, X_test = X_trains
    _, y_val, y_test = y_trains

    X_role, y_role, years_role = _conditioned_role_data(X, y, years)
    X_role_trains, y_role_trains = split_data_temporal(X_role, y_role, years_role)
    role_candidates, role_metrics = train_role_conditioned_candidates(X_role_trains, y_role_trains)
    best_role_name = max(role_metrics, key=lambda name: role_metrics[name]["validation_macro_f1"])
    best_role_model = role_candidates[best_role_name]

    y_gate = make_gate_target(y)
    gate_trains = split_data_temporal(X, y_gate, years)[1]
    gate_candidates = train_gate_candidates(X_trains, gate_trains)

    combined_validation: dict[str, dict[str, float]] = {}
    for gate_name, gate_model in gate_candidates.items():
        threshold, validation_macro_f1 = tune_gate_threshold_for_combined_macro_f1(
            gate_model=gate_model,
            role_model=best_role_model,
            X_val=X_val,
            y_val=y_val,
        )
        combined_validation[gate_name] = {
            "threshold": threshold,
            "validation_macro_f1": validation_macro_f1,
        }

    best_gate_name = max(combined_validation, key=lambda name: combined_validation[name]["validation_macro_f1"])
    best_gate_model = gate_candidates[best_gate_name]
    best_threshold = combined_validation[best_gate_name]["threshold"]

    y_val_pred = combine_two_stage_predictions(best_gate_model, best_role_model, X_val, best_threshold)
    y_test_pred = combine_two_stage_predictions(best_gate_model, best_role_model, X_test, best_threshold)
    validation_metrics = evaluate_multiclass_predictions(y_val, y_val_pred)
    test_metrics = evaluate_multiclass_predictions(y_test, y_test_pred)

    return {
        "model": "two_stage_temporary",
        "selected_gate_model": best_gate_name,
        "selected_role_model": best_role_name,
        "gate_threshold": best_threshold,
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "validation_starter_plus_recall": _starter_plus_recall(y_val, y_val_pred),
        "test_starter_plus_recall": _starter_plus_recall(y_test, y_test_pred),
        "role_conditioned_metrics": role_metrics,
        "combined_validation": combined_validation,
    }


def _train_base_temporary(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
) -> list[dict[str, Any]]:
    X_trains, y_trains = split_data_temporal(X, y, years)
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    train_sets = {
        "no_smote": (X_trains, y_trains),
        "production_smote": ((X_train_smote, X_val, X_test), (y_train_smote, y_val, y_test)),
    }

    rows: list[dict[str, Any]] = []
    for sampling_name, (X_train_tuple, y_train_tuple) in train_sets.items():
        for name, train_func in [("xgboost", train_xgboost), ("catboost", train_catboost)]:
            model, metrics = train_func(X_train_tuple, y_train_tuple)
            y_val_pred = pd.Series(np.asarray(model.predict(X_val)).ravel(), index=y_val.index)
            y_test_pred = pd.Series(np.asarray(model.predict(X_test)).ravel(), index=y_test.index)
            rows.append(
                {
                    "model": name,
                    "sampling": sampling_name,
                    "validation_macro_f1": metrics["validation_macro_f1"],
                    "validation_balanced_accuracy": metrics["validation_balanced_accuracy"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_balanced_accuracy": metrics["balanced_accuracy"],
                    "test_accuracy": metrics["accuracy"],
                    "validation_starter_plus_recall": _starter_plus_recall(y_val, y_val_pred),
                    "test_starter_plus_recall": _starter_plus_recall(y_test, y_test_pred),
                }
            )
    return rows


def run_feature_ablation_experiment() -> dict[str, Any]:
    X, y, years = _load_processed()
    _, y_trains = split_data_temporal(X, y, years)
    payload: dict[str, Any] = {
        "split_distribution": {
            "train": _class_support(y_trains[0]),
            "validation": _class_support(y_trains[1]),
            "test": _class_support(y_trains[2]),
        },
        "official_feature_columns": [column for column in OFFICIAL_FEATURES if column in X.columns],
        "variants": [],
    }

    rows: list[dict[str, Any]] = []
    for variant in ["full_features", "official_zeroed"]:
        X_variant = _feature_variant(X, variant)
        variant_rows = _train_base_temporary(X_variant, y, years)
        variant_rows.append(_train_two_stage_temporary(X_variant, y, years))
        for row in variant_rows:
            row["feature_variant"] = variant
            rows.append(row)
        payload["variants"].append({"name": variant, "results": variant_rows})

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "feature_ablation_experiment.json"
    csv_path = config.REPORTS_DIR / "feature_ablation_experiment.csv"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return payload


def main() -> None:
    payload = run_feature_ablation_experiment()
    print("=== Feature ablation experiment ===")
    for variant in payload["variants"]:
        print(f"\n{variant['name']}")
        rows = pd.DataFrame(variant["results"])
        print(
            rows[
                [
                    "model",
                    "sampling",
                    "validation_macro_f1",
                    "test_macro_f1",
                    "validation_balanced_accuracy",
                    "test_balanced_accuracy",
                    "test_starter_plus_recall",
                ]
            ].to_string(index=False)
        )
    print(f"\nJSON: {config.REPORTS_DIR / 'feature_ablation_experiment.json'}")
    print(f"CSV:  {config.REPORTS_DIR / 'feature_ablation_experiment.csv'}")


if __name__ == "__main__":
    main()
