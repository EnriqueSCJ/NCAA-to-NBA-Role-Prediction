"""Run controlled modeling experiments without overwriting the stable pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

import src.config as config
from src.data.build_master import build_master_table, build_processed_features
from src.models.ordinal_model import train_ordinal_xgboost
from src.models.probability_adjusted_model import ProbabilityAdjustedRoleClassifier
from src.models.train import (
    XGBoostRoleClassifier,
    apply_smote,
    get_training_data,
    save_model,
    split_data_temporal,
    train_xgboost,
)


DEFAULT_XGBOOST_TUNING_TRIALS = 30


@dataclass
class ExperimentResult:
    """Compact record for one experiment."""

    name: str
    target_horizon: int
    target_start_offset: int
    model_type: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    test_distribution: dict[str, int]
    confusion_matrix: list[list[int]]
    comparable_to_current_target: bool
    notes: str


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return the shared imbalance-aware metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)),
    }


def _record_result(
    name: str,
    target_horizon: int,
    target_start_offset: int,
    model_type: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    comparable_to_current_target: bool,
    notes: str,
) -> ExperimentResult:
    """Evaluate and package one fitted model."""
    y_pred = np.asarray(model.predict(X_test)).ravel()
    metrics = _evaluate_predictions(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=config.ROLE_LABELS)
    return ExperimentResult(
        name=name,
        target_horizon=target_horizon,
        target_start_offset=target_start_offset,
        model_type=model_type,
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        test_distribution=y_test.value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict(),
        confusion_matrix=cm.astype(int).tolist(),
        comparable_to_current_target=comparable_to_current_target,
        notes=notes,
    )


def _predict_labels_from_probabilities(probabilities: np.ndarray, labels: list[str]) -> np.ndarray:
    """Return labels from a softprob matrix."""
    values = np.asarray(probabilities)
    if values.ndim == 1:
        class_ids = values.astype(int)
    else:
        class_ids = values.argmax(axis=1)
    return np.asarray(labels)[class_ids]


def _encode_roles(y: pd.Series, labels: list[str] | None = None) -> np.ndarray:
    """Encode role labels in the project class order."""
    role_labels = labels or config.ROLE_LABELS
    mapping = {label: index for index, label in enumerate(role_labels)}
    encoded = y.map(mapping)
    if encoded.isna().any():
        unknown = sorted(set(y[encoded.isna()]))
        raise ValueError(f"Etiquetas desconocidas para tuning: {unknown}")
    return encoded.astype(int).to_numpy()


def _class_weight_params(trial: Any) -> dict[str, float]:
    """Suggest class multipliers used as XGBoost row weights."""
    return {
        "NON_NBA": trial.suggest_float("non_nba_weight", 0.70, 1.30),
        "BENCH": trial.suggest_float("bench_weight", 0.80, 2.50),
        "ROTATION": trial.suggest_float("rotation_weight", 0.80, 2.80),
        "STARTER_PLUS": trial.suggest_float("starter_plus_weight", 0.80, 4.00),
    }


def _sample_weights(y: pd.Series, class_weights: dict[str, float]) -> np.ndarray:
    """Build row weights from role labels."""
    return y.astype(str).map(class_weights).fillna(1.0).astype(float).to_numpy()


def _suggest_xgboost_tuning_values(trial: Any) -> dict[str, float | int]:
    """Suggest raw search-space values for one XGBoost configuration."""
    values: dict[str, float | int] = {
        "eta": trial.suggest_float("eta", 0.025, 0.180, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
        "gamma": trial.suggest_float("gamma", 0.0, 4.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.50, 8.00, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "num_boost_round": trial.suggest_int("num_boost_round", 180, 450, step=30),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 15, 45, step=5),
    }
    values.update(
        {
            "non_nba_weight": trial.suggest_float("non_nba_weight", 0.70, 1.30),
            "bench_weight": trial.suggest_float("bench_weight", 0.80, 2.50),
            "rotation_weight": trial.suggest_float("rotation_weight", 0.80, 2.80),
            "starter_plus_weight": trial.suggest_float("starter_plus_weight", 0.80, 4.00),
        }
    )
    return values


def _materialize_xgboost_tuning_params(
    values: dict[str, float | int],
) -> tuple[dict[str, Any], int, int, dict[str, float]]:
    """Convert search-space values into XGBoost params and row weights."""
    class_weights = {
        "NON_NBA": float(values["non_nba_weight"]),
        "BENCH": float(values["bench_weight"]),
        "ROTATION": float(values["rotation_weight"]),
        "STARTER_PLUS": float(values["starter_plus_weight"]),
    }
    params = {
        "objective": "multi:softprob",
        "num_class": len(config.ROLE_LABELS),
        "eval_metric": "mlogloss",
        "eta": float(values["eta"]),
        "max_depth": int(values["max_depth"]),
        "min_child_weight": float(values["min_child_weight"]),
        "subsample": float(values["subsample"]),
        "colsample_bytree": float(values["colsample_bytree"]),
        "gamma": float(values["gamma"]),
        "lambda": float(values["reg_lambda"]),
        "alpha": float(values["reg_alpha"]),
        "seed": config.RANDOM_STATE,
        "tree_method": "hist",
    }
    return (
        params,
        int(values["num_boost_round"]),
        int(values["early_stopping_rounds"]),
        class_weights,
    )


def _suggest_xgboost_tuning_params(trial: Any) -> tuple[dict[str, Any], int, int, dict[str, float]]:
    """Suggest one XGBoost configuration for validation Macro F1 search."""
    return _materialize_xgboost_tuning_params(_suggest_xgboost_tuning_values(trial))


def _train_tuned_xgboost(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
    n_trials: int,
) -> tuple[XGBoostRoleClassifier, dict[str, Any]]:
    """Tune XGBoost hyperparameters on validation Macro F1."""
    import xgboost as xgb

    if n_trials < 1:
        raise ValueError("n_trials debe ser al menos 1")

    X_train, X_val, _ = X_trains
    y_train, y_val, _ = y_trains
    y_train_encoded = _encode_roles(y_train)
    y_val_encoded = _encode_roles(y_val)

    def fit_booster(values: dict[str, float | int]) -> Any:
        params, num_boost_round, early_stopping_rounds, class_weights = _materialize_xgboost_tuning_params(values)
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train_encoded,
            weight=_sample_weights(y_train, class_weights),
            feature_names=list(X_train.columns),
        )
        dval = xgb.DMatrix(
            X_val,
            label=y_val_encoded,
            feature_names=list(X_val.columns),
        )
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        return booster

    dval_for_score = xgb.DMatrix(
        X_val,
        label=y_val_encoded,
        feature_names=list(X_val.columns),
    )

    def score_booster(booster: Any) -> float:
        probabilities = booster.predict(dval_for_score)
        y_pred = _predict_labels_from_probabilities(probabilities, config.ROLE_LABELS)
        return float(f1_score(y_val, y_pred, labels=config.ROLE_LABELS, average="macro", zero_division=0))

    trials: list[dict[str, Any]] = []
    search_method = "optuna_tpe"
    best_score = -1.0
    best_values: dict[str, float | int] | None = None

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            values = _suggest_xgboost_tuning_values(trial)
            booster = fit_booster(values)
            score = score_booster(booster)
            trial.set_user_attr("best_iteration", int(getattr(booster, "best_iteration", values["num_boost_round"])))
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_score = float(study.best_value)
        best_values = dict(study.best_params)
        trials = [
            {
                "number": int(trial.number),
                "value": float(trial.value) if trial.value is not None else None,
                "state": str(trial.state),
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
            }
            for trial in study.trials
        ]
    except ModuleNotFoundError:
        search_method = "sklearn_parameter_sampler"
        from scipy.stats import loguniform, uniform
        from sklearn.model_selection import ParameterSampler

        distributions = {
            "eta": loguniform(0.025, 0.180),
            "max_depth": list(range(3, 8)),
            "min_child_weight": loguniform(1.0, 10.0),
            "subsample": uniform(0.65, 0.30),
            "colsample_bytree": uniform(0.65, 0.30),
            "gamma": uniform(0.0, 4.0),
            "reg_lambda": loguniform(0.50, 8.00),
            "reg_alpha": uniform(0.0, 3.0),
            "num_boost_round": list(range(180, 451, 30)),
            "early_stopping_rounds": list(range(15, 46, 5)),
            "non_nba_weight": uniform(0.70, 0.60),
            "bench_weight": uniform(0.80, 1.70),
            "rotation_weight": uniform(0.80, 2.00),
            "starter_plus_weight": uniform(0.80, 3.20),
        }
        sampler = ParameterSampler(
            distributions,
            n_iter=n_trials,
            random_state=config.RANDOM_STATE,
        )
        for number, raw_values in enumerate(sampler):
            values = {
                key: int(value) if key in {"max_depth", "num_boost_round", "early_stopping_rounds"} else float(value)
                for key, value in raw_values.items()
            }
            booster = fit_booster(values)
            score = score_booster(booster)
            best_iteration = int(getattr(booster, "best_iteration", values["num_boost_round"]))
            trials.append(
                {
                    "number": number,
                    "value": score,
                    "state": "COMPLETE",
                    "params": values,
                    "user_attrs": {"best_iteration": best_iteration},
                }
            )
            if score > best_score:
                best_score = score
                best_values = values

    if best_values is None:
        raise RuntimeError("No se pudo completar ningun trial de tuning XGBoost")

    best_params, num_boost_round, early_stopping_rounds, class_weights = _materialize_xgboost_tuning_params(best_values)
    booster = fit_booster(best_values)
    model = XGBoostRoleClassifier(booster=booster, classes=config.ROLE_LABELS)
    metadata = {
        "n_trials": int(n_trials),
        "search_method": search_method,
        "best_validation_macro_f1": float(best_score),
        "best_search_params": best_values,
        "best_params": best_params,
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "class_weights": class_weights,
        "best_iteration": int(getattr(booster, "best_iteration", num_boost_round)),
        "trials": trials,
    }
    return model, metadata


def _tune_probability_multipliers(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    """Tune class probability multipliers on validation Macro F1."""
    grid = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    probabilities = np.asarray(model.predict_proba(X_val), dtype=float)
    labels = [str(label) for label in getattr(model, "classes_", config.ROLE_LABELS)]
    aligned = np.zeros((len(X_val), len(config.ROLE_LABELS)), dtype=float)
    for source_index, label in enumerate(labels):
        if label in config.ROLE_LABELS and source_index < probabilities.shape[1]:
            aligned[:, config.ROLE_LABELS.index(label)] = probabilities[:, source_index]

    best_score = -1.0
    best_multipliers = {role: 1.0 for role in config.ROLE_LABELS}
    for bench_multiplier in grid:
        for rotation_multiplier in grid:
            for starter_multiplier in grid:
                multipliers = np.array([1.0, bench_multiplier, rotation_multiplier, starter_multiplier])
                y_pred = np.array(config.ROLE_LABELS)[(aligned * multipliers).argmax(axis=1)]
                score = f1_score(y_val, y_pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)
                if score > best_score:
                    best_score = float(score)
                    best_multipliers = {
                        "NON_NBA": 1.0,
                        "BENCH": bench_multiplier,
                        "ROTATION": rotation_multiplier,
                        "STARTER_PLUS": starter_multiplier,
                    }
    print(f"Best validation probability multipliers: {best_multipliers} (macro_f1={best_score:.4f})")
    return best_multipliers


def _training_blocks(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
) -> tuple[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tuple[pd.Series, pd.Series, pd.Series],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    tuple[pd.Series, pd.Series, pd.Series],
]:
    """Return original and SMOTE-balanced temporal splits."""
    X_trains, y_trains = split_data_temporal(X, y, years)
    X_train_smote, y_train_smote = apply_smote(X_trains[0], y_trains[0])
    return X_trains, y_trains, (X_train_smote, X_trains[1], X_trains[2]), (y_train_smote, y_trains[1], y_trains[2])


def _load_processed_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the current processed dataset used by the production model."""
    X, y, years, source = get_training_data()
    if source != "processed":
        raise RuntimeError("Experiment lab requires processed data artifacts.")
    return X, y, years


def _build_dataset_for_horizon(
    horizon: int,
    start_offset: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build an in-memory dataset with a different active-season target slice."""
    previous_horizon = config.NBA_TARGET_ACTIVE_SEASONS
    previous_offset = config.NBA_TARGET_ACTIVE_SEASON_OFFSET
    try:
        config.NBA_TARGET_ACTIVE_SEASONS = horizon
        config.NBA_TARGET_ACTIVE_SEASON_OFFSET = start_offset
        master = build_master_table()
        X, y = build_processed_features(master)
        years = master[["draft_year"]].copy()
        return X, y, years
    finally:
        config.NBA_TARGET_ACTIVE_SEASONS = previous_horizon
        config.NBA_TARGET_ACTIVE_SEASON_OFFSET = previous_offset


def _run_xgboost_experiment(
    name: str,
    horizon: int,
    start_offset: int,
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
    comparable_to_current_target: bool,
    notes: str,
) -> tuple[ExperimentResult, Any]:
    """Train the standard XGBoost model for one dataset."""
    _, y_trains, X_trains_smote, y_trains_smote = _training_blocks(X, y, years)
    model, _ = train_xgboost(X_trains_smote, y_trains_smote)
    result = _record_result(
        name=name,
        target_horizon=horizon,
        target_start_offset=start_offset,
        model_type="xgboost_multiclass",
        model=model,
        X_test=X_trains_smote[2],
        y_test=y_trains[2],
        comparable_to_current_target=comparable_to_current_target,
        notes=notes,
    )
    return result, model


def _run_ordinal_experiment(
    name: str,
    horizon: int,
    start_offset: int,
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
    comparable_to_current_target: bool,
    notes: str,
) -> tuple[ExperimentResult, Any]:
    """Train the cumulative ordinal XGBoost model for one dataset."""
    _, y_trains, X_trains_smote, y_trains_smote = _training_blocks(X, y, years)
    model = train_ordinal_xgboost(X_trains_smote, y_trains_smote)
    result = _record_result(
        name=name,
        target_horizon=horizon,
        target_start_offset=start_offset,
        model_type="ordinal_xgboost",
        model=model,
        X_test=X_trains_smote[2],
        y_test=y_trains[2],
        comparable_to_current_target=comparable_to_current_target,
        notes=notes,
    )
    return result, model


def _run_probability_adjusted_experiment(
    name: str,
    horizon: int,
    base_model: Any,
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[ExperimentResult, Any]:
    """Tune validation multipliers and evaluate adjusted probabilities."""
    multipliers = _tune_probability_multipliers(base_model, X_trains[1], y_trains[1])
    model = ProbabilityAdjustedRoleClassifier(base_model=base_model, class_multipliers=multipliers)
    result = _record_result(
        name=name,
        target_horizon=horizon,
        target_start_offset=0,
        model_type="xgboost_probability_adjusted",
        model=model,
        X_test=X_trains[2],
        y_test=y_trains[2],
        comparable_to_current_target=True,
        notes=f"Ajuste de probabilidades aprendido en validacion: {multipliers}",
    )
    return result, model


def _persist_results(results: list[ExperimentResult]) -> None:
    """Write experiment tables to reports/."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [result.__dict__ for result in results]
    json_path = config.REPORTS_DIR / "experiment_lab_results.json"
    csv_path = config.REPORTS_DIR / "experiment_lab_results.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, sort_keys=True)

    flat_rows = []
    for result in results:
        row = result.__dict__.copy()
        row["test_distribution"] = json.dumps(row["test_distribution"], sort_keys=True)
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        flat_rows.append(row)
    pd.DataFrame(flat_rows).sort_values("macro_f1", ascending=False).to_csv(csv_path, index=False)
    print(f"Experiment results saved: {csv_path}")
    print(f"Experiment details saved: {json_path}")


def _load_persisted_results() -> list[ExperimentResult]:
    """Load previously persisted experiment results when present."""
    path = config.REPORTS_DIR / "experiment_lab_results.json"
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    return [ExperimentResult(**row) for row in rows]


def _merge_results(existing: list[ExperimentResult], updates: list[ExperimentResult]) -> list[ExperimentResult]:
    """Merge results by experiment name, preserving old rows not rerun."""
    by_name = {result.name: result for result in existing}
    for result in updates:
        if result.name == "current_target_xgboost_hyperparameter_tuned":
            by_name.pop("current_target_xgboost_optuna_tuned", None)
        by_name[result.name] = result
    return list(by_name.values())


def _baseline_macro_f1() -> float:
    """Return the current production XGBoost validation Macro F1."""
    metrics_path = config.REPORTS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return 0.0
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    xgboost_metrics = metrics.get("xgboost_model", {})
    return float(xgboost_metrics.get("validation_macro_f1", xgboost_metrics.get("macro_f1", 0.0)))


def _persist_tuning_artifacts(result: ExperimentResult, metadata: dict[str, Any]) -> None:
    """Persist detailed tuning metadata and a flat trial table."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "xgboost_tuning_experiment.json"
    csv_path = config.REPORTS_DIR / "xgboost_tuning_trials.csv"
    payload = {
        "result": result.__dict__,
        "metadata": metadata,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    trial_rows = []
    for trial in metadata["trials"]:
        row = {
            "number": trial["number"],
            "value": trial["value"],
            "state": trial["state"],
            **trial["params"],
            **{f"user_{key}": value for key, value in trial["user_attrs"].items()},
        }
        trial_rows.append(row)
    pd.DataFrame(trial_rows).sort_values("value", ascending=False).to_csv(csv_path, index=False)
    print(f"XGBoost tuning details saved: {json_path}")
    print(f"XGBoost tuning trials saved: {csv_path}")


def run_xgboost_tuning_experiment(
    promote_if_better: bool = False,
    n_trials: int | None = None,
) -> tuple[ExperimentResult, dict[str, Any], Any]:
    """Run only the current-target XGBoost tuning experiment."""
    current_X, current_y, current_years = _load_processed_dataset()
    baseline_macro = _baseline_macro_f1()
    trials = n_trials or int(os.getenv("XGBOOST_TUNING_TRIALS", str(DEFAULT_XGBOOST_TUNING_TRIALS)))

    _, y_trains, X_trains_smote, y_trains_smote = _training_blocks(current_X, current_y, current_years)
    model, metadata = _train_tuned_xgboost(X_trains_smote, y_trains_smote, n_trials=trials)
    search_label = "Optuna TPE" if metadata["search_method"] == "optuna_tpe" else "ParameterSampler determinista"
    result = _record_result(
        name="current_target_xgboost_hyperparameter_tuned",
        target_horizon=3,
        target_start_offset=0,
        model_type="xgboost_hyperparameter_tuned",
        model=model,
        X_test=X_trains_smote[2],
        y_test=y_trains[2],
        comparable_to_current_target=True,
        notes=(
            f"{search_label} sobre hiperparametros, pesos por clase y parada temprana; "
            f"mejor Macro F1 de validacion={metadata['best_validation_macro_f1']:.4f}."
        ),
    )

    promoted = promote_if_better and metadata["best_validation_macro_f1"] > baseline_macro
    metadata["baseline_macro_f1"] = float(baseline_macro)
    metadata["promoted_to_best_model"] = bool(promoted)

    _persist_results(_merge_results(_load_persisted_results(), [result]))
    _persist_tuning_artifacts(result, metadata)

    if promoted:
        save_model(model, "best_model.pkl")
        save_model(model, f"{result.name}.pkl")
        print(
            "Promoted tuned XGBoost to best_model.pkl: "
            f"{metadata['best_validation_macro_f1']:.4f} validation > {baseline_macro:.4f}"
        )
    else:
        print(
            "Tuned XGBoost did not improve production Macro F1: "
            f"tuned_validation={metadata['best_validation_macro_f1']:.4f}, current_validation={baseline_macro:.4f}"
        )

    return result, metadata, model


def run_experiments(promote_if_better: bool = False) -> list[ExperimentResult]:
    """Run target and modeling experiments, promoting only comparable improvements."""
    current_X, current_y, current_years = _load_processed_dataset()
    baseline_macro = _baseline_macro_f1()

    results: list[ExperimentResult] = []
    fitted_models: dict[str, Any] = {}

    xgb_current, xgb_current_model = _run_xgboost_experiment(
        name="current_target_xgboost_recheck",
        horizon=3,
        start_offset=0,
        X=current_X,
        y=current_y,
        years=current_years,
        comparable_to_current_target=True,
        notes="Reentrenamiento de control con el target actual.",
    )
    results.append(xgb_current)
    fitted_models[xgb_current.name] = xgb_current_model

    tuned_current, _, tuned_model = run_xgboost_tuning_experiment(
        promote_if_better=False,
    )
    results.append(tuned_current)
    fitted_models[tuned_current.name] = tuned_model

    current_X_trains, current_y_trains = split_data_temporal(current_X, current_y, current_years)
    adjusted_current, adjusted_current_model = _run_probability_adjusted_experiment(
        name="current_target_xgboost_probability_adjusted",
        horizon=3,
        base_model=xgb_current_model,
        X_trains=current_X_trains,
        y_trains=current_y_trains,
    )
    results.append(adjusted_current)
    fitted_models[adjusted_current.name] = adjusted_current_model

    ordinal_current, ordinal_current_model = _run_ordinal_experiment(
        name="current_target_ordinal_xgboost",
        horizon=3,
        start_offset=0,
        X=current_X,
        y=current_y,
        years=current_years,
        comparable_to_current_target=True,
        notes="Modelo ordinal acumulativo sobre la jerarquia de roles actual.",
    )
    results.append(ordinal_current)
    fitted_models[ordinal_current.name] = ordinal_current_model

    for horizon in [4, 5]:
        X_horizon, y_horizon, years_horizon = _build_dataset_for_horizon(horizon)
        result, model = _run_xgboost_experiment(
            name=f"target_{horizon}_active_seasons_xgboost",
            horizon=horizon,
            start_offset=0,
            X=X_horizon,
            y=y_horizon,
            years=years_horizon,
            comparable_to_current_target=False,
            notes=(
                "Redefine el target agregando mas temporadas activas NBA; "
                "la metrica no es estrictamente comparable con el target actual."
            ),
        )
        results.append(result)
        fitted_models[result.name] = model

    X_third, y_third, years_third = _build_dataset_for_horizon(horizon=1, start_offset=2)
    third_season_result, third_season_model = _run_xgboost_experiment(
        name="target_third_active_season_only_xgboost",
        horizon=1,
        start_offset=2,
        X=X_third,
        y=y_third,
        years=years_third,
        comparable_to_current_target=False,
        notes=(
            "Redefine el target usando solo la tercera temporada activa NBA. "
            "Evalua si esperar a un jugador mas asentado mejora la senial, "
            "pero no es estrictamente comparable con el target actual."
        ),
    )
    results.append(third_season_result)
    fitted_models[third_season_result.name] = third_season_model

    _persist_results(results)

    comparable_results = [result for result in results if result.comparable_to_current_target]
    best_comparable = max(comparable_results, key=lambda result: result.macro_f1)
    if promote_if_better:
        print(
            "Experiment lab does not promote models automatically because these rows are "
            "ranked by test metrics. Use the validation-tuned path for promotion."
        )
    print(
        "Best comparable experiment on test, for audit only: "
        f"{best_comparable.name} macro_f1={best_comparable.macro_f1:.4f}; "
        f"current_validation_macro_f1={baseline_macro:.4f}"
    )

    print("\nExperiment summary:")
    summary = pd.DataFrame([result.__dict__ for result in results])
    print(summary[["name", "target_horizon", "target_start_offset", "model_type", "balanced_accuracy", "macro_f1", "comparable_to_current_target"]].to_string(index=False))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled modeling experiments.")
    parser.add_argument(
        "--only-tuning",
        action="store_true",
        help="Run only the current-target XGBoost Optuna tuning experiment.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Deprecated compatibility flag; experiment lab does not promote unless --promote is passed.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Allow a validation-improving tuning run to overwrite best_model.pkl.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials for the XGBoost tuning experiment.",
    )
    args = parser.parse_args()

    if args.only_tuning:
        run_xgboost_tuning_experiment(promote_if_better=args.promote and not args.no_promote, n_trials=args.trials)
    else:
        run_experiments(promote_if_better=args.promote and not args.no_promote)


if __name__ == "__main__":
    main()
