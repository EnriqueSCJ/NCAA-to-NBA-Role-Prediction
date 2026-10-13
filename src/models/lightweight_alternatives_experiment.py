"""Cheap alternative model experiments for the corrected target.

The goal is to test a small neural network and a few inexpensive tabular
alternatives without changing the production artifact unless validation Macro
F1 improves over the current selected model.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src.config as config
from src.models.explain import load_model
from src.models.probability_adjusted_model import ProbabilityAdjustedRoleClassifier
from src.models.train import get_training_data, save_model, split_data_temporal


@dataclass
class LightweightResult:
    """Serializable metrics for one lightweight candidate."""

    name: str
    model_type: str
    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_macro_f1: float
    validation_starter_plus_recall: float
    validation_non_nba_to_starter_plus: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    starter_plus_recall: float
    non_nba_to_starter_plus: int
    confusion_matrix: list[list[int]]
    notes: str


def _evaluate_predictions(y_true: pd.Series, y_pred: Any) -> dict[str, Any]:
    pred = np.asarray(y_pred).ravel()
    cm = confusion_matrix(y_true, pred, labels=config.ROLE_LABELS)
    starter_index = config.ROLE_LABELS.index("STARTER_PLUS")
    starter_support = int(cm[starter_index].sum())
    starter_recall = float(cm[starter_index, starter_index] / starter_support) if starter_support else 0.0
    non_nba_to_starter = int(((y_true.to_numpy() == "NON_NBA") & (pred == "STARTER_PLUS")).sum())
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)),
        "starter_plus_recall": starter_recall,
        "non_nba_to_starter_plus": non_nba_to_starter,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _record_result(
    *,
    name: str,
    model_type: str,
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    notes: str,
) -> LightweightResult:
    validation = _evaluate_predictions(y_val, model.predict(X_val))
    test = _evaluate_predictions(y_test, model.predict(X_test))
    return LightweightResult(
        name=name,
        model_type=model_type,
        validation_accuracy=validation["accuracy"],
        validation_balanced_accuracy=validation["balanced_accuracy"],
        validation_macro_f1=validation["macro_f1"],
        validation_starter_plus_recall=validation["starter_plus_recall"],
        validation_non_nba_to_starter_plus=validation["non_nba_to_starter_plus"],
        accuracy=test["accuracy"],
        balanced_accuracy=test["balanced_accuracy"],
        macro_f1=test["macro_f1"],
        starter_plus_recall=test["starter_plus_recall"],
        non_nba_to_starter_plus=test["non_nba_to_starter_plus"],
        confusion_matrix=test["confusion_matrix"],
        notes=notes,
    )


def _capped_resample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    targets: dict[str, int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a small, deterministic balanced-ish training sample."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    sampled_indices: list[int] = []
    for label in config.ROLE_LABELS:
        label_indices = y_train[y_train.astype(str).eq(label)].index.to_numpy()
        if len(label_indices) == 0:
            continue
        target = int(targets.get(label, len(label_indices)))
        replace = target > len(label_indices)
        chosen = rng.choice(label_indices, size=target, replace=replace)
        sampled_indices.extend(chosen.tolist())

    rng.shuffle(sampled_indices)
    return X_train.loc[sampled_indices].reset_index(drop=True), y_train.loc[sampled_indices].reset_index(drop=True)


def _class_weight_dict(y_train: pd.Series) -> dict[str, float]:
    counts = y_train.value_counts().reindex(config.ROLE_LABELS, fill_value=0)
    total = float(counts.sum())
    weights: dict[str, float] = {}
    for label, count in counts.items():
        weights[str(label)] = total / (len(config.ROLE_LABELS) * max(float(count), 1.0))
    return weights


def _tune_probability_multipliers(
    base_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[ProbabilityAdjustedRoleClassifier, dict[str, float], float]:
    """Tune simple class probability multipliers on validation Macro F1."""
    probabilities = np.asarray(base_model.predict_proba(X_val), dtype=float)
    labels = [str(label) for label in getattr(base_model, "classes_", config.ROLE_LABELS)]
    aligned = np.zeros((len(X_val), len(config.ROLE_LABELS)), dtype=float)
    for source_index, label in enumerate(labels):
        if label in config.ROLE_LABELS and source_index < probabilities.shape[1]:
            aligned[:, config.ROLE_LABELS.index(label)] = probabilities[:, source_index]

    grids = {
        "NON_NBA": [0.85, 1.0, 1.15],
        "BENCH": [0.75, 1.0, 1.25, 1.50, 1.75, 2.00],
        "ROTATION": [0.75, 1.0, 1.25, 1.50, 1.75, 2.00],
        "STARTER_PLUS": [0.50, 0.75, 1.0, 1.25, 1.50, 2.00],
    }
    best_score = -1.0
    best_multipliers = {label: 1.0 for label in config.ROLE_LABELS}
    for non_nba in grids["NON_NBA"]:
        for bench in grids["BENCH"]:
            for rotation in grids["ROTATION"]:
                for starter in grids["STARTER_PLUS"]:
                    multipliers = np.array([non_nba, bench, rotation, starter], dtype=float)
                    pred = np.array(config.ROLE_LABELS)[(aligned * multipliers).argmax(axis=1)]
                    score = float(
                        f1_score(y_val, pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)
                    )
                    if score > best_score:
                        best_score = score
                        best_multipliers = {
                            "NON_NBA": float(non_nba),
                            "BENCH": float(bench),
                            "ROTATION": float(rotation),
                            "STARTER_PLUS": float(starter),
                        }

    return ProbabilityAdjustedRoleClassifier(base_model, best_multipliers), best_multipliers, best_score


def _candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[tuple[str, str, Any, pd.DataFrame, pd.Series, str]]:
    """Build cheap model candidates with their training data."""
    class_weights = _class_weight_dict(y_train)
    X_mlp_balanced, y_mlp_balanced = _capped_resample(
        X_train,
        y_train,
        {
            "NON_NBA": 4000,
            "BENCH": 1200,
            "ROTATION": 1200,
            "STARTER_PLUS": 1200,
        },
    )

    mlp_small = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            alpha=0.02,
            learning_rate_init=0.001,
            max_iter=140,
            early_stopping=False,
            n_iter_no_change=12,
            random_state=config.RANDOM_STATE,
        ),
    )
    mlp_medium = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 24),
            activation="relu",
            alpha=0.03,
            learning_rate_init=0.0008,
            max_iter=160,
            early_stopping=False,
            n_iter_no_change=14,
            random_state=config.RANDOM_STATE,
        ),
    )

    candidates: list[tuple[str, str, Any, pd.DataFrame, pd.Series, str]] = [
        (
            "mlp_small_balanced_sample",
            "mlp",
            mlp_small,
            X_mlp_balanced,
            y_mlp_balanced,
            "Small MLP on capped balanced sample; cheap neural-network sanity check.",
        ),
        (
            "mlp_medium_balanced_sample",
            "mlp",
            mlp_medium,
            X_mlp_balanced,
            y_mlp_balanced,
            "Two-layer MLP on the same capped sample.",
        ),
        (
            "extra_trees_balanced",
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=260,
                max_depth=10,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
            ),
            X_train,
            y_train,
            "ExtraTrees with balanced_subsample class weights.",
        ),
        (
            "hist_gradient_boosting_balanced",
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.045,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=0.10,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
            X_train,
            y_train,
            "Scikit-learn histogram gradient boosting with class weights.",
        ),
    ]

    try:
        from lightgbm import LGBMClassifier

        candidates.append(
            (
                "lightgbm_balanced",
                "lightgbm",
                LGBMClassifier(
                    objective="multiclass",
                    n_estimators=260,
                    learning_rate=0.035,
                    num_leaves=15,
                    min_child_samples=35,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    class_weight=class_weights,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
                X_train,
                y_train,
                "LightGBM multiclass with conservative tree size and class weights.",
            )
        )
    except ModuleNotFoundError:
        pass

    return candidates


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


def run_lightweight_experiments(promote_if_better: bool = False) -> dict[str, Any]:
    """Run cheap alternatives and optionally promote a validation winner."""
    X, y, years, source = get_training_data()
    X_trains, y_trains = split_data_temporal(X, y, years)
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    current_model = load_model()
    current_result = _record_result(
        name="current_best_model",
        model_type=type(current_model).__name__,
        model=current_model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        notes="Current models/best_model.pkl baseline.",
    )

    results = [current_result]
    fitted_models: dict[str, Any] = {"current_best_model": current_model}

    adjusted_model, multipliers, validation_score = _tune_probability_multipliers(current_model, X_val, y_val)
    probability_result = _record_result(
        name="xgboost_probability_adjusted_light",
        model_type="probability_adjusted",
        model=adjusted_model,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        notes=f"Validation-tuned class multipliers: {multipliers}; validation grid score={validation_score:.4f}.",
    )
    results.append(probability_result)
    fitted_models[probability_result.name] = adjusted_model

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name, model_type, model, train_X, train_y, notes in _candidate_models(X_train, y_train):
        model.fit(train_X, train_y)
        result = _record_result(
            name=name,
            model_type=model_type,
            model=model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            notes=notes,
        )
        results.append(result)
        fitted_models[name] = model

    best_result = max(results, key=lambda item: item.validation_macro_f1)
    promoted = False
    promoted_path = ""
    if promote_if_better and best_result.name != "current_best_model":
        current_score = current_result.validation_macro_f1
        if best_result.validation_macro_f1 > current_score:
            promoted_path = str(save_model(fitted_models[best_result.name], "best_model.pkl"))
            promoted = True

    rows = [asdict(result) for result in results]
    rows = sorted(rows, key=lambda row: row["validation_macro_f1"], reverse=True)

    payload: dict[str, Any] = {
        "data_source": source,
        "split": {
            "train": f"draft_year <= {config.TRAIN_MAX_DRAFT_YEAR}",
            "validation": f"{config.VALIDATION_START_DRAFT_YEAR}-{config.VALIDATION_END_DRAFT_YEAR}",
            "test": f"{config.TEST_START_DRAFT_YEAR}-{config.TEST_END_DRAFT_YEAR}",
        },
        "selection_rule": "validation_macro_f1",
        "recommendation": best_result.name,
        "promoted_to_best_model": promoted,
        "promoted_path": promoted_path,
        "results": rows,
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "lightweight_alternatives_experiment.json"
    csv_path = config.REPORTS_DIR / "lightweight_alternatives_experiment.csv"
    json_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    csv_rows = pd.DataFrame(rows).drop(columns=["confusion_matrix"], errors="ignore")
    csv_rows.to_csv(csv_path, index=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the validation winner to models/best_model.pkl if it beats the current model.",
    )
    args = parser.parse_args()
    payload = run_lightweight_experiments(promote_if_better=args.promote)

    print("=== Lightweight alternatives experiment ===")
    rows = pd.DataFrame(payload["results"])
    print(
        rows[
            [
                "name",
                "validation_macro_f1",
                "validation_balanced_accuracy",
                "macro_f1",
                "balanced_accuracy",
                "starter_plus_recall",
            ]
        ].to_string(index=False)
    )
    print(f"\nRecommendation: {payload['recommendation']}")
    print(f"Promoted: {payload['promoted_to_best_model']}")
    print(f"JSON: {config.REPORTS_DIR / 'lightweight_alternatives_experiment.json'}")
    print(f"CSV:  {config.REPORTS_DIR / 'lightweight_alternatives_experiment.csv'}")


if __name__ == "__main__":
    main()
