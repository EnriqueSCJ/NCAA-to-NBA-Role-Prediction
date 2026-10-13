"""Two-stage modeling experiment for NCAA to NBA role prediction.

The baseline multiclass model is intentionally kept intact. This module adds an
experimental architecture that first gates NBA likelihood and then predicts the
role among NBA outcomes only.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import src.config as config
from src.models.train import (
    TEST_START_DRAFT_YEAR,
    TEST_END_DRAFT_YEAR,
    TRAIN_MAX_DRAFT_YEAR,
    VALIDATION_END_DRAFT_YEAR,
    VALIDATION_START_DRAFT_YEAR,
    XGBoostRoleClassifier,
    apply_smote,
    get_training_data,
    save_model,
    split_data_temporal,
)
from src.models.two_stage_model import (
    GATE_LABELS,
    ROLE_CONDITIONED_LABELS,
    BinaryXGBoostClassifier,
    TwoStageRoleClassifier,
)


def make_gate_target(y: pd.Series) -> pd.Series:
    """Collapse role labels into NON_NBA vs NBA."""
    return pd.Series(
        np.where(y.astype(str).eq("NON_NBA"), "NON_NBA", "NBA"),
        index=y.index,
        name="nba_gate",
    )


def _nba_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the probability of the NBA class from a gate model."""
    probabilities = np.asarray(model.predict_proba(X))
    classes = [str(label) for label in getattr(model, "classes_", GATE_LABELS)]
    if "NBA" in classes:
        return probabilities[:, classes.index("NBA")]
    return probabilities[:, -1]


def evaluate_gate_model(model: Any, X_test: pd.DataFrame, y_test_gate: pd.Series) -> dict[str, float]:
    """Evaluate NBA/non-NBA gate performance."""
    y_pred = pd.Series(model.predict(X_test), index=y_test_gate.index)
    y_true_binary = y_test_gate.eq("NBA").astype(int)
    nba_probability = _nba_probability(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test_gate, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_gate, y_pred)),
        "macro_f1": float(f1_score(y_test_gate, y_pred, labels=GATE_LABELS, average="macro", zero_division=0)),
        "nba_precision": float(precision_score(y_test_gate, y_pred, pos_label="NBA", zero_division=0)),
        "nba_recall": float(recall_score(y_test_gate, y_pred, pos_label="NBA", zero_division=0)),
        "nba_f1": float(f1_score(y_test_gate, y_pred, pos_label="NBA", zero_division=0)),
        "average_precision": float(average_precision_score(y_true_binary, nba_probability)),
    }
    if y_true_binary.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_binary, nba_probability))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def train_gate_candidates(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_gate_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> dict[str, Any]:
    """Train several binary gate candidates."""
    import xgboost as xgb

    X_train, X_val, _ = X_trains
    y_train, y_val, _ = y_gate_trains

    candidates: dict[str, Any] = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    candidates["gate_dummy"] = dummy

    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE)
    logistic.fit(X_train, y_train)
    candidates["gate_logistic"] = logistic

    forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    forest.fit(X_train, y_train)
    candidates["gate_random_forest"] = forest

    encoded_train = y_train.eq("NBA").astype(int).to_numpy()
    encoded_val = y_val.eq("NBA").astype(int).to_numpy()
    positive_count = max(int(encoded_train.sum()), 1)
    negative_count = max(int((encoded_train == 0).sum()), 1)
    dtrain = xgb.DMatrix(X_train, label=encoded_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_val, label=encoded_val, feature_names=list(X_val.columns))
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.07,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": negative_count / positive_count,
            "seed": config.RANDOM_STATE,
        },
        dtrain=dtrain,
        num_boost_round=350,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    candidates["gate_xgboost"] = BinaryXGBoostClassifier(booster=booster)

    return candidates


def _conditioned_role_data(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return rows with observed NBA roles only."""
    mask = y.astype(str).ne("NON_NBA")
    return X.loc[mask].copy(), y.loc[mask].copy(), years.loc[mask].copy()


def train_role_conditioned_candidates(
    X_role_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_role_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Train role models after removing NON_NBA."""
    import xgboost as xgb

    X_train, X_val, X_test = X_role_trains
    y_train, y_val, y_test = y_role_trains
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    candidates: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}

    forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    forest.fit(X_train_balanced, y_train_balanced)
    candidates["role_random_forest"] = forest

    mapping = {label: index for index, label in enumerate(ROLE_CONDITIONED_LABELS)}
    dtrain = xgb.DMatrix(
        X_train_balanced,
        label=y_train_balanced.map(mapping).astype(int),
        feature_names=list(X_train_balanced.columns),
    )
    dval = xgb.DMatrix(
        X_val,
        label=y_val.map(mapping).astype(int),
        feature_names=list(X_val.columns),
    )
    booster = xgb.train(
        params={
            "objective": "multi:softprob",
            "num_class": len(ROLE_CONDITIONED_LABELS),
            "eval_metric": "mlogloss",
            "eta": 0.08,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "seed": config.RANDOM_STATE,
        },
        dtrain=dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    candidates["role_xgboost"] = XGBoostRoleClassifier(booster=booster, classes=ROLE_CONDITIONED_LABELS)

    for name, model in candidates.items():
        y_val_pred = pd.Series(model.predict(X_val), index=y_val.index)
        y_test_pred = pd.Series(model.predict(X_test), index=y_test.index)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_test_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
            "macro_f1": float(
                f1_score(
                    y_test,
                    y_test_pred,
                    labels=ROLE_CONDITIONED_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "validation_accuracy": float(accuracy_score(y_val, y_val_pred)),
            "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, y_val_pred)),
            "validation_macro_f1": float(
                f1_score(
                    y_val,
                    y_val_pred,
                    labels=ROLE_CONDITIONED_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    return candidates, metrics


def combine_two_stage_predictions(
    gate_model: Any,
    role_model: Any,
    X: pd.DataFrame,
    gate_threshold: float,
) -> pd.Series:
    """Run the gate and then role model for NBA-gated players."""
    nba_probability = _nba_probability(gate_model, X)
    predictions = pd.Series("NON_NBA", index=X.index, name="two_stage_role")
    nba_mask = nba_probability >= gate_threshold
    if nba_mask.any():
        predictions.loc[nba_mask] = role_model.predict(X.loc[nba_mask])
    return predictions


def tune_gate_threshold_for_combined_macro_f1(
    gate_model: Any,
    role_model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, float]:
    """Choose the gate threshold that maximizes combined validation Macro F1."""
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        y_pred = combine_two_stage_predictions(gate_model, role_model, X_val, float(threshold))
        score = f1_score(y_val, y_pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def evaluate_multiclass_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Evaluate predictions over the original five project roles."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=config.ROLE_LABELS, average="weighted", zero_division=0)),
    }


def _save_confusion_matrix(cm: np.ndarray, labels: list[str], path: str, title: str) -> None:
    """Persist a compact confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Real",
        xlabel="Predicho",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(
                column,
                row,
                int(cm[row, column]),
                ha="center",
                va="center",
                color="white" if cm[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def train_two_stage_models() -> dict[str, Any]:
    """Train and evaluate the two-stage experiment."""
    X, y, years, source = get_training_data()
    y_gate = make_gate_target(y)
    X_trains, y_trains = split_data_temporal(X, y, years)
    gate_trains = split_data_temporal(X, y_gate, years)[1]
    X_train, X_val, X_test = X_trains
    _, y_val, y_test = y_trains

    X_role, y_role, years_role = _conditioned_role_data(X, y, years)
    X_role_trains, y_role_trains = split_data_temporal(X_role, y_role, years_role)
    role_candidates, role_metrics = train_role_conditioned_candidates(X_role_trains, y_role_trains)
    best_role_name = max(role_metrics, key=lambda name: role_metrics[name]["validation_macro_f1"])
    best_role_model = role_candidates[best_role_name]

    gate_candidates = train_gate_candidates(X_trains, gate_trains)
    gate_metrics = {}
    _, y_val_gate, y_test_gate = gate_trains
    for name, model in gate_candidates.items():
        test_gate_metrics = evaluate_gate_model(model, X_test, y_test_gate)
        validation_gate_metrics = evaluate_gate_model(model, X_val, y_val_gate)
        gate_metrics[name] = {
            **test_gate_metrics,
            **{f"validation_{key}": value for key, value in validation_gate_metrics.items()},
        }

    combined_validation: dict[str, dict[str, float]] = {}
    for name, gate_model in gate_candidates.items():
        threshold, validation_macro_f1 = tune_gate_threshold_for_combined_macro_f1(
            gate_model=gate_model,
            role_model=best_role_model,
            X_val=X_val,
            y_val=y_val,
        )
        combined_validation[name] = {
            "threshold": threshold,
            "validation_macro_f1": validation_macro_f1,
        }

    best_gate_name = max(combined_validation, key=lambda name: combined_validation[name]["validation_macro_f1"])
    best_gate_model = gate_candidates[best_gate_name]
    best_threshold = combined_validation[best_gate_name]["threshold"]
    if hasattr(best_gate_model, "threshold"):
        best_gate_model.threshold = best_threshold

    y_pred_combined = combine_two_stage_predictions(best_gate_model, best_role_model, X_test, best_threshold)
    combined_metrics = evaluate_multiclass_predictions(y_test, y_pred_combined)
    combined_model = TwoStageRoleClassifier(
        gate_model=best_gate_model,
        role_model=best_role_model,
        gate_threshold=best_threshold,
    )

    cm = confusion_matrix(y_test, y_pred_combined, labels=config.ROLE_LABELS)
    cm_df = pd.DataFrame(cm, index=config.ROLE_LABELS, columns=config.ROLE_LABELS)
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred_combined,
            labels=config.ROLE_LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).T

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    gate_path = save_model(best_gate_model, "nba_gate_model.pkl")
    role_path = save_model(best_role_model, "role_conditioned_model.pkl")
    combined_model_path = save_model(combined_model, "two_stage_best_model.pkl")

    baseline_best_validation_macro_f1 = -1.0
    model_metrics_path = config.REPORTS_DIR / "model_metrics.json"
    if model_metrics_path.exists():
        with model_metrics_path.open("r", encoding="utf-8") as file:
            model_metrics = json.load(file)
        baseline_best_validation_macro_f1 = max(
            float(values.get("validation_macro_f1", -1.0))
            for values in model_metrics.values()
        )

    combined_validation_macro_f1 = combined_validation[best_gate_name]["validation_macro_f1"]
    promoted_to_best_model = combined_validation_macro_f1 >= baseline_best_validation_macro_f1
    best_model_path = None
    if promoted_to_best_model:
        best_model_path = save_model(combined_model, "best_model.pkl")

    metrics_payload: dict[str, Any] = {
        "data_source": source,
        "selected_gate_model": best_gate_name,
        "selected_role_model": best_role_name,
        "gate_threshold": best_threshold,
        "split": {
            "train": f"draft_year <= {TRAIN_MAX_DRAFT_YEAR}",
            "validation": f"{VALIDATION_START_DRAFT_YEAR}-{VALIDATION_END_DRAFT_YEAR}",
            "test": f"{TEST_START_DRAFT_YEAR}-{TEST_END_DRAFT_YEAR}",
        },
        "gate_metrics": gate_metrics,
        "role_conditioned_metrics": role_metrics,
        "combined_validation": combined_validation,
        "combined_test_metrics": combined_metrics,
        "test_distribution": y_test.value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict(),
        "gate_model_path": str(gate_path),
        "role_model_path": str(role_path),
        "combined_model_path": str(combined_model_path),
        "promoted_to_best_model": promoted_to_best_model,
        "best_model_path": str(best_model_path) if best_model_path is not None else "",
    }

    metrics_path = config.REPORTS_DIR / "two_stage_metrics.json"
    report_path = config.REPORTS_DIR / "two_stage_classification_report.csv"
    cm_csv_path = config.REPORTS_DIR / "two_stage_confusion_matrix.csv"
    cm_png_path = config.REPORTS_DIR / "two_stage_confusion_matrix.png"
    card_path = config.REPORTS_DIR / "two_stage_model_card.md"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2, sort_keys=True)
    report_df.to_csv(report_path)
    cm_df.to_csv(cm_csv_path)
    _save_confusion_matrix(
        cm,
        list(config.ROLE_LABELS),
        str(cm_png_path),
        "Matriz de confusion - modelo en dos etapas",
    )
    card_path.write_text(_build_two_stage_model_card(metrics_payload, cm_df), encoding="utf-8")

    print("=== Entrenamiento en dos etapas ===")
    print(f"Fuente de datos: {source}")
    print(f"Modelo gate seleccionado: {best_gate_name} (threshold={best_threshold:.2f})")
    print(f"Modelo de rol condicionado: {best_role_name}")
    print("Metricas combinadas en test:")
    for name, value in combined_metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Artefactos:")
    for path in [gate_path, role_path, combined_model_path, metrics_path, report_path, cm_csv_path, cm_png_path, card_path]:
        print(f"  {path}")
    if promoted_to_best_model:
        print(f"  {best_model_path} (promoted by validation Macro F1)")

    return metrics_payload


def _build_two_stage_model_card(metrics: dict[str, Any], cm_df: pd.DataFrame) -> str:
    """Create a concise markdown card for the two-stage experiment."""
    combined = metrics["combined_test_metrics"]
    gate = metrics["gate_metrics"][metrics["selected_gate_model"]]
    role = metrics["role_conditioned_metrics"][metrics["selected_role_model"]]
    header = "| Real \\ Predicho | " + " | ".join(cm_df.columns.astype(str)) + " |"
    separator = "| --- | " + " | ".join(["---:"] * len(cm_df.columns)) + " |"
    rows = [
        "| " + str(index) + " | " + " | ".join(str(int(value)) for value in row) + " |"
        for index, row in cm_df.iterrows()
    ]
    cm_markdown = "\n".join([header, separator, *rows])
    return f"""# Two-Stage Model Card - NCAA to NBA Role Predictor

## Objetivo

Este experimento mantiene el baseline multiclase intacto y prueba una arquitectura en dos etapas para reducir el efecto de inundacion de `NON_NBA`.

## Arquitectura

| Etapa | Modelo seleccionado | Funcion |
| --- | --- | --- |
| Gate NBA | `{metrics["selected_gate_model"]}` | Clasifica `NON_NBA` vs `NBA`. |
| Rol condicionado | `{metrics["selected_role_model"]}` | Clasifica `BENCH`, `ROTATION`, `STARTER_PLUS` cuando el gate detecta perfil NBA. |

Threshold del gate elegido en validacion: `{metrics["gate_threshold"]:.2f}`.

## Metricas del gate NBA

| Metrica | Valor |
| --- | ---: |
| Balanced Accuracy | {gate["balanced_accuracy"]:.4f} |
| Macro F1 | {gate["macro_f1"]:.4f} |
| Precision NBA | {gate["nba_precision"]:.4f} |
| Recall NBA | {gate["nba_recall"]:.4f} |
| F1 NBA | {gate["nba_f1"]:.4f} |
| Average Precision | {gate["average_precision"]:.4f} |

## Metricas del rol condicionado

| Metrica | Valor |
| --- | ---: |
| Accuracy | {role["accuracy"]:.4f} |
| Balanced Accuracy | {role["balanced_accuracy"]:.4f} |
| Macro F1 | {role["macro_f1"]:.4f} |

## Metricas combinadas en test

| Metrica | Valor |
| --- | ---: |
| Accuracy | {combined["accuracy"]:.4f} |
| Balanced Accuracy | {combined["balanced_accuracy"]:.4f} |
| Macro F1 | {combined["macro_f1"]:.4f} |
| Weighted F1 | {combined["weighted_f1"]:.4f} |

## Matriz de confusion combinada

{cm_markdown}

## Lectura

Este experimento comprueba si los datos actuales permiten separar primero llegada a NBA y despues rol. Debe compararse contra `reports/model_card.md`, que conserva el baseline multiclase completo.
"""


def main() -> None:
    """CLI entrypoint."""
    train_two_stage_models()


if __name__ == "__main__":
    main()
