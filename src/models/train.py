"""Baseline model training for NCAA to NBA role prediction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

import src.config as config


FEATURE_COLUMNS = list(config.MODEL_FEATURE_COLUMNS)
TRAIN_MAX_DRAFT_YEAR = config.TRAIN_MAX_DRAFT_YEAR
VALIDATION_START_DRAFT_YEAR = config.VALIDATION_START_DRAFT_YEAR
VALIDATION_END_DRAFT_YEAR = config.VALIDATION_END_DRAFT_YEAR
TEST_START_DRAFT_YEAR = config.TEST_START_DRAFT_YEAR
TEST_END_DRAFT_YEAR = config.TEST_END_DRAFT_YEAR


class XGBoostRoleClassifier:
    """Small wrapper that exposes label predictions from an XGBoost Booster."""

    def __init__(self, booster: Any, classes: list[str]):
        self.booster = booster
        self.classes_ = np.array(classes)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        predictions = self.booster.predict(dmatrix)
        return np.asarray(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if probabilities.ndim == 1:
            class_ids = probabilities.astype(int)
        else:
            class_ids = probabilities.argmax(axis=1)
        return self.classes_[class_ids]


def split_data_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    df_with_year: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.Series, pd.Series, pd.Series]]:
    """Split by draft year to reduce temporal leakage."""
    if "draft_year" not in df_with_year.columns:
        raise ValueError("df_with_year debe incluir la columna 'draft_year'")

    years = pd.to_numeric(df_with_year["draft_year"], errors="coerce")
    train_idx = years <= TRAIN_MAX_DRAFT_YEAR
    val_idx = (years >= VALIDATION_START_DRAFT_YEAR) & (years <= VALIDATION_END_DRAFT_YEAR)
    test_idx = (years >= TEST_START_DRAFT_YEAR) & (years <= TEST_END_DRAFT_YEAR)

    if not train_idx.any() or not val_idx.any() or not test_idx.any():
        raise ValueError("El split temporal produjo algun bloque vacio")

    return (
        X.loc[train_idx].copy(),
        X.loc[val_idx].copy(),
        X.loc[test_idx].copy(),
    ), (
        y.loc[train_idx].copy(),
        y.loc[val_idx].copy(),
        y.loc[test_idx].copy(),
    )


def _smote_sampling_strategy(y_train: pd.Series) -> dict[str, int] | str:
    """Build a multi-class SMOTE strategy from the configured ratio."""
    counts = y_train.value_counts()
    if len(counts) < 2:
        return "auto"

    configured = config.SMOTE_SAMPLING_STRATEGY
    if not isinstance(configured, float):
        return configured

    majority_count = int(counts.max())
    target_count = max(2, int(round(majority_count * configured)))

    strategy: dict[str, int] = {}
    for label, count in counts.items():
        if count < target_count:
            strategy[str(label)] = target_count

    return strategy or "auto"


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to partially balance minority classes."""
    counts = y_train.value_counts()
    min_count = int(counts.min())

    if len(counts) < 2 or min_count < 2:
        print("SMOTE omitido: no hay suficientes muestras por clase.")
        return X_train.copy(), y_train.copy()

    k_neighbors = min(5, min_count - 1)
    smote = SMOTE(
        sampling_strategy=_smote_sampling_strategy(y_train),
        random_state=config.RANDOM_STATE,
        k_neighbors=k_neighbors,
    )
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
    y_balanced = pd.Series(y_balanced, name=y_train.name or "target_role")

    print("SMOTE aplicado.")
    print("Distribucion original:")
    print(counts.reindex(config.ROLE_LABELS, fill_value=0).to_string())
    print("Distribucion balanceada:")
    print(y_balanced.value_counts().reindex(config.ROLE_LABELS, fill_value=0).to_string())

    return X_balanced, y_balanced


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict[str, float]:
    """Evaluate a fitted classifier with imbalance-aware metrics."""
    y_pred = np.asarray(model.predict(X_test)).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(
                y_test,
                y_pred,
                labels=config.ROLE_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=config.ROLE_LABELS, zero_division=0))

    return metrics


def _with_validation_metrics(
    test_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> dict[str, float]:
    """Keep test metrics primary while exposing validation metrics for selection."""
    merged = dict(test_metrics)
    for key, value in validation_metrics.items():
        merged[f"validation_{key}"] = value
    return merged


def select_best_model_name(metrics: dict[str, dict[str, float]]) -> str:
    """Select a model using validation Macro F1, leaving test for final audit."""
    if not metrics:
        raise ValueError("No hay metricas de modelos para seleccionar")
    return max(metrics, key=lambda name: metrics[name].get("validation_macro_f1", float("-inf")))


def print_feature_importances(
    importances: list[tuple[str, float]],
    title: str = "Top 10 feature importances",
) -> None:
    """Print the largest feature importance values."""
    print(f"\n{title}:")
    for feature, importance in importances[:10]:
        print(f"  {feature}: {importance:.4f}")


def train_dummy_baseline(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[DummyClassifier, dict[str, float]]:
    """Train a most-frequent baseline classifier."""
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    validation_metrics = evaluate_model(model, X_val, y_val, "DUMMY BASELINE VALIDATION")
    test_metrics = evaluate_model(model, X_test, y_test, "DUMMY BASELINE TEST")
    metrics = _with_validation_metrics(test_metrics, validation_metrics)
    return model, metrics


def train_logistic_regression(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[LogisticRegression, dict[str, float]]:
    """Train a balanced multinomial logistic regression baseline."""
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    validation_metrics = evaluate_model(model, X_val, y_val, "LOGISTIC REGRESSION VALIDATION")
    test_metrics = evaluate_model(model, X_test, y_test, "LOGISTIC REGRESSION TEST")
    metrics = _with_validation_metrics(test_metrics, validation_metrics)
    return model, metrics


def train_random_forest(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Train a balanced Random Forest baseline."""
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    validation_metrics = evaluate_model(model, X_val, y_val, "RANDOM FOREST VALIDATION")
    test_metrics = evaluate_model(model, X_test, y_test, "RANDOM FOREST TEST")
    metrics = _with_validation_metrics(test_metrics, validation_metrics)

    importances = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    print_feature_importances(importances)

    return model, metrics


def _encode_roles(y: pd.Series) -> np.ndarray:
    """Encode role labels using the project ordinal role order."""
    mapping = {label: index for index, label in enumerate(config.ROLE_LABELS)}
    encoded = y.map(mapping)
    if encoded.isna().any():
        unknown = sorted(set(y[encoded.isna()]))
        raise ValueError(f"Etiquetas desconocidas para entrenamiento: {unknown}")
    return encoded.astype(int).to_numpy()


def _xgboost_row_weights(y: pd.Series) -> np.ndarray:
    """Return deterministic class weights for multiclass XGBoost training."""
    class_weights = {
        "NON_NBA": 0.70,
        "BENCH": 1.40,
        "ROTATION": 2.00,
        "STARTER_PLUS": 1.80,
    }
    return y.astype(str).map(class_weights).fillna(1.0).astype(float).to_numpy()


def train_xgboost(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[XGBoostRoleClassifier, dict[str, float]]:
    """Train an XGBoost multiclass model with validation early stopping."""
    import xgboost as xgb

    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    dtrain = xgb.DMatrix(
        X_train,
        label=_encode_roles(y_train),
        weight=_xgboost_row_weights(y_train),
        feature_names=list(X_train.columns),
    )
    dval = xgb.DMatrix(
        X_val,
        label=_encode_roles(y_val),
        feature_names=list(X_val.columns),
    )

    params = {
        "objective": "multi:softprob",
        "num_class": len(config.ROLE_LABELS),
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "seed": config.RANDOM_STATE,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=250,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=25,
        verbose_eval=False,
    )
    model = XGBoostRoleClassifier(booster=booster, classes=config.ROLE_LABELS)
    validation_metrics = evaluate_model(model, X_val, y_val, "XGBOOST VALIDATION")
    test_metrics = evaluate_model(model, X_test, y_test, "XGBOOST TEST")
    metrics = _with_validation_metrics(test_metrics, validation_metrics)

    gains = booster.get_score(importance_type="gain")
    importances = sorted(gains.items(), key=lambda item: item[1], reverse=True)
    print_feature_importances(importances, title="Top 10 XGBoost gain importances")

    return model, metrics


def train_catboost(
    X_trains: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    y_trains: tuple[pd.Series, pd.Series, pd.Series],
) -> tuple[Any, dict[str, float]]:
    """Train a CatBoost multiclass model with balanced class weights."""
    from catboost import CatBoostClassifier

    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    model = CatBoostClassifier(
        iterations=250,
        depth=6,
        learning_rate=0.1,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        random_seed=config.RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=20,
        use_best_model=True,
    )
    validation_metrics = evaluate_model(model, X_val, y_val, "CATBOOST VALIDATION")
    test_metrics = evaluate_model(model, X_test, y_test, "CATBOOST TEST")
    metrics = _with_validation_metrics(test_metrics, validation_metrics)

    importances = sorted(
        zip(X_train.columns, model.get_feature_importance()),
        key=lambda item: item[1],
        reverse=True,
    )
    print_feature_importances(importances, title="Top 10 CatBoost importances")

    return model, metrics


def save_model(model: Any, filename: str) -> Path:
    """Persist a trained model under the project models directory."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / filename
    with path.open("wb") as file:
        pickle.dump(model, file)
    return path


def save_metrics(metrics: dict[str, dict[str, float]], filename: str = "model_metrics.json") -> Path:
    """Persist model metrics as JSON for later reporting."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / filename
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    return path


def save_comparison(metrics: dict[str, dict[str, float]]) -> Path:
    """Persist a CSV model-comparison table."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / "model_comparison.csv"
    comparison = pd.DataFrame(metrics).T.sort_values("validation_macro_f1", ascending=False)
    comparison.to_csv(path, index_label="model")
    return path


def load_processed_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame] | None:
    """Load processed training files when they already exist."""
    x_path = config.DATA_PROCESSED / "X.csv"
    y_path = config.DATA_PROCESSED / "y.csv"
    master_path = config.DATA_PROCESSED / "master_table.csv"

    if not (x_path.exists() and y_path.exists() and master_path.exists()):
        return None

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze("columns")
    master = pd.read_csv(master_path, low_memory=False)

    if "draft_year" not in master.columns:
        raise ValueError("master_table.csv debe contener 'draft_year'")

    X = X.select_dtypes(include=[np.number]).copy()
    y = pd.Series(y, name="target_role")
    return X, y, master[["draft_year"]]


def make_synthetic_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create deterministic imbalanced data for pipeline validation."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    role_counts = {
        "NON_NBA": 15,
        "BENCH": 6,
        "ROTATION": 4,
        "STARTER_PLUS": 4,
    }

    rows: list[dict[str, float]] = []
    labels: list[str] = []
    years: list[int] = []
    samples_per_year = sum(role_counts.values())
    role_order_for_scores = ["STARTER_PLUS", "ROTATION", "BENCH", "NON_NBA"]

    def nonlinear_score(features: np.ndarray, draft_year: int) -> float:
        """Score synthetic players through interactions that favor tree models."""
        column_index = {column: index for index, column in enumerate(FEATURE_COLUMNS)}
        pts = features[column_index["pts_per40_z"]]
        ast = features[column_index["ast_per40_z"]]
        reb = features[column_index["reb_per40_z"]]
        blk = features[column_index["blk_per40_z"]]
        tov = features[column_index["tov_per40_z"]]
        ts = features[column_index["ts_pct_z"]]
        usg = features[column_index["usg_pct_z"]]
        stocks = features[column_index["stocks_per40_z"]]
        height = features[column_index["height_in_z"]]
        wingspan = features[column_index["wingspan_in_z"]]
        adj_o = features[column_index["team_adj_o_z"]]
        adj_d = features[column_index["team_adj_d_z"]]
        sos = features[column_index["strength_of_schedule_z"]]

        age_shift = (draft_year - 2015) * 0.015
        return float(
            1.35 * pts * ts
            + 1.10 * ast * stocks
            + 0.85 * height * wingspan
            + 0.75 * adj_o * sos
            + 0.55 * (reb > 0.25) * blk
            + 0.45 * (usg > 0.5) * ts
            - 0.70 * max(tov, 0.0) * max(adj_d, 0.0)
            + age_shift
        )

    for draft_year in range(2009, 2022):
        year_features = rng.normal(loc=0.0, scale=1.0, size=(samples_per_year, len(FEATURE_COLUMNS)))
        scores = np.array(
            [nonlinear_score(features, draft_year) for features in year_features]
        )
        ranked_indices = list(np.argsort(scores)[::-1])

        assigned_roles: dict[int, str] = {}
        cursor = 0
        for role in role_order_for_scores:
            count = role_counts[role]
            for row_index in ranked_indices[cursor : cursor + count]:
                assigned_roles[int(row_index)] = role
            cursor += count

        for row_index, features in enumerate(year_features):
            row = {column: float(value) for column, value in zip(FEATURE_COLUMNS, features)}
            rows.append(row)
            labels.append(assigned_roles[row_index])
            years.append(draft_year)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="target_role")
    years_df = pd.DataFrame({"draft_year": years})
    return X, y, years_df


def get_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, str]:
    """Load processed data or fall back to synthetic checkpoint data."""
    processed = load_processed_training_data()
    if processed is not None:
        X, y, years = processed
        return X, y, years, "processed"

    X, y, years = make_synthetic_training_data()
    return X, y, years, "synthetic"


def train_all_models() -> dict[str, dict[str, float]]:
    """Train baseline and advanced models, then persist artifacts."""
    X, y, years, source = get_training_data()
    print(f"Training data source: {source}")
    print(f"X shape: {X.shape}")
    print("Class distribution:")
    print(y.value_counts().reindex(config.ROLE_LABELS, fill_value=0).to_string())

    X_trains, y_trains = split_data_temporal(X, y, years)
    X_train, X_val, X_test = X_trains
    y_train, y_val, y_test = y_trains

    print("\nTemporal split:")
    print(f"Train: {X_train.shape}, {y_train.value_counts().to_dict()}")
    print(f"Val:   {X_val.shape}, {y_val.value_counts().to_dict()}")
    print(f"Test:  {X_test.shape}, {y_test.value_counts().to_dict()}")

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    X_trains_smote = (X_train_smote, X_val, X_test)
    y_trains_smote = (y_train_smote, y_val, y_test)

    metrics: dict[str, dict[str, float]] = {}

    trained_models: dict[str, Any] = {}

    trained_models["dummy_baseline"], metrics["dummy_baseline"] = train_dummy_baseline(X_trains, y_trains)
    trained_models["lr_baseline"], metrics["lr_baseline"] = train_logistic_regression(X_trains_smote, y_trains_smote)
    trained_models["rf_baseline"], metrics["rf_baseline"] = train_random_forest(X_trains_smote, y_trains_smote)
    trained_models["xgboost_model"], metrics["xgboost_model"] = train_xgboost(X_trains, y_trains)
    trained_models["xgboost_smote_model"], metrics["xgboost_smote_model"] = train_xgboost(
        X_trains_smote,
        y_trains_smote,
    )
    trained_models["catboost_model"], metrics["catboost_model"] = train_catboost(X_trains_smote, y_trains_smote)

    best_model_name = select_best_model_name(metrics)
    saved_paths = {
        model_name: save_model(model, f"{model_name}.pkl")
        for model_name, model in trained_models.items()
    }
    saved_paths["best_model"] = save_model(trained_models[best_model_name], "best_model.pkl")

    baseline_metrics = {
        name: metrics[name]
        for name in ["dummy_baseline", "lr_baseline", "rf_baseline"]
    }
    baseline_metrics_path = save_metrics(baseline_metrics, filename="baseline_metrics.json")
    metrics_path = save_metrics(metrics, filename="model_metrics.json")
    comparison_path = save_comparison(metrics)

    print("\nModel artifacts saved:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")
    print(f"Baseline metrics saved: {baseline_metrics_path}")
    print(f"All metrics saved: {metrics_path}")
    print(f"Comparison saved: {comparison_path}")
    print(f"Best model by validation Macro F1: {best_model_name}")

    print("\nModel comparison:")
    comparison = pd.DataFrame(metrics).T.sort_values("validation_macro_f1", ascending=False)
    print(comparison.to_string(float_format=lambda value: f"{value:.4f}"))

    return metrics


def train_all_baselines() -> dict[str, dict[str, float]]:
    """Backward-compatible entry point for the complete training pipeline."""
    return train_all_models()


def main() -> None:
    """Run the training checkpoint."""
    train_all_models()
    print("\nTraining checkpoint completed.")


if __name__ == "__main__":
    main()
