"""SHAP explanations for trained role-prediction models."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import src.config as config
from src.models.guarded_model import StarterPlusFamilyVetoRoleClassifier, StarterPlusGuardedRoleClassifier
from src.models.grouped_model import GroupedXGBoostClassifier, RoleGroupCascadeClassifier
from src.models.probability_adjusted_model import ProbabilityAdjustedRoleClassifier
from src.models.train import XGBoostRoleClassifier, get_training_data, split_data_temporal
from src.models.two_stage_model import BinaryXGBoostClassifier, TwoStageRoleClassifier


def load_model(path: Path | None = None) -> Any:
    """Load the best trained model artifact."""
    env_path = os.getenv("SHAP_MODEL_PATH")
    model_path = path or (Path(env_path) if env_path else config.MODELS_DIR / "best_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo entrenado: {model_path}. "
            "Ejecuta primero `docker compose run --rm model-train`."
        )

    # Models saved from `python -m src.models.train` can carry a `__main__`
    # reference for this small wrapper. Register it for robust loading.
    sys.modules["__main__"].XGBoostRoleClassifier = XGBoostRoleClassifier
    sys.modules["__main__"].StarterPlusGuardedRoleClassifier = StarterPlusGuardedRoleClassifier
    sys.modules["__main__"].StarterPlusFamilyVetoRoleClassifier = StarterPlusFamilyVetoRoleClassifier
    sys.modules["__main__"].GroupedXGBoostClassifier = GroupedXGBoostClassifier
    sys.modules["__main__"].RoleGroupCascadeClassifier = RoleGroupCascadeClassifier
    sys.modules["__main__"].ProbabilityAdjustedRoleClassifier = ProbabilityAdjustedRoleClassifier
    sys.modules["__main__"].BinaryXGBoostClassifier = BinaryXGBoostClassifier
    sys.modules["__main__"].TwoStageRoleClassifier = TwoStageRoleClassifier

    with model_path.open("rb") as file:
        return pickle.load(file)


def get_test_data() -> tuple[pd.DataFrame, pd.Series, str]:
    """Return the test split from processed or synthetic training data."""
    X, y, years, source = get_training_data()
    X_trains, y_trains = split_data_temporal(X, y, years)
    _, _, X_test = X_trains
    _, _, y_test = y_trains
    return X_test, y_test, source


def get_tree_model(model: Any) -> Any:
    """Return the object SHAP should explain for tree-based models."""
    if isinstance(model, ProbabilityAdjustedRoleClassifier):
        return get_tree_model(model.base_model)
    if isinstance(model, StarterPlusGuardedRoleClassifier):
        return get_tree_model(model.base_model)
    if hasattr(model, "booster"):
        return model.booster
    return model


def get_predicted_label(model: Any, X_row: pd.DataFrame) -> str:
    """Return a string label for one prediction."""
    prediction = np.asarray(model.predict(X_row)).ravel()[0]
    if isinstance(prediction, (np.integer, int)):
        return config.ROLE_LABELS[int(prediction)]
    return str(prediction)


def get_predicted_class_index(model: Any, X_row: pd.DataFrame) -> int:
    """Return the project role index for the model prediction."""
    label = get_predicted_label(model, X_row)
    if label in config.ROLE_LABELS:
        return config.ROLE_LABELS.index(label)
    raise ValueError(f"Prediccion fuera de ROLE_LABELS: {label}")


def compute_shap_values(model: Any, X: pd.DataFrame) -> Any:
    """Compute SHAP values with a TreeExplainer when possible."""
    import shap

    explainer = shap.TreeExplainer(get_tree_model(model))
    return explainer.shap_values(X)


def _values_array(shap_values: Any) -> np.ndarray:
    """Extract a numpy array from SHAP outputs across versions."""
    if hasattr(shap_values, "values"):
        return np.asarray(shap_values.values)
    return np.asarray(shap_values)


def select_shap_values_for_class(
    shap_values: Any,
    sample_idx: int,
    class_idx: int,
    n_features: int,
) -> np.ndarray:
    """Select one sample/class SHAP vector from list or ndarray outputs."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx][sample_idx], dtype=float)

    values = _values_array(shap_values)

    if values.ndim == 3:
        if values.shape[1] == n_features:
            return np.asarray(values[sample_idx, :, class_idx], dtype=float)
        if values.shape[2] == n_features:
            return np.asarray(values[sample_idx, class_idx, :], dtype=float)

    if values.ndim == 2:
        if values.shape[0] > sample_idx and values.shape[1] == n_features:
            return np.asarray(values[sample_idx, :], dtype=float)
        if values.shape[0] == n_features:
            return np.asarray(values[:, class_idx], dtype=float)

    if values.ndim == 1 and values.shape[0] == n_features:
        return np.asarray(values, dtype=float)

    raise ValueError(f"Forma de SHAP no soportada: {values.shape}")


def mean_abs_shap_by_class(
    shap_values: Any,
    class_idx: int,
    n_features: int,
) -> np.ndarray:
    """Return mean absolute SHAP values for one class."""
    if isinstance(shap_values, list):
        return np.abs(np.asarray(shap_values[class_idx], dtype=float)).mean(axis=0)

    values = _values_array(shap_values)

    if values.ndim == 3:
        if values.shape[1] == n_features:
            return np.abs(values[:, :, class_idx]).mean(axis=0)
        if values.shape[2] == n_features:
            return np.abs(values[:, class_idx, :]).mean(axis=0)

    if values.ndim == 2 and values.shape[1] == n_features:
        return np.abs(values).mean(axis=0)

    raise ValueError(f"Forma de SHAP no soportada para resumen: {values.shape}")


def top_shap_features(
    values: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
    positive: bool = True,
) -> list[dict[str, float | str]]:
    """Return top SHAP contributors for one prediction."""
    items = list(zip(feature_names, values))
    if positive:
        selected = sorted(items, key=lambda item: item[1], reverse=True)[:top_n]
    else:
        selected = sorted(items, key=lambda item: item[1])[:top_n]

    return [
        {"feature": feature, "shap_value": float(value)}
        for feature, value in selected
    ]


def explain_prediction_shap(
    model: Any,
    X_test: pd.DataFrame,
    player_idx: int,
    shap_values: Any | None = None,
) -> dict[str, Any]:
    """Explain one test prediction with SHAP values."""
    if player_idx < 0 or player_idx >= len(X_test):
        raise IndexError(f"player_idx fuera de rango: {player_idx}")

    X_row = X_test.iloc[[player_idx]]
    predicted_label = get_predicted_label(model, X_row)
    class_idx = get_predicted_class_index(model, X_row)
    all_shap_values = shap_values if shap_values is not None else compute_shap_values(model, X_test)
    values = select_shap_values_for_class(
        all_shap_values,
        sample_idx=player_idx,
        class_idx=class_idx,
        n_features=X_test.shape[1],
    )

    positive = top_shap_features(values, X_test.columns.tolist(), top_n=5, positive=True)
    negative = top_shap_features(values, X_test.columns.tolist(), top_n=5, positive=False)

    explanation = {
        "player_idx": int(player_idx),
        "predicted_role": predicted_label,
        "positive": positive,
        "negative": negative,
    }

    print(f"\n=== EXPLICACION SHAP: muestra {player_idx} ===")
    print(f"Rol predicho: {predicted_label}")
    print("Top factores positivos:")
    for item in positive:
        print(f"  + {item['feature']}: {item['shap_value']:.4f}")
    print("Top factores negativos:")
    for item in negative:
        print(f"  - {item['feature']}: {item['shap_value']:.4f}")

    return explanation


def summary_plot(
    model: Any,
    X_test: pd.DataFrame,
    shap_values: Any | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build a global SHAP importance summary by role."""
    all_shap_values = shap_values if shap_values is not None else compute_shap_values(model, X_test)
    rows: list[dict[str, float | str | int]] = []

    for class_idx, role in enumerate(config.ROLE_LABELS):
        try:
            importances = mean_abs_shap_by_class(all_shap_values, class_idx, X_test.shape[1])
        except ValueError:
            if class_idx > 0:
                break
            raise

        ranked = sorted(
            zip(X_test.columns.tolist(), importances),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]
        for rank, (feature, value) in enumerate(ranked, start=1):
            rows.append(
                {
                    "role": role,
                    "rank": rank,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                }
            )

    summary = pd.DataFrame(rows)

    print("\n=== RESUMEN GLOBAL SHAP ===")
    if summary.empty:
        print("No se genero resumen SHAP.")
    else:
        for role in summary["role"].unique():
            print(f"\nClase: {role}")
            role_summary = summary[summary["role"] == role].head(top_n)
            print(role_summary[["rank", "feature", "mean_abs_shap"]].to_string(index=False))

    return summary


def save_explanations(
    explanations: list[dict[str, Any]],
    summary: pd.DataFrame,
) -> tuple[Path, Path]:
    """Persist individual and global SHAP explanation artifacts."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    explanations_path = config.REPORTS_DIR / "shap_explanations.json"
    with explanations_path.open("w", encoding="utf-8") as file:
        json.dump(explanations, file, indent=2, sort_keys=True)

    summary_path = config.REPORTS_DIR / "shap_summary.csv"
    summary.to_csv(summary_path, index=False)
    return explanations_path, summary_path


def main() -> None:
    """Run the SHAP explanation checkpoint."""
    model = load_model()
    X_test, y_test, source = get_test_data()

    print("=== SHAP interpretability checkpoint ===")
    print(f"Data source: {source}")
    print(f"Test shape: {X_test.shape}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")

    shap_values = compute_shap_values(model, X_test)
    sample_indices = list(range(min(3, len(X_test))))
    explanations = [
        explain_prediction_shap(model, X_test, player_idx=index, shap_values=shap_values)
        for index in sample_indices
    ]
    summary = summary_plot(model, X_test, shap_values=shap_values, top_n=10)
    explanations_path, summary_path = save_explanations(explanations, summary)

    print("\nArtifacts saved:")
    print(f"  {explanations_path}")
    print(f"  {summary_path}")
    print("\nSHAP checkpoint completed.")


if __name__ == "__main__":
    main()
