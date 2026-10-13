"""Generate final evaluation artifacts for the NCAA to NBA role model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import src.config as config
from src.models.explain import load_model
from src.models.train import (
    TEST_START_DRAFT_YEAR,
    TEST_END_DRAFT_YEAR,
    TRAIN_MAX_DRAFT_YEAR,
    VALIDATION_END_DRAFT_YEAR,
    VALIDATION_START_DRAFT_YEAR,
    XGBoostRoleClassifier,
    get_training_data,
    split_data_temporal,
)
from src.models.probability_adjusted_model import ProbabilityAdjustedRoleClassifier
from src.models.guarded_model import StarterPlusFamilyVetoRoleClassifier, StarterPlusGuardedRoleClassifier
from src.models.grouped_model import GROUPED_LABELS, GroupedXGBoostClassifier, RoleGroupCascadeClassifier, group_roles
from src.models.two_stage_model import TwoStageRoleClassifier


POST_TRAIN_MODEL_NAMES = {
    "xgboost_starter_plus_guarded",
    "xgboost_starter_plus_threshold_guarded",
    "xgboost_starter_plus_family_veto",
    "grouped_to_fine_cascade",
    "grouped_xgboost_model",
    "xgboost_probability_adjusted",
}


def _model_name(model: Any) -> str:
    """Return a readable model name for reports."""
    if isinstance(model, TwoStageRoleClassifier):
        return "two_stage_role_model"
    if isinstance(model, RoleGroupCascadeClassifier):
        return "grouped_to_fine_cascade"
    if isinstance(model, GroupedXGBoostClassifier):
        return "grouped_xgboost_model"
    if isinstance(model, StarterPlusFamilyVetoRoleClassifier):
        return "xgboost_starter_plus_family_veto"
    if isinstance(model, StarterPlusGuardedRoleClassifier):
        if float(getattr(model, "minimum_starter_plus_probability", 0.0)) > 0.0:
            return "xgboost_starter_plus_threshold_guarded"
        return "xgboost_starter_plus_guarded"
    if isinstance(model, ProbabilityAdjustedRoleClassifier):
        return "xgboost_probability_adjusted"
    if isinstance(model, XGBoostRoleClassifier):
        return "xgboost_model"
    return type(model).__name__


def _comparison_with_current_model(
    comparison: pd.DataFrame,
    model_name: str,
    metrics: dict[str, float],
) -> pd.DataFrame:
    """Add the final promoted model to the comparison table when needed."""
    rows = comparison.copy()
    if rows.empty:
        rows = pd.DataFrame(columns=["accuracy", "balanced_accuracy", "macro_f1"])
    stale_post_train_rows = POST_TRAIN_MODEL_NAMES.intersection(rows.index).difference({model_name})
    if stale_post_train_rows:
        rows = rows.drop(index=list(stale_post_train_rows), errors="ignore")
    rows.loc[model_name, ["accuracy", "balanced_accuracy", "macro_f1"]] = [
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
    ]
    validation_macro = _current_model_validation_macro_f1(model_name)
    if validation_macro is not None:
        rows.loc[model_name, "validation_macro_f1"] = validation_macro
    sort_column = "validation_macro_f1" if "validation_macro_f1" in rows.columns else "macro_f1"
    return rows.sort_values(sort_column, ascending=False)


def _current_model_validation_macro_f1(model_name: str) -> float | None:
    """Return validation Macro F1 for post-training wrappers when available."""
    if model_name == "two_stage_role_model":
        guard_path = config.REPORTS_DIR / "starter_plus_guard_experiment.json"
        if guard_path.exists():
            with guard_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if payload.get("recommendation") == "two_stage_model":
                for result in payload.get("validation_results", []):
                    if result.get("variant") == "two_stage_model":
                        return float(result.get("macro_f1", 0.0))

        two_stage_path = config.REPORTS_DIR / "two_stage_metrics.json"
        if not two_stage_path.exists():
            return None
        with two_stage_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        selected_gate = payload.get("selected_gate_model")
        combined = payload.get("combined_validation", {})
        if selected_gate in combined:
            return float(combined[selected_gate].get("validation_macro_f1", 0.0))

    if model_name in {"xgboost_starter_plus_guarded", "xgboost_starter_plus_threshold_guarded"}:
        guard_path = config.REPORTS_DIR / "starter_plus_guard_experiment.json"
        if not guard_path.exists():
            return None

        with guard_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        recommendation = payload.get("recommendation")
        for result in payload.get("validation_results", []):
            if result.get("variant") == recommendation:
                return float(result.get("macro_f1", 0.0))

    if model_name in {"grouped_to_fine_cascade", "grouped_xgboost_model"}:
        grouped_path = config.REPORTS_DIR / "bench_rotation_grouping_experiment.json"
        if not grouped_path.exists():
            return None

        with grouped_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        result_key = "cascade_results" if model_name == "grouped_to_fine_cascade" else "grouped_results"
        expected_name = "grouped_to_fine_cascade" if model_name == "grouped_to_fine_cascade" else "xgboost_trained_on_grouped_roles"
        for result in payload.get(result_key, []):
            if result.get("name") == expected_name and result.get("split") == "validation":
                return float(result.get("macro_f1", 0.0))

    if model_name == "xgboost_probability_adjusted":
        lightweight_path = config.REPORTS_DIR / "lightweight_alternatives_experiment.json"
        if not lightweight_path.exists():
            return None

        with lightweight_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        for result in payload.get("results", []):
            if result.get("name") == "xgboost_probability_adjusted_light":
                return float(result.get("validation_macro_f1", 0.0))
    return None


def _evaluation_labels(model: Any) -> list[str]:
    """Return the label set that should be used to evaluate the loaded model."""
    classes = getattr(model, "classes_", None)
    labels = [str(label) for label in list(classes)] if classes is not None else list(config.ROLE_LABELS)
    if set(labels) == set(GROUPED_LABELS):
        return list(GROUPED_LABELS)
    return list(config.ROLE_LABELS)


def _evaluation_target(y_true: pd.Series, labels: list[str]) -> pd.Series:
    """Map the fine target to grouped labels when the final model is grouped."""
    if labels == list(GROUPED_LABELS):
        grouped = group_roles(y_true)
        grouped.index = y_true.index
        return grouped
    return y_true


def _as_label_series(predictions: Any, index: pd.Index, labels: list[str] | None = None) -> pd.Series:
    """Normalize estimator predictions to project role labels."""
    labels = labels or list(config.ROLE_LABELS)
    values = np.asarray(predictions).ravel()
    normalized: list[str] = []
    for value in values:
        try:
            numeric_value = int(value)
        except (TypeError, ValueError):
            numeric_value = -1

        if 0 <= numeric_value < len(labels) and str(value) not in labels:
            normalized.append(labels[numeric_value])
            continue

        normalized.append(str(value))
    return pd.Series(normalized, index=index, name="predicted_role")


def _probability_frame(model: Any, X_test: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Return predicted probabilities aligned to the evaluation labels."""
    raw = np.asarray(model.predict_proba(X_test))
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    classes = getattr(model, "classes_", None)
    model_labels = [str(label) for label in list(classes)] if classes is not None else list(labels)
    if len(model_labels) != raw.shape[1]:
        model_labels = list(labels[: raw.shape[1]])

    probabilities = pd.DataFrame(raw, columns=model_labels, index=X_test.index)
    for label in labels:
        if label not in probabilities.columns:
            probabilities[label] = 0.0
    probabilities = probabilities[labels].clip(lower=1e-12, upper=1.0)
    row_sums = probabilities.sum(axis=1).replace(0, np.nan)
    return probabilities.div(row_sums, axis=0).fillna(1.0 / len(labels))


def _top_k_accuracy(y_true: pd.Series, probabilities: pd.DataFrame, k: int = 2) -> float:
    """Compute top-k accuracy without depending on label presence in y_true."""
    top_labels = probabilities.apply(
        lambda row: set(row.nlargest(k).index),
        axis=1,
    )
    hits = [
        str(true_label) in predicted_labels
        for true_label, predicted_labels in zip(y_true, top_labels)
    ]
    return float(np.mean(hits)) if hits else 0.0


def _multiclass_log_loss(y_true: pd.Series, probabilities: pd.DataFrame) -> float:
    """Compute multiclass log loss with the project's explicit label order."""
    true_probabilities = []
    for label, (_, row) in zip(y_true, probabilities.iterrows()):
        true_probabilities.append(float(row.get(str(label), 1e-12)))
    return float(-np.mean(np.log(np.clip(true_probabilities, 1e-12, 1.0))))


def _load_master_summary() -> dict[str, Any]:
    """Load the processed-data summary if it exists."""
    summary_path = config.REPORTS_DIR / "master_table_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_confusion_matrix_plot(cm: np.ndarray, labels: list[str], path: Path) -> None:
    """Save a readable confusion matrix PNG."""
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
        title="Matriz de confusion - test temporal",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _row_normalized_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    """Normalize a confusion matrix by real-class row totals."""
    row_totals = cm.sum(axis=1, keepdims=True)
    return np.divide(
        cm.astype(float),
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals > 0,
    )


def _save_normalized_confusion_matrix_plot(
    cm: np.ndarray,
    cm_normalized: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    """Save a row-normalized confusion matrix with count annotations."""
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    image = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Porcentaje dentro de la clase real", rotation=270, labelpad=18)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Real",
        xlabel="Predicho",
        title="Matriz de confusion normalizada por clase real",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        row_total = int(cm[i].sum())
        for j in range(cm.shape[1]):
            percentage = cm_normalized[i, j] * 100.0
            count = int(cm[i, j])
            text = f"{percentage:.1f}%\n({count}/{row_total})" if row_total else "0.0%\n(0/0)"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] >= 0.55 else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_model_card(
    *,
    model_name: str,
    metrics: dict[str, float],
    master_summary: dict[str, Any],
    test_distribution: dict[str, int],
    comparison: pd.DataFrame,
    labels: list[str],
) -> str:
    """Create the final model card markdown content."""
    model_rows = "\n".join(
        f"| {index} | {row.get('validation_macro_f1', np.nan):.4f} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} |"
        for index, row in comparison.iterrows()
    )
    target_distribution = master_summary.get("target_distribution", {})
    target_rows = "\n".join(
        f"| `{label}` | {int(target_distribution.get(label, 0))} |"
        for label in config.ROLE_LABELS
    )
    test_rows = "\n".join(
        f"| `{label}` | {int(test_distribution.get(label, 0))} |"
        for label in labels
    )

    draft_source = master_summary.get("draft_source_distribution", {})
    friendly_source_names = {
        "basketball_reference_draft": "Basketball Reference Draft",
        "ncaa_nba_excel_2009_2021": "Excel historico NCAA/NBA 2009-2021",
        "twenty_years_nba_draft_data": "20 Years of NBA Draft Data",
        "not_drafted": "No drafteados",
    }
    draft_source_rows = "\n".join(
        f"| {friendly_source_names.get(str(source), str(source))} | {int(count):,} |"
        for source, count in draft_source.items()
    )
    undrafted_relabel_distribution = master_summary.get("undrafted_nba_relabel_target_distribution", {})
    undrafted_relabel_rows = "\n".join(
        f"| `{label}` | {int(undrafted_relabel_distribution.get(label, 0))} |"
        for label in config.ROLE_LABELS
    )

    return f"""# Model Card - NCAA to NBA Role Predictor

## Objetivo

Este modelo estima el rol NBA temprano de un jugador a partir de su perfil NCAA: produccion por 40 minutos, eficiencia, fisico y contexto competitivo del equipo. La salida final del proyecto usa tres clases:

```text
NON_NBA < NBA_ROLE_PLAYER < STARTER_PLUS
```

`NBA_ROLE_PLAYER` agrupa las etiquetas finas `BENCH` y `ROTATION`, que eran la frontera mas ruidosa del problema. La taxonomia fina se conserva para construir el target y documentar experimentos, pero la API y la demo deben leerse en la salida de 3 clases.

El sistema esta pensado como herramienta academica y de exploracion: acompania la prediccion con comparables en Milvus y explicaciones SHAP, no como sustituto de scouting humano.

## Arquitectura

- Modelo seleccionado: `{model_name}`.
- Entrenamiento con split temporal para reducir fuga de informacion.
- El modelo final entrena directamente `NON_NBA`, `NBA_ROLE_PLAYER` y `STARTER_PLUS` sin SMOTE, para tratar `STARTER_PLUS` como una etiqueta conservadora.
- La salida fina de 4 clases queda como material comparativo e historico.
- Se probaron variantes con SMOTE, dos etapas, guardarrailes para `STARTER_PLUS`, ajuste de probabilidades, embeddings y modelos ligeros.
- Variables normalizadas con z-score ajustado solo en entrenamiento y reutilizadas por API, entrenamiento y busqueda vectorial.
- Milvus almacena vectores numericos y texto narrativo para recuperar perfiles historicos similares.

## Dataset

| Elemento | Valor |
| --- | ---: |
| Filas totales | {int(master_summary.get("master_rows", 0)):,} |
| Filas drafteadas enlazadas | {int(master_summary.get("drafted_rows", 0)):,} |
| Filas no drafteadas | {int(master_summary.get("non_drafted_rows", 0)):,} |
| Anio minimo modelado | {int(master_summary.get("draft_year_min", 0))} |
| Anio maximo modelado | {int(master_summary.get("draft_year_max", 0))} |
| Variables de entrada | {len(master_summary.get("feature_columns", config.MODEL_FEATURE_COLUMNS))} |

Procedencia de draft:

| Fuente | Filas |
| --- | ---: |
{draft_source_rows}

Distribucion global del target:

| Rol | Filas |
| --- | ---: |
{target_rows}

Correccion de no drafteados con evidencia NBA:

| Rol tras relabel conservador | Filas |
| --- | ---: |
{undrafted_relabel_rows}

El relabel de no drafteados usa datos NBA solo para construir el target, no como variables de entrada. La correccion exige match de nombre y universidad compatible contra `NBA Dataset Box Scores and Stats/Players.csv`, y despues calcula el outcome con las mismas reglas de primeras temporadas activas usadas para drafteados.

Split temporal:

| Bloque | Rango |
| --- | --- |
| Entrenamiento | `draft_year <= {TRAIN_MAX_DRAFT_YEAR}` |
| Validacion | `{VALIDATION_START_DRAFT_YEAR}-{VALIDATION_END_DRAFT_YEAR}` |
| Test final | `{TEST_START_DRAFT_YEAR}-{TEST_END_DRAFT_YEAR}` |

Distribucion del test en la evaluacion final:

| Rol | Filas |
| --- | ---: |
{test_rows}

## Metricas del mejor modelo

| Metrica | Valor |
| --- | ---: |
| Accuracy | {metrics["accuracy"]:.4f} |
| Balanced Accuracy | {metrics["balanced_accuracy"]:.4f} |
| Macro F1 | {metrics["macro_f1"]:.4f} |
| Weighted F1 | {metrics["weighted_f1"]:.4f} |
| Top-2 Accuracy | {metrics["top_2_accuracy"]:.4f} |
| Log Loss | {metrics["log_loss"]:.4f} |

Comparacion de modelos:

| Modelo | Validacion Macro F1 | Test Accuracy | Test Balanced Accuracy | Test Macro F1 |
| --- | ---: | ---: | ---: | ---: |
{model_rows}

## Limitaciones

- El problema esta fuertemente desbalanceado: `NON_NBA` domina la cohorte.
- La clase `STARTER_PLUS` agrupa titulares y estrellas para evitar una division artificial con pocos casos.
- Los jugadores internacionales o rutas no NCAA quedan fuera del alcance del predictor tabular.
- Las cohortes recientes deben tratarse con cuidado: 2023 queda fuera del test final porque su horizonte NBA es menos estable.
- Las predicciones deben interpretarse junto con probabilidades, SHAP y comparables, no como una sentencia unica.

## Diagnostico de la matriz de confusion

La matriz de confusion final se lee sobre tres clases. Agrupar `BENCH` y `ROTATION` reduce el ruido de una frontera que no era estable en test y centra la decision en la pregunta defendible: sin nivel NBA, jugador NBA de rol o perfil de titular/estrella temprana.

La auditoria de features muestra que las variables oficiales de BigQuery aportan algo de senial, pero su cobertura se concentra en 2014-2018. Por eso quedan documentadas como senial util con riesgo de covariate shift.

## Mapeo de etiquetas

| Clase fina | Clase agrupada |
| --- | --- |
| `NON_NBA` | `NON_NBA` |
| `BENCH` | `NBA_ROLE_PLAYER` |
| `ROTATION` | `NBA_ROLE_PLAYER` |
| `STARTER_PLUS` | `STARTER_PLUS` |

El modelo final se entrena ya sobre la clase agrupada, no solo como un mapeo posterior.

## Enfoque Comparativo

El proyecto conserva el enfoque en dos etapas como comparacion metodologica. La arquitectura alternativa sigue siendo util para analizar el problema:

| Etapa | Pregunta | Target |
| --- | --- | --- |
| 1 | Tiene perfil NBA? | `NON_NBA` vs `NBA` |
| 2 | Si tiene perfil NBA, que rol alcanza? | `BENCH`, `ROTATION`, `STARTER_PLUS` |

Esta separacion evita que `NON_NBA` inunde la decision de rol. En la ejecucion actual queda como comparativa, no como modelo final.

## Uso Recomendado

El uso mas defendible es mostrar la salida de 3 clases junto con probabilidades, comparables y explicaciones. En presentacion academica, el valor principal esta en la trazabilidad del flujo completo: fuentes, target, features, modelo, explicabilidad y recuperacion vectorial.
"""


def generate_final_report() -> dict[str, str]:
    """Generate final reports, metrics and confusion matrix artifacts."""
    model = load_model()
    X, y, years, source = get_training_data()
    X_trains, y_trains = split_data_temporal(X, y, years)
    _, _, X_test = X_trains
    _, _, y_test = y_trains

    labels = _evaluation_labels(model)
    y_eval = _evaluation_target(y_test, labels)
    y_pred = _as_label_series(model.predict(X_test), y_test.index, labels)
    probabilities = _probability_frame(model, X_test, labels)

    metrics = {
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        "macro_f1": float(
            f1_score(y_eval, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_eval, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "top_2_accuracy": _top_k_accuracy(y_eval, probabilities, k=2),
        "log_loss": _multiclass_log_loss(y_eval, probabilities),
    }

    report_dict = classification_report(
        y_eval,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    cm = confusion_matrix(y_eval, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.REPORTS_DIR / "metrics.json"
    classification_report_path = config.REPORTS_DIR / "classification_report.csv"
    confusion_matrix_csv_path = config.REPORTS_DIR / "confusion_matrix.csv"
    confusion_matrix_png_path = config.REPORTS_DIR / "confusion_matrix.png"
    confusion_matrix_normalized_csv_path = config.REPORTS_DIR / "confusion_matrix_normalized.csv"
    confusion_matrix_normalized_png_path = config.REPORTS_DIR / "confusion_matrix_normalized.png"
    model_card_path = config.REPORTS_DIR / "model_card.md"

    master_summary = _load_master_summary()
    test_distribution = y_eval.value_counts().reindex(labels, fill_value=0).astype(int).to_dict()
    comparison_path = config.REPORTS_DIR / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path, index_col="model") if comparison_path.exists() else pd.DataFrame()

    payload = {
        "model_name": _model_name(model),
        "data_source": source,
        "split": {
            "train": f"draft_year <= {TRAIN_MAX_DRAFT_YEAR}",
            "validation": f"{VALIDATION_START_DRAFT_YEAR}-{VALIDATION_END_DRAFT_YEAR}",
            "test": f"{TEST_START_DRAFT_YEAR}-{TEST_END_DRAFT_YEAR}",
        },
        "test_rows": int(len(y_test)),
        "test_distribution": test_distribution,
        "labels": labels,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    report_df.to_csv(classification_report_path)
    cm_df.to_csv(confusion_matrix_csv_path)
    _save_confusion_matrix_plot(cm, labels, confusion_matrix_png_path)
    cm_normalized = _row_normalized_confusion_matrix(cm)
    cm_normalized_df = pd.DataFrame(cm_normalized, index=labels, columns=labels)
    cm_normalized_df.to_csv(confusion_matrix_normalized_csv_path)
    _save_normalized_confusion_matrix_plot(
        cm,
        cm_normalized,
        labels,
        confusion_matrix_normalized_png_path,
    )

    model_name = _model_name(model)
    comparison = _comparison_with_current_model(comparison, model_name, metrics)
    comparison.to_csv(comparison_path, index_label="model")

    model_card = _build_model_card(
        model_name=model_name,
        metrics=metrics,
        master_summary=master_summary,
        test_distribution=test_distribution,
        comparison=comparison,
        labels=labels,
    )
    model_card_path.write_text(model_card, encoding="utf-8")

    print("=== Reporte final generado ===")
    print(f"Modelo: {_model_name(model)}")
    print(f"Fuente de datos: {source}")
    print(f"Test rows: {len(y_test)}")
    print("Metricas:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("Artefactos:")
    for path in [
        metrics_path,
        classification_report_path,
        confusion_matrix_csv_path,
        confusion_matrix_png_path,
        confusion_matrix_normalized_csv_path,
        confusion_matrix_normalized_png_path,
        model_card_path,
    ]:
        print(f"  {path}")

    return {
        "metrics": str(metrics_path),
        "classification_report": str(classification_report_path),
        "confusion_matrix_csv": str(confusion_matrix_csv_path),
        "confusion_matrix_png": str(confusion_matrix_png_path),
        "confusion_matrix_normalized_csv": str(confusion_matrix_normalized_csv_path),
        "confusion_matrix_normalized_png": str(confusion_matrix_normalized_png_path),
        "model_card": str(model_card_path),
    }


def main() -> None:
    """CLI entrypoint for Docker and notebooks."""
    generate_final_report()


if __name__ == "__main__":
    main()
