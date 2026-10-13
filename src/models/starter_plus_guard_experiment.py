"""Evaluate conservative guards for STARTER_PLUS predictions."""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from src import config
from src.models.guarded_model import StarterPlusFamilyVetoRoleClassifier, StarterPlusGuardedRoleClassifier
from src.models.train import XGBoostRoleClassifier, split_data_temporal
from src.models.two_stage_model import TwoStageRoleClassifier


@dataclass
class GuardResult:
    """Serializable metrics for one STARTER_PLUS guard variant."""

    variant: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    non_nba_to_starter_plus: int
    starter_plus_recall: float
    confusion_matrix: list[list[int]]
    notes: str


def _load_pickle(path: str) -> Any:
    """Load project pickles that may have been saved from a script entrypoint."""
    sys.modules["__main__"].XGBoostRoleClassifier = XGBoostRoleClassifier
    sys.modules["__main__"].TwoStageRoleClassifier = TwoStageRoleClassifier
    with open(path, "rb") as file:
        return pickle.load(file)


def _evaluate_variant(
    *,
    variant: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    notes: str,
) -> GuardResult:
    cm = confusion_matrix(y_true, y_pred, labels=config.ROLE_LABELS)
    starter_index = config.ROLE_LABELS.index("STARTER_PLUS")
    starter_support = cm[starter_index].sum()
    starter_recall = float(cm[starter_index, starter_index] / starter_support) if starter_support else 0.0

    return GuardResult(
        variant=variant,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(
            f1_score(
                y_true,
                y_pred,
                labels=config.ROLE_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        non_nba_to_starter_plus=int(((y_true.to_numpy() == "NON_NBA") & (y_pred == "STARTER_PLUS")).sum()),
        starter_plus_recall=starter_recall,
        confusion_matrix=cm.astype(int).tolist(),
        notes=notes,
    )


def _to_csv_rows(results: list[GuardResult]) -> pd.DataFrame:
    """Flatten result objects for a compact CSV report."""
    rows = []
    for result in results:
        row = asdict(result)
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_predictions(base_pred: np.ndarray, two_stage_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Build guard variants without looking at ground-truth labels."""
    guard_to_rotation = base_pred.copy()
    guard_to_rotation[(guard_to_rotation == "STARTER_PLUS") & (two_stage_pred != "STARTER_PLUS")] = "ROTATION"

    guard_to_two_stage_family = base_pred.copy()
    veto_mask = (guard_to_two_stage_family == "STARTER_PLUS") & (two_stage_pred != "STARTER_PLUS")
    guard_to_two_stage_family[veto_mask & (two_stage_pred == "NON_NBA")] = "NON_NBA"
    guard_to_two_stage_family[veto_mask & (two_stage_pred != "NON_NBA")] = "ROTATION"

    return {
        "xgboost_base": base_pred,
        "two_stage_model": two_stage_pred,
        "starter_plus_requires_two_stage_confirmation": guard_to_rotation,
        "starter_plus_two_stage_veto_family": guard_to_two_stage_family,
    }


def _validation_priority_key(result: GuardResult) -> tuple[float, float, float, float]:
    """Rank guards by STARTER_PLUS recall first, then false positives and general quality."""
    return (
        result.starter_plus_recall,
        -float(result.non_nba_to_starter_plus),
        result.macro_f1,
        result.balanced_accuracy,
    )


def _structural_guard_priority(result: GuardResult) -> int:
    """Prefer the simplest model when validation metrics are otherwise tied."""
    if result.variant == "xgboost_base":
        return 3
    if result.variant == "two_stage_model":
        return 2
    if result.variant in {"starter_plus_requires_two_stage_confirmation", "starter_plus_two_stage_veto_family"}:
        return 1
    return 0


def _select_recommended_result(
    results: list[GuardResult],
    starter_plus_support: int,
) -> GuardResult:
    """Select a guard using validation, with a stability rule for tiny STARTER_PLUS support."""
    threshold_variant = "starter_plus_confirmed_min_base_probability"
    candidate_results = results

    if starter_plus_support < 3:
        candidate_results = [result for result in results if result.variant != threshold_variant]
        zero_false_positive = [
            result
            for result in candidate_results
            if result.non_nba_to_starter_plus == 0 and result.starter_plus_recall > 0.0
        ]
        if zero_false_positive:
            candidate_results = zero_false_positive

    return max(
        candidate_results,
        key=lambda result: (
            *_validation_priority_key(result),
            _structural_guard_priority(result),
        ),
    )


def _starter_plus_threshold_grid() -> np.ndarray:
    """Return the validation-only grid for the STARTER_PLUS confidence guard."""
    return np.round(np.linspace(0.0, 0.99, 100), 2)


def _threshold_guard_predictions(
    base_pred: np.ndarray,
    two_stage_pred: np.ndarray,
    base_probabilities: np.ndarray,
    minimum_probability: float,
    downgrade_role: str = "ROTATION",
) -> np.ndarray:
    """Veto low-confidence STARTER_PLUS predictions without using labels."""
    guarded = base_pred.copy()
    starter_index = config.ROLE_LABELS.index("STARTER_PLUS")
    veto_mask = (
        (guarded == "STARTER_PLUS")
        & (
            (two_stage_pred != "STARTER_PLUS")
            | (base_probabilities[:, starter_index] < minimum_probability)
        )
    )
    guarded[veto_mask] = downgrade_role
    return guarded


def _variant_notes() -> dict[str, str]:
    """Return stable notes for guard variants."""
    return {
        "xgboost_base": "Current 4-class XGBoost predictions.",
        "two_stage_model": "Existing two-stage model used as a conservative comparison.",
        "starter_plus_requires_two_stage_confirmation": (
            "Allow STARTER_PLUS only if the two-stage model also predicts STARTER_PLUS; "
            "otherwise downgrade to ROTATION."
        ),
        "starter_plus_two_stage_veto_family": (
            "If two-stage vetoes STARTER_PLUS, downgrade to NON_NBA when the gate rejects "
            "NBA profile, otherwise downgrade to ROTATION."
        ),
    }


def run_experiment() -> dict[str, Any]:
    """Run STARTER_PLUS guard variants and persist comparison reports."""
    X = pd.read_csv(config.DATA_PROCESSED / "X.csv")
    y = pd.read_csv(config.DATA_PROCESSED / "y.csv").squeeze("columns")
    master = pd.read_csv(config.DATA_PROCESSED / "master_table.csv", low_memory=False)
    X = X.select_dtypes(include=[np.number]).copy()

    (X_train, X_val, X_test), (_, y_val, y_test) = split_data_temporal(X, y, master[["draft_year"]])

    best_model = _load_pickle(str(config.MODELS_DIR / "xgboost_model.pkl"))
    two_stage_model = _load_pickle(str(config.MODELS_DIR / "two_stage_best_model.pkl"))

    labels = np.array(config.ROLE_LABELS)
    validation_base_probabilities = np.asarray(best_model.predict_proba(X_val), dtype=float)
    test_base_probabilities = np.asarray(best_model.predict_proba(X_test), dtype=float)
    validation_base_pred = labels[validation_base_probabilities.argmax(axis=1)]
    validation_two_stage_pred = np.asarray(two_stage_model.predict(X_val)).ravel()
    test_base_pred = labels[test_base_probabilities.argmax(axis=1)]
    test_two_stage_pred = np.asarray(two_stage_model.predict(X_test)).ravel()

    notes = _variant_notes()
    validation_predictions = _variant_predictions(validation_base_pred, validation_two_stage_pred)
    test_predictions = _variant_predictions(test_base_pred, test_two_stage_pred)
    validation_results = [
        _evaluate_variant(variant=variant, y_true=y_val, y_pred=prediction, notes=notes[variant])
        for variant, prediction in validation_predictions.items()
    ]
    test_results = [
        _evaluate_variant(variant=variant, y_true=y_test, y_pred=prediction, notes=notes[variant])
        for variant, prediction in test_predictions.items()
    ]

    threshold_variant = "starter_plus_confirmed_min_base_probability"
    best_threshold = 0.0
    best_threshold_result: GuardResult | None = None
    for threshold in _starter_plus_threshold_grid():
        threshold_prediction = _threshold_guard_predictions(
            validation_base_pred,
            validation_two_stage_pred,
            validation_base_probabilities,
            minimum_probability=float(threshold),
        )
        threshold_result = _evaluate_variant(
            variant=threshold_variant,
            y_true=y_val,
            y_pred=threshold_prediction,
            notes=(
                "Allow STARTER_PLUS only if the two-stage model also predicts STARTER_PLUS "
                f"and the base STARTER_PLUS probability is at least {float(threshold):.2f}; "
                "otherwise downgrade to ROTATION."
            ),
        )
        threshold_key = _validation_priority_key(threshold_result) + (-float(threshold),)
        best_key = (
            _validation_priority_key(best_threshold_result) + (-best_threshold,)
        ) if best_threshold_result is not None else None
        if best_key is None or threshold_key > best_key:
            best_threshold = float(threshold)
            best_threshold_result = threshold_result

    if best_threshold_result is None:
        raise RuntimeError("No STARTER_PLUS threshold candidate was evaluated.")

    validation_results.append(best_threshold_result)
    test_threshold_prediction = _threshold_guard_predictions(
        test_base_pred,
        test_two_stage_pred,
        test_base_probabilities,
        minimum_probability=best_threshold,
    )
    test_results.append(
        _evaluate_variant(
            variant=threshold_variant,
            y_true=y_test,
            y_pred=test_threshold_prediction,
            notes=(
                "Validation-selected STARTER_PLUS confidence guard "
                f"(minimum base probability {best_threshold:.2f})."
            ),
        )
    )
    starter_plus_support = int((y_val == "STARTER_PLUS").sum())
    recommended = _select_recommended_result(validation_results, starter_plus_support)

    payload = {
        "labels": config.ROLE_LABELS,
        "selection_set": f"{config.VALIDATION_START_DRAFT_YEAR}-{config.VALIDATION_END_DRAFT_YEAR}",
        "test_set": f"{config.TEST_START_DRAFT_YEAR}-{config.TEST_END_DRAFT_YEAR}",
        "validation_results": [asdict(result) for result in validation_results],
        "test_results": [asdict(result) for result in test_results],
        "recommendation": recommended.variant,
        "selected_thresholds": {threshold_variant: best_threshold},
        "interpretation": (
            "The guard recommendation is selected on validation only. Test metrics are reported "
            "after selection and are not used to choose the guard."
        ),
    }

    json_path = config.REPORTS_DIR / "starter_plus_guard_experiment.json"
    csv_path = config.REPORTS_DIR / "starter_plus_guard_experiment.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_rows = []
    for split_name, split_results in {"validation": validation_results, "test": test_results}.items():
        split_rows = _to_csv_rows(split_results)
        split_rows.insert(0, "split", split_name)
        csv_rows.append(split_rows)
    pd.concat(csv_rows, ignore_index=True).to_csv(csv_path, index=False)

    print("=== STARTER_PLUS guard experiment ===")
    print("Validation:")
    print(_to_csv_rows(validation_results).drop(columns=["confusion_matrix"]).to_string(index=False))
    print("\nTest:")
    print(_to_csv_rows(test_results).drop(columns=["confusion_matrix"]).to_string(index=False))
    print(f"\nRecommended from validation: {recommended.variant}")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return {"json": str(json_path), "csv": str(csv_path)}


def promote_guarded_model() -> dict[str, str]:
    """Persist the recommended guarded model as the selected best model."""
    base_model = _load_pickle(str(config.MODELS_DIR / "xgboost_model.pkl"))
    confirmation_model = _load_pickle(str(config.MODELS_DIR / "two_stage_best_model.pkl"))
    report_path = config.REPORTS_DIR / "starter_plus_guard_experiment.json"
    recommendation = "starter_plus_requires_two_stage_confirmation"
    selected_thresholds: dict[str, float] = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        recommendation = payload.get("recommendation", recommendation)
        selected_thresholds = payload.get("selected_thresholds", {})

    report_results = payload.get("validation_results", []) if report_path.exists() else []
    if report_results:
        validation_results = [
            GuardResult(
                variant=item["variant"],
                accuracy=float(item["accuracy"]),
                balanced_accuracy=float(item["balanced_accuracy"]),
                macro_f1=float(item["macro_f1"]),
                non_nba_to_starter_plus=int(item["non_nba_to_starter_plus"]),
                starter_plus_recall=float(item["starter_plus_recall"]),
                confusion_matrix=item["confusion_matrix"],
                notes=item["notes"],
            )
            for item in report_results
        ]
        starter_plus_support = int(sum(validation_results[0].confusion_matrix[-1])) if validation_results else 0
        best_validation_result = _select_recommended_result(validation_results, starter_plus_support)
        recommendation = best_validation_result.variant

    if recommendation == "xgboost_base":
        selected_model = base_model
    elif recommendation == "two_stage_model":
        selected_model = confirmation_model
    elif recommendation == "starter_plus_two_stage_veto_family":
        selected_model = StarterPlusFamilyVetoRoleClassifier(
            base_model=base_model,
            confirmation_model=confirmation_model,
            downgrade_role="ROTATION",
        )
    elif recommendation == "starter_plus_confirmed_min_base_probability":
        selected_model = StarterPlusGuardedRoleClassifier(
            base_model=base_model,
            confirmation_model=confirmation_model,
            downgrade_role="ROTATION",
            minimum_starter_plus_probability=float(selected_thresholds.get(recommendation, 0.0)),
        )
    else:
        selected_model = StarterPlusGuardedRoleClassifier(
            base_model=base_model,
            confirmation_model=confirmation_model,
            downgrade_role="ROTATION",
        )

    best_path = config.MODELS_DIR / "best_model.pkl"
    promoted_path = config.MODELS_DIR / "starter_plus_guarded_model.pkl"
    with promoted_path.open("wb") as file:
        pickle.dump(selected_model, file)
    with best_path.open("wb") as file:
        pickle.dump(selected_model, file)

    print(f"Promoted validation-selected model ({recommendation}): {best_path}")
    print(f"Standalone guarded model: {promoted_path}")
    return {"best_model": str(best_path), "guarded_model": str(promoted_path)}


def main() -> None:
    """CLI entrypoint."""
    run_experiment()


if __name__ == "__main__":
    main()
