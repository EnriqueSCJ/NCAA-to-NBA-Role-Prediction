"""Walk-forward temporal backtesting for the role model.

This module evaluates the stable XGBoost setup across several historical draft
years. It is an evaluation tool, not a model-promotion path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

import src.config as config
from src.models.train import apply_smote, get_training_data, train_xgboost


DEFAULT_TEST_START_YEAR = 2017
DEFAULT_TEST_END_YEAR = 2023


@dataclass
class BacktestFold:
    """One walk-forward temporal evaluation fold."""

    test_year: int
    train_rule: str
    validation_year: int
    test_rule: str
    train_rows: int
    validation_rows: int
    test_rows: int
    train_distribution: dict[str, int]
    validation_distribution: dict[str, int]
    test_distribution: dict[str, int]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    skipped: bool = False
    skip_reason: str = ""


def _role_counts(y: pd.Series) -> dict[str, int]:
    """Return role counts aligned to project label order."""
    return y.value_counts().reindex(config.ROLE_LABELS, fill_value=0).astype(int).to_dict()


def make_walk_forward_masks(
    years: pd.Series,
    test_year: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return train/validation/test masks for one draft-year fold."""
    numeric_years = pd.to_numeric(years, errors="coerce")
    validation_year = test_year - 1
    train_mask = numeric_years <= test_year - 2
    validation_mask = numeric_years == validation_year
    test_mask = numeric_years == test_year
    return train_mask, validation_mask, test_mask


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return shared metrics for one fold."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=config.ROLE_LABELS, average="macro", zero_division=0)),
    }


def _skipped_fold(
    test_year: int,
    train_mask: pd.Series,
    validation_mask: pd.Series,
    test_mask: pd.Series,
    y: pd.Series,
    reason: str,
) -> BacktestFold:
    """Build a fold record when a split cannot be trained."""
    validation_year = test_year - 1
    return BacktestFold(
        test_year=test_year,
        train_rule=f"draft_year <= {test_year - 2}",
        validation_year=validation_year,
        test_rule=f"draft_year == {test_year}",
        train_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        test_rows=int(test_mask.sum()),
        train_distribution=_role_counts(y.loc[train_mask]),
        validation_distribution=_role_counts(y.loc[validation_mask]),
        test_distribution=_role_counts(y.loc[test_mask]),
        accuracy=0.0,
        balanced_accuracy=0.0,
        macro_f1=0.0,
        confusion_matrix=np.zeros((len(config.ROLE_LABELS), len(config.ROLE_LABELS)), dtype=int).tolist(),
        skipped=True,
        skip_reason=reason,
    )


def run_fold(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.DataFrame,
    test_year: int,
) -> BacktestFold:
    """Train and evaluate the stable XGBoost setup for one test year."""
    if "draft_year" not in years.columns:
        raise ValueError("years debe incluir la columna 'draft_year'")

    train_mask, validation_mask, test_mask = make_walk_forward_masks(years["draft_year"], test_year)
    validation_year = test_year - 1

    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        return _skipped_fold(
            test_year=test_year,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            y=y,
            reason="split_empty",
        )

    X_train = X.loc[train_mask].copy()
    X_val = X.loc[validation_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_val = y.loc[validation_mask].copy()
    y_test = y.loc[test_mask].copy()

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    model, _ = train_xgboost(
        (X_train_smote, X_val, X_test),
        (y_train_smote, y_val, y_test),
    )
    y_pred = np.asarray(model.predict(X_test)).ravel()
    metrics = _evaluate_predictions(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=config.ROLE_LABELS)

    return BacktestFold(
        test_year=test_year,
        train_rule=f"draft_year <= {test_year - 2}",
        validation_year=validation_year,
        test_rule=f"draft_year == {test_year}",
        train_rows=int(train_mask.sum()),
        validation_rows=int(validation_mask.sum()),
        test_rows=int(test_mask.sum()),
        train_distribution=_role_counts(y_train),
        validation_distribution=_role_counts(y_val),
        test_distribution=_role_counts(y_test),
        accuracy=metrics["accuracy"],
        balanced_accuracy=metrics["balanced_accuracy"],
        macro_f1=metrics["macro_f1"],
        confusion_matrix=cm.astype(int).tolist(),
    )


def _summarize_folds(folds: list[BacktestFold]) -> dict[str, Any]:
    """Build aggregate statistics for non-skipped folds."""
    completed = [fold for fold in folds if not fold.skipped]
    if not completed:
        return {
            "completed_folds": 0,
            "skipped_folds": len(folds),
        }

    summary = _summary_for_completed_folds(completed)
    summary["completed_folds"] = len(completed)
    summary["skipped_folds"] = len(folds) - len(completed)
    full_cohort_folds = [fold for fold in completed if fold.test_rows >= 1000]
    summary["full_cohort_subset"] = {
        "definition": "folds with at least 1000 test rows; excludes tiny recent draft-only slices",
        **_summary_for_completed_folds(full_cohort_folds),
    }
    summary["small_recent_fold_years"] = [
        fold.test_year
        for fold in completed
        if fold.test_rows < 1000
    ]
    return summary


def _summary_for_completed_folds(completed: list[BacktestFold]) -> dict[str, Any]:
    """Build aggregate statistics for an already filtered list of folds."""
    if not completed:
        return {
            "fold_count": 0,
            "test_years": [],
        }

    metric_frame = pd.DataFrame(
        [
            {
                "accuracy": fold.accuracy,
                "balanced_accuracy": fold.balanced_accuracy,
                "macro_f1": fold.macro_f1,
            }
            for fold in completed
        ]
    )
    aggregate_cm = np.sum(
        [np.asarray(fold.confusion_matrix, dtype=int) for fold in completed],
        axis=0,
    )
    summary: dict[str, Any] = {
        "fold_count": len(completed),
        "test_years": [fold.test_year for fold in completed],
        "metrics": {
            metric: {
                "mean": float(metric_frame[metric].mean()),
                "std": float(metric_frame[metric].std(ddof=0)),
                "min": float(metric_frame[metric].min()),
                "max": float(metric_frame[metric].max()),
            }
            for metric in ["accuracy", "balanced_accuracy", "macro_f1"]
        },
        "aggregate_confusion_matrix": aggregate_cm.astype(int).tolist(),
        "aggregate_test_distribution": {
            role: int(sum(fold.test_distribution.get(role, 0) for fold in completed))
            for role in config.ROLE_LABELS
        },
    }
    return summary


def _persist_backtest(folds: list[BacktestFold], summary: dict[str, Any]) -> None:
    """Persist detailed and flat backtest reports."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "temporal_backtest_results.json"
    csv_path = config.REPORTS_DIR / "temporal_backtest_results.csv"
    payload = {
        "model": "stable_xgboost_multiclass",
        "split_strategy": "walk_forward_one_year_test",
        "role_labels": config.ROLE_LABELS,
        "summary": summary,
        "folds": [fold.__dict__ for fold in folds],
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    flat_rows = []
    for fold in folds:
        row = fold.__dict__.copy()
        row["train_distribution"] = json.dumps(row["train_distribution"], sort_keys=True)
        row["validation_distribution"] = json.dumps(row["validation_distribution"], sort_keys=True)
        row["test_distribution"] = json.dumps(row["test_distribution"], sort_keys=True)
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"])
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"Temporal backtest details saved: {json_path}")
    print(f"Temporal backtest table saved: {csv_path}")


def run_temporal_backtest(
    test_start_year: int = DEFAULT_TEST_START_YEAR,
    test_end_year: int = DEFAULT_TEST_END_YEAR,
) -> tuple[list[BacktestFold], dict[str, Any]]:
    """Run walk-forward temporal backtesting over a test-year range."""
    X, y, years, source = get_training_data()
    if source != "processed":
        raise RuntimeError("Temporal backtest requires processed data artifacts.")

    folds: list[BacktestFold] = []
    for test_year in range(test_start_year, test_end_year + 1):
        print(f"\n=== Temporal backtest fold: test_year={test_year} ===")
        fold = run_fold(X, y, years, test_year)
        folds.append(fold)
        if fold.skipped:
            print(f"Fold skipped: {fold.skip_reason}")
        else:
            print(
                "Fold metrics: "
                f"accuracy={fold.accuracy:.4f}, "
                f"balanced_accuracy={fold.balanced_accuracy:.4f}, "
                f"macro_f1={fold.macro_f1:.4f}"
            )

    summary = _summarize_folds(folds)
    _persist_backtest(folds, summary)

    print("\n=== Temporal backtest summary ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return folds, summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run walk-forward temporal backtesting.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_TEST_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_TEST_END_YEAR)
    args = parser.parse_args()
    run_temporal_backtest(test_start_year=args.start_year, test_end_year=args.end_year)


if __name__ == "__main__":
    main()
