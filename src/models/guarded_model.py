"""Guarded model wrappers for conservative role inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import src.config as config


class StarterPlusGuardedRoleClassifier:
    """Wrap a base role model and require a confirmation model for STARTER_PLUS.

    The wrapper keeps the base model probabilities for ordinary decisions. When
    the base model's argmax is STARTER_PLUS but the confirmation model does not
    also predict STARTER_PLUS, the STARTER_PLUS probability mass is moved to
    ROTATION. This keeps the output aligned with APIs that select the role from
    ``predict_proba`` rather than ``predict``.
    """

    def __init__(
        self,
        base_model: Any,
        confirmation_model: Any,
        downgrade_role: str = "ROTATION",
        minimum_starter_plus_probability: float = 0.0,
    ):
        if downgrade_role not in config.ROLE_LABELS:
            raise ValueError(f"Unknown downgrade role: {downgrade_role}")
        self.base_model = base_model
        self.confirmation_model = confirmation_model
        self.downgrade_role = downgrade_role
        self.minimum_starter_plus_probability = float(minimum_starter_plus_probability)
        self.classes_ = np.array(config.ROLE_LABELS)

    def _aligned_base_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.asarray(self.base_model.predict_proba(X), dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)

        base_classes = [str(label) for label in getattr(self.base_model, "classes_", config.ROLE_LABELS)]
        if base_classes == config.ROLE_LABELS and probabilities.shape[1] == len(config.ROLE_LABELS):
            return probabilities.copy()

        aligned = np.zeros((len(X), len(config.ROLE_LABELS)), dtype=float)
        for source_index, label in enumerate(base_classes):
            if label in config.ROLE_LABELS and source_index < probabilities.shape[1]:
                aligned[:, config.ROLE_LABELS.index(label)] = probabilities[:, source_index]

        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        return aligned / row_sums

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self._aligned_base_proba(X)
        labels = np.asarray(config.ROLE_LABELS)
        base_predictions = labels[probabilities.argmax(axis=1)]
        confirmation_predictions = np.asarray(self.confirmation_model.predict(X)).ravel()

        starter_index = config.ROLE_LABELS.index("STARTER_PLUS")
        downgrade_index = config.ROLE_LABELS.index(self.downgrade_role)
        minimum_probability = float(getattr(self, "minimum_starter_plus_probability", 0.0))
        low_confidence_mask = probabilities[:, starter_index] < minimum_probability
        veto_mask = (
            (base_predictions == "STARTER_PLUS")
            & ((confirmation_predictions != "STARTER_PLUS") | low_confidence_mask)
        )

        adjusted = probabilities.copy()
        adjusted[veto_mask, downgrade_index] += adjusted[veto_mask, starter_index]
        adjusted[veto_mask, starter_index] = 0.0
        return adjusted

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class StarterPlusFamilyVetoRoleClassifier(StarterPlusGuardedRoleClassifier):
    """Downgrade STARTER_PLUS according to the confirmation model's family."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self._aligned_base_proba(X)
        labels = np.asarray(config.ROLE_LABELS)
        base_predictions = labels[probabilities.argmax(axis=1)]
        confirmation_predictions = np.asarray(self.confirmation_model.predict(X)).ravel()

        starter_index = config.ROLE_LABELS.index("STARTER_PLUS")
        rotation_index = config.ROLE_LABELS.index("ROTATION")
        non_nba_index = config.ROLE_LABELS.index("NON_NBA")
        minimum_probability = float(getattr(self, "minimum_starter_plus_probability", 0.0))
        low_confidence_mask = probabilities[:, starter_index] < minimum_probability
        veto_mask = (
            (base_predictions == "STARTER_PLUS")
            & ((confirmation_predictions != "STARTER_PLUS") | low_confidence_mask)
        )
        non_nba_veto = veto_mask & (confirmation_predictions == "NON_NBA")
        rotation_veto = veto_mask & (confirmation_predictions != "NON_NBA")

        adjusted = probabilities.copy()
        adjusted[non_nba_veto, non_nba_index] += adjusted[non_nba_veto, starter_index]
        adjusted[rotation_veto, rotation_index] += adjusted[rotation_veto, starter_index]
        adjusted[veto_mask, starter_index] = 0.0
        return adjusted
