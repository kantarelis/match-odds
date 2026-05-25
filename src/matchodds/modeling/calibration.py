"""Calibrate the discriminative models so their probabilities are honest, not merely well-ranked.

A classifier can rank outcomes correctly yet be over- or under-confident; calibration maps its raw
scores to probabilities that match observed frequencies (CLAUDE.md -> "Calibration is part of the
model"). :func:`calibrate` wraps a discriminative model's underlying scikit-learn estimator in
``CalibratedClassifierCV`` over the **temporal** folds from :mod:`matchodds.modeling.cv` (never random
folds), so the calibrator only ever sees data held out from the estimator's training — no leakage. The
result stays an :class:`~matchodds.modeling.base.OutcomeModel`, emitting ``[H, D, A]`` via the same
class-scatter as the wrapped models.

``method="auto"`` picks sigmoid (Platt) vs isotonic by mean held-out log-loss under **nested** temporal
CV: the inner calibration split is rebuilt from each outer training fold's own dates, so both levels
stay strictly forward-chaining. An outer fold whose training slice has too few distinct dates to form
an inner split is skipped; if none are usable, selection falls back to sigmoid (robust on small data).

Dixon-Coles is generative and naturally calibrated — it is reliability-*checked* (Task 10), never
wrapped (PLAN Decision 5), so it is not a target of this function.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from matchodds.config import settings
from matchodds.modeling import base, metrics
from matchodds.modeling.cv import TimeOrderedSplit

_METHODS = ("sigmoid", "isotonic")
_DEFAULT_METHOD = "sigmoid"


class Discriminative(Protocol):
    """A model exposing the scikit-learn estimator that calibration wraps (LR / XGBoost)."""

    @property
    def estimator(self) -> object:
        """The underlying scikit-learn estimator; cloned (params only, fitted state discarded)."""
        ...


class CalibratedOutcome(base.OutcomeModel, Protocol):
    """An :class:`OutcomeModel` that also records which calibration method it applied."""

    method: str


def _features(table: pd.DataFrame) -> base.FloatArray:
    return np.asarray(table[base.feature_columns()].to_numpy(), dtype=np.float64)


def _scatter(calibrated: Any, features: base.FloatArray) -> base.FloatArray:
    """Map a calibrated estimator's ``predict_proba`` back into the fixed ``(n, 3)`` [H, D, A] grid."""
    raw = np.asarray(calibrated.predict_proba(features), dtype=np.float64)
    classes = np.asarray(calibrated.classes_, dtype=np.intp)
    proba = np.zeros((features.shape[0], len(metrics.CLASSES)), dtype=np.float64)
    proba[:, classes] = raw
    return proba


def _fit_calibrated(estimator: object, table: pd.DataFrame, method: str, cv: TimeOrderedSplit) -> Any:
    calibrated = CalibratedClassifierCV(clone(estimator), method=method, cv=cv)
    calibrated.fit(_features(table), metrics.encode_labels(table["result"]))
    return calibrated


def safe_calibration_cv(table: pd.DataFrame) -> TimeOrderedSplit | None:
    """A temporal calibration split valid for ``table``, or ``None`` if it is too small.

    Capped by the rarest class's count (``CalibratedClassifierCV`` requires at least ``n_splits``
    examples per class) and by the number of distinct dates; a table missing a class yields ``None``.
    Used both for the inner split of method selection and (by ``train.py``) for the per-fold and final
    calibration folds, so a cold-start slice never over-requests folds.
    """
    labels = metrics.encode_labels(table["result"])
    min_class = int(np.bincount(labels, minlength=len(metrics.CLASSES)).min())
    n_splits = min(settings.cv_splits, int(table["date"].nunique()) - 1, min_class)
    return TimeOrderedSplit(table["date"], n_splits=n_splits) if n_splits >= 2 else None


def _select_method(estimator: object, table: pd.DataFrame, cv: TimeOrderedSplit) -> str:
    """Pick sigmoid vs isotonic by mean held-out log-loss under nested temporal CV."""
    scores: dict[str, list[float]] = {method: [] for method in _METHODS}
    for train_idx, test_idx in cv.split():
        train_table = table.iloc[train_idx]
        inner = safe_calibration_cv(train_table)
        if inner is None:
            continue
        test_table = table.iloc[test_idx]
        y_test = metrics.encode_labels(test_table["result"])
        test_features = _features(test_table)
        for method in _METHODS:
            proba = _scatter(_fit_calibrated(estimator, train_table, method, inner), test_features)
            scores[method].append(metrics.log_loss(y_test, proba))
    means = {method: float(np.mean(values)) for method, values in scores.items() if values}
    return min(means, key=lambda method: means[method]) if means else _DEFAULT_METHOD


class _CalibratedModel(base.OutcomeModel):
    """An :class:`OutcomeModel` backed by a fitted ``CalibratedClassifierCV``, emitting ``[H, D, A]``."""

    def __init__(self, calibrated: Any, method: str) -> None:
        self._calibrated = calibrated
        self.method = method

    def fit(self, _table: pd.DataFrame, /) -> _CalibratedModel:
        """No-op: :func:`calibrate` returns this already fitted (interface conformance only)."""
        return self

    def predict_proba(self, table: pd.DataFrame, /) -> base.FloatArray:
        return _scatter(self._calibrated, _features(table))


def calibrate(
    model: Discriminative,
    table: pd.DataFrame,
    *,
    method: str = "auto",
    cv: TimeOrderedSplit,
) -> CalibratedOutcome:
    """Calibrate a discriminative model over temporal folds; return a calibrated ``OutcomeModel``.

    ``method`` is ``"sigmoid"``, ``"isotonic"``, or ``"auto"`` (choose by held-out log-loss). The
    wrapped estimator is cloned, so the input model's fitted state — if any — is irrelevant.
    """
    chosen = _select_method(model.estimator, table, cv) if method == "auto" else method
    return _CalibratedModel(_fit_calibrated(model.estimator, table, chosen, cv), chosen)
