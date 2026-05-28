"""Calibrate the discriminative models so their probabilities are honest, not merely well-ranked.

A classifier can rank outcomes correctly yet be over- or under-confident; calibration maps its raw
scores to probabilities that match observed frequencies (CLAUDE.md -> "Calibration is part of the
model"). :func:`calibrate` uses the **final** fold of a :class:`TimeOrderedSplit` as a temporal
hold-out: the underlying scikit-learn estimator is fit on the fold's expanding-window training slice,
then wrapped in :class:`~sklearn.frozen.FrozenEstimator` and handed to
``CalibratedClassifierCV``, which fits the calibration regressor on the strictly-later held-out tail.
The calibrator therefore only ever sees data dated **after** the base's training window — leakage-free
by construction. (The previous ``CalibratedClassifierCV(cv=temporal_split)`` ensemble averaged
fold-subset submodels and sigmoid-squashed confident predictions, flattening enough to invert the
real home advantage on near-even derbies — Epic 06.5.) The result stays an
:class:`~matchodds.modeling.base.OutcomeModel`, emitting ``[H, D, A]`` via the same class-scatter as
the wrapped models.

``method="auto"`` picks sigmoid (Platt) vs isotonic by held-out log-loss on a **temporal sub-split of
the holdout itself** — fit each calibrator on the holdout's earlier portion, score on its strictly
later tail, pick the winner, refit on the full holdout. If the holdout is too small to sub-split,
selection falls back to sigmoid (robust on small data).

Dixon-Coles is generative and naturally calibrated — it is reliability-*checked* (Task 10), never
wrapped (PLAN Decision 5), so it is not a target of this function.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

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


def _fit_base(estimator: object, train_table: pd.DataFrame) -> Any:
    """Clone the estimator (parameters only) and fit on the training slice; return the fitted base."""
    fitted = clone(estimator)
    fitted.fit(_features(train_table), metrics.encode_labels(train_table["result"]))
    return fitted


def _fit_calibrator(frozen_base: Any, holdout_table: pd.DataFrame, method: str) -> Any:
    """Fit ``CalibratedClassifierCV`` on the held-out slice using the already-fitted base.

    ``FrozenEstimator`` keeps the base pre-trained (re-``fit`` is a no-op) so the calibrator sees the
    base's predictions on the whole holdout. A single-fold ``cv`` (all indices in both train and test)
    is passed explicitly: it bypasses ``CalibratedClassifierCV``'s default ``StratifiedKFold`` and its
    per-fold class-diversity check — which is meaningless under a frozen base — and lets a tiny / class-
    imbalanced holdout still calibrate.
    """
    features = _features(holdout_table)
    labels = metrics.encode_labels(holdout_table["result"])
    single_fold = [(np.arange(features.shape[0]), np.arange(features.shape[0]))]
    calibrated = CalibratedClassifierCV(FrozenEstimator(frozen_base), method=method, cv=single_fold)
    calibrated.fit(features, labels)
    return calibrated


def safe_calibration_cv(table: pd.DataFrame) -> TimeOrderedSplit | None:
    """A temporal calibration split valid for ``table``, or ``None`` if it is too small / skewed.

    Capped by the rarest class's count (``CalibratedClassifierCV`` needs at least ``n_splits`` examples
    per class) and by the number of distinct dates. Then every fold's **training** portion must contain
    all classes — an expanding-window fold whose early dates miss an outcome would fit a single-class
    base estimator and raise — and the **final fold's hold-out** portion must also carry every class
    so the calibrator covers all three outcomes. Used by ``train.py`` to derive the calibration cv;
    a cold-start or class-skewed slice falls back to the raw estimator rather than crashing.
    """
    labels = metrics.encode_labels(table["result"])
    n_classes = len(metrics.CLASSES)
    n_splits = min(
        settings.cv_splits, int(table["date"].nunique()) - 1, int(np.bincount(labels, minlength=n_classes).min())
    )
    if n_splits < 2:
        return None
    splitter = TimeOrderedSplit(table["date"], n_splits=n_splits)
    splits = list(splitter.split())
    if any(len(np.unique(labels[train_idx])) < n_classes for train_idx, _ in splits):
        return None
    _, last_test_idx = splits[-1]
    if len(np.unique(labels[last_test_idx])) < n_classes:
        return None
    return splitter


def _select_method(frozen_base: Any, holdout_table: pd.DataFrame) -> str:
    """Pick sigmoid vs isotonic by held-out log-loss on a **temporal sub-split of the holdout itself**.

    Each method's calibrator is fit on the holdout's earlier sub-slice and scored on its strictly-later
    sub-slice, preserving the forward-chaining discipline at the inner level. If the holdout is too
    small for a valid sub-split, selection falls back to :data:`_DEFAULT_METHOD` (sigmoid).
    """
    inner = safe_calibration_cv(holdout_table)
    if inner is None:
        return _DEFAULT_METHOD
    sub_train_idx, sub_test_idx = list(inner.split())[-1]
    sub_train_table = holdout_table.iloc[sub_train_idx]
    sub_test_table = holdout_table.iloc[sub_test_idx]
    y_sub_test = metrics.encode_labels(sub_test_table["result"])
    sub_test_features = _features(sub_test_table)
    scores: dict[str, float] = {}
    for method in _METHODS:
        calibrator = _fit_calibrator(frozen_base, sub_train_table, method)
        proba = _scatter(calibrator, sub_test_features)
        scores[method] = metrics.log_loss(y_sub_test, proba)
    return min(scores, key=lambda method: scores[method])


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
    """Calibrate a discriminative model via a temporal hold-out; return a calibrated ``OutcomeModel``.

    The **final fold** of ``cv`` defines the split: the base is fit on the fold's training slice, and
    the calibrator on the strictly-later hold-out slice (``FrozenEstimator`` + prefit pattern).
    ``method`` is ``"sigmoid"``, ``"isotonic"``, or ``"auto"`` (chosen by held-out log-loss on a
    temporal sub-split of the hold-out itself). The wrapped estimator is cloned, so the input model's
    fitted state — if any — is irrelevant.
    """
    train_idx, holdout_idx = list(cv.split())[-1]
    train_table = table.iloc[train_idx]
    holdout_table = table.iloc[holdout_idx]
    frozen_base = _fit_base(model.estimator, train_table)
    chosen = _select_method(frozen_base, holdout_table) if method == "auto" else method
    return _CalibratedModel(_fit_calibrator(frozen_base, holdout_table, chosen), chosen)
