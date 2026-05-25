"""The model bake-off: compare the four approaches under temporal CV, freeze the best deployable one.

This is the convergence point of Epic 04. Every candidate — the bookmaker **baseline**, **logistic
regression**, **XGBoost**, and **Dixon-Coles** — is scored under the same forward-chaining temporal CV
(:mod:`matchodds.modeling.cv`) on the same feature table, on proper scoring rules (log-loss primary,
Brier + accuracy reported). The discriminative models are evaluated in their *deployable*, calibrated
form. Selection minimises mean CV **log-loss** among the **deployable** models — the baseline needs
closing odds the serving request never carries, so it is scored as a benchmark but never shipped
(PLAN Decisions 2-3). The winner is refit on all data and frozen to ``models/v1.joblib`` with a
``v1.metadata.json`` sidecar (per-model CV scores, selected model + calibration method, feature
columns, seed, data version, library versions) so Epic 05 can load it and call ``predict_proba``.

Determinism (CLAUDE.md): every split, estimator, and calibrator is seeded, so ``make repro`` on a
pinned data version reproduces the **metric scores** recorded in the metadata (not the pickled bytes).
The model classes live in importable ``matchodds.modeling`` (never here), so the frozen object unpickles.

Per-fold calibration uses a fixed sigmoid (robust + cheap inside the CV loop); the final shipped model
selects sigmoid vs isotonic by CV log-loss on all data (``method="auto"``) — see the decisions in PLAN.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as importlib_metadata
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from matchodds.config import settings
from matchodds.modeling import base, calibration, metrics
from matchodds.modeling.baselines import BookmakerBaseline
from matchodds.modeling.cv import TimeOrderedSplit
from matchodds.modeling.dixon_coles import DixonColesModel
from matchodds.modeling.logistic import LogisticModel
from matchodds.modeling.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

_ARTIFACT = "v1.joblib"
_METADATA = "v1.metadata.json"
_FEATURES = "features.parquet"
_LIBRARIES = ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib")
_BAKE_OFF_METHOD = "sigmoid"  # per-fold calibration in the CV loop
_FINAL_METHOD = "auto"  # the shipped model selects sigmoid vs isotonic by CV log-loss


@dataclass(frozen=True)
class _Candidate:
    """One contender in the bake-off: how to build it, and whether it ships / gets calibrated."""

    name: str
    factory: Callable[[], base.OutcomeModel]
    deployable: bool
    calibrated: bool


_CANDIDATES: tuple[_Candidate, ...] = (
    _Candidate("baseline", BookmakerBaseline, deployable=False, calibrated=False),
    _Candidate("logistic", LogisticModel, deployable=True, calibrated=True),
    _Candidate("xgboost", XGBoostModel, deployable=True, calibrated=True),
    _Candidate("dixon_coles", DixonColesModel, deployable=True, calibrated=False),
)
_BY_NAME = {candidate.name: candidate for candidate in _CANDIDATES}


def _fit_calibrated(candidate: _Candidate, table: pd.DataFrame, method: str) -> calibration.CalibratedOutcome | None:
    """Calibrate a discriminative candidate on ``table``, or ``None`` if the slice is too small."""
    cv = calibration.safe_calibration_cv(table)
    if cv is None:
        return None
    return calibration.calibrate(cast(calibration.Discriminative, candidate.factory()), table, method=method, cv=cv)


def _fit_for_eval(candidate: _Candidate, table: pd.DataFrame) -> base.OutcomeModel:
    """Fit a candidate's deployable form on a training fold (calibrated for the discriminative ones).

    A cold-start fold too small to calibrate falls back to the raw estimator — only the synthetic
    fixture's earliest folds hit this; real-data folds are always large enough.
    """
    if candidate.calibrated:
        calibrated = _fit_calibrated(candidate, table, _BAKE_OFF_METHOD)
        if calibrated is not None:
            return calibrated
    return candidate.factory().fit(table)


def _cv_scores(candidate: _Candidate, table: pd.DataFrame, splitter: TimeOrderedSplit) -> dict[str, float]:
    """Mean per-fold ``score_summary`` (log-loss, Brier, accuracy) under temporal CV."""
    folds: list[dict[str, float]] = []
    for train_idx, test_idx in splitter.split():
        model = _fit_for_eval(candidate, table.iloc[train_idx])
        test_table = table.iloc[test_idx]
        proba = model.predict_proba(test_table)
        labels = metrics.encode_labels(test_table["result"])
        folds.append(metrics.score_summary(labels, proba))
    return {metric: float(np.mean([fold[metric] for fold in folds])) for metric in folds[0]}


def _fit_final(candidate: _Candidate, table: pd.DataFrame) -> tuple[base.OutcomeModel, str | None]:
    """Refit the winner on all data; calibrate the discriminative ones (method chosen by CV log-loss)."""
    if candidate.calibrated:
        calibrated = _fit_calibrated(candidate, table, _FINAL_METHOD)
        if calibrated is not None:
            return calibrated, calibrated.method
    return candidate.factory().fit(table), None


def _data_version() -> dict[str, Any] | None:
    """Provenance from the matches / features meta sidecars, or ``None`` if they are absent."""
    version: dict[str, Any] = {}
    for key, name in (("matches", "matches.meta.json"), ("features", "features.meta.json")):
        path = settings.processed_dir / name
        if path.exists():
            version[key] = json.loads(path.read_text())
    return version or None


def _library_versions() -> dict[str, str]:
    return {library: importlib_metadata.version(library) for library in _LIBRARIES}


def _metadata(scores: dict[str, dict[str, float]], selected: str, method: str | None) -> dict[str, Any]:
    return {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "random_seed": settings.random_seed,
        "cv_splits": settings.cv_splits,
        "data_version": _data_version(),
        "selected_model": selected,
        "calibration_method": method,
        "feature_columns": base.feature_columns(),
        "models": {name: {**scores[name], "deployable": _BY_NAME[name].deployable} for name in scores},
        "library_versions": _library_versions(),
    }


def run(table: pd.DataFrame, models_dir: Path) -> dict[str, Any]:
    """Score every candidate under temporal CV, freeze the best deployable model, return the metadata."""
    splitter = TimeOrderedSplit(table["date"])
    scores = {candidate.name: _cv_scores(candidate, table, splitter) for candidate in _CANDIDATES}
    deployable = {name: score for name, score in scores.items() if _BY_NAME[name].deployable}
    selected = min(deployable, key=lambda name: deployable[name]["log_loss"])
    logger.info("selected %s (mean CV log-loss %.4f)", selected, deployable[selected]["log_loss"])

    final_model, method = _fit_final(_BY_NAME[selected], table)
    metadata = _metadata(scores, selected, method)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, models_dir / _ARTIFACT)
    (models_dir / _METADATA).write_text(json.dumps(metadata, indent=2))
    return metadata


def _load_features() -> pd.DataFrame:
    return pd.read_parquet(settings.processed_dir / _FEATURES)


def main() -> None:
    metadata = run(_load_features(), settings.models_dir)
    logger.info("froze %s -> %s (+ metadata)", metadata["selected_model"], settings.models_dir / _ARTIFACT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
