"""Unit tests for calibration of the discriminative models.

The synthetic fixture is far too small for calibration to *improve* log-loss, so — per the Task-8
acceptance criterion — these assert the mechanics instead: the wrapper is applied, output stays a
valid ``(n, 3)`` [H, D, A] grid, calibration uses only temporally-held-out folds, the chosen method is
one of the two, and refits are deterministic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from matchodds.features import pipeline
from matchodds.modeling.calibration import calibrate
from matchodds.modeling.cv import TimeOrderedSplit
from matchodds.modeling.logistic import LogisticModel
from matchodds.modeling.xgboost_model import XGBoostModel

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"


def _feature_table() -> pd.DataFrame:
    matches = pd.read_csv(_SYNTHETIC, parse_dates=["date"])
    return pipeline.build_feature_table(matches, pipeline.default_accumulators())


def _cv(table: pd.DataFrame) -> TimeOrderedSplit:
    return TimeOrderedSplit(table["date"])


def test_calibrated_logistic_outputs_valid_probability_grid():
    table = _feature_table()
    calibrated = calibrate(LogisticModel(), table, method="sigmoid", cv=_cv(table))
    proba = calibrated.predict_proba(table)
    assert proba.shape == (len(table), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_calibrated_xgboost_outputs_valid_probability_grid():
    table = _feature_table()
    calibrated = calibrate(XGBoostModel(), table, method="sigmoid", cv=_cv(table))
    proba = calibrated.predict_proba(table)
    assert proba.shape == (len(table), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_wrapper_is_a_calibrated_classifier_over_temporal_folds():
    table = _feature_table()
    cv = _cv(table)
    calibrated = calibrate(LogisticModel(), table, method="sigmoid", cv=cv)
    # The shipped object wraps CalibratedClassifierCV, fit with our forward-chaining splitter.
    assert isinstance(calibrated._calibrated, CalibratedClassifierCV)
    assert calibrated._calibrated.cv is cv
    assert isinstance(cv, TimeOrderedSplit)  # never a random KFold/StratifiedKFold


def test_auto_selects_one_of_the_two_methods():
    table = _feature_table()
    calibrated = calibrate(LogisticModel(), table, method="auto", cv=_cv(table))
    assert calibrated.method in {"sigmoid", "isotonic"}


def test_deterministic_across_two_calibrations():
    table = _feature_table()
    first = calibrate(LogisticModel(), table, method="sigmoid", cv=_cv(table)).predict_proba(table)
    second = calibrate(LogisticModel(), table, method="sigmoid", cv=_cv(table)).predict_proba(table)
    np.testing.assert_array_equal(first, second)
