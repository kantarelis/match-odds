"""Unit tests for calibration of the discriminative models.

The synthetic fixture is far too small for calibration to *improve* log-loss, so — per the Task-8
acceptance criterion — these assert the mechanics instead: the wrapper is applied, output stays a
valid ``(n, 3)`` [H, D, A] grid, calibration is built as a leakage-free temporal hold-out (base fit on
earlier dates, calibrator fit on a strictly-later slice via :class:`FrozenEstimator`), the chosen
method is one of the two, and refits are deterministic. Plus a home-advantage no-inversion guard
(Epic 06.5 regression) on a purpose-built strongly-home-biased fixture.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from matchodds.features import pipeline
from matchodds.modeling import base
from matchodds.modeling.calibration import calibrate
from matchodds.modeling.cv import TimeOrderedSplit
from matchodds.modeling.logistic import LogisticModel
from matchodds.modeling.xgboost_model import XGBoostModel

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_SYNTHETIC = _FIXTURES / "synthetic_matches.csv"
_HOME_ADVANTAGE = _FIXTURES / "home_advantage_matches.csv"


def _feature_table(path: Path = _SYNTHETIC) -> pd.DataFrame:
    matches = pd.read_csv(path, parse_dates=["date"])
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


def test_wrapper_is_a_prefit_calibrator_on_a_temporal_holdout():
    """Structural test: the new hold-out (prefit) calibration of Epic 06.5.

    The wrapper is a ``CalibratedClassifierCV`` whose inner estimator is a ``FrozenEstimator`` — the
    prefit pattern — and the final fold of the splitter places the calibrator's data strictly **after**
    the base's training window, so no future leaks into the calibration step. The splitter is the
    forward-chaining :class:`TimeOrderedSplit`, never a random ``KFold`` / ``StratifiedKFold``.
    """
    table = _feature_table()
    cv = _cv(table)
    train_idx, holdout_idx = list(cv.split())[-1]
    train_max_date = table.iloc[train_idx]["date"].max()
    holdout_min_date = table.iloc[holdout_idx]["date"].min()
    assert holdout_min_date > train_max_date

    calibrated = calibrate(LogisticModel(), table, method="sigmoid", cv=cv)
    assert isinstance(calibrated._calibrated, CalibratedClassifierCV)
    assert isinstance(calibrated._calibrated.estimator, FrozenEstimator)
    assert isinstance(cv, TimeOrderedSplit)


def test_auto_selects_one_of_the_two_methods():
    table = _feature_table()
    calibrated = calibrate(LogisticModel(), table, method="auto", cv=_cv(table))
    assert calibrated.method in {"sigmoid", "isotonic"}


def test_deterministic_across_two_calibrations():
    table = _feature_table()
    first = calibrate(LogisticModel(), table, method="sigmoid", cv=_cv(table)).predict_proba(table)
    second = calibrate(LogisticModel(), table, method="sigmoid", cv=_cv(table)).predict_proba(table)
    np.testing.assert_array_equal(first, second)


def test_calibration_does_not_invert_home_advantage_on_balanced_fixture():
    """Epic 06.5 regression guard.

    On a strongly home-biased fixture (~60/20/20 H/D/A), the calibrated model must keep
    ``P(home) > P(away)`` for a perfectly balanced fixture (one whose feature row is the median of the
    training-set features — i.e. every signal sits at parity, so any remaining preference comes from
    the base rate). The shipped ensemble-over-folds calibration in Epic 06 inverted this; the
    hold-out (prefit) calibration of Epic 06.5 must not.
    """
    table = _feature_table(_HOME_ADVANTAGE)
    calibrated = calibrate(LogisticModel(), table, method="sigmoid", cv=_cv(table))
    feature_cols = base.feature_columns()
    balanced = table.iloc[[0]].copy()
    for col in feature_cols:
        balanced[col] = float(table[col].median())
    p_home, _p_draw, p_away = calibrated.predict_proba(balanced)[0]
    assert p_home > p_away, f"home advantage inverted: P(home)={p_home:.3f} <= P(away)={p_away:.3f}"
