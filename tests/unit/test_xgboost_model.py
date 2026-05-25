"""Unit tests for the XGBoost gradient-boosted-trees model."""

from pathlib import Path

import numpy as np
import pandas as pd

from matchodds.features import pipeline
from matchodds.modeling.xgboost_model import XGBoostModel

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"


def _feature_table() -> pd.DataFrame:
    matches = pd.read_csv(_SYNTHETIC, parse_dates=["date"])
    return pipeline.build_feature_table(matches, pipeline.default_accumulators())


def test_predict_proba_shape_and_sums_to_one():
    table = _feature_table()
    proba = XGBoostModel().fit(table).predict_proba(table)
    assert proba.shape == (len(table), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all()


def test_accepts_nan_features_natively():
    table = _feature_table()
    assert table.iloc[0].isna().any()  # sanity: cold-start NaNs are fed straight to the booster
    proba = XGBoostModel().fit(table).predict_proba(table)
    assert not np.isnan(proba).any()


def test_deterministic_with_seed():
    table = _feature_table()
    first = XGBoostModel().fit(table).predict_proba(table)
    second = XGBoostModel().fit(table).predict_proba(table)
    np.testing.assert_array_equal(first, second)
