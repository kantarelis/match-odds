"""Unit tests for the multinomial logistic-regression model."""

from pathlib import Path

import numpy as np
import pandas as pd

from matchodds.features import pipeline
from matchodds.modeling.logistic import LogisticModel

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"


def _feature_table() -> pd.DataFrame:
    matches = pd.read_csv(_SYNTHETIC, parse_dates=["date"])
    return pipeline.build_feature_table(matches, pipeline.default_accumulators())


def test_predict_proba_shape_and_sums_to_one():
    table = _feature_table()
    proba = LogisticModel().fit(table).predict_proba(table)
    assert proba.shape == (len(table), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_handles_nan_cold_start_rows():
    table = _feature_table()
    assert table.iloc[0].isna().any()  # sanity: the first row has cold-start NaNs to impute
    proba = LogisticModel().fit(table).predict_proba(table)
    assert not np.isnan(proba).any()


def test_deterministic_with_seed():
    table = _feature_table()
    first = LogisticModel().fit(table).predict_proba(table)
    second = LogisticModel().fit(table).predict_proba(table)
    np.testing.assert_array_equal(first, second)
