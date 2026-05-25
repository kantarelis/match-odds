"""Unit tests for the bookmaker-implied-odds baseline."""

import numpy as np
import pandas as pd

from matchodds.modeling.baselines import BookmakerBaseline


def _table(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["odds_home", "odds_draw", "odds_away"])


def test_implied_probabilities_remove_overround():
    # Implied 1/2, 1/3, 1/3 -> overround 7/6 -> normalised to 3/7, 2/7, 2/7.
    table = _table([[2.0, 3.0, 3.0]])
    proba = BookmakerBaseline().fit(table).predict_proba(table)
    assert proba.shape == (1, 3)
    np.testing.assert_allclose(proba[0], [3 / 7, 2 / 7, 2 / 7])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_rows_sum_to_one_and_order_is_home_draw_away():
    table = _table([[1.5, 4.0, 7.0], [10.0, 5.0, 1.3]])
    proba = BookmakerBaseline().predict_proba(table)
    assert proba.shape == (2, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    # Shortest odds -> highest implied probability: row 0 favours home, row 1 favours away.
    assert proba[0].argmax() == 0
    assert proba[1].argmax() == 2


def test_fit_is_a_noop_returning_self():
    baseline = BookmakerBaseline()
    assert baseline.fit(_table([[2.0, 3.5, 3.5]])) is baseline


def test_missing_odds_yield_a_nan_row():
    table = _table([[2.0, 3.0, 3.0], [np.nan, np.nan, np.nan]])
    proba = BookmakerBaseline().predict_proba(table)
    assert not np.isnan(proba[0]).any()
    assert np.isnan(proba[1]).all()
