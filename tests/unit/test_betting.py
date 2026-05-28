"""Unit tests for the historical-only betting helpers (`matchodds.modeling.betting`)."""

import numpy as np
import pytest

from matchodds.modeling.betting import edge, expected_value, implied_probabilities


def test_implied_probabilities_remove_overround_on_a_pinned_example():
    # 1/2.0 + 1/3.0 + 1/3.0 = 7/6 (overround); normalised -> 3/7, 2/7, 2/7. Matches the BookmakerBaseline.
    proba = implied_probabilities(np.array([[2.0, 3.0, 3.0]]))
    assert proba.shape == (1, 3)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba[0], [3 / 7, 2 / 7, 2 / 7])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_implied_probabilities_handle_a_realistic_overround():
    # Typical football odds; the raw inverses sum to ~1.066 (a 6.6% bookmaker hold) before normalising.
    odds = np.array([[2.0, 3.3, 3.8]])
    raw = 1.0 / odds
    assert raw.sum() > 1.0  # overround exists
    proba = implied_probabilities(odds)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    # Shortest odds -> highest implied probability.
    assert proba[0].argmax() == 0


def test_implied_probabilities_reject_invalid_odds():
    # Odds <= 1.0 imply a probability >= 100% (impossible) or no profit (no real market does this).
    for bad in (1.0, 0.5, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and strictly greater than 1.0"):
            implied_probabilities(np.array([[bad, 3.0, 3.0]]))


def test_implied_probabilities_deterministic_across_calls():
    odds = np.array([[2.0, 3.3, 3.8], [1.5, 4.0, 7.0]])
    np.testing.assert_array_equal(implied_probabilities(odds), implied_probabilities(odds))


def test_edge_is_model_minus_implied_market_probability():
    odds = np.array([[2.0, 3.0, 3.0]])
    model = np.array([[0.50, 0.25, 0.25]])
    # Implied is (3/7, 2/7, 2/7); model edge is element-wise difference.
    np.testing.assert_allclose(edge(model, odds), model - np.array([[3 / 7, 2 / 7, 2 / 7]]))


def test_expected_value_uses_decimal_odds_directly():
    # EV_i = P_i * odds_i - 1. P = (0.5, 0.25, 0.25) at odds (2.0, 3.0, 3.0) -> (0, -0.25, -0.25).
    ev = expected_value(np.array([[0.50, 0.25, 0.25]]), np.array([[2.0, 3.0, 3.0]]))
    np.testing.assert_allclose(ev[0], [0.0, -0.25, -0.25])


def test_expected_value_positive_iff_model_beats_break_even():
    # An outcome the model rates 0.5 at decimal odds 2.2 has EV = 0.5 * 2.2 - 1 = 0.10 (favourable bet).
    ev = expected_value(np.array([[0.5, 0.3, 0.2]]), np.array([[2.2, 3.5, 5.5]]))
    np.testing.assert_allclose(ev[0], [0.10, 0.05, 0.10])
    assert (ev > 0).all()


def test_expected_value_rejects_invalid_odds():
    with pytest.raises(ValueError, match="finite and strictly greater than 1.0"):
        expected_value(np.array([[0.5, 0.3, 0.2]]), np.array([[1.0, 3.0, 3.0]]))
