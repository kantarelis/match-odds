"""Unit tests for the historical-only betting helpers (`matchodds.modeling.betting`)."""

import numpy as np
import pytest

from matchodds.modeling.betting import backtest, edge, expected_value, implied_probabilities

# A 4-match synthetic fixture pinned by hand:
#   Match 0: pick H @ 2.0  -> bet wins   (result H)             profit +1.0
#   Match 1: pick D @ 3.0  -> bet loses  (result H)             profit -1.0
#   Match 2: max EV is -0.1, all <= 0    -> NO BET              (result A)
#   Match 3: pick A @ 2.5  -> bet wins   (result A)             profit +1.5
# So at the default edge_threshold=0 + stake=1: n_bets=3, n_wins=2, pnl=+1.5, roi=+0.5.
_FIXTURE_PROBS = np.array(
    [
        [0.60, 0.25, 0.15],
        [0.30, 0.40, 0.30],
        [0.40, 0.30, 0.30],
        [0.20, 0.30, 0.50],
    ]
)
_FIXTURE_ODDS = np.array(
    [
        [2.0, 3.5, 6.0],
        [2.5, 3.0, 3.5],
        [2.0, 3.0, 3.0],
        [4.0, 3.5, 2.5],
    ]
)
_FIXTURE_RESULTS = np.array([0, 0, 2, 2], dtype=np.intp)  # H, H, A, A


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


def test_backtest_pins_summary_on_a_synthetic_fixture():
    summary = backtest(_FIXTURE_PROBS, _FIXTURE_ODDS, _FIXTURE_RESULTS)
    assert summary["n_matches"] == 4
    assert summary["n_bets"] == 3
    assert summary["n_wins"] == 2
    assert summary["total_staked"] == pytest.approx(3.0)
    # Winning returns: 1 * 2.0 (match 0) + 1 * 2.5 (match 3) = 4.5.
    assert summary["total_return"] == pytest.approx(4.5)
    assert summary["pnl"] == pytest.approx(1.5)
    assert summary["roi"] == pytest.approx(0.5)
    assert summary["win_rate"] == pytest.approx(2 / 3)


def test_backtest_places_zero_bets_when_threshold_is_unrealistic():
    # No outcome in the fixture has EV >= 1.0; the strategy "does nothing", roi/win_rate fall back to 0.
    summary = backtest(_FIXTURE_PROBS, _FIXTURE_ODDS, _FIXTURE_RESULTS, edge_threshold=1.0)
    assert summary["n_matches"] == 4
    assert summary["n_bets"] == 0
    assert summary["n_wins"] == 0
    assert summary["total_staked"] == 0.0
    assert summary["total_return"] == 0.0
    assert summary["pnl"] == 0.0
    assert summary["roi"] == 0.0
    assert summary["win_rate"] == 0.0


def test_backtest_honours_stake_size():
    # Doubling the stake should double total_staked, total_return, and pnl; roi / win_rate are invariant.
    base = backtest(_FIXTURE_PROBS, _FIXTURE_ODDS, _FIXTURE_RESULTS)
    doubled = backtest(_FIXTURE_PROBS, _FIXTURE_ODDS, _FIXTURE_RESULTS, stake=2.0)
    assert doubled["total_staked"] == pytest.approx(2 * base["total_staked"])
    assert doubled["total_return"] == pytest.approx(2 * base["total_return"])
    assert doubled["pnl"] == pytest.approx(2 * base["pnl"])
    assert doubled["roi"] == pytest.approx(base["roi"])
    assert doubled["win_rate"] == pytest.approx(base["win_rate"])


def test_backtest_breaks_ties_to_the_lowest_outcome_index():
    # EVs for D and A both equal +0.125; numpy.argmax tie-break picks the lower index (D).
    # Result is A — so if backtest picked D the bet loses, if it picked A the bet wins. Verify D loses.
    probs = np.array([[0.20, 0.45, 0.45]])
    odds = np.array([[4.0, 2.5, 2.5]])
    results = np.array([2], dtype=np.intp)  # A
    summary = backtest(probs, odds, results)
    assert summary["n_bets"] == 1
    assert summary["n_wins"] == 0
    assert summary["pnl"] == pytest.approx(-1.0)


def test_backtest_deterministic_across_calls():
    first = backtest(_FIXTURE_PROBS, _FIXTURE_ODDS, _FIXTURE_RESULTS)
    second = backtest(_FIXTURE_PROBS, _FIXTURE_ODDS, _FIXTURE_RESULTS)
    assert first == second


def test_backtest_rejects_invalid_odds():
    with pytest.raises(ValueError, match="finite and strictly greater than 1.0"):
        backtest(
            np.array([[0.5, 0.3, 0.2]]),
            np.array([[1.0, 3.0, 3.0]]),
            np.array([0], dtype=np.intp),
        )
