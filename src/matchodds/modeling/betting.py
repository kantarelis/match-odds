"""Pure numeric helpers for the historical-only betting-edge analysis (Epic 07).

A handful of stateless functions over ``(n, 3)`` arrays in the fixed ``[H, D, A]`` order: turn
bookmaker decimal odds into normalised implied probabilities (overround removed), measure the
per-outcome edge of a model versus the market, and compute the per-unit expected value of a bet.
They operate on numpy arrays, do no I/O, and have no model dependency — so they slot into the
notebook narrative without dragging the modeling stack with them.

Strict on invalid odds: a row whose odds are not all ``> 1.0`` (or that contains a non-finite value)
raises :class:`ValueError`. The notebook filters such rows out **before** calling these helpers; the
strictness is a deliberate boundary that keeps the analysis honest (CLAUDE.md → *"No live betting /
real-money framing"* — historical, illustrative only).
"""

from __future__ import annotations

import numpy as np

from matchodds.modeling.base import FloatArray
from matchodds.modeling.metrics import IntArray


def _as_odds(odds: FloatArray) -> FloatArray:
    """Validate and coerce a decimal-odds matrix: every entry must be finite and strictly > 1.0."""
    array = np.asarray(odds, dtype=np.float64)
    if not np.isfinite(array).all() or (array <= 1.0).any():
        raise ValueError("decimal odds must be finite and strictly greater than 1.0")
    return array


def _as_probabilities(probabilities: FloatArray) -> FloatArray:
    return np.asarray(probabilities, dtype=np.float64)


def implied_probabilities(odds: FloatArray) -> FloatArray:
    """Overround-removed implied probabilities from decimal odds, in the fixed ``[H, D, A]`` order.

    Computes ``1 / odds`` per cell, then normalises each row by its sum — the standard
    "remove the bookmaker hold" transform. The raw ``1 / odds`` row sums to more than 1 (the
    overround / bookie margin); the normalised row sums to exactly 1.
    """
    array = _as_odds(odds)
    raw = 1.0 / array
    normalised: FloatArray = raw / raw.sum(axis=1, keepdims=True)
    return normalised


def edge(model_prob: FloatArray, market_odds: FloatArray) -> FloatArray:
    """Per-outcome edge: model probability minus the market's overround-removed implied probability.

    Positive means the model thinks the outcome is more likely than the (de-overrounded) market does.
    Edge is a useful summary statistic but is *not* the bet-selection criterion — :func:`expected_value`
    is, because the same edge can be favourable or unfavourable depending on the decimal odds offered.
    """
    return _as_probabilities(model_prob) - implied_probabilities(market_odds)


def expected_value(model_prob: FloatArray, market_odds: FloatArray) -> FloatArray:
    """Expected value of a one-unit bet on each outcome at the given decimal odds.

    Per unit stake on outcome *i*: profit on win is ``odds_i - 1``, loss otherwise is ``1``, giving
    ``EV_i = P_i * odds_i - 1`` (equivalently ``P_i * (odds_i - 1) - (1 - P_i)``). A positive value
    is a model-favourable bet; this is the right criterion for selection, since it accounts for both
    the model's edge **and** the size of the payout being offered.
    """
    return _as_probabilities(model_prob) * _as_odds(market_odds) - 1.0


def backtest(
    model_probs: FloatArray,
    market_odds: FloatArray,
    results: IntArray,
    *,
    edge_threshold: float = 0.0,
    stake: float = 1.0,
) -> dict[str, float | int]:
    """Flat-stake positive-EV backtest summary on out-of-sample matches.

    For each match, pick the outcome (H / D / A) with the largest :func:`expected_value`; place a
    bet of size ``stake`` on it iff that EV is **strictly greater than** ``edge_threshold``. Ties in
    the maximum EV are broken by lowest outcome index (numpy's ``argmax`` default — so H beats D,
    D beats A). A match with no qualifying outcome contributes to ``n_matches`` but never to
    ``n_bets``. Returns a numeric summary; the per-bet ledger is the caller's concern.

    ``results`` is the encoded outcome per match (``H=0 / D=1 / A=2`` per
    :data:`matchodds.modeling.metrics.CLASSES`). ``roi`` and ``win_rate`` are ``0.0`` when no bets
    were placed (a "did nothing" strategy, not a divide-by-zero).
    """
    odds = _as_odds(market_odds)
    labels = np.asarray(results, dtype=np.intp)
    ev = expected_value(model_probs, odds)
    picks = np.asarray(ev.argmax(axis=1), dtype=np.intp)
    rows = np.arange(ev.shape[0], dtype=np.intp)
    bet_mask = ev[rows, picks] > edge_threshold
    win_mask = bet_mask & (picks == labels)
    n_matches = int(ev.shape[0])
    n_bets = int(bet_mask.sum())
    n_wins = int(win_mask.sum())
    total_staked = float(stake) * n_bets
    total_return = float(stake) * float(odds[win_mask, picks[win_mask]].sum())
    pnl = total_return - total_staked
    return {
        "n_matches": n_matches,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "total_staked": total_staked,
        "total_return": total_return,
        "pnl": pnl,
        "roi": pnl / total_staked if total_staked > 0 else 0.0,
        "win_rate": n_wins / n_bets if n_bets > 0 else 0.0,
    }
