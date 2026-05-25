"""The bookmaker-implied-odds baseline — the benchmark every model must beat.

Closing odds already encode the market's probability estimate; ``1 / odds`` normalised to remove the
bookmaker's overround gives a strong, well-calibrated reference. This is a **benchmark only**: the
serving request (Epic 05) carries no odds, so the baseline is never the shipped artifact — just the
yardstick the deployable models are scored against under temporal CV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchodds.modeling.base import FloatArray, OutcomeModel

_ODDS_COLUMNS = ["odds_home", "odds_draw", "odds_away"]


class BookmakerBaseline(OutcomeModel):
    """Implied 1X2 probabilities from closing odds, with the overround removed (stateless).

    A fixture with missing odds yields an all-``NaN`` probability row — an explicit "no market
    prediction here" rather than a fabricated guess.
    """

    def fit(self, _table: pd.DataFrame, /) -> BookmakerBaseline:
        """No-op: the baseline has no parameters to learn. Returns ``self`` for a uniform API."""
        return self

    def predict_proba(self, table: pd.DataFrame, /) -> FloatArray:
        odds = np.asarray(table[_ODDS_COLUMNS].to_numpy(), dtype=np.float64)
        implied = 1.0 / odds
        normalised: FloatArray = implied / implied.sum(axis=1, keepdims=True)
        return normalised
