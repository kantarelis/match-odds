"""The model interface shared across every approach in this epic.

Each model — the bookmaker baseline, logistic regression, XGBoost, Dixon-Coles — implements
:class:`OutcomeModel`: ``fit(table)`` then ``predict_proba(table)`` returning an ``(n, 3)`` array of
``[home_win, draw, away_win]`` probabilities in the fixed :data:`matchodds.modeling.metrics.CLASSES`
order. One interface over one feature table is what keeps training and serving on the same code path.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from matchodds.features.pipeline import default_accumulators

FloatArray = npt.NDArray[np.float64]


class OutcomeModel(Protocol):
    """Structural interface for a 1X2 (home / draw / away) probability model."""

    def fit(self, table: pd.DataFrame, /) -> OutcomeModel:
        """Fit on a feature table and return ``self``."""
        ...

    def predict_proba(self, table: pd.DataFrame, /) -> FloatArray:
        """``[H, D, A]`` probabilities, one row per fixture; each row sums to 1."""
        ...


def feature_columns() -> list[str]:
    """The engineered feature columns the discriminative models consume, in pipeline order.

    Read from the live accumulator set, so it stays in lock-step with the feature table and excludes
    the identifiers, odds, goals, and the label.
    """
    return [name for accumulator in default_accumulators() for name in accumulator.feature_names]
