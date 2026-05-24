"""Shared contracts for the leakage-free feature pipeline.

A *feature accumulator* is the unit of the pipeline: a stateful object that, walking the match
history in chronological order, emits a fixture's **pre-match** feature values (``pre_match``) and
then folds a played match into its state (``update``). The pipeline snapshots every match *before*
updating with that day's results, so a match's features depend only on strictly-earlier matches —
leakage-free by construction (see :mod:`matchodds.features.pipeline`).

The same accumulators build the training table and the single feature row served at inference time,
so there is no train/serve skew.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

from matchodds.data import sources


@dataclass(frozen=True, slots=True)
class MatchRow:
    """A played match, as fed to :meth:`FeatureAccumulator.update` (home perspective for ``result``)."""

    home: str
    away: str
    league: str
    date: dt.date
    ft_home_goals: int
    ft_away_goals: int
    result: str


class FeatureAccumulator(Protocol):
    """A stateful, chronologically-driven source of pre-match features for one feature family."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The feature column names this accumulator emits (fixes column order in the table)."""
        ...

    def pre_match(self, home: str, away: str, league: str, date: dt.date, /) -> dict[str, float]:
        """Pre-kickoff feature values for a fixture, from state built only from earlier matches.

        Arguments are positional-only, so an accumulator that does not need one (e.g. Elo ignores
        ``date``) may rename it ``_date`` without breaking structural conformance.
        """
        ...

    def update(self, match: MatchRow, /) -> None:
        """Fold a played match into the accumulator's state."""
        ...


def season_of(date: dt.date) -> str:
    """Season code (e.g. ``'2324'``) containing ``date``; seasons roll over in July (Aug–May)."""
    start_year = date.year if date.month >= 7 else date.year - 1
    return sources.season_code(start_year)
