"""Pre-match Elo ratings as a feature accumulator.

Each team carries an Elo rating per league. A match's snapshot reads the two ratings (and their
home-advantage-adjusted difference) *before* the result is folded in, so the feature is leakage-free.
Ratings update by the standard Elo rule: the expected home score comes from the rating gap plus a
home-advantage bonus, and the realised score is 1 / 0.5 / 0 for a home win / draw / away win. The
update is zero-sum (the home team's gain is the away team's loss). New teams start at
``settings.elo_base``.
"""

from __future__ import annotations

import datetime as dt

from matchodds.config import settings
from matchodds.features.base import MatchRow

_RESULT_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}
_LOGISTIC_SCALE = 400.0


class EloAccumulator:
    """Per-``(league, team)`` Elo ratings; emits ``elo_home`` / ``elo_away`` / ``elo_diff``."""

    feature_names = ("elo_home", "elo_away", "elo_diff")

    def __init__(self, base: float | None = None, k: float | None = None, home_advantage: float | None = None) -> None:
        self.base = settings.elo_base if base is None else base
        self.k = settings.elo_k if k is None else k
        self.home_advantage = settings.elo_home_advantage if home_advantage is None else home_advantage
        self._ratings: dict[tuple[str, str], float] = {}

    def _rating(self, league: str, team: str) -> float:
        return self._ratings.get((league, team), self.base)

    def pre_match(self, home: str, away: str, league: str, _date: dt.date, /) -> dict[str, float]:
        elo_home = self._rating(league, home)
        elo_away = self._rating(league, away)
        return {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home + self.home_advantage - elo_away,
        }

    def update(self, match: MatchRow, /) -> None:
        elo_home = self._rating(match.league, match.home)
        elo_away = self._rating(match.league, match.away)
        expected_home = 1.0 / (1.0 + 10.0 ** (-(elo_home + self.home_advantage - elo_away) / _LOGISTIC_SCALE))
        delta = self.k * (_RESULT_SCORE[match.result] - expected_home)
        self._ratings[(match.league, match.home)] = elo_home + delta
        self._ratings[(match.league, match.away)] = elo_away - delta
