"""Rolling last-N form as a feature accumulator.

For each team the pipeline keeps the points, goals scored and goals conceded from its most recent
N matches (``settings.rolling_window_n``), in either venue. A match's snapshot reads the home and
away teams' rolling means *before* the match is folded in, so the feature is leakage-free. A team
with no prior matches snapshots NaN (with a companion ``form_*_n`` count of 0) so the downstream
model can tell genuine form from a cold start.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from math import nan

from matchodds.config import settings
from matchodds.features.base import MatchRow

_HOME_POINTS = {"H": 3.0, "D": 1.0, "A": 0.0}
_AWAY_POINTS = {"H": 0.0, "D": 1.0, "A": 3.0}


class FormAccumulator:
    """Per-``(league, team)`` rolling form over the last N matches (any venue)."""

    feature_names = (
        "form_home_ppg",
        "form_home_gf",
        "form_home_ga",
        "form_home_n",
        "form_away_ppg",
        "form_away_gf",
        "form_away_ga",
        "form_away_n",
    )

    def __init__(self, window: int | None = None) -> None:
        self.window = settings.rolling_window_n if window is None else window
        self._history: dict[tuple[str, str], deque[tuple[float, float, float]]] = {}

    def _form(self, league: str, team: str) -> dict[str, float]:
        recent = self._history.get((league, team))
        if not recent:
            return {"ppg": nan, "gf": nan, "ga": nan, "n": 0.0}
        n = float(len(recent))
        return {
            "ppg": sum(rec[0] for rec in recent) / n,
            "gf": sum(rec[1] for rec in recent) / n,
            "ga": sum(rec[2] for rec in recent) / n,
            "n": n,
        }

    def pre_match(self, home: str, away: str, league: str, _date: dt.date, /) -> dict[str, float]:
        home_form = self._form(league, home)
        away_form = self._form(league, away)
        return {
            "form_home_ppg": home_form["ppg"],
            "form_home_gf": home_form["gf"],
            "form_home_ga": home_form["ga"],
            "form_home_n": home_form["n"],
            "form_away_ppg": away_form["ppg"],
            "form_away_gf": away_form["gf"],
            "form_away_ga": away_form["ga"],
            "form_away_n": away_form["n"],
        }

    def update(self, match: MatchRow, /) -> None:
        self._record(match.league, match.home, _HOME_POINTS[match.result], match.ft_home_goals, match.ft_away_goals)
        self._record(match.league, match.away, _AWAY_POINTS[match.result], match.ft_away_goals, match.ft_home_goals)

    def _record(self, league: str, team: str, points: float, gf: int, ga: int) -> None:
        bucket = self._history.get((league, team))
        if bucket is None:
            bucket = deque(maxlen=self.window)
            self._history[(league, team)] = bucket
        bucket.append((points, float(gf), float(ga)))
