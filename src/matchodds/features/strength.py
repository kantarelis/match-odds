"""Home/away venue strength as a season-scoped feature accumulator.

Within each season the pipeline keeps, per team, a record of how it performs **at home** and a
separate record of how it performs **away** — points, goals scored and goals conceded. A match's
snapshot reads the home team's home record and the away team's away record *before* the match is
folded in, so the feature is leakage-free. Records are keyed by season (``season_of``) and therefore
reset at each season boundary; a team with no qualifying matches yet this season snapshots NaN with
a companion ``strength_*_n`` count of 0.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from math import nan

from matchodds.features.base import MatchRow, season_of

_HOME_POINTS = {"H": 3, "D": 1, "A": 0}
_AWAY_POINTS = {"H": 0, "D": 1, "A": 3}


@dataclass
class _VenueRecord:
    """Running points/goals for one team's matches at a single venue within one season."""

    n: int = 0
    points: int = 0
    gf: int = 0
    ga: int = 0


def _stats(rec: _VenueRecord | None) -> dict[str, float]:
    if rec is None or rec.n == 0:
        return {"ppg": nan, "gf": nan, "ga": nan, "n": 0.0}
    n = float(rec.n)
    return {"ppg": rec.points / n, "gf": rec.gf / n, "ga": rec.ga / n, "n": n}


class StrengthAccumulator:
    """Per-``(league, team, season, venue)`` home/away strength records."""

    feature_names = (
        "strength_home_ppg",
        "strength_home_gf",
        "strength_home_ga",
        "strength_home_n",
        "strength_away_ppg",
        "strength_away_gf",
        "strength_away_ga",
        "strength_away_n",
    )

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, str], _VenueRecord] = {}

    def pre_match(self, home: str, away: str, league: str, date: dt.date, /) -> dict[str, float]:
        season = season_of(date)
        home_stats = _stats(self._records.get((league, home, season, "H")))
        away_stats = _stats(self._records.get((league, away, season, "A")))
        return {
            "strength_home_ppg": home_stats["ppg"],
            "strength_home_gf": home_stats["gf"],
            "strength_home_ga": home_stats["ga"],
            "strength_home_n": home_stats["n"],
            "strength_away_ppg": away_stats["ppg"],
            "strength_away_gf": away_stats["gf"],
            "strength_away_ga": away_stats["ga"],
            "strength_away_n": away_stats["n"],
        }

    def update(self, match: MatchRow, /) -> None:
        season = season_of(match.date)
        home_key = (match.league, match.home, season, "H")
        away_key = (match.league, match.away, season, "A")
        self._add(home_key, _HOME_POINTS[match.result], match.ft_home_goals, match.ft_away_goals)
        self._add(away_key, _AWAY_POINTS[match.result], match.ft_away_goals, match.ft_home_goals)

    def _add(self, key: tuple[str, str, str, str], points: int, gf: int, ga: int) -> None:
        rec = self._records.get(key)
        if rec is None:
            rec = _VenueRecord()
            self._records[key] = rec
        rec.n += 1
        rec.points += points
        rec.gf += gf
        rec.ga += ga
