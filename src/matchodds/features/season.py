"""Season-progress features as an accumulator.

For each team within a season the pipeline tracks how many matches it has already played and the date
of its last one. A match's snapshot reads, for both teams *before* the match is folded in: the
matchday it is about to play (matches played this season + 1), the rest since its previous match, and
a single normalised season-progress fraction. Records are keyed by season (``season_of``) and reset
at each season boundary; a team's first appearance in a season has no rest (NaN).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from math import nan

from matchodds.features.base import MatchRow, season_of

_SEASON_MATCHDAYS = 38.0


@dataclass
class _TeamSeason:
    """How many matches a team has played in a season, and when it last played."""

    played: int = 0
    last_date: dt.date | None = None


def _matchday(rec: _TeamSeason | None) -> int:
    return (rec.played if rec is not None else 0) + 1


def _rest_days(rec: _TeamSeason | None, date: dt.date) -> float:
    if rec is None or rec.last_date is None:
        return nan
    return float((date - rec.last_date).days)


class SeasonAccumulator:
    """Per-``(league, team, season)`` matchday, rest days, and season-progress fraction."""

    feature_names = (
        "season_home_matchday",
        "season_away_matchday",
        "season_fraction",
        "days_since_last_match_home",
        "days_since_last_match_away",
    )

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], _TeamSeason] = {}

    def pre_match(self, home: str, away: str, league: str, date: dt.date, /) -> dict[str, float]:
        season = season_of(date)
        home_rec = self._records.get((league, home, season))
        away_rec = self._records.get((league, away, season))
        home_matchday = _matchday(home_rec)
        away_matchday = _matchday(away_rec)
        return {
            "season_home_matchday": float(home_matchday),
            "season_away_matchday": float(away_matchday),
            "season_fraction": min(1.0, max(home_matchday, away_matchday) / _SEASON_MATCHDAYS),
            "days_since_last_match_home": _rest_days(home_rec, date),
            "days_since_last_match_away": _rest_days(away_rec, date),
        }

    def update(self, match: MatchRow, /) -> None:
        season = season_of(match.date)
        for team in (match.home, match.away):
            rec = self._records.get((match.league, team, season))
            if rec is None:
                rec = _TeamSeason()
                self._records[(match.league, team, season)] = rec
            rec.played += 1
            rec.last_date = match.date
