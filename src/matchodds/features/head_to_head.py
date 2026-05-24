"""Pairwise head-to-head history as a feature accumulator.

For each unordered club pair the pipeline keeps a running tally of their previous meetings: how many
each side won, how many were drawn, and how many goals each side scored. A match's snapshot reads
that tally *before* the meeting is folded in (reported from the current home team's perspective), so
the feature is leakage-free. A pair that has never met snapshots zero counts and NaN goal averages.

All prior meetings count, regardless of venue or recency (a recency window is a possible later
refinement).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from math import nan

from matchodds.features.base import MatchRow


@dataclass
class _PairRecord:
    """Running head-to-head tally for one canonical ``(team_lo, team_hi)`` pair."""

    matches: int = 0
    lo_wins: int = 0
    hi_wins: int = 0
    draws: int = 0
    lo_goals: int = 0
    hi_goals: int = 0


class HeadToHeadAccumulator:
    """Per-``(league, team_lo, team_hi)`` head-to-head tallies, reported home-team-first."""

    feature_names = (
        "h2h_matches",
        "h2h_home_wins",
        "h2h_draws",
        "h2h_away_wins",
        "h2h_home_goals_avg",
        "h2h_away_goals_avg",
    )

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], _PairRecord] = {}

    def pre_match(self, home: str, away: str, league: str, _date: dt.date, /) -> dict[str, float]:
        lo, hi = sorted((home, away))
        rec = self._records.get((league, lo, hi))
        if rec is None or rec.matches == 0:
            return {
                "h2h_matches": 0.0,
                "h2h_home_wins": 0.0,
                "h2h_draws": 0.0,
                "h2h_away_wins": 0.0,
                "h2h_home_goals_avg": nan,
                "h2h_away_goals_avg": nan,
            }
        home_is_lo = home == lo
        home_wins = rec.lo_wins if home_is_lo else rec.hi_wins
        away_wins = rec.hi_wins if home_is_lo else rec.lo_wins
        home_goals = rec.lo_goals if home_is_lo else rec.hi_goals
        away_goals = rec.hi_goals if home_is_lo else rec.lo_goals
        matches = float(rec.matches)
        return {
            "h2h_matches": matches,
            "h2h_home_wins": float(home_wins),
            "h2h_draws": float(rec.draws),
            "h2h_away_wins": float(away_wins),
            "h2h_home_goals_avg": home_goals / matches,
            "h2h_away_goals_avg": away_goals / matches,
        }

    def update(self, match: MatchRow, /) -> None:
        lo, hi = sorted((match.home, match.away))
        rec = self._records.get((match.league, lo, hi))
        if rec is None:
            rec = _PairRecord()
            self._records[(match.league, lo, hi)] = rec
        rec.matches += 1
        if match.result == "D":
            rec.draws += 1
        else:
            winner = match.home if match.result == "H" else match.away
            if winner == lo:
                rec.lo_wins += 1
            else:
                rec.hi_wins += 1
        if match.home == lo:
            rec.lo_goals += match.ft_home_goals
            rec.hi_goals += match.ft_away_goals
        else:
            rec.hi_goals += match.ft_home_goals
            rec.lo_goals += match.ft_away_goals
