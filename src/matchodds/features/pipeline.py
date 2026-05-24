"""The leakage-free feature pipeline: chronological driver + the ``features`` contract.

:func:`build_feature_table` walks the master matches table in date order and, for every match,
records the accumulators' **pre-match** snapshot (computed only from strictly-earlier matches)
alongside the identifiers, the label, and the bookmaker odds. :func:`features` is the point-in-time
entrypoint used at serving time: it replays history before a cutoff, then snapshots upcoming
fixtures with the *same* accumulators — so training and serving share one code path (no skew).

Matches sharing a date are snapshotted as a block *before* any of them updates state, so a match's
features never depend on another match played the same day (kickoff order within a day is unknown).
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd

from matchodds.features.base import FeatureAccumulator, MatchRow
from matchodds.features.elo import EloAccumulator

_IDENTIFIERS = ["league", "date", "home", "away"]
_ODDS = ["odds_home", "odds_draw", "odds_away"]
_LABEL = "result"
_SORT_KEYS = ["date", "league", "home", "away"]


def default_accumulators() -> tuple[FeatureAccumulator, ...]:
    """The feature families wired into the pipeline (filled in as Epic 03 tasks land)."""
    return (EloAccumulator(),)


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(key): val for key, val in rec.items()} for rec in frame.to_dict("records")]


def _records_by_date(matches: pd.DataFrame) -> Iterator[tuple[dt.date, list[dict[str, Any]]]]:
    """Yield ``(date, records)`` groups in chronological order."""
    ordered = matches.sort_values(_SORT_KEYS, kind="stable")
    for date, group in itertools.groupby(_records(ordered), key=lambda rec: _as_date(rec["date"])):
        yield date, list(group)


def _match_row(record: dict[str, Any]) -> MatchRow:
    return MatchRow(
        home=str(record["home"]),
        away=str(record["away"]),
        league=str(record["league"]),
        date=_as_date(record["date"]),
        ft_home_goals=int(record["ft_home_goals"]),
        ft_away_goals=int(record["ft_away_goals"]),
        result=str(record["result"]),
    )


def _feature_columns(accumulators: Sequence[FeatureAccumulator]) -> list[str]:
    return [name for acc in accumulators for name in acc.feature_names]


def build_feature_table(
    matches: pd.DataFrame,
    accumulators: Sequence[FeatureAccumulator] | None = None,
) -> pd.DataFrame:
    """One pre-match feature row per match, in chronological order (leakage-free by construction)."""
    accs = default_accumulators() if accumulators is None else tuple(accumulators)
    rows: list[dict[str, Any]] = []
    for _, day in _records_by_date(matches):
        played: list[MatchRow] = []
        for record in day:
            match = _match_row(record)
            snapshot: dict[str, float] = {}
            for acc in accs:
                snapshot.update(acc.pre_match(match.home, match.away, match.league, match.date))
            rows.append(
                {
                    "league": match.league,
                    "date": match.date,
                    "home": match.home,
                    "away": match.away,
                    **snapshot,
                    "odds_home": record.get("odds_home"),
                    "odds_draw": record.get("odds_draw"),
                    "odds_away": record.get("odds_away"),
                    _LABEL: match.result,
                }
            )
            played.append(match)
        for played_match in played:
            for acc in accs:
                acc.update(played_match)
    columns = _IDENTIFIERS + _feature_columns(accs) + _ODDS + [_LABEL]
    return pd.DataFrame(rows, columns=columns)


def features(
    matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    *,
    date_cutoff: dt.date,
    accumulators: Sequence[FeatureAccumulator] | None = None,
) -> pd.DataFrame:
    """Pre-match features for ``fixtures``, using only matches strictly before ``date_cutoff``.

    The serving / point-in-time entrypoint: replays history into fresh accumulators, then snapshots
    each fixture. Shares accumulator code with :func:`build_feature_table` — no train/serve skew.
    """
    accs = default_accumulators() if accumulators is None else tuple(accumulators)
    for date, day in _records_by_date(matches):
        if date >= date_cutoff:
            break
        for record in day:
            match = _match_row(record)
            for acc in accs:
                acc.update(match)
    rows: list[dict[str, Any]] = []
    for fixture in _records(fixtures):
        home, away, league = str(fixture["home"]), str(fixture["away"]), str(fixture["league"])
        date = _as_date(fixture["date"])
        snapshot: dict[str, float] = {}
        for acc in accs:
            snapshot.update(acc.pre_match(home, away, league, date))
        rows.append({"league": league, "date": date, "home": home, "away": away, **snapshot})
    columns = _IDENTIFIERS + _feature_columns(accs)
    return pd.DataFrame(rows, columns=columns)
