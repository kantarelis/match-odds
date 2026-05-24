"""Tests for the chronological feature-pipeline driver and the ``features`` contract.

Offline and deterministic: every assertion runs against the committed synthetic fixture
(4 teams, one league, two seasons, full home/away double round-robin → 24 matches, two per date).
"""

import datetime as dt
from pathlib import Path

import pandas as pd

from matchodds.features import pipeline
from matchodds.features.base import season_of

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


class _Probe:
    """A feature accumulator that records how many matches it has folded in at each snapshot."""

    feature_names = ("probe_updates_seen",)

    def __init__(self) -> None:
        self.updates = 0

    def pre_match(self, home, away, league, date):
        return {"probe_updates_seen": float(self.updates)}

    def update(self, match):
        self.updates += 1


def test_build_emits_one_row_per_match_in_date_order():
    matches = _load()
    table = pipeline.build_feature_table(matches)
    assert len(table) == len(matches)
    # No accumulators wired yet: identifiers + odds + label only, label last.
    assert list(table.columns) == ["league", "date", "home", "away", "odds_home", "odds_draw", "odds_away", "result"]
    dates = list(table["date"])
    assert dates == sorted(dates)


def test_label_and_odds_passthrough():
    matches = _load()
    table = pipeline.build_feature_table(matches)
    assert set(table["result"]) == {"H", "D", "A"}
    assert (table["odds_home"] == 2.0).all()
    # The first Alpha-vs-Beta meeting is a 2-0 home win.
    alpha_beta = table[(table["home"] == "Alpha") & (table["away"] == "Beta")].iloc[0]
    assert alpha_beta["result"] == "H"


def test_probe_sees_only_strictly_earlier_dated_matches():
    matches = _load()
    probe = _Probe()
    table = pipeline.build_feature_table(matches, accumulators=[probe])
    match_dates = pd.to_datetime(matches["date"]).dt.date
    for _, row in table.iterrows():
        expected = int((match_dates < row["date"]).sum())
        # Day-batched: every match snapshots after exactly the strictly-earlier-dated matches,
        # never after a same-day match (kickoff order within a day is unknown).
        assert row["probe_updates_seen"] == expected


def test_appending_a_future_match_leaves_earlier_rows_unchanged():
    matches = _load()
    base = pipeline.build_feature_table(matches, accumulators=[_Probe()])
    future = pd.DataFrame(
        [
            {
                "league": "Test League",
                "date": pd.Timestamp("2025-05-01"),
                "home": "Alpha",
                "away": "Beta",
                "ft_home_goals": 5,
                "ft_away_goals": 0,
                "result": "H",
                "odds_home": 2.00,
                "odds_draw": 3.30,
                "odds_away": 3.80,
                "source": "football-data",
            }
        ]
    )
    extended = pd.concat([matches, future], ignore_index=True)
    rebuilt = pipeline.build_feature_table(extended, accumulators=[_Probe()])
    rebuilt_earlier = rebuilt[rebuilt["date"] < dt.date(2025, 5, 1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(base, rebuilt_earlier)


def test_features_uses_only_pre_cutoff_history():
    matches = _load()
    probe = _Probe()
    fixtures = pd.DataFrame([{"league": "Test League", "home": "Alpha", "away": "Gamma", "date": dt.date(2023, 2, 4)}])
    out = pipeline.features(matches, fixtures, date_cutoff=dt.date(2023, 1, 1), accumulators=[probe])
    assert list(out.columns) == ["league", "date", "home", "away", "probe_updates_seen"]
    assert len(out) == 1
    # Six matches fall strictly before 2023-01-01 (the three autumn rounds of season one).
    assert out.iloc[0]["probe_updates_seen"] == 6.0


def test_season_of_july_cutover():
    assert season_of(dt.date(2023, 6, 30)) == "2223"  # June → prior season
    assert season_of(dt.date(2023, 7, 1)) == "2324"  # July → new season (cutover)
    assert season_of(dt.date(2023, 8, 1)) == "2324"
    assert season_of(dt.date(2024, 5, 20)) == "2324"  # May → still 23/24
