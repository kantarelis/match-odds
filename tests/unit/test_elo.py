"""Tests for the Elo feature accumulator (offline, hand-computable)."""

import datetime as dt
from pathlib import Path

import pandas as pd

from matchodds.config import settings
from matchodds.features import pipeline
from matchodds.features.base import MatchRow
from matchodds.features.elo import EloAccumulator

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"
_D = dt.date(2024, 1, 1)


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


def _match(home: str, away: str, hg: int, ag: int, league: str = "L") -> MatchRow:
    result = "H" if hg > ag else "A" if hg < ag else "D"
    return MatchRow(home=home, away=away, league=league, date=_D, ft_home_goals=hg, ft_away_goals=ag, result=result)


def test_cold_start_is_base_and_diff_includes_home_advantage():
    elo = EloAccumulator(base=1500.0, k=20.0, home_advantage=65.0)
    snap = elo.pre_match("A", "B", "L", _D)
    assert snap["elo_home"] == 1500.0
    assert snap["elo_away"] == 1500.0
    assert snap["elo_diff"] == 65.0


def test_update_uses_result_score_and_k_factor():
    elo = EloAccumulator(base=1500.0, k=20.0, home_advantage=0.0)
    elo.update(_match("A", "B", 2, 0))  # equal ratings -> expected_home 0.5, delta = 20 * (1 - 0.5)
    snap = elo.pre_match("A", "B", "L", _D)
    assert snap["elo_home"] == 1510.0
    assert snap["elo_away"] == 1490.0


def test_update_is_zero_sum():
    elo = EloAccumulator(base=1500.0, k=32.0, home_advantage=50.0)
    elo.update(_match("A", "B", 0, 1))  # away win despite the home-advantage bonus
    snap = elo.pre_match("A", "B", "L", _D)
    assert round((snap["elo_home"] - 1500.0) + (snap["elo_away"] - 1500.0), 9) == 0.0


def test_sequential_updates_carry_forward():
    elo = EloAccumulator(base=1500.0, k=20.0, home_advantage=0.0)
    elo.update(_match("A", "B", 1, 0))  # A -> 1510
    snap = elo.pre_match("A", "C", "L", _D)
    assert snap["elo_home"] == 1510.0  # A carried forward
    assert snap["elo_away"] == 1500.0  # C cold-starts at base
    assert snap["elo_diff"] == 10.0


def test_ratings_are_isolated_by_league():
    elo = EloAccumulator(base=1500.0, k=20.0, home_advantage=0.0)
    elo.update(_match("A", "B", 3, 0, league="L1"))
    snap = elo.pre_match("A", "B", "L2", _D)
    assert snap["elo_home"] == 1500.0  # untouched in a different league
    assert snap["elo_away"] == 1500.0


def test_pipeline_emits_elo_columns_with_cold_start_first():
    table = pipeline.build_feature_table(_load())
    assert {"elo_home", "elo_away", "elo_diff"} <= set(table.columns)
    first = table.sort_values(["date", "home"]).iloc[0]
    assert first["elo_home"] == settings.elo_base
    assert first["elo_away"] == settings.elo_base


def test_appending_future_match_does_not_change_earlier_elo():
    matches = _load()
    base_table = pipeline.build_feature_table(matches)
    future = pd.DataFrame(
        [
            {
                "league": "Test League",
                "date": pd.Timestamp("2025-05-01"),
                "home": "Alpha",
                "away": "Beta",
                "ft_home_goals": 4,
                "ft_away_goals": 0,
                "result": "H",
                "odds_home": 2.00,
                "odds_draw": 3.30,
                "odds_away": 3.80,
                "source": "football-data",
            }
        ]
    )
    rebuilt = pipeline.build_feature_table(pd.concat([matches, future], ignore_index=True))
    earlier = rebuilt[rebuilt["date"] < dt.date(2025, 5, 1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(base_table, earlier)
