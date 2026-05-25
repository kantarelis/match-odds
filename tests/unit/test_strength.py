"""Tests for the season-scoped home/away strength accumulator (offline, hand-computable)."""

import datetime as dt
import math
from pathlib import Path

import pandas as pd

from matchodds.features import pipeline
from matchodds.features.base import MatchRow
from matchodds.features.strength import StrengthAccumulator

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"
_D = dt.date(2024, 1, 1)
_STRENGTH_COLUMNS = {
    "strength_home_ppg",
    "strength_home_gf",
    "strength_home_ga",
    "strength_home_n",
    "strength_away_ppg",
    "strength_away_gf",
    "strength_away_ga",
    "strength_away_n",
}


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


def _match(home: str, away: str, hg: int, ag: int, date: dt.date = _D, league: str = "L") -> MatchRow:
    result = "H" if hg > ag else "A" if hg < ag else "D"
    return MatchRow(home=home, away=away, league=league, date=date, ft_home_goals=hg, ft_away_goals=ag, result=result)


def test_cold_start_is_nan_with_zero_count():
    strength = StrengthAccumulator()
    snap = strength.pre_match("A", "B", "L", dt.date(2023, 9, 1))
    assert snap["strength_home_n"] == 0.0
    assert snap["strength_away_n"] == 0.0
    assert math.isnan(snap["strength_home_ppg"])
    assert math.isnan(snap["strength_away_gf"])


def test_home_record_uses_only_home_matches():
    strength = StrengthAccumulator()
    day = dt.date(2023, 9, 1)
    strength.update(_match("A", "B", 2, 0, date=day))  # A home win
    strength.update(_match("C", "A", 1, 1, date=day))  # A away draw — must not touch A's home record
    snap = strength.pre_match("A", "X", "L", day)
    assert snap["strength_home_n"] == 1.0
    assert snap["strength_home_ppg"] == 3.0
    assert snap["strength_home_gf"] == 2.0
    assert snap["strength_home_ga"] == 0.0


def test_away_record_uses_only_away_matches():
    strength = StrengthAccumulator()
    day = dt.date(2023, 9, 1)
    strength.update(_match("H", "A", 0, 2, date=day))  # A away win
    strength.update(_match("A", "Z", 0, 0, date=day))  # A home draw — must not touch A's away record
    snap = strength.pre_match("X", "A", "L", day)
    assert snap["strength_away_n"] == 1.0
    assert snap["strength_away_ppg"] == 3.0
    assert snap["strength_away_gf"] == 2.0
    assert snap["strength_away_ga"] == 0.0


def test_records_reset_each_season():
    strength = StrengthAccumulator()
    strength.update(_match("A", "B", 3, 0, date=dt.date(2023, 9, 1)))  # season 2324
    snap = strength.pre_match("A", "C", "L", dt.date(2024, 9, 1))  # season 2425 — fresh
    assert snap["strength_home_n"] == 0.0
    assert math.isnan(snap["strength_home_ppg"])


def test_pipeline_emits_strength_columns_with_season_reset():
    table = pipeline.build_feature_table(_load())
    assert _STRENGTH_COLUMNS <= set(table.columns)
    alpha_home = table[table["home"] == "Alpha"].sort_values("date").reset_index(drop=True)
    assert alpha_home.loc[0, "strength_home_n"] == 0.0  # season-one opener
    assert alpha_home.loc[2, "strength_home_n"] == 2.0  # third home match of season one
    assert alpha_home.loc[3, "strength_home_n"] == 0.0  # season-two opener resets


def test_appending_future_match_does_not_change_earlier_strength():
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
