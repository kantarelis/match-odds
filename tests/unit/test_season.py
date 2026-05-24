"""Tests for the season-progress feature accumulator (offline, hand-computable)."""

import datetime as dt
import math
from pathlib import Path

import pandas as pd

from matchodds.features import pipeline
from matchodds.features.base import MatchRow
from matchodds.features.season import SeasonAccumulator

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"
_D = dt.date(2024, 1, 1)
_SEASON_COLUMNS = {
    "season_home_matchday",
    "season_away_matchday",
    "season_fraction",
    "days_since_last_match_home",
    "days_since_last_match_away",
}


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


def _match(home: str, away: str, hg: int, ag: int, date: dt.date = _D, league: str = "L") -> MatchRow:
    result = "H" if hg > ag else "A" if hg < ag else "D"
    return MatchRow(home=home, away=away, league=league, date=date, ft_home_goals=hg, ft_away_goals=ag, result=result)


def test_first_appearance_is_matchday_one_with_nan_rest():
    season = SeasonAccumulator()
    snap = season.pre_match("A", "B", "L", dt.date(2023, 8, 1))
    assert snap["season_home_matchday"] == 1.0
    assert snap["season_away_matchday"] == 1.0
    assert math.isnan(snap["days_since_last_match_home"])
    assert math.isnan(snap["days_since_last_match_away"])
    assert snap["season_fraction"] == 1.0 / 38.0


def test_matchday_increments_and_rest_days():
    season = SeasonAccumulator()
    season.update(_match("A", "B", 1, 0, date=dt.date(2023, 8, 5)))
    snap = season.pre_match("A", "C", "L", dt.date(2023, 8, 12))
    assert snap["season_home_matchday"] == 2.0  # A's second match
    assert snap["days_since_last_match_home"] == 7.0
    assert snap["season_away_matchday"] == 1.0  # C is fresh
    assert math.isnan(snap["days_since_last_match_away"])


def test_matchday_resets_each_season():
    season = SeasonAccumulator()
    season.update(_match("A", "B", 1, 0, date=dt.date(2023, 9, 1)))  # season 2324
    snap = season.pre_match("A", "C", "L", dt.date(2024, 9, 1))  # season 2425 — fresh
    assert snap["season_home_matchday"] == 1.0
    assert math.isnan(snap["days_since_last_match_home"])


def test_fraction_uses_furthest_matchday_and_caps_at_one():
    season = SeasonAccumulator()
    day = dt.date(2023, 9, 1)
    season.update(_match("A", "X", 1, 0, date=day))  # A matchday 1 played
    season.update(_match("B", "Y", 1, 0, date=day))  # B matchday 1 played
    season.update(_match("B", "Z", 1, 0, date=day))  # B matchday 2 played
    snap = season.pre_match("A", "B", "L", day)
    assert snap["season_home_matchday"] == 2.0
    assert snap["season_away_matchday"] == 3.0
    assert snap["season_fraction"] == 3.0 / 38.0  # furthest-along team
    for _ in range(40):
        season.update(_match("A", "W", 1, 0, date=day))
    assert season.pre_match("A", "V", "L", day)["season_fraction"] == 1.0  # capped


def test_pipeline_emits_season_columns_with_reset():
    table = pipeline.build_feature_table(_load())
    assert _SEASON_COLUMNS <= set(table.columns)
    alpha_home = table[table["home"] == "Alpha"].sort_values("date").reset_index(drop=True)
    assert math.isnan(alpha_home.loc[0, "days_since_last_match_home"])  # season-one opener
    assert alpha_home.loc[0, "season_home_matchday"] == 1.0
    assert alpha_home.loc[1, "season_home_matchday"] == 2.0
    assert alpha_home.loc[1, "days_since_last_match_home"] == 35.0  # 2022-08-06 -> 2022-09-10
    assert alpha_home.loc[2, "season_home_matchday"] == 3.0
    assert alpha_home.loc[3, "season_home_matchday"] == 1.0  # season-two opener resets


def test_appending_future_match_does_not_change_earlier_season_features():
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
