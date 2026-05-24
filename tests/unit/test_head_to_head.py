"""Tests for the head-to-head feature accumulator (offline, hand-computable)."""

import datetime as dt
import math
from pathlib import Path

import pandas as pd

from matchodds.features import pipeline
from matchodds.features.base import MatchRow
from matchodds.features.head_to_head import HeadToHeadAccumulator

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"
_D = dt.date(2024, 1, 1)
_H2H_COLUMNS = {
    "h2h_matches",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
}


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


def _match(home: str, away: str, hg: int, ag: int, league: str = "L") -> MatchRow:
    result = "H" if hg > ag else "A" if hg < ag else "D"
    return MatchRow(home=home, away=away, league=league, date=_D, ft_home_goals=hg, ft_away_goals=ag, result=result)


def test_first_meeting_is_zero_and_nan():
    h2h = HeadToHeadAccumulator()
    snap = h2h.pre_match("A", "B", "L", _D)
    assert snap["h2h_matches"] == 0.0
    assert snap["h2h_home_wins"] == 0.0
    assert snap["h2h_draws"] == 0.0
    assert snap["h2h_away_wins"] == 0.0
    assert math.isnan(snap["h2h_home_goals_avg"])
    assert math.isnan(snap["h2h_away_goals_avg"])


def test_counts_and_goals_from_current_home_perspective():
    h2h = HeadToHeadAccumulator()
    h2h.update(_match("A", "B", 2, 1))  # A home win
    h2h.update(_match("B", "A", 0, 0))  # draw
    snap = h2h.pre_match("A", "B", "L", _D)
    assert snap["h2h_matches"] == 2.0
    assert snap["h2h_home_wins"] == 1.0  # A won once
    assert snap["h2h_draws"] == 1.0
    assert snap["h2h_away_wins"] == 0.0  # B never won
    assert snap["h2h_home_goals_avg"] == 1.0  # A scored 2 + 0 over 2 meetings
    assert snap["h2h_away_goals_avg"] == 0.5  # B scored 1 + 0 over 2 meetings


def test_perspective_flips_for_reverse_fixture():
    h2h = HeadToHeadAccumulator()
    h2h.update(_match("A", "B", 3, 0))  # A home win
    snap = h2h.pre_match("B", "A", "L", _D)  # B now hosts A
    assert snap["h2h_matches"] == 1.0
    assert snap["h2h_home_wins"] == 0.0  # B (home) never won
    assert snap["h2h_away_wins"] == 1.0  # A (away) won
    assert snap["h2h_home_goals_avg"] == 0.0  # B scored 0
    assert snap["h2h_away_goals_avg"] == 3.0  # A scored 3


def test_pairs_are_isolated():
    h2h = HeadToHeadAccumulator()
    h2h.update(_match("A", "B", 1, 0))
    snap = h2h.pre_match("A", "C", "L", _D)  # A and C have never met
    assert snap["h2h_matches"] == 0.0
    assert math.isnan(snap["h2h_home_goals_avg"])


def test_pipeline_emits_h2h_columns_and_counts_rematches():
    table = pipeline.build_feature_table(_load())
    assert _H2H_COLUMNS <= set(table.columns)
    alpha_beta = table[(table["home"] == "Alpha") & (table["away"] == "Beta")].sort_values("date")
    first = alpha_beta.iloc[0]  # 2022-08-06, first-ever meeting
    assert first["h2h_matches"] == 0.0
    assert math.isnan(first["h2h_home_goals_avg"])
    second = alpha_beta.iloc[1]  # 2023-08-12, after two prior meetings of the pair
    assert second["h2h_matches"] == 2.0


def test_appending_future_match_does_not_change_earlier_h2h():
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
