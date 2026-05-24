"""Tests for the rolling-form feature accumulator (offline, hand-computable)."""

import datetime as dt
import math
from pathlib import Path

import pandas as pd

from matchodds.features import pipeline
from matchodds.features.base import MatchRow
from matchodds.features.form import FormAccumulator

_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic_matches.csv"
_D = dt.date(2024, 1, 1)
_FORM_COLUMNS = {
    "form_home_ppg",
    "form_home_gf",
    "form_home_ga",
    "form_home_n",
    "form_away_ppg",
    "form_away_gf",
    "form_away_ga",
    "form_away_n",
}


def _load() -> pd.DataFrame:
    return pd.read_csv(_SYNTHETIC, parse_dates=["date"])


def _match(home: str, away: str, hg: int, ag: int, league: str = "L") -> MatchRow:
    result = "H" if hg > ag else "A" if hg < ag else "D"
    return MatchRow(home=home, away=away, league=league, date=_D, ft_home_goals=hg, ft_away_goals=ag, result=result)


def test_cold_start_is_nan_with_zero_count():
    form = FormAccumulator(window=5)
    snap = form.pre_match("A", "B", "L", _D)
    assert snap["form_home_n"] == 0.0
    assert snap["form_away_n"] == 0.0
    for key in ("form_home_ppg", "form_home_gf", "form_home_ga"):
        assert math.isnan(snap[key])


def test_rolling_means_over_both_venues():
    form = FormAccumulator(window=5)
    form.update(_match("A", "B", 2, 0))  # A home win: pts 3, gf 2, ga 0
    form.update(_match("C", "A", 3, 1))  # A away loss: pts 0, gf 1, ga 3
    snap = form.pre_match("A", "X", "L", _D)
    assert snap["form_home_n"] == 2.0
    assert snap["form_home_ppg"] == 1.5  # (3 + 0) / 2
    assert snap["form_home_gf"] == 1.5  # (2 + 1) / 2
    assert snap["form_home_ga"] == 1.5  # (0 + 3) / 2


def test_window_bounds_to_last_n():
    form = FormAccumulator(window=2)
    form.update(_match("A", "B", 5, 0))  # dropped once the window fills
    form.update(_match("A", "C", 0, 0))  # pts 1, gf 0, ga 0
    form.update(_match("A", "D", 1, 0))  # pts 3, gf 1, ga 0
    snap = form.pre_match("A", "X", "L", _D)
    assert snap["form_home_n"] == 2.0
    assert snap["form_home_ppg"] == 2.0  # (1 + 3) / 2
    assert snap["form_home_gf"] == 0.5  # (0 + 1) / 2
    assert snap["form_home_ga"] == 0.0


def test_away_side_uses_away_perspective():
    form = FormAccumulator(window=5)
    form.update(_match("H", "A", 0, 2))  # A away win: pts 3, gf 2, ga 0
    snap = form.pre_match("X", "A", "L", _D)
    assert snap["form_away_n"] == 1.0
    assert snap["form_away_ppg"] == 3.0
    assert snap["form_away_gf"] == 2.0
    assert snap["form_away_ga"] == 0.0


def test_pipeline_emits_form_columns_with_cold_start_first():
    table = pipeline.build_feature_table(_load())
    assert _FORM_COLUMNS <= set(table.columns)
    first = table.sort_values(["date", "home"]).iloc[0]
    assert first["form_home_n"] == 0.0
    assert first["form_away_n"] == 0.0


def test_appending_future_match_does_not_change_earlier_form():
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
