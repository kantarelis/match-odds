"""Validation tests for the Match ingestion model."""

import datetime as dt

import pytest
from pydantic import ValidationError

from matchodds.data.schema import Match


def _valid_kwargs(**overrides):
    base = dict(
        home="Olympiacos",
        away="Panathinaikos",
        league="Greek Super League",
        date=dt.date(2023, 9, 1),
        ft_home_goals=2,
        ft_away_goals=1,
        result="H",
        source="football-data",
    )
    base.update(overrides)
    return base


def test_valid_match():
    match = Match(**_valid_kwargs())
    assert match.result == "H"
    assert match.odds_home is None


def test_valid_match_with_odds():
    match = Match(**_valid_kwargs(odds_home=2.1, odds_draw=3.4, odds_away=3.0))
    assert match.odds_home == 2.1


def test_negative_goals_rejected():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(ft_home_goals=-1))


def test_result_inconsistent_with_score_rejected():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(ft_home_goals=1, ft_away_goals=1, result="H"))  # draw, not H


def test_same_team_rejected():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(away="Olympiacos"))


def test_odds_must_exceed_one():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(odds_home=0.5))


def test_unknown_source_rejected():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(source="scraped"))


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(foo="bar"))


def test_out_of_range_date_rejected():
    with pytest.raises(ValidationError):
        Match(**_valid_kwargs(date=dt.date(1900, 1, 1)))
