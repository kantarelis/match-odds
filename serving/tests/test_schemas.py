"""Contract tests for the serving schemas: date parsing and probability-range guards."""

import datetime as dt

import pytest
from pydantic import ValidationError

from serving.app.schemas import OutcomeProbabilities, PredictRequest


def _valid_request_kwargs(**overrides):
    base = dict(
        home="Arsenal",
        away="Chelsea",
        league="English Premier League",
        match_date="2024-05-01",
    )
    base.update(overrides)
    return base


def test_predict_request_parses_iso_match_date():
    request = PredictRequest(**_valid_request_kwargs())
    assert request.match_date == dt.date(2024, 5, 1)


def test_predict_request_rejects_malformed_match_date():
    with pytest.raises(ValidationError):
        PredictRequest(**_valid_request_kwargs(match_date="not-a-date"))


def test_predict_request_forbids_extra_fields():
    with pytest.raises(ValidationError):
        PredictRequest(**_valid_request_kwargs(foo="bar"))


def test_outcome_probabilities_round_trip():
    probs = OutcomeProbabilities(home_win=0.5, draw=0.3, away_win=0.2)
    assert probs.model_dump() == {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}


def test_outcome_probabilities_rejects_out_of_range():
    with pytest.raises(ValidationError):
        OutcomeProbabilities(home_win=1.5, draw=0.0, away_win=0.0)
