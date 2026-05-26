"""End-to-end inference tests against the committed artifact + the sample EPL season — offline.

Points the config singleton at ``tests/fixtures/sample`` (PLAN Decision 7) so ``matches.load`` reads
the committed sample ``matches.parquet``; ``models_dir`` is left untouched so the **shipped**
``models/v1.joblib`` is exercised. No network, no real data.
"""

import datetime as dt

import pytest

from matchodds import config
from serving.app.inference import Inference, UnknownFixtureError
from serving.app.schemas import PredictRequest

_SAMPLE = config.settings.repo_root / "tests" / "fixtures" / "sample"
# A date after every sample match (sample ends 2023-12-09), so the fixture has full pre-match history.
_MATCH_DATE = dt.date(2024, 5, 1)


@pytest.fixture
def inference(monkeypatch):
    monkeypatch.setattr(config.settings, "data_dir", _SAMPLE)
    return Inference()


def _request(**overrides):
    base = dict(home="Arsenal", away="Chelsea", league="English Premier League", match_date=_MATCH_DATE)
    base.update(overrides)
    return PredictRequest(**base)


def test_known_fixture_returns_normalized_probabilities(inference):
    probs = inference.predict(_request())
    values = (probs.home_win, probs.draw, probs.away_win)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert sum(values) == pytest.approx(1.0)


def test_prediction_is_deterministic(inference):
    first = inference.predict(_request())
    second = inference.predict(_request())
    assert first.model_dump() == second.model_dump()


def test_unknown_team_raises(inference):
    with pytest.raises(UnknownFixtureError):
        inference.predict(_request(home="Nonexistent FC"))


def test_unknown_league_raises(inference):
    with pytest.raises(UnknownFixtureError):
        inference.predict(_request(league="Martian Premier League"))
