"""API tests for ``POST /predict`` — valid fixture, unknown team/league, malformed date — offline.

The ``sample_data_dir`` fixture (conftest) builds the sample ``matches.parquet`` from the committed
raw CSV into a tmp data dir and points the config singleton there; the ``get_inference`` cache is
cleared so the app builds inference against that table + the shipped ``models/v1.joblib``. No
network, no real data.
"""

import pytest
from fastapi.testclient import TestClient

from serving.app import inference
from serving.app.main import create_app

# After every sample match (sample ends 2023-12-09), so the fixture has full pre-match history.
_MATCH_DATE = "2024-05-01"


@pytest.fixture
def client(sample_data_dir):
    inference.get_inference.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    inference.get_inference.cache_clear()


def _payload(**overrides):
    body = {"home": "Arsenal", "away": "Chelsea", "league": "English Premier League", "match_date": _MATCH_DATE}
    body.update(overrides)
    return body


def test_predict_known_fixture_sums_to_one(client):
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["home_win", "draw", "away_win"]
    assert all(0.0 <= value <= 1.0 for value in body.values())
    assert sum(body.values()) == pytest.approx(1.0)


def test_predict_unknown_team_returns_422(client):
    response = client.post("/predict", json=_payload(home="Nonexistent FC"))
    assert response.status_code == 422
    assert "Nonexistent FC" in response.json()["detail"]


def test_predict_unknown_league_returns_422(client):
    response = client.post("/predict", json=_payload(league="Martian Premier League"))
    assert response.status_code == 422


def test_predict_malformed_date_returns_422(client):
    response = client.post("/predict", json=_payload(match_date="not-a-date"))
    assert response.status_code == 422


def test_openapi_exposes_predict_match(client):
    schema = client.get("/openapi.json").json()
    operations = {op.get("operationId") for path in schema["paths"].values() for op in path.values()}
    assert "predict_match" in operations
