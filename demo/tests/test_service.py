"""Unit tests for the demo's service client + option helpers — offline via httpx.MockTransport.

No real network: a mock transport plays the inference service, so we assert the exact request body,
a parsed 200, and that connection failures / a 422 each raise a ServiceError with a useful message.
"""

import datetime as dt
import json

import httpx
import pytest

from demo.service import PredictClient, Probabilities, ServiceError, available_leagues, teams_for
from matchodds.config import settings
from matchodds.data import teams

_FIXTURE = {
    "home": "Arsenal",
    "away": "Chelsea",
    "league": "English Premier League",
    "match_date": dt.date(2024, 5, 1),
}


def _client(handler):
    transport = httpx.MockTransport(handler)
    return PredictClient(httpx.Client(transport=transport, base_url="http://testserver"))


def test_predict_posts_fixture_and_parses_probabilities():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"home_win": 0.6, "draw": 0.25, "away_win": 0.15})

    probs = _client(handler).predict(**_FIXTURE)

    assert captured["url"].endswith("/predict")
    assert captured["body"] == {
        "home": "Arsenal",
        "away": "Chelsea",
        "league": "English Premier League",
        "match_date": "2024-05-01",
    }
    assert probs == Probabilities(home_win=0.6, draw=0.25, away_win=0.15)


def test_predict_omitting_match_date_sends_a_future_cutoff():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"home_win": 0.5, "draw": 0.3, "away_win": 0.2})

    _client(handler).predict(home="Arsenal", away="Chelsea", league="English Premier League")

    sent = dt.date.fromisoformat(captured["body"]["match_date"])
    assert sent > dt.date.today()


def test_predict_connection_error_raises_service_error():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(ServiceError, match="unavailable"):
        _client(handler).predict(**_FIXTURE)


def test_predict_422_surfaces_service_detail():
    def handler(request):
        return httpx.Response(422, json={"detail": "unknown team 'Nonexistent FC'"})

    with pytest.raises(ServiceError, match="Nonexistent FC"):
        _client(handler).predict(**_FIXTURE)


def test_available_leagues_matches_config():
    assert available_leagues() == list(settings.leagues)


def test_teams_for_returns_sorted_registry_names():
    result = teams_for("English Premier League")

    assert result == sorted(teams.canonical_names("English Premier League"))
    assert result == sorted(result)
