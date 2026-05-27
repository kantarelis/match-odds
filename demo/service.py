"""Typed HTTP client for the inference service — the demo's only contact with serving.

The Streamlit demo is a thin HTTP client (PLAN Decision 2): it never imports ``serving.app`` and
never loads the model. :class:`PredictClient` POSTs a fixture to ``f"{settings.service_url}/predict"``
and parses the calibrated ``{home_win, draw, away_win}`` response into a :class:`Probabilities`.
Connection / timeout / non-200 failures map to a :class:`ServiceError` carrying a human-readable
message (the service's ``detail`` on a 422). The option helpers read the canonical registry shipped
in ``matchodds`` so the demo stays data- and model-free.

:class:`PredictClient` accepts an optional ``httpx.Client`` (Decision 5) so tests inject an
``httpx.MockTransport`` — no real network, no extra mocking dependency.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import httpx

from matchodds.config import settings
from matchodds.data import teams

# Short timeout: a recruiter-facing demo should fail fast and degrade gracefully, not hang.
_PREDICT_TIMEOUT_S = 10.0

_UNREACHABLE_MESSAGE = (
    "Inference service unavailable — is it running? Start it with `make up` (Docker) or `make serve`."
)

# The demo forecasts the teams' *next*, unscheduled meeting — which has no calendar date. The frozen
# service still requires a match_date, so we send a near-future cutoff: it makes the service score the
# fixture with each side's latest known form, and every sufficiently-future date collapses to that
# same snapshot, so the exact value is irrelevant.
_NEXT_MATCH_HORIZON = dt.timedelta(days=365)


def _next_match_date() -> dt.date:
    """A point-in-time cutoff representing the teams' next (unscheduled) meeting."""
    return dt.date.today() + _NEXT_MATCH_HORIZON


@dataclass(frozen=True)
class Probabilities:
    """Calibrated 1X2 probabilities for a fixture; the three values sum to ~1.0."""

    home_win: float
    draw: float
    away_win: float


class ServiceError(Exception):
    """The inference service was unreachable, timed out, or returned a non-200 response."""


def _error_message(response: httpx.Response) -> str:
    """Best-effort human message from a non-200 response (the service's ``detail`` on a 422)."""
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str):
        return detail
    if detail is not None:
        return f"Service returned {response.status_code}: {detail}"
    return f"Service returned an unexpected {response.status_code} response."


class PredictClient:
    """POSTs fixtures to the inference service's ``/predict`` and parses calibrated probabilities."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(base_url=settings.service_url, timeout=_PREDICT_TIMEOUT_S)

    def predict(self, home: str, away: str, league: str, match_date: dt.date | None = None) -> Probabilities:
        """Calibrated home/draw/away probabilities for the teams' next meeting.

        ``match_date`` is the point-in-time cutoff the frozen service requires; the demo never asks
        the user for it (all future dates give the same forecast — see :func:`_next_match_date`), so
        it defaults to a near-future date. Raises ServiceError on any failure.
        """
        cutoff = match_date if match_date is not None else _next_match_date()
        payload = {"home": home, "away": away, "league": league, "match_date": cutoff.isoformat()}
        try:
            response = self._client.post("/predict", json=payload)
        except httpx.HTTPError as exc:
            raise ServiceError(_UNREACHABLE_MESSAGE) from exc
        if response.status_code != 200:
            raise ServiceError(_error_message(response))
        body = response.json()
        return Probabilities(
            home_win=float(body["home_win"]),
            draw=float(body["draw"]),
            away_win=float(body["away_win"]),
        )


def available_leagues() -> list[str]:
    """In-scope leagues for the league dropdown (from the typed config)."""
    return list(settings.leagues)


def teams_for(league: str) -> list[str]:
    """Sorted canonical team names known for ``league`` (from the committed registry)."""
    return sorted(teams.canonical_names(league))
