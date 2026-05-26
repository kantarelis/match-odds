"""Pydantic v2 request/response models — the typed HTTP contract for the inference service.

One module holds every schema (PLAN Decision 4): the ``POST /predict`` request and response, plus
the ``/health`` and ``/env`` payloads. Keeping the contract in one place (rather than quake-feed's
per-module ``models.py``) is the deliberate simplification for this smaller service.
"""

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """A fixture to score: canonical team + league names and the kickoff date.

    Names are expected in their canonical spellings (the ones stored in ``matches.parquet``); the
    inference layer validates ``league`` and the teams against the in-scope registry and rejects an
    unknown fixture with HTTP 422 rather than fabricating a prediction (PLAN Decision 5).
    """

    model_config = ConfigDict(extra="forbid")

    home: str = Field(min_length=1)
    away: str = Field(min_length=1)
    league: str = Field(min_length=1)
    match_date: dt.date


class OutcomeProbabilities(BaseModel):
    """Calibrated 1X2 probabilities for a fixture; the three values sum to ~1.0."""

    home_win: float = Field(ge=0.0, le=1.0)
    draw: float = Field(ge=0.0, le=1.0)
    away_win: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Liveness payload for ``GET /health``."""

    status: str
    service: str
    version: str


class EnvResponse(BaseModel):
    """Running-environment payload for ``GET /env``."""

    environment: str
    application_name: str
    version: str
