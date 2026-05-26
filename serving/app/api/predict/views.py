"""Endpoint logic for ``POST /predict``: turn a fixture request into calibrated probabilities.

A **sync** ``def`` (PLAN Decision 11): the feature reconstruction + ``predict_proba`` are CPU-bound,
so FastAPI runs this in a threadpool, off the event loop. The view stays HTTP-agnostic — an
out-of-scope fixture raises :class:`~serving.app.inference.UnknownFixtureError`, which the app-level
handler maps to HTTP 422.
"""

from __future__ import annotations

import logging

from serving.app.inference import get_inference
from serving.app.metrics import predictions_total
from serving.app.schemas import OutcomeProbabilities, PredictRequest


class PredictViews:
    """Handler for ``POST /predict``."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger else logging.getLogger("PredictViews")

    def predict(self, request: PredictRequest) -> OutcomeProbabilities:
        """Calibrated home/draw/away probabilities for a fixture."""
        probabilities = get_inference().predict(request)
        predictions_total.inc()
        self.logger.debug("predicted %s vs %s (%s)", request.home, request.away, request.league)
        return probabilities
