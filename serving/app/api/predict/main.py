"""Predict router Manager (quake-feed Manager/Views): owns ``APIRouter(prefix="/predict")``.

``run()`` wires ``POST /predict`` to :meth:`PredictViews.predict` and returns the router for the app
factory to mount.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from serving.app.api.predict.views import PredictViews


class PredictManager:
    """Owns the ``/predict`` router and instantiates the :class:`PredictViews`."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger else logging.getLogger("PredictManager")
        self.router = APIRouter(prefix="/predict")
        self.views = PredictViews(logger=self.logger)

    def run(self) -> APIRouter:
        """Wire ``POST /predict`` onto the router and return it."""
        self.router.add_api_route(
            "",
            endpoint=self.views.predict,
            methods=["POST"],
            summary="Match-outcome probabilities",
            description="Calibrated home_win / draw / away_win probabilities for a fixture.",
            operation_id="predict_match",
            tags=["Predict"],
        )
        return self.router
