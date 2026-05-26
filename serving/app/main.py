"""The FastAPI app factory: build the service and mount every Manager's router.

Mirrors the quake-feed app-factory + Manager/Views structure: :class:`MatchOddsService` builds the
``FastAPI`` app from the project metadata, constructs each Manager, and ``.run()``s its router onto
the app. :func:`create_app` is the entrypoint used by ``__main__`` and the tests.
"""

from __future__ import annotations

import logging
from typing import Protocol

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from matchodds import __metadata__
from serving.app.api.main.main import MainManager
from serving.app.api.predict.main import PredictManager
from serving.app.inference import UnknownFixtureError


class _Manager(Protocol):
    """A quake-feed-style API Manager: wires its routes and returns the router to mount."""

    def run(self) -> APIRouter:
        """Wire routes onto the manager's router and return it."""
        ...


def _unknown_fixture_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Map an out-of-scope fixture (:class:`UnknownFixtureError`) to HTTP 422 with the message."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


class MatchOddsService:
    """Builds the FastAPI app and mounts every API Manager's router."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger else logging.getLogger("MatchOddsService")
        self.app = FastAPI(
            title="match-odds inference service",
            description="Calibrated football match-outcome probabilities from the frozen model artifact.",
            version=__metadata__.__version__,
        )
        self.app.add_exception_handler(UnknownFixtureError, _unknown_fixture_handler)
        self.managers: list[_Manager] = [MainManager(logger=self.logger), PredictManager(logger=self.logger)]

    def run(self) -> FastAPI:
        """Mount every Manager's router onto the app and return it."""
        for manager in self.managers:
            self.app.include_router(manager.run())
        return self.app


def create_app() -> FastAPI:
    """Construct the fully-wired FastAPI app (factory for uvicorn and the tests)."""
    return MatchOddsService().run()
