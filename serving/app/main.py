"""The FastAPI app factory: build the service and mount every Manager's router.

Mirrors the quake-feed app-factory + Manager/Views structure: :class:`MatchOddsService` builds the
``FastAPI`` app from the project metadata, constructs each Manager, and ``.run()``s its router onto
the app. :func:`create_app` is the entrypoint used by ``__main__`` and the tests.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from matchodds import __metadata__
from serving.app.api.main.main import MainManager


class MatchOddsService:
    """Builds the FastAPI app and mounts every API Manager's router."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger else logging.getLogger("MatchOddsService")
        self.app = FastAPI(
            title="match-odds inference service",
            description="Calibrated football match-outcome probabilities from the frozen model artifact.",
            version=__metadata__.__version__,
        )
        self.managers = [MainManager(logger=self.logger)]

    def run(self) -> FastAPI:
        """Mount every Manager's router onto the app and return it."""
        for manager in self.managers:
            self.app.include_router(manager.run())
        return self.app


def create_app() -> FastAPI:
    """Construct the fully-wired FastAPI app (factory for uvicorn and the tests)."""
    return MatchOddsService().run()
