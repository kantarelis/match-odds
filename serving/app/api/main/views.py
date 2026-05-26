"""Endpoint logic for the main router: health, env, and Prometheus metrics.

These are trivial (no model work), so they stay ``async`` and run on the event loop (PLAN
Decision 11). Payloads are the typed schemas from :mod:`serving.app.schemas`.
"""

from __future__ import annotations

import logging

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from matchodds import __metadata__
from matchodds.config import settings
from serving.app.schemas import EnvResponse, HealthResponse

_SERVICE_NAME = "match-odds"


class MainManagerViews:
    """Handlers for ``/health``, ``/env``, and ``/metrics``."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger else logging.getLogger("MainManagerViews")

    async def health(self) -> HealthResponse:
        """Liveness: the service is up and serving."""
        return HealthResponse(status="ok", service=_SERVICE_NAME, version=__metadata__.__version__)

    async def env(self) -> EnvResponse:
        """The configured environment label, application name, and version."""
        self.logger.debug("env requested")
        return EnvResponse(
            environment=settings.environment,
            application_name=_SERVICE_NAME,
            version=__metadata__.__version__,
        )

    async def metrics(self) -> Response:
        """Default-registry metrics in Prometheus text exposition format."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
