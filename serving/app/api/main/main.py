"""Main router Manager (quake-feed Manager/Views): owns the ``APIRouter`` and wires its routes.

``run()`` registers ``/health``, ``/env``, and ``/metrics`` on the router (each with a summary,
operation id, and tag) and returns it for the app factory to mount.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from serving.app.api.main.views import MainManagerViews


class MainManager:
    """Owns the main ``APIRouter`` and instantiates the :class:`MainManagerViews`."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger if logger else logging.getLogger("MainManager")
        self.router = APIRouter()
        self.views = MainManagerViews(logger=self.logger)

    def run(self) -> APIRouter:
        """Wire the health / env / metrics routes onto the router and return it."""
        self.logger.debug("wiring main router (health, env, metrics)")
        self.router.add_api_route(
            "/health",
            endpoint=self.views.health,
            methods=["GET"],
            summary="Liveness probe",
            description="Returns ok with the service name and version.",
            operation_id="health",
            tags=["Main"],
        )
        self.router.add_api_route(
            "/env",
            endpoint=self.views.env,
            methods=["GET"],
            summary="Running environment",
            description="The configured environment label, application name, and version.",
            operation_id="env",
            tags=["Main"],
        )
        self.router.add_api_route(
            "/metrics",
            endpoint=self.views.metrics,
            methods=["GET"],
            summary="Prometheus metrics",
            description="Default-registry metrics in Prometheus text exposition format.",
            operation_id="metrics",
            tags=["Main"],
        )
        return self.router
