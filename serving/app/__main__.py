"""``python -m serving.app`` — run the inference service with uvicorn.

Bind host/port come from the typed config (``MATCHODDS_SERVE_HOST`` / ``MATCHODDS_SERVE_PORT``).
The ``local`` environment gets hot reload (uvicorn re-imports the app factory on change); any other
environment — e.g. the Docker image — serves a single pre-built app with no reloader.
"""

from __future__ import annotations

import uvicorn

from matchodds.config import settings
from serving.app.main import create_app


def main() -> None:
    if settings.environment == "local":
        uvicorn.run(
            "serving.app.main:create_app",
            factory=True,
            host=settings.serve_host,
            port=settings.serve_port,
            reload=True,
        )
    else:
        uvicorn.run(create_app(), host=settings.serve_host, port=settings.serve_port)


if __name__ == "__main__":
    main()
