"""API tests for the main router — health, env, metrics — via an in-process TestClient (no server).

The main router does no model work, so the app boots without any data or artifact.
"""

from fastapi.testclient import TestClient

from matchodds import __metadata__
from matchodds.config import settings
from serving.app.main import create_app

client = TestClient(create_app())


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"]
    assert body["version"] == __metadata__.__version__


def test_env_reflects_settings():
    response = client.get("/env")
    assert response.status_code == 200
    body = response.json()
    assert body["environment"] == settings.environment
    assert body["application_name"]
    assert body["version"] == __metadata__.__version__


def test_metrics_prometheus_content_type():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_openapi_documents_main_operations():
    schema = client.get("/openapi.json").json()
    operations = {op.get("operationId") for path in schema["paths"].values() for op in path.values()}
    assert {"health", "env", "metrics"} <= operations
