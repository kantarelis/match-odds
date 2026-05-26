"""Prometheus metrics for the inference service, on the default registry.

A single counter for now; the predict view (Task 4) increments it and ``GET /metrics`` exposes the
default registry in Prometheus text format.
"""

from prometheus_client import Counter

predictions_total = Counter("predictions_total", "Total /predict requests served.")
