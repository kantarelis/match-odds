# Inference service image — frozen-artifact serving, deliberately data-free.
# The committed models/v1.* are baked in; the master matches table is mounted at runtime
# (docker-compose: ./data:/app/data:ro), so the image stays reproducible and carries no data.
# Build context is the repo root: `docker compose build` (see docker-compose.yml) or
# `docker build -f serving/Dockerfile .`.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MATCHODDS_ENVIRONMENT=docker \
    MATCHODDS_SERVE_HOST=0.0.0.0 \
    MATCHODDS_MODELS_DIR=/app/models \
    MATCHODDS_DATA_DIR=/app/data

WORKDIR /app

# 1) Runtime dependencies first, so this layer is cached unless the requirements change.
COPY requirements.txt requirements-serving.txt ./
RUN pip install --no-cache-dir -r requirements-serving.txt

# 2) The matchodds package (editable, so config's repo-root paths resolve under /app).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# 3) The frozen artifact + the service code (most-churn layers last).
COPY models/v1.joblib models/v1.metadata.json ./models/
COPY serving/ ./serving/

# 4) Drop privileges.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "serving.app"]
