# Streamlit demo image — a thin HTTP client of the inference service, deliberately data-free AND
# model-free: it carries only the matchodds package (canonical registry + typed config), streamlit,
# httpx, and demo/. It POSTs fixtures to the serving container over the compose network and never
# loads models/v1.* or the matches table. Build context is the repo root (`docker compose build`;
# see docker-compose.yml) or `docker build -f demo/Dockerfile .`.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MATCHODDS_SERVICE_URL=http://serving:8000

WORKDIR /app

# 1) Demo dependencies first, so this layer is cached unless the requirements change.
COPY requirements.txt requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt

# 2) The matchodds package (editable, for the canonical registry + config). --no-deps: the runtime
#    deps already came from requirements-demo.txt above.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

# 3) The demo code (most-churn layer last; demo/tests is excluded via .dockerignore).
COPY demo/ ./demo/

# 4) Drop privileges.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "demo/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
