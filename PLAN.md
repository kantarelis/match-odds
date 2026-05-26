# PLAN — Epic 05: FastAPI Inference Service (Dockerised)

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `implement-fastapi-inference-service`

**Goal.** Serve the frozen `models/v1.joblib` behind `POST /predict`, reusing the **quake-feed Manager/Views** FastAPI pattern, with **no train/serve skew**: the request `{home, away, league, match_date}` is turned into a single feature row by the *same* `matchodds.features.pipeline.features()` that built the training table, then scored by the loaded artifact. The service is read-only inference — it never trains. Ships with `/health`, `/env`, `/metrics`, a Dockerfile + docker-compose, and `make serve` / `make up` / `make down`, all local-only.

## Progress

| Task | Description                                                                 | Status         | Commit |
|------|-----------------------------------------------------------------------------|----------------|--------|
| 1    | Serving scaffold + deps + gate extension + `schemas.py` (the Pydantic contract) | ⬜ Not started | —      |
| 2    | `inference.py` — load artifact + matches table; reconstruct features → `predict_proba` | ⬜ Not started | —      |
| 3    | `api/main/` (health/env/metrics) + app factory `main.py` + entrypoint + `make serve` | ⬜ Not started | —      |
| 4    | `api/predict/` — `POST /predict` (Manager + Views) wired to inference        | ⬜ Not started | —      |
| 5    | `Dockerfile` + `docker-compose.yml` + `.dockerignore` + `make up` / `make down` | ⬜ Not started | —      |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Task 1 **extends the gate to cover `serving/`** (mypy strict + bandit + flake8/isort/black + vulture) and adds `serving/tests/` to `pytest` — from then on the serving code is held to the same bar as `src/`. Because the app is wired incrementally (health-only in Task 3, `/predict` in Task 4), each task's TestClient suite must pass on its own.

**Consolidation note (flag at review).** The split keeps one named concern per commit. If you prefer fewer: Tasks 3+4 (the two API modules + the app factory) are one "build the app" unit and can merge; the gate-extension half of Task 1 can be its own commit before the contract lands.

## Decisions baked in (flag at review to change)

1. **No train/serve skew — the whole point.** `predict` builds the feature row with `matchodds.features.pipeline.features(matches, fixtures, date_cutoff=match_date)` — the identical accumulators that built the training table. The service therefore needs the **master matches table** (`data/processed/matches.parquet`, via `matchodds.data.matches.load()`) to replay pre-match history; it does **not** need the feature table. `make data` is the prerequisite.
2. **Matches table is mounted, not baked.** `data/` is gitignored, so the Docker image stays data-free; `docker-compose` mounts `./data:/app/data:ro` and the service reads `matches.parquet` at startup. The committed `models/v1.*` *are* copied into the image. (Alternative: bake the parquet in — rejected to keep the image reproducible and data-free.)
3. **One typed config.** Extend `matchodds.config.Settings` with `serve_host` / `serve_port` / `environment` rather than a second settings object — the repo's "single typed config" principle. Paths (`models_dir`, `processed_dir`) already live there.
4. **Single `serving/app/schemas.py`** (per the CLAUDE.md layout) holds every Pydantic model — `PredictRequest`, `OutcomeProbabilities`, `HealthResponse`, `EnvResponse` — rather than quake-feed's per-module `models.py`. The Manager/Views *pattern* is mirrored; this one simplification is deliberate for a smaller service.
5. **Unknown team / league → 422, never a fabricated prediction.** `predict` validates `league ∈ settings.leagues` and `home`/`away ∈ teams.canonical_names(league)` up front; an unknown fixture raises a typed error mapped to HTTP 422 with a helpful message (and, where cheap, the valid options). Canonical spellings are expected (the names stored in `matches.parquet`).
6. **The artifact is opaque.** The service loads whatever `v1.joblib` is — currently a sigmoid-calibrated **logistic** model (feature-driven, league-agnostic), but it only ever calls `.predict_proba(feature_row) -> (1, 3)` in `[H, D, A]` and maps to `{home_win, draw, away_win}`. A future Dixon-Coles artifact serves through the same path unchanged.
7. **Offline, deterministic tests.** Serving tests point the live config singleton at the committed sample season (`tests/fixtures/sample` — one real `E0_2324.csv` plus its already-built `processed/matches.parquet`) via `monkeypatch.setattr(config.settings, "data_dir", <sample>)`, the established unit-test pattern (`test_matches`, `test_features_table`). **Setting `MATCHODDS_DATA_DIR` in a fixture does *not* work** — `settings` is a module-level singleton constructed at first import, before any fixture body runs. The tests read the **committed** `matches.parquet` via `matches.load()` (do **not** call `matches.build()`: it writes into the committed fixture dir and dirties the tree) and exercise the **committed** `models/v1.joblib` end-to-end — no network, no real data. Because `get_inference()` is cached, each test constructs `Inference()` directly or resets that cache so it reads the patched path. (Alternative: train a tiny model into `tmp_path` — heavier, and it wouldn't test the shipped artifact.)
8. **Manager/Views reused; quake-feed's heavy infra is not.** No auth, no database, no Celery, no SPA, no Vault/monitoring stack. The app factory in `serving/app/main.py` constructs each Manager and `.run()`s its router onto the FastAPI app — identical muscle memory, nothing more.
9. **`/metrics` via `prometheus_client`** (mirroring quake-feed): a minimal predictions `Counter` plus the default-registry text exposition. Adds `prometheus-client` to the serving runtime.
10. **Per-request history replay is accepted.** `features()` rebuilds accumulator state from scratch each call (correct, skew-free). At the in-scope local/demo data size this is fine; a startup-warmed accumulator cache is a deliberate **future** optimization, out of scope here.
11. **CPU-bound `predict` view is a sync `def`** (FastAPI runs it in a threadpool, off the event loop); the trivial health/env/metrics views stay `async` like quake-feed. A small, documented divergence justified by the synchronous feature-build + `predict_proba`.

**Out of scope (their own epics):** the Streamlit demo + a combined `make up` with the demo (Epic 06); the betting-edge notebook (Epic 07); the model card / architecture / evaluation docs and the final reproducibility pass (Epic 08). No live odds, no cloud hosting, no auth, no scraping. No accumulator-cache optimization. No new model training (the artifact is frozen by Epic 04).

---

## Task 1 — Serving scaffold, deps, gate extension + `schemas.py`

**Scope (files to touch).**
- `serving/__init__.py`, `serving/app/__init__.py`, `serving/app/api/__init__.py`, `serving/app/api/main/__init__.py`, `serving/app/api/predict/__init__.py`, `serving/tests/__init__.py` — the package skeleton so `serving.app.*` imports cleanly (drop `serving/.gitkeep`).
- `serving/app/schemas.py` (new): `PredictRequest{home: str, away: str, league: str, match_date: datetime.date}`; `OutcomeProbabilities{home_win: float, draw: float, away_win: float}` (each in `[0, 1]`; docstring notes they sum to ~1.0); `HealthResponse{status, service, version}`; `EnvResponse{environment, application_name, version}`. Pydantic v2.
- `src/matchodds/config.py`: add `serve_host: str = "127.0.0.1"`, `serve_port: int = 8000`, `environment: str = "local"` to `Settings` (env-overridable, `MATCHODDS_` prefix).
- `requirements-serving.txt` (new): `-r requirements.txt` + `fastapi`, `uvicorn`, `prometheus-client`. Use **plain `uvicorn`**, not `uvicorn[standard]` — the `[standard]` extra pulls compiled `uvloop`/`httptools`/`websockets` that may lack cp314 wheels (the same reason streamlit is deferred in `requirements-dev.txt`); flag at review to change. `requirements-test.txt`: add `-r requirements-serving.txt` **and `httpx`** (FastAPI's `TestClient` imports httpx and it is absent today). `requirements-dev.txt`: add `-r requirements-serving.txt` so `make check` can import fastapi without the test file. (`make install-dev` installs dev + test, so the gate and `make test` both see fastapi/httpx; CI runs the suite under `install-dev`.)
- **Gate extension — the `makefile` `check` recipe is the source of truth: it passes explicit path args (`mypy src`, `bandit -q -r src`, `vulture src .vulture_allowlist.py`) that *override* the pyproject `files`/`paths` config, so editing pyproject alone does nothing.** Edit the recipe to `mypy src serving/app`, `bandit -q -r src serving/app`, `vulture src serving/app .vulture_allowlist.py`; set `CODE_PATHS += serving` so isort/black/flake8 also cover `serving` (incl. `serving/tests`). Keep `pyproject.toml` `[tool.mypy] files = ["src", "serving/app"]` and `[tool.vulture] paths` in sync for config-driven runs, and add `prometheus_client` to the mypy `ignore_missing_imports` override if it ships no stubs. `pytest.ini` `testpaths` += `serving/tests`. `makefile` adds an `install-serve` target.
- `serving/tests/test_schemas.py` (new): `PredictRequest` parses an ISO `match_date` and rejects a malformed one (422-style `ValidationError`); `OutcomeProbabilities` round-trips and rejects out-of-range.

**Acceptance criteria.**
- `make check` now lints/type-checks `serving/app` and stays green; `make test` collects + passes `serving/tests`.
- `fastapi` / `uvicorn` / `prometheus_client` import under the dev env; schemas validate as specified.

**Gate.** `make check` → PASS (serving in scope); `make test` → all green.

---

## Task 2 — `serving/app/inference.py`: load the artifact + matches, reconstruct features, predict

**Scope (files to touch).**
- `serving/app/inference.py` (new): an `Inference` class that on construction loads `models/v1.joblib` (`joblib.load` from `settings.models_dir`), reads `v1.metadata.json` (for `feature_columns` / selected model — exposed for `/env` later), and loads the master matches table once (`matchodds.data.matches.load()`). Method `predict(request: PredictRequest) -> OutcomeProbabilities`: validate `league` + `home`/`away` against `settings.leagues` and `matchodds.data.teams.canonical_names(league)` (raise a module-level `UnknownFixtureError` on miss); build a one-row `fixtures` DataFrame `{league, date, home, away}`; call `pipeline.features(self._matches, fixtures, date_cutoff=request.match_date)`; `self._model.predict_proba(row) -> (1, 3)`; map `[H, D, A]` → `OutcomeProbabilities`. A cached accessor (`get_inference()`) constructs it once.
- `.vulture_allowlist.py`: add `inference`/`UnknownFixtureError`/`get_inference` if no `src` consumer yet (removed as the predict view lands in Task 4).
- `serving/tests/test_inference.py` (new): a fixture monkeypatches `config.settings.data_dir` → `tests/fixtures/sample` (Decision 7) and constructs `Inference()` against the committed `matches.parquet`; a valid EPL fixture (e.g. `Arsenal` vs `Chelsea`, with a late-2023/24 `match_date` so pre-match history exists in the sample) → probabilities in `[0, 1]` summing to 1.0; unknown team and unknown league each raise `UnknownFixtureError`; deterministic across two calls.

**Acceptance criteria.**
- Predict returns a valid `[home_win, draw, away_win]` summing to 1.0 on the sample, using the **committed** `v1.joblib`; unknown team/league raises; offline + deterministic.
- No train/serve skew: features come from `pipeline.features`, not a re-implementation.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 3 — `api/main/` + app factory `serving/app/main.py` + entrypoint + `make serve`

**Scope (files to touch).**
- `serving/app/api/main/main.py` (`MainManager`) + `views.py` (`MainManagerViews`): `GET /health`, `GET /env`, `GET /metrics` — mirroring quake-feed's `MainManager` exactly (router, `run()` wiring routes with summaries/tags/operation_ids). `/metrics` returns `prometheus_client.generate_latest()`; `/env` reads `settings.environment` + `__metadata__`.
- `serving/app/metrics.py` (new, small): a `prometheus_client` `Counter` (e.g. `predictions_total`) on the default registry, imported by the predict view in Task 4.
- `serving/app/main.py` (new): an app-factory class (e.g. `MatchOddsService`) that builds `FastAPI(title/description/version)` from `__metadata__`, constructs `MainManager`, and `.run()`s it onto the app; a `create_app()` helper for tests.
- `serving/app/__main__.py` (new): `uvicorn.run` reading `settings.serve_host` / `serve_port`. `makefile`: `serve` → `python -m serving.app` (uvicorn, `--reload`).
- `serving/tests/test_main_api.py` (new): `TestClient(create_app())` — `/health` 200 + `HealthResponse` shape; `/env` reflects `settings.environment` + version; `/metrics` returns the Prometheus text content-type.

**Acceptance criteria.**
- The app boots with only the main router; `/health`, `/env`, `/metrics` respond as specified; `make serve` launches uvicorn locally.
- `serving/app/main.py` mirrors the quake-feed app-factory + Manager/Views structure.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 4 — `api/predict/`: `POST /predict` (Manager + Views) wired to inference

**Scope (files to touch).**
- `serving/app/api/predict/main.py` (`PredictManager`, `APIRouter(prefix="/predict")`) + `views.py` (`PredictViews`): the CLAUDE.md-specified manager; `POST ""` → `views.predict(request: PredictRequest) -> OutcomeProbabilities` (sync `def`, threadpool) calling `get_inference().predict(request)` and incrementing `predictions_total`.
- Map `UnknownFixtureError` → **HTTP 422** with a helpful message (a FastAPI exception handler registered in `main.py`, or caught in the view).
- `serving/app/main.py`: construct + mount `PredictManager`; register the exception handler.
- `.vulture_allowlist.py`: remove the Task-2 inference entries now consumed by the view.
- `serving/tests/test_predict_api.py` (new): `TestClient` (applying the Decision 7 sample-data monkeypatch + `get_inference` cache reset) — valid EPL fixture → 200, body sums to 1.0, order `home_win/draw/away_win`; unknown team → 422 + message; unknown league → 422; malformed `match_date` → 422 (Pydantic); the OpenAPI schema exposes `predict_match`.

**Acceptance criteria.**
- `POST /predict {home, away, league, match_date}` returns calibrated `{home_win, draw, away_win}` summing to 1.0 for a known fixture; unknown fixture → 422; Swagger `/docs` documents the contract.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 5 — `Dockerfile` + `docker-compose.yml` + `.dockerignore` + `make up` / `make down`

**Scope (files to touch).**
- `serving/Dockerfile` (new): `python:3.14-slim`; install `requirements-serving.txt`; `COPY` `src/`, `serving/`, `models/v1.*`, `pyproject.toml`; non-root user; `CMD` launches uvicorn (`python -m serving.app`). Layer the requirements copy first for cache reuse.
- `docker-compose.yml` (root, new): one `serving` service building the image, `ports: "${MATCHODDS_SERVE_PORT}:8000"`, `env_file`/env for `MATCHODDS_*`, and a read-only volume `./data:/app/data:ro` (the matches table — Decision 2).
- `.dockerignore` (new): exclude `data/`, `.venv`, `notebooks/`, `tests/`, `serving/tests/`, `.git`, caches, `htmlcov/`.
- `makefile`: `up` → `docker compose up -d --build`; `down` → `docker compose down`. Keep `demo` a stub (Epic 06).
- `README.md` / `.env.template`: note `make data` (matches table) is a prerequisite for `make up`; document the port env.

**Acceptance criteria.**
- `make up` (manual, local) builds the image and serves `/predict` returning calibrated probabilities for a known fixture against the mounted matches table; `make down` stops it. (Docker is **not** run in unit tests — manual verification, like `make train` in Epic 04.)
- The image is data-free; the matches table arrives via the mounted volume.

**Gate.** `make check` → PASS; `make test` → all green (Docker build/run verified manually and noted in the post-task summary).

---

## Definition of done (Epic 05)

`make serve` runs the FastAPI app locally and `make up` serves it in Docker; `POST /predict {home, away, league, match_date}` reconstructs the feature row with the *same* `matchodds.features` code that trained the model and returns calibrated `{home_win, draw, away_win}` summing to 1.0; unknown teams/leagues return a helpful 422; `/health`, `/env`, `/metrics`, and Swagger `/docs` all work; the serving code passes the (now serving-aware) `make check` + `make test` gate. On close-out: mark Epic 05 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-05-serving.md`.
