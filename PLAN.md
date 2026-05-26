# PLAN — Epic 05: FastAPI Inference Service (Dockerised)

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `implement-fastapi-inference-service`

**Goal.** Serve the frozen `models/v1.joblib` behind `POST /predict`, reusing the **quake-feed Manager/Views** FastAPI pattern, with **no train/serve skew**: the request `{home, away, league, match_date}` is turned into a single feature row by the *same* `matchodds.features.pipeline.features()` that built the training table, then scored by the loaded artifact. The service is read-only inference — it never trains. Ships with `/health`, `/env`, `/metrics`, a Dockerfile + docker-compose, and `make serve` / `make up` / `make down`, all local-only.

## Progress

| Task | Description                                                                 | Status         | Commit |
|------|-----------------------------------------------------------------------------|----------------|--------|
| 1    | Serving scaffold + deps + gate extension + `schemas.py` (the Pydantic contract) | ✅ Done | `65cf406` |
| 2    | `inference.py` — load artifact + matches table; reconstruct features → `predict_proba` | ✅ Done | `9e4c75c` |
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

## Task 1 — Serving scaffold, deps, gate extension + `schemas.py` ✅ `65cf406`

**Outcome.** Stood up the `serving/` package skeleton, the Pydantic contract, and extended the static-analysis gate + pytest to hold serving code to the same bar as `src/`. `make check` → PASS (mypy now covers 31 files incl. `serving/app`); `make test` → 116 passed (5 new in `serving/tests/test_schemas.py`).

**What shipped.**
- **Package skeleton:** `serving/{__init__, app/__init__, app/api/__init__, app/api/main/__init__, app/api/predict/__init__, tests/__init__}.py` (each a one-line docstring); `serving/.gitkeep` deleted.
- **`serving/app/schemas.py`:** Pydantic v2 `PredictRequest{home, away, league, match_date: date}` (`extra="forbid"`), `OutcomeProbabilities{home_win, draw, away_win}` (each `[0, 1]`), `HealthResponse{status, service, version}`, `EnvResponse{environment, application_name, version}`.
- **`src/matchodds/config.py`:** added `serve_host="127.0.0.1"`, `serve_port=8000`, `environment="local"` (env-overridable, `MATCHODDS_` prefix).
- **Deps:** `requirements-serving.txt` (new) = `-r requirements.txt` + `fastapi`, plain `uvicorn`, `prometheus-client`; wired into both `requirements-dev.txt` and `requirements-test.txt`. Verified the file resolves on Python 3.14.5: `fastapi 0.136.3`, `uvicorn 0.48.0`, `prometheus-client 0.25.0` — plain `uvicorn` installed without compiled-wheel trouble, confirming the cp314 decision.
- **Gate extension:** `makefile` `check` recipe now runs `mypy src serving/app`, `bandit -q -r src serving/app`, `vulture src serving/app .vulture_allowlist.py`; `CODE_PATHS += serving`; new `install-serve` target + help line. `pyproject.toml` mypy `files=["src","serving/app"]`, vulture `paths` += `serving/app`. `pytest.ini` `testpaths = tests serving/tests`. `.vulture_allowlist.py` covers the not-yet-consumed schema classes/fields + the three config fields.
- **Tests:** `serving/tests/test_schemas.py` — ISO `match_date` parses, malformed rejected, extra fields forbidden, `OutcomeProbabilities` round-trips, out-of-range rejected.

**Deviations (all minor, within scope).**
- **httpx already present.** A plan-review claim that `httpx` was absent was wrong — it is already a core dep in `requirements.txt` (the data-acquisition client). So `requirements-test.txt` only gained `-r requirements-serving.txt`; no redundant `httpx`. Plan text corrected.
- **`extra="forbid"` on `PredictRequest`** (mirrors the `Match` ingestion model) + an extra-fields rejection test, beyond the literal test list.
- **`prometheus_client` not added** to mypy's `ignore_missing_imports`: it ships `py.typed` and isn't imported until Task 3, so the "if it ships no stubs" condition didn't apply.
- `install-serve` installs `-e .` + `requirements-serving.txt` (mirrors `install` / `install-dev`).

---

## Task 2 — `serving/app/inference.py`: load the artifact + matches, reconstruct features, predict

**Scope (files to touch).**
**Outcome.** `serving/app/inference.py` loads the frozen artifact + matches table once and reconstructs the single feature row through the same `pipeline.features` that built the training table — verified end-to-end on the committed `v1.joblib` + sample EPL season. `make check` → PASS (mypy 32 files); `make test` → 120 passed (4 new). A home/away swap (Arsenal-home `0.31/0.34/0.35` vs Chelsea-home `0.16/0.35/0.49`) confirms the prediction is genuinely feature-driven, not constant.

**What shipped.**
- **`serving/app/inference.py`:** `Inference` loads `models/v1.joblib` (`joblib.load`), reads `feature_columns` from `v1.metadata.json`, loads the master matches table (`matches.load()`), and runs a **startup no-skew guard** (`_guard_no_skew`) asserting the live `base.feature_columns()` equals the trained artifact's columns. `predict(request) -> OutcomeProbabilities`: validates `league ∈ settings.leagues` and `home`/`away ∈ teams.canonical_names(league)` (raising `UnknownFixtureError`), builds the one-row `fixtures` frame, calls `pipeline.features(self._matches, fixtures, date_cutoff=request.match_date)`, maps `predict_proba(...)[0]` `[H, D, A]` → `OutcomeProbabilities`. `get_inference()` builds the shared instance once (`functools.lru_cache`).
- **`serving/tests/test_inference.py`:** monkeypatches `config.settings.data_dir` → the sample, leaves `models_dir` untouched (exercises the shipped artifact). Asserts normalized probabilities (`[0, 1]`, sum ≈ 1.0), determinism across two calls, and `UnknownFixtureError` for an unknown team and an unknown league. `match_date = 2024-05-01` (after the sample's last match, 2023-12-09) so the fixture has full pre-match history.
- **`.vulture_allowlist.py`:** allowlisted `Inference.predict` + `get_inference` (consumed by the Task 4 view); `UnknownFixtureError` needs none (in-module consumer).

**Deviations (minor, flagged).**
- **`selected_model` not read.** Plan said read `v1.metadata.json` for "feature_columns / selected model — exposed for /env later." Only `feature_columns` was read, given an **immediate** consumer (the skew guard). `selected_model` was left out because nothing consumes it — the Task-1 `EnvResponse` is `{environment, application_name, version}` with no model field, so storing it would be dead code needing a permanent allowlist entry. **Open question for Task 3:** if `/env` (or a future `/model-info`) should surface the loaded model, add the metadata read + schema field there alongside its consumer.
- **`_guard_no_skew` added** beyond the literal spec — a fail-fast startup assertion that serving columns still match the trained artifact (on-theme for the repo's no-skew principle); it is also what gives `feature_columns` its consumer.
- **`UnknownFixtureError(ValueError)`** — plan left the base unspecified; `ValueError` maps cleanly to the HTTP 422 wired in Task 4.

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
