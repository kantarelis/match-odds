# PLAN — Epic 06: Streamlit Demo App

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-06-demo`

**Goal.** A small, recruiter-friendly **"odds this weekend"** UI over the Epic 05 inference service: pick a league, home + away team, and a date; hit predict; see calibrated `home / draw / away` probabilities as a chart plus a one-line plain-English read. The demo is a **thin HTTP client** of the service — it never imports `serving.app` and never loads the model; it `POST`s to `/predict` and renders the response, degrading gracefully when the service is down. `make demo` runs it locally; `make up` brings up **serving + demo together** in Docker.

## Progress

| Task | Description                                                                 | Status         | Commit |
|------|-----------------------------------------------------------------------------|----------------|--------|
| 1    | Streamlit dep + `service_url` config + gate/test extension + demo service-client & summary (logic, no UI) | ⬜ Not started | — |
| 2    | `demo/app.py` Streamlit UI (inputs → predict → chart + plain-English read + graceful errors) + `make demo` | ⬜ Not started | — |
| 3    | `demo/Dockerfile` + docker-compose `demo` service + `make up` (serving + demo) + docs | ⬜ Not started | — |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Task 1 **extends the gate to cover `demo/`** (mypy + bandit + flake8/isort/black + vulture over the demo *source*, excluding `demo/tests/` just as `serving/tests/` is excluded) and adds `demo/tests/` to `pytest` — from then on the demo code is held to the same bar as `src/` and `serving/`. The UI lands in Task 2 and the container in Task 3; each task's tests pass on their own.

**Consolidation note (flag at review).** The split keeps one named concern per commit (foundation/logic → UI → container). If you prefer fewer, Tasks 1+2 (logic + the thin UI that wires it) can merge into one "build the demo" commit; the gate-extension half of Task 1 can instead be its own commit before the logic lands.

## Decisions baked in (flag at review to change)

1. **Streamlit is feasible on Python 3.14 — verified.** `streamlit 1.57.0` and its whole dependency tree install from **cp314 wheels** (wheels-only resolution succeeds; the compiled deps `httptools`/`websockets` ship cp314 wheels; `pyarrow 24` — the original Epic 01 blocker — is already a core dep). Task 1 adds a new `requirements-demo.txt` and **removes the now-stale "streamlit deferred / no cp314 pyarrow wheel" note** from `requirements-dev.txt`.
2. **The demo is a thin HTTP client; it imports `matchodds` for option lists but never `serving.app`.** It `POST`s `{home, away, league, match_date}` to `f"{settings.service_url}/predict"` and renders `{home_win, draw, away_win}` — the service stays a black box (the demo never loads the model or the matches table). It imports `matchodds.config` (`leagues`, `service_url`) and `matchodds.data.teams.canonical_names(league)` to populate the league/team dropdowns from the **canonical registry** (committed in `src/`, so the demo is data- and model-free). *(Rejected alternative: add a `/leagues` discovery endpoint to the service — keeps the frozen Epic-05 serving layer untouched, and the demo legitimately shares the domain package. Also rejected: importing `serving.app.schemas` for the contract — keeps the demo a genuinely standalone client; the tiny contract is re-expressed locally as a typed `Probabilities`.)*
3. **One typed config.** Add `service_url: str = "http://127.0.0.1:8000"` to `matchodds.config.Settings` (env `MATCHODDS_SERVICE_URL`), per the repo's single-config principle — rather than a separate demo settings object. `docker-compose` overrides it to `http://serving:8000` for the demo service.
4. **Logic is importable + tested; the Streamlit script is thin.** The non-UI logic — the service client, response parsing/validation, the plain-English summary, and the dropdown option helpers — lives in importable modules (`demo/service.py`, `demo/summary.py`). `demo/app.py` is glue that calls them and draws widgets. This mirrors the repo's "notebooks/UI stay thin; logic is importable" rule and makes the demo unit-testable without a running browser.
5. **HTTP client uses `httpx` (already a core dep), with dependency injection for tests.** `PredictClient` accepts an optional `httpx.Client` (default: one bound to `settings.service_url` with a short timeout). Tests inject an `httpx.MockTransport` client — no real network, no extra mocking dependency. A connection failure / timeout / non-200 raises a typed `ServiceError` carrying a human message (the service's `detail` on a 422).
6. **Graceful degradation is a feature, not an afterthought.** Service unreachable → the UI shows a friendly "service unavailable — is it running? (`make up` / `make serve`)" notice, never a stack trace. A 422 (e.g. an out-of-scope fixture, or `home == away`) → the service's message is surfaced. The dropdowns only offer registry-valid teams per league, so unknown-team 422s shouldn't arise through the UI; `home == away` is blocked client-side with a hint.
7. **`make up` brings up serving + demo together.** Task 3 adds a `demo` service to `docker-compose.yml`; the existing `up` recipe (`docker compose up -d --build`) then starts both. The demo container talks to the serving container at `http://serving:8000` over the compose network and publishes Streamlit on `${MATCHODDS_DEMO_PORT:-8501}`. `make demo` runs Streamlit locally against `http://127.0.0.1:8000`.
8. **Strict mypy + Streamlit.** Streamlit's public API is largely untyped, which strict mypy can flag (untyped calls). Because the typed logic lives in `service.py`/`summary.py`, `demo/app.py`'s Streamlit surface is small; if the gate flags it, treat `streamlit.*` as a stub-less import via a **project-wide** `[[tool.mypy.overrides]]` (the CLAUDE.md-approved config path) — never a line-level silencer. Resolved in Task 2 when `app.py` first imports streamlit.
9. **Demo image is data-free and model-free.** It carries the `matchodds` package (for the canonical registry + config) + `streamlit` + `httpx` + `demo/` — no `models/`, no `./data` mount. Only the serving container needs those.

**Out of scope (their own epics / frozen):** changes to the serving layer (frozen after Epic 05 — no new endpoints), the model or feature pipeline; the betting-edge notebook (Epic 07); the README per-role polish, model card, architecture/eval docs, and the final reproducibility pass (Epic 08). No live odds, no cloud hosting, no auth, no scraping.

---

## Task 1 — Streamlit dep, `service_url` config, gate/test extension + the demo service-client & summary (logic, no UI)

**Scope (files to touch).**
- `requirements-demo.txt` (new): `-r requirements.txt` + `streamlit` (`httpx` is already core). Wire `-r requirements-demo.txt` into **both** `requirements-dev.txt` (so `make check` can import it) and `requirements-test.txt` (so `make test` — incl. the Task 2 `AppTest` — has streamlit). **Remove the stale streamlit/pyarrow deferral comment** in `requirements-dev.txt`.
- `src/matchodds/config.py`: add `service_url: str = "http://127.0.0.1:8000"` to `Settings` (env `MATCHODDS_SERVICE_URL`).
- `demo/__init__.py`, `demo/tests/__init__.py` — package skeleton so `demo.*` imports cleanly (drop `demo/.gitkeep`).
- `demo/service.py` (new): `Probabilities` (a small typed `home_win/draw/away_win` value, e.g. a frozen dataclass); `ServiceError(Exception)`; `PredictClient` — `__init__(self, client: httpx.Client | None = None)` (default bound to `settings.service_url` with a short timeout); `predict(home, away, league, match_date) -> Probabilities` that POSTs to `/predict`, maps connection/timeout/non-200 to `ServiceError` (using the response `detail` on 422). Option helpers: `available_leagues() -> list[str]` (`settings.leagues`) and `teams_for(league) -> list[str]` (`sorted(teams.canonical_names(league))`).
- `demo/summary.py` (new): `verbal_summary(probs: Probabilities) -> str` — names the favourite with its percentage when one outcome clearly leads, or calls it a close match when the spread is small.
- **Gate extension** (makefile recipe is the source of truth — explicit path args override pyproject config; see the Epic 05 lesson): `CODE_PATHS += demo`; `mypy`/`vulture` over the demo *source* via a `demo/*.py` glob (which excludes `demo/tests/`); `bandit -r … demo` (its `exclude_dirs` already drops any `tests/`); keep `pyproject.toml` `[tool.mypy] files`/`[tool.vulture] paths` in sync (+ a `tests/` `exclude` for config-driven runs); `pytest.ini` `testpaths += demo/tests`; add an `install-demo` target.
- `demo/tests/test_service.py` (new): with an `httpx.MockTransport`, `PredictClient.predict` builds the right request body and parses a 200 → `Probabilities`; a connection error and a 422 (with `detail`) each raise `ServiceError`; `available_leagues()` and `teams_for("English Premier League")` return the expected registry lists.
- `demo/tests/test_summary.py` (new): a clear favourite → the summary names the team + percentage; a near-uniform distribution → the "close match" phrasing.

**Acceptance criteria.**
- `make check` now lints/type-checks the demo source and stays green; `make test` collects + passes `demo/tests`.
- `streamlit` imports under the dev env; `PredictClient` works against a mocked service (offline) and raises `ServiceError` on failure; `verbal_summary` reads sensibly.

**Gate.** `make check` → PASS (demo source in scope); `make test` → all green.

---

## Task 2 — `demo/app.py`: the Streamlit UI + `make demo`

**Scope (files to touch).**
- `demo/app.py` (new): the Streamlit app. A title + one-line intro; a **league** selectbox (`available_leagues()`), **home**/**away** selectboxes (`teams_for(league)`), and a **match date** input; a **Predict** button that calls `PredictClient().predict(...)`. On success: render the three probabilities as a bar chart (e.g. `st.bar_chart`/altair) + `st.metric` trio + the `verbal_summary` line. Guard `home == away` with a hint (no call). Catch `ServiceError` → `st.error(...)` with the friendly/service `detail` message. Keep the script thin — all logic comes from `demo/service.py` + `demo/summary.py`.
- `makefile`: `demo` → `$(PY) -m streamlit run demo/app.py` (local; honour `MATCHODDS_DEMO_PORT` if set).
- `pyproject.toml`: if strict mypy flags the streamlit surface, add `streamlit.*` to the `ignore_missing_imports` override (Decision 8) — project-wide, not line-level.
- `demo/tests/test_app.py` (new): a headless `streamlit.testing.v1.AppTest` smoke — initial render populates the league/team selectboxes without error; with `PredictClient.predict` monkeypatched to return fixed `Probabilities`, exercising the predict path renders the metrics + summary; with it raising `ServiceError`, an `st.error` is shown. (Scale back to a render-only smoke if `AppTest` interaction proves brittle — the predict/error logic is already covered by Task 1.)

**Acceptance criteria.**
- `make demo` launches Streamlit locally; choosing a fixture and predicting renders calibrated probabilities + the plain-English read; a down service shows a friendly message, not a traceback.
- The `AppTest` smoke passes headlessly under `make test`.

**Gate.** `make check` → PASS; `make test` → all green (local end-to-end with `make serve` noted in the post-task summary).

---

## Task 3 — `demo/Dockerfile` + docker-compose `demo` service + `make up` (serving + demo) + docs

**Scope (files to touch).**
- `demo/Dockerfile` (new): `python:3.14-slim`; install `requirements-demo.txt`; editable-install `matchodds` (for the registry + config); `COPY demo/`; non-root user; `EXPOSE 8501`; `CMD` runs `streamlit run demo/app.py --server.address=0.0.0.0 --server.port=8501`. Layer the requirements copy first for cache reuse. Image env defaults: `MATCHODDS_SERVICE_URL=http://serving:8000`.
- `docker-compose.yml`: add a `demo` service (build `demo/Dockerfile`, `depends_on: serving`, `environment: MATCHODDS_SERVICE_URL=http://serving:8000`, `ports: "${MATCHODDS_DEMO_PORT:-8501}:8501"`). The existing `up`/`down` recipes now manage both services.
- `.dockerignore`: **remove the `demo/` exclusion** (the demo image must COPY it; the serving image is unaffected since it never COPYs `demo/`); add `demo/tests/` to keep the demo image test-free.
- `makefile`: `up`/`down` unchanged (they already build/run every compose service); `demo` stays the local Streamlit target.
- `README.md` / `.env.template`: document `make demo` (local), `make up` now serving **+** demo, `MATCHODDS_DEMO_PORT`, and `MATCHODDS_SERVICE_URL`.

**Acceptance criteria.**
- `make up` (manual, local) builds + runs **serving + demo**; opening the demo (`localhost:8501`) renders calibrated probabilities for a known fixture end-to-end (demo → serving → model, against the mounted matches table); `make down` stops both. (Docker is **not** run in unit tests — manual verification, as in Epic 05.)
- The demo image is data-free and model-free; it reaches the serving container over the compose network.

**Gate.** `make check` → PASS; `make test` → all green (Docker build/run verified manually and noted in the post-task summary).

---

## Definition of done (Epic 06)

`make demo` runs the Streamlit UI locally and `make up` brings up serving + demo together; selecting a league / home / away / date and predicting renders calibrated `{home_win, draw, away_win}` (chart + plain-English read) end-to-end through the real inference service; a down service degrades gracefully; the demo code passes the (now demo-aware) `make check` + `make test` gate. On close-out: mark Epic 06 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-06-demo.md`.
