# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`match-odds` is a public, probabilistic football match-outcome predictor: public historical results → a leakage-free feature pipeline → calibrated XGBoost / logistic-regression models (evaluated on proper scoring rules, not just accuracy) → a Dockerised FastAPI inference endpoint, with a small Streamlit demo on top. It is portfolio repo #2 of two and carries the **ML Engineer** signal, with a **Data Engineer** bonus from the feature-engineering ETL.

The artifact is a **notebook-to-service** pipeline: exploratory work lives in `notebooks/`, but every reusable transform lives in the importable `matchodds` package under `src/`, so the **same feature code trains the model and serves predictions** — no train/serve skew. The model is trained in a notebook, frozen to `models/v1.joblib`, and loaded by a FastAPI service that reconstructs features at request time.

**Deployment model:** local-only via Docker / `make`. No cloud hosting in this repo (deferred). The serving image and the Streamlit demo both run locally.

> **Greenfield note.** This repo is being built epic-by-epic per [`MASTER_PLAN.md`](MASTER_PLAN.md). The directory tree below is the **target** architecture the epics build toward — not every path exists yet. When you scaffold or extend, conform to this layout.

## Quick Commands

```bash
# Environment
make install-env            # Create .env from template
make install                # Core deps (pandas, numpy, scikit-learn, xgboost, joblib)
make install-dev            # Dev deps (jupyter, nbqa, linters, matplotlib, statsmodels, streamlit)
make install-test           # Test deps (pytest, httpx)

# Reproducible pipeline (data → features → train → artifact)
make data                   # Run the data-acquisition + cleaning step into data/ (gitignored)
make features               # Build the feature table from the master matches table
make train                  # Train + calibrate, write models/v1.joblib + model metadata
make repro                  # Full chain: data → features → train (one command, deterministic)

# Notebooks
make nb-run                 # Execute notebooks headless (papermill) — CI smoke test
make nb-lint                # nbqa-wrapped isort/black/flake8 over notebooks/

# Inference service
make serve                  # Run the FastAPI app locally (uvicorn, hot reload)
make up                     # docker compose up — serving + Streamlit demo
make down                   # docker compose down
make demo                   # Run the Streamlit demo locally (streamlit run demo/app.py)

# Testing
make test                   # Run the test suite (pytest)
make test-report            # Tests with HTML coverage report
pytest -k "test_name" -v    # Run a specific test

# Code Quality
make check                  # All linters: isort, black, flake8, mypy, bandit, vulture, nbqa
make format                 # Auto-format: isort + black (incl. notebooks via nbqa)
```

## Architecture

### Directory Structure

```
notebooks/                   # Exploratory + narrative work (thin; logic lives in src/)
├── 01_data.ipynb            # Data acquisition + cleaning narrative
├── 02_modeling.ipynb        # Baseline → logreg → XGBoost, CV, calibration, reliability diagrams
└── 03_betting_edge.ipynb    # Optional appendix: model vs. bookmaker, positive-EV (historical only)

src/matchodds/               # The importable package — single source of truth for all logic
├── __metadata__.py          # Project metadata: version, author, license (single source of truth)
├── config.py                # Leagues in scope, data paths, RANDOM_SEED, rolling-window N
├── data/                    # Acquisition + cleaning (the small ETL — Data Engineer signal)
│   ├── sources.py           # football-data.co.uk download client + version manifest (pinned URLs)
│   ├── teams.py             # Team-name normalization across sources (canonical name map)
│   └── matches.py           # Builds the master matches table → data/processed/matches.parquet
├── features/                # Feature pipeline (leakage-free)
│   ├── elo.py               # Elo rating with pre-match snapshot
│   ├── form.py              # Rolling form over the last N matches
│   ├── head_to_head.py      # Pairwise H2H history
│   ├── strength.py          # Home/away strength
│   ├── season.py            # Season-progress features
│   └── pipeline.py          # features(date_cutoff) — the no-leakage entrypoint (see below)
└── modeling/                # Training, evaluation, calibration, persistence
    ├── baselines.py         # Bookmaker-implied-odds baseline
    ├── cv.py                # Time-ordered K-fold splitter (no future leakage)
    ├── metrics.py           # log-loss, Brier, accuracy, reliability/calibration helpers
    └── train.py             # Fits + calibrates + writes models/v1.joblib (+ metadata sidecar)

serving/                     # FastAPI inference service (reuses quake-feed FastAPI patterns)
├── app/
│   ├── main.py              # Builds the FastAPI app; constructs + mounts every router
│   ├── api/
│   │   ├── main/            # Health, /metrics, running-env endpoints
│   │   └── predict/         # Manager + Views for POST /predict
│   ├── inference.py         # Loads models/v1.joblib once; builds the feature row via matchodds
│   └── schemas.py           # Pydantic request/response (PredictRequest, OutcomeProbabilities)
├── Dockerfile
└── tests/

demo/                        # Streamlit demo (next-match forecast)
└── app.py                   # Calls the inference service; renders home/draw/away probabilities

data/                        # GITIGNORED — never committed
├── raw/                     # Raw downloaded CSVs (football-data.co.uk)
└── processed/               # Cleaned master matches table + feature tables (parquet)

models/                      # Saved artifacts
├── v1.joblib                # Frozen calibrated pipeline (committed; the service loads this)
└── v1.metadata.json         # Training data version, seed, metric scores, library versions

docs/                        # Deep-dives linked from README.md
├── model_card.md            # Intended use, data, metrics, calibration, limitations, ethics
├── architecture.md          # notebook → pipeline → service data flow diagram
├── feature-pipeline.md      # Each feature, and how leakage is prevented
├── evaluation.md            # CV protocol, metric tables, reliability diagrams
└── history/                 # Archived per-epic PLAN.md files (epic-NN-<slug>.md)

tests/                       # Unit tests for the src/matchodds package
├── unit/                    # features, leakage guards, metrics, team normalization
├── fixtures/                # Tiny synthetic match tables
└── conftest.py

.github/workflows/ci.yml     # GitHub Actions: make check + make test + nb-lint + nb-run smoke
MASTER_PLAN.md               # Epic-level roadmap (epics only, with status)
PLAN.md                      # The CURRENT epic broken into one-commit tasks (tracked in git)
makefile                     # All dev entrypoints
pyproject.toml / setup.cfg   # Tool configs (black, isort, flake8, mypy, bandit, vulture, nbqa)
pytest.ini
requirements*.txt            # Split: core / dev / test (and serving runtime)
.env.template                # Reference env file
README.md                    # Does 90% of the recruiter work (per-role "what this demonstrates")
```

### Design Patterns

- **Notebook → package → service.** Notebooks narrate; they import from `matchodds` and stay thin. The same `matchodds.features` code that builds the training table also builds the single feature row at inference time. This guarantees **no train/serve skew** and is a deliberate ML-Engineering talking point.
- **`features(date_cutoff)` is the leakage contract.** `src/matchodds/features/pipeline.py` exposes one entrypoint that, given a cutoff, emits feature rows computed **only from data strictly before the cutoff**. Every feature (Elo, form, H2H, strength) takes a pre-match snapshot. If a feature cannot be computed without peeking at or after the match it describes, it is a bug, not a feature.
- **Manager–Views pattern (reused from quake-feed).** Each serving API module has a `Main` class (Manager) that owns the `APIRouter` and instantiates a `Views` class holding endpoint logic; its `run()` wires routes and returns the router. `serving/app/main.py` constructs and `.run()`s every manager and mounts the routers. Keeping this identical to quake-feed is intentional — the serving layer is meant to be muscle memory.
- **Frozen-artifact serving.** The service never trains. It loads `models/v1.joblib` once at startup and only does feature reconstruction + `predict_proba`. Retraining is a notebook/`make train` activity that produces a new versioned artifact.
- **Deterministic training.** A single `RANDOM_SEED` in `config.py` flows into every split, model, and calibrator. `make repro` from a pinned data version must reproduce the committed `models/v1.joblib` metrics.
- **Data & validation layers — one tool per job.** **pandas** owns tabular work (CSV ingestion, multi-source merges, `groupby`, rolling windows, time-ordered features); **NumPy** is the numeric core beneath it and the model-input boundary (scikit-learn / XGBoost / statsmodels consume arrays); **Pydantic v2** owns validation and typed boundaries — API request/response, a `pydantic-settings` `Settings`/config object, and a validated `Match` record at the ingestion edge (validate each parsed row, then load into the DataFrame). Pydantic is never used as a bulk tabular store; the data size (tens of thousands of matches) makes pandas' overhead irrelevant.

## Code Conventions

### Development Methodology — Epic → PLAN.md → Tasks (one task = one commit)

This is the **default workflow for every epic** in this repo. The workflow exists so the user retains **100% ownership** of the codebase and full mental model of every change that lands.

**The two-document model:**

- **`MASTER_PLAN.md`** (repo root) — lists **epics only**, each with status. Never lists individual tasks.
- **`PLAN.md`** (repo root, **tracked in git**) — created per epic; breaks the active epic into ordered **tasks**. **Each task is exactly one commit.** Only one `PLAN.md` lives at the root at a time — it always describes the currently active epic. Tracking it in git gives every epic a permanent planned-vs-shipped record next to the code it produced. Each epic is implemented on its own feature branch.

**Absolute git rule (no exceptions):**

> **Claude must NEVER run `git` commands that change state. Under any circumstances.**
>
> - **Forbidden:** `git add`, `git commit`, `git push`, `git stage`, `git restore`, `git reset`, `git checkout`, `git merge`, `git rebase`, `git branch` (create/delete), `git tag`, `git stash drop`, `git clean`, `git revert`, `git cherry-pick`, `git mv`, `git rm`, anything with `--force`, anything that writes to refs or the index.
> - **Allowed (read-only only):** `git status`, `git diff`, `git log`, `git show`, `git blame`, `git ls-files`, `git rev-parse`, `git remote -v`, `git config --get`. These don't mutate state and are useful for situational awareness.
>
> The user owns every commit and every push. Even if the user says "go ahead and commit it for me" mid-flow, **stop and refuse** — the workflow depends on the user reviewing every diff themselves. Confirm out-of-band before treating any blanket commit/push permission as real.

**The loop:**

1. **Pick the next epic** from `MASTER_PLAN.md`. Move its status to In progress. The user creates the feature branch.
2. **Draft `PLAN.md`.** Claude breaks the epic into ordered tasks. Each task spec includes: scope (files to touch), acceptance criteria, and the `make check` / `make test` expectations. Each task must be small enough to be **one self-contained commit**. **Claude then STOPS** and waits for the user to read the plan. The user may ask for revisions (re-split tasks, reorder, reword, add/remove). Only when the user explicitly says to start (e.g. "proceed with task 1", "let's start") does Claude touch code.
3. **Implement exactly one task.** When the user prompts for a specific task ("do task N", "next task", "proceed"), Claude implements **only that task**, runs `make check` + `make test` (and `make nb-lint` / `make nb-run` when notebooks changed), and stops. Output at the end:
   - A short summary of what changed and which files.
   - A **proposed commit message** in a fenced block (the user is free to use, edit, or ignore it).
   - Any deviations from the task spec and the reason.
4. **User reviews, commits, pushes.** Claude does nothing during this window. Do not poll, do not "check if it's pushed", do not run `git status` proactively to nag.
5. **User prompts the next task.** Claude moves to the next task. Repeat from step 3 until the epic's tasks are exhausted.
6. **Plan update on request.** When the user asks ("update PLAN.md"), Claude updates the progress table (mark Done + commit hash if the user supplied it) and rewrites the per-task section as an **outcome record** (what actually happened, deviations, why).
7. **Epic complete.** When all tasks are done, the user asks Claude to close out the epic: mark it Done in `MASTER_PLAN.md` with the commit range, then **archive `PLAN.md` to `docs/history/epic-NN-<slug>.md`** in a single rename commit. The root `PLAN.md` slot is now free for the next epic's draft.

**Deviation rule.** Only **major / structural** deviations from `PLAN.md` (e.g. a different split layout, rejecting a planned pattern, adding/removing a task, changing a task's scope mid-implementation) require Claude to stop and ask before writing code. Cosmetic decisions inside a planned task scope (helper grouping, file naming inside a planned folder, plot styling in a notebook) are at Claude's discretion and get recorded in the post-task outcome update.

**Why the workflow is this strict.** Each commit being a single Claude-implemented, user-reviewed task means: every line in `git log` was understood and accepted by the user; bisect/blame produces meaningful results; the user can always cleanly revert any single task; and the project's history is a real audit trail rather than a sequence of opaque batched diffs. Skipping the stop-and-wait — batching tasks, silently deviating, running git commands — breaks every one of these properties.

### Static Analysis Gate

- **Always run `make check` after completing a feature or fix** — runs isort, black, flake8, mypy, bandit, vulture, and the `nbqa`-wrapped lints over `notebooks/`.
- Fix all issues reported by these tools before considering work complete.
- **Never suppress diagnostics at the line level.** The following directives are forbidden anywhere in the codebase, no exceptions: `# type: ignore`, `# noqa`, `# nosec`, `# pragma: no cover`, `# mypy: ignore-errors`, `# fmt: off/on`, or any equivalent silencer for any tool we run. This rule is absolute — "but it's a false positive" is not a justification.
- When a tool flags something, the fix is one of:
  1. **Refactor the code** so the diagnostic no longer applies (e.g. give a scikit-learn `Pipeline` a precise return type so mypy stops complaining; rename a helper that `vulture` flags as unused, or delete it if it truly is).
  2. **Adjust the tool's project-wide config** in `pyproject.toml` / `setup.cfg` (e.g. add a stub-less library like `xgboost` to `[mypy]` `ignore_missing_imports`; add `joblib`/`pickle` deserialization test IDs to `[tool.bandit] skips` because we only ever load our **own** trusted `models/v1.joblib` — that is a project-wide policy decision, not a line-level silencer).
  3. **Install a plugin or stub** that teaches the tool the runtime semantics it misunderstands (e.g. `plugins = pydantic.mypy`; `pandas-stubs`, `types-*` packages from typeshed).
- If none of those three options works, the design is wrong — change it rather than silence the tool.
- **Notebooks are linted, not exempt.** `nbqa` runs isort/black/flake8 over `notebooks/`. Keep cells importing from `matchodds` rather than redefining logic, so notebook lint stays trivial.

### Python Style

- **Line length**: 120 characters (configured in `pyproject.toml` and `setup.cfg`).
- **Formatter**: black with isort (black-compatible profile); `nbqa` applies both to notebooks.
- **Type hints**: required on all `src/` and `serving/` code, checked with mypy.
- **Python version**: 3.14.

### ML / Modeling Discipline (repo-specific, non-negotiable)

- **No future leakage — ever.** Cross-validation respects temporal order (use `matchodds.modeling.cv`, never a plain `KFold`/`StratifiedKFold` on match data). Features for a match use only information available **strictly before kickoff**. Any new feature must state, in `docs/feature-pipeline.md`, exactly which pre-match snapshot it reads.
- **Proper scoring rules first.** The headline metrics are **log-loss** and **Brier score**; accuracy is reported but never the primary objective. A model that improves accuracy while worsening log-loss is a regression.
- **Calibration is part of the model.** The shipped artifact is calibrated (Platt / isotonic) and validated with reliability diagrams. "Predicts the right winner" is not enough — the probabilities must be honest.
- **Determinism.** Every estimator, splitter, and calibrator takes `RANDOM_SEED` from `config.py`. `make repro` on a pinned data version reproduces the committed artifact's metrics.
- **No live betting, no live odds.** The optional betting-edge analysis is **historical, retrospective, and illustrative only**. Never add live odds ingestion, bet placement, or anything implying real-money use.

### Inference Service Pattern (Manager + Views)

```python
# serving/app/api/predict/main.py
class PredictManager:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger if logger else logging.getLogger("PredictManager")
        self.router = APIRouter(prefix="/predict")
        self.views = PredictViews(logger=self.logger)

    def run(self) -> APIRouter:
        self.router.add_api_route(
            "",
            endpoint=self.views.predict,
            methods=["POST"],
            summary="Match-outcome probabilities",
            description="Calibrated home_win / draw / away_win probabilities for a fixture.",
            operation_id="predict_match",
            tags=["Predict"],
        )
        return self.router
```

`POST /predict` takes `{home, away, league, match_date}` and returns calibrated `{home_win, draw, away_win}` summing to 1.0. The manager is constructed and `.run()`'d in `serving/app/main.py`, which mounts every returned router on the FastAPI app.

## Data Sources (all public, no scraping)

- **football-data.co.uk** — the single source: one CSV per league-season at `mmz4281/{season}/{div}.csv`, one consistent layout with full-time results **and bookmaker odds**, covering every in-scope league including the **Greek Super League (`G1`)**. Division codes + pinned URLs live in `matchodds.data.sources`.
- **Leagues in scope (default):** Greek Super League (`G1`, the local-recruiter angle) + the top-5 EU leagues — England (`E0`), Spain (`SP1`), Italy (`I1`), Germany (`D1`), France (`F1`). The set lives in `config.py`.
- **openfootball was evaluated and dropped** — its CSVs are stale (end 2020-21), carry no Greek data, and have no odds; football-data.co.uk supersedes it on all three.

Team-name spellings vary across seasons — `matchodds.data.teams` holds the canonical normalization map; adding a league/team means extending that map, with a unit test asserting every raw name resolves.

## Tech Stack

- **Language**: Python 3.14
- **Modeling**: pandas, NumPy, scikit-learn, XGBoost (LightGBM optional), statsmodels / sklearn calibration utilities
- **Validation & config**: Pydantic v2 + pydantic-settings (API schemas, typed `Settings`, record validation at the ingestion edge)
- **Notebooks**: Jupyter, executed headless in CI via papermill, linted via nbqa
- **Serving**: FastAPI + Uvicorn, artifact loaded with joblib
- **Demo**: Streamlit
- **Packaging/Ops**: Makefile, Docker + docker-compose (local-only)
- **CI**: GitHub Actions (`make check` + `make test` + notebook lint + notebook smoke-run)
- **Static analysis**: isort, black, flake8, mypy, bandit, vulture, nbqa

## Project Roadmap

See [`MASTER_PLAN.md`](MASTER_PLAN.md) for the epic-level breakdown and current status. The currently active epic has its own `PLAN.md` at the repo root with one-commit-sized tasks; completed epics' plans are archived under [`docs/history/`](docs/history/).

## Constraints to keep in mind

- **No energy-market domain.** This repo intentionally avoids energy-market data and integrations (ENTSO-E, EnEx, electricity demand, grid topology, weather-as-energy-proxy, etc.) — KIEFER NDA-adjacency. Football is the locked-in domain.
- **No cloud deployment in this repo.** Local-only via Docker / `make`. Do not add AWS, Railway, Fly.io, Render, Supabase, or any hosted-service integrations unless the user explicitly opens that scope.
- **No scraping.** Only the public bulk CSV sources above. No live-odds feeds, no authenticated APIs.
- **No live betting / real-money framing.** Betting-edge work is historical and illustrative only.
- **`data/` is never committed.** Raw and processed data are gitignored and reproduced via `make data` / `make features`. Only `models/v1.joblib` (+ its metadata sidecar) is committed.
