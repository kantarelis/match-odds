# MASTER_PLAN — match-odds

Epic-level roadmap for the football match-outcome predictor. This document lists **epics only** — never individual tasks. When an epic becomes active, it is broken into one-commit tasks in a root `PLAN.md` on its own feature branch, then archived to `docs/history/epic-NN-<slug>.md` on completion.

See [`CLAUDE.md`](CLAUDE.md) → *Development Methodology* for the full Epic → PLAN.md → one-task-one-commit workflow and the absolute git rule. The short version: pick an epic here → draft `PLAN.md` → user reviews → implement one task per commit → user commits/pushes → repeat → archive `PLAN.md` and mark the epic Done.

## What this repo demonstrates (recruiter map)

- **ML Engineer (primary):** leakage-free temporal CV, model comparison spanning discriminative (logistic regression, XGBoost) and the domain-canonical generative approach (Dixon-Coles bivariate-Poisson), calibration + reliability diagrams + proper scoring rules, a reproducible saved pipeline, and a serving artifact.
- **Data Engineer (bonus):** the feature-engineering pipeline is a small, clean ETL over multi-source public data.

## Epic Status

| #  | Epic                                         | Plan steps | Status         | Branch              | Commit range |
|----|----------------------------------------------|------------|----------------|---------------------|--------------|
| 01 | Repo Scaffold, Tooling & CI Skeleton         | 2.1        | ✅ Done        | `implement-scaffold-tooling-and-ci-skeleton` | `e84115d..f793d2f` |
| 02 | Data Acquisition & Cleaning                  | 2.2        | 🔄 In progress | `epic-02-data`      | —            |
| 03 | Feature Pipeline (leakage-free)              | 2.3        | ⬜ Not started | `epic-03-features`  | —            |
| 04 | Modeling, Evaluation & Calibration           | 2.4, 2.5   | ⬜ Not started | `epic-04-modeling`  | —            |
| 05 | FastAPI Inference Service (Dockerised)       | 2.6        | ⬜ Not started | `epic-05-serving`   | —            |
| 06 | Streamlit Demo App                           | —          | ⬜ Not started | `epic-06-demo`      | —            |
| 07 | Betting-Edge Analysis (optional appendix)    | 2.7        | ⬜ Not started | `epic-07-betting`   | —            |
| 08 | Documentation, Model Card & Reproducibility  | 2.8, 2.9   | ⬜ Not started | `epic-08-docs`      | —            |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started. Branch names are suggestions; the user creates branches.

---

## Epic 01 — Repo Scaffold, Tooling & CI Skeleton

**Goal.** Stand up the empty-but-rigorous skeleton every later epic plugs into: directory layout, the strict static-analysis gate, the Makefile entrypoints, dependency split, and a green CI pipeline that has something trivial to lint and test.

**Scope.**
- Directory tree per `CLAUDE.md`: `notebooks/`, `src/matchodds/` (package with `config.py`), `serving/`, `demo/`, `data/` (gitignored), `models/`, `docs/`, `tests/`.
- `pyproject.toml` / `setup.cfg` with the full toolset configured at 120-col: isort, black, flake8, mypy (strict), bandit, vulture, nbqa.
- `requirements.txt` / `requirements-dev.txt` / `requirements-test.txt` split.
- `makefile` with the targets documented in `CLAUDE.md` (install*, check, format, test, plus stubs for data/features/train/serve/up/down/demo/nb-* that later epics flesh out).
- `.gitignore` (data/, models/* except v1.*, notebook checkpoints, `.env`, caches), `.env.template`, `__metadata__.py`, `pytest.ini`.
- `.github/workflows/ci.yml`: `make check` + `make test` + `make nb-lint` on push/PR.
- `README.md` skeleton with the per-role "What this demonstrates" section.
- One trivial unit test so `make test` and CI are genuinely green.

**Depends on.** Nothing.

**Exit criteria.** `make install-dev && make check && make test` pass locally and in CI on a fresh clone; the layout matches `CLAUDE.md`.

---

## Epic 02 — Data Acquisition & Cleaning

**Goal.** Turn public football-data.co.uk CSVs into one clean, canonical master matches table — reproducibly, with no committed data.

**Scope.**
- `matchodds.data.sources`: pinned-URL download client for football-data.co.uk — one CSV per league-season (`E0`/`SP1`/`I1`/`D1`/`F1`/`G1`) into `data/raw/`, with caching + a version manifest.
- `matchodds.data.teams`: canonical team-name normalization map for the football-data.co.uk spellings, with a test asserting every raw name in scope resolves.
- `matchodds.data.matches`: build + de-duplicate the master matches table → `data/processed/matches.parquet` (one row per match, normalized teams, league, date, full-time result, **full-time goals (FTHG/FTAG) — required by the Dixon-Coles goals model in Epic 04**, bookmaker closing odds where available).
- `matchodds.data.schema`: a Pydantic v2 `Match` model validating each parsed row at the ingestion edge before loading into the DataFrame. This epic adds `pydantic` + `pydantic-settings` to `requirements.txt` and graduates `config.py` to a typed `Settings` (pydantic-settings).
- `make data` runs the whole acquisition + cleaning chain.
- `notebooks/01_data.ipynb`: a thin narrative that imports the above, shows coverage per league/season, and sanity-checks the cleaned table.
- Data versioning: record the source snapshot date / file versions used.

**Depends on.** Epic 01.

**Exit criteria.** `make data` produces `data/processed/matches.parquet` from scratch; name-normalization tests pass; `01_data.ipynb` runs headless in CI.

---

## Epic 03 — Feature Pipeline (leakage-free)

**Goal.** Implement the `features(date_cutoff)` contract — the heart of the ML-Engineering signal — producing rows computed strictly from pre-kickoff information.

**Scope.**
- `matchodds.features`: `elo.py` (pre-match Elo snapshot), `form.py` (rolling last-N form), `head_to_head.py`, `strength.py` (home/away), `season.py` (season-progress).
- `matchodds.features.pipeline.features(date_cutoff)`: the single no-leakage entrypoint assembling all features for matches, using only data strictly before each match.
- Leakage guards: unit tests that fabricate a future result and assert it cannot influence a prior match's features; tests pinning expected Elo/form values on a tiny synthetic fixture.
- `make features` materializes the feature table for the in-scope data.

**Depends on.** Epic 02.

**Exit criteria.** `make features` emits a feature table; leakage tests pass; every feature is documented (which pre-match snapshot it reads) in `docs/feature-pipeline.md`.

---

## Epic 04 — Modeling, Evaluation & Calibration

**Goal.** Train, compare, calibrate, and freeze the shipped model — judged on proper scoring rules across discriminative (LR / XGBoost) and the domain-canonical generative (Dixon-Coles) approaches — and persist a reproducible artifact. (Covers plan steps 2.4 **and** 2.5.)

**Scope.**
- `matchodds.modeling.cv`: time-ordered K-fold splitter (no future leakage).
- `matchodds.modeling.baselines`: bookmaker-implied-odds baseline.
- `matchodds.modeling.dixon_coles`: domain-canonical bivariate-Poisson goals model — per-team attack/defense strengths + home advantage + the low-score (draw) correction, optionally time-weighted, fit by MLE; derives 1X2 probabilities by summing the score matrix. Naturally calibrated; CPU-only, fits in seconds (no GPU).
- `matchodds.modeling.metrics`: log-loss, Brier, accuracy, reliability/calibration helpers.
- `matchodds.modeling.train`: fit logistic regression + XGBoost + Dixon-Coles, calibrate the discriminative models (Platt / isotonic) and reliability-check the generative one, select on proper scoring, and write `models/v1.joblib` + `models/v1.metadata.json` (data version, seed, scores, library versions).
- `notebooks/02_modeling.ipynb`: baseline → logreg → XGBoost → Dixon-Coles narrative; four-way CV comparison on log-loss + Brier + accuracy; reliability diagrams before/after calibration.
- `make train` and `make repro` (data → features → train) are deterministic from the pinned data version.

**Depends on.** Epic 03.

**Exit criteria.** `make repro` reproduces `models/v1.joblib` with metrics matching the metadata sidecar; the four approaches (baseline → LR → XGBoost → Dixon-Coles) are compared head-to-head on log-loss + Brier under temporal CV, and the winner — whichever it is — is the one frozen into the artifact; reliability diagrams render in the notebook.

---

## Epic 05 — FastAPI Inference Service (Dockerised)

**Goal.** Serve the frozen artifact behind `POST /predict`, reusing the quake-feed Manager/Views FastAPI patterns, with no train/serve skew.

**Scope.**
- `serving/app/main.py` (app factory, mounts routers), `serving/app/api/main/` (health, /metrics, env), `serving/app/api/predict/` (Manager + Views).
- `serving/app/inference.py`: load `models/v1.joblib` once; build the single feature row at request time via `matchodds.features` (same code path as training).
- `serving/app/schemas.py`: `PredictRequest{home, away, league, match_date}` → `OutcomeProbabilities{home_win, draw, away_win}` summing to 1.0; validation + helpful errors for unknown teams/leagues.
- `serving/Dockerfile`, docker-compose service, `make serve` / `make up` / `make down`.
- `serving/tests/`: endpoint tests (valid fixture, unknown team, probability-sum invariant) via httpx.

**Depends on.** Epic 04 (needs `models/v1.joblib`).

**Exit criteria.** `make up` serves `/predict` returning calibrated probabilities for a known fixture; service tests pass; Swagger `/docs` documents the contract.

---

## Epic 06 — Streamlit Demo App

**Goal.** A small, recruiter-friendly "odds this weekend" UI over the inference service.

**Scope.**
- `demo/app.py`: team/league/date inputs → calls `POST /predict` → renders home/draw/away probabilities (bar/gauge) and a one-line plain-English read.
- `make demo` (local) and a docker-compose demo service wired to the serving container.
- Graceful handling when the service is down or inputs are unknown.

**Depends on.** Epic 05.

**Exit criteria.** `make up` brings up serving + demo together; selecting a fixture renders calibrated probabilities end-to-end.

---

## Epic 07 — Betting-Edge Analysis (optional appendix)

**Goal.** Retrospective, historical-only comparison of model probabilities vs. bookmaker closing odds, identifying positive-EV bets on past data. **Optional — schedulable or skippable.**

**Scope.**
- `notebooks/03_betting_edge.ipynb`: implied-probability extraction from closing odds, model-vs-book edge, a backtest of a flat-stake positive-EV strategy on historical matches only, with clear caveats.
- Any reusable math lifted into `matchodds.modeling` (e.g. overround removal) with tests.

**Depends on.** Epic 04 (and odds columns from Epic 02).

**Exit criteria.** Notebook runs headless; results framed explicitly as historical/illustrative; **no** live-odds or bet-placement code anywhere.

---

## Epic 08 — Documentation, Model Card & Reproducibility

**Goal.** Make the repo do the recruiter work in 30 seconds and prove it reproduces end-to-end. (Covers plan steps 2.8 and the 2.9 GitHub-pin closeout.)

**Scope.**
- `README.md`: per-role "what this demonstrates", quickstart, `make repro` reproducibility instructions, eval-results table, demo screenshot/GIF.
- `docs/model_card.md` (intended use, data, metrics, calibration, limitations, ethics), `docs/architecture.md` (notebook → pipeline → service diagram), `docs/evaluation.md` (CV protocol + metric tables + reliability diagrams).
- Final reproducibility pass: fresh clone → `make install-dev` → `make repro` → `make up` documented and verified.
- CV integration follow-ups (a `data/projects.yaml` entry in `~/dev/cv` tagged `science`, `data`; verify with `make cv-all`) noted as an out-of-repo action.
- **Manual closeout (2.9):** the user pins the repo on the GitHub profile — not a code task.

**Depends on.** Epics 01–06 (07 optional).

**Exit criteria.** A newcomer can reproduce the artifact and run the demo from the README alone; model card and eval tables are complete.

---

## Sequencing notes

- Epics are strictly ordered 01 → 08; each depends on the prior data/feature/model artifacts. Epic 07 is optional and can be deferred or skipped.
- The serving layer (Epic 05) deliberately mirrors `quake-feed`'s FastAPI + Docker patterns — build that repo first so this is muscle memory.
- One `PLAN.md` at a time at the repo root; archive to `docs/history/` before drafting the next.
