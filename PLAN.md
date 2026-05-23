# PLAN — Epic 01: Repo Scaffold, Tooling & CI Skeleton

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-01-scaffold`

**Goal.** Stand up the empty-but-rigorous skeleton every later epic plugs into: the directory layout, the strict static-analysis gate, the Makefile entrypoints, the dependency split, and a green CI pipeline with something real to lint and test.

## Progress

| Task | Description                                          | Status      | Commit |
|------|------------------------------------------------------|-------------|--------|
| 1    | Package skeleton, metadata & config module           | Not started | —      |
| 2    | Tooling config, dependency manifests & repo hygiene  | Not started | —      |
| 3    | Makefile + test scaffolding (first green check+test) | Not started | —      |
| 4    | Complete the directory tree (placeholders)           | Not started | —      |
| 5    | README skeleton (per-role "what this demonstrates")  | Not started | —      |
| 6    | GitHub Actions CI                                    | Not started | —      |

**Green-state rule.** Tasks 1–2 are bootstrap (no Makefile yet — verified by running tools/imports directly). **From Task 3 onward, every commit must keep `make check` + `make test` green.**

**Out of scope (deferred to the epics that own them):** real data/feature/model/serving/demo code, actual notebooks, `docs/` deep-dives, the `joblib`-deserialization `bandit` config (Epic 05, when the first `joblib.load` lands), and the Pydantic mypy plugin (Epic 05, when schemas appear).

---

## Task 1 — Package skeleton, metadata & config module

**Scope (files created):**
- `src/matchodds/__init__.py` — re-exports `__version__` from `__metadata__`.
- `src/matchodds/config.py` — typed module-level constants: `RANDOM_SEED: int`, `DATA_DIR` / `RAW_DIR` / `PROCESSED_DIR` / `MODELS_DIR` (`pathlib.Path`), `ROLLING_WINDOW_N: int`, `LEAGUES: tuple[str, ...]` (default: Greek Superleague + EPL, La Liga, Serie A, Bundesliga, Ligue 1).
- `src/matchodds/data/__init__.py`, `src/matchodds/features/__init__.py`, `src/matchodds/modeling/__init__.py` — empty package markers (filled by Epics 02–04).
- `__metadata__.py` — `__version__`, `__author__` (Spyro Kantareli), `__license__` ("MIT"), repo URL.

**Details.** Pure, type-hinted Python; no third-party imports. `config.py` paths are derived from the repo root via `pathlib`, not hard-coded absolutes. `LEAGUES` is a sensible default that Epic 02 may refine.

**Acceptance criteria.**
- `PYTHONPATH=src python -c "import matchodds, matchodds.config; print(matchodds.__version__)"` succeeds.
- Every config constant is present with the documented type.

**Verification (no Makefile yet).** The import line above.

---

## Task 2 — Tooling config, dependency manifests & repo hygiene

**Scope (files created):**
- `pyproject.toml` — `[build-system]` (setuptools, src layout: `package-dir = {"" = "src"}`, `packages = ["matchodds"]` + subpackages); `[tool.black]`, `[tool.isort]` (black profile), `[tool.mypy]` (strict, `ignore_missing_imports` for `xgboost`/`statsmodels`/`streamlit`/`lightgbm`), `[tool.bandit]`, `[tool.vulture]` — all at **120 cols**.
- `setup.cfg` — `[flake8]` (max-line-length 120, black-compatible ignores).
- `pytest.ini` — `testpaths = tests`, strict markers.
- `requirements.txt` — runtime core: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `pyarrow`.
- `requirements-dev.txt` — `-r requirements.txt` + `jupyter`, `papermill`, `nbqa`, `black`, `isort`, `flake8`, `mypy`, `bandit`, `vulture`, `pandas-stubs`, `matplotlib`, `statsmodels`, `streamlit` (LightGBM optional, commented).
- `requirements-test.txt` — `-r requirements.txt` + `pytest`, `pytest-cov`, `httpx`.
- `.gitignore` — `data/` contents (with `!data/raw/.gitkeep`, `!data/processed/.gitkeep` exceptions), `models/*` except `models/v1.*`, `__pycache__`, `.ipynb_checkpoints`, `.env`, `.pytest_cache`, `.mypy_cache`, coverage artifacts, `dist/`/`build/`/`*.egg-info`.
- `.env.template` — reference env (placeholder; no secrets needed locally yet).
- `.vulture_allowlist.py` — checked-in allowlist for the not-yet-consumed `config.py` constants (project-wide config, **not** a line-level silence).

**Acceptance criteria.** Run directly against the Task-1 package and pass clean:
`black --check src`, `isort --check-only src`, `flake8 src`, `mypy src`, `bandit -r src`, `vulture src .vulture_allowlist.py`.

**Verification.** The six commands above (Makefile wraps them in Task 3).

**Notes.** No `joblib.load` exists yet, so `bandit` is clean now; the deserialization-skip config is deferred to Epic 05. Pin major versions in the requirements files; record exact pins when committing.

---

## Task 3 — Makefile + test scaffolding (first green check + test)

**Scope (files created):**
- `makefile` — real targets: `install-env`, `install`, `install-dev`, `install-test`, `format` (isort+black, incl. nbqa), `check` (isort, black, flake8, mypy, bandit, vulture, nbqa), `test` (pytest), `test-report` (pytest-cov HTML). Stub targets that print "not implemented until Epic NN" and exit 0: `data`, `features`, `train`, `repro`, `serve`, `up`, `down`, `demo`, `nb-run`. `nb-lint` runs nbqa only if `notebooks/*.ipynb` exist, else no-ops cleanly.
- `tests/conftest.py` — ensures the installed `matchodds` package is importable under pytest.
- `tests/unit/test_metadata.py` — trivial but real: `__version__` is a non-empty str; `RANDOM_SEED` is an int; `LEAGUES` is non-empty.
- `tests/fixtures/.gitkeep`.

**Acceptance criteria (the epic's exit-criteria triple, achieved here).**
- `make install-dev` installs the editable package + dev deps.
- `make check` passes (all seven tools).
- `make test` passes (≥ 1 test, exit 0 — not pytest's "no tests collected" exit 5).
- `make format` is idempotent.
- Stub targets exit 0 with a clear message; `make nb-lint` no-ops with no notebooks.

**Verification.** `make install-dev && make check && make test`.

---

## Task 4 — Complete the directory tree (placeholders)

**Scope (files created):** `.gitkeep` (or minimal) markers so the full [`CLAUDE.md`](CLAUDE.md) target layout exists and is tracked:
`notebooks/.gitkeep`, `serving/.gitkeep`, `demo/.gitkeep`, `models/.gitkeep`, `docs/.gitkeep`, `docs/history/.gitkeep`, `data/raw/.gitkeep`, `data/processed/.gitkeep`.

**Acceptance criteria.**
- Directory tree matches the CLAUDE.md target (dirs present; code/notebooks arrive with their epics).
- `data/` contents stay ignored while the two `data/` subdirs remain tracked via their `.gitkeep` exceptions.
- `make check` + `make test` still green.

**Verification.** `git status` shows no stray ignored files tracked; `make check && make test`.

---

## Task 5 — README skeleton (per-role "what this demonstrates")

**Scope (files created):** `README.md` — title + one-line pitch; **"What this demonstrates per role"** section (ML Engineer primary, Data Engineer bonus); a quickstart stub referencing the `make` targets; a status line pointing at `MASTER_PLAN.md` for epic progress; license note; links to `MASTER_PLAN.md` and `CLAUDE.md`.

**Acceptance criteria.** Renders as valid Markdown; per-role section present; no broken relative links. Full polish (eval tables, demo GIF) is Epic 08 — this is the skeleton only.

**Verification.** `make check` + `make test` still green (README is not linted, but the commit must not regress them).

---

## Task 6 — GitHub Actions CI

**Scope (files created):** `.github/workflows/ci.yml` — triggers on push and pull_request; Python 3.13; steps: checkout → setup-python (pip cache) → `make install-dev` → `make check` → `make test` → `make nb-lint`.

**Acceptance criteria.**
- Valid workflow YAML mirroring the local `make` gate.
- Green on the scaffold from a fresh clone (`nb-lint` no-ops with no notebooks; `nb-run` is **not** wired into CI until Epic 02 adds a runnable notebook).

**Verification.** `make install-dev && make check && make test && make nb-lint` locally reproduces the CI steps.

---

## Definition of done (Epic 01)

`make install-dev && make check && make test` pass locally and in CI on a fresh clone; the directory layout matches `CLAUDE.md`; README carries the per-role section. On close-out: mark Epic 01 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-01-scaffold.md`.
