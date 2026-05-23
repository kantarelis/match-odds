# PLAN — Epic 01: Repo Scaffold, Tooling & CI Skeleton

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-01-scaffold`

**Goal.** Stand up the empty-but-rigorous skeleton every later epic plugs into: the directory layout, the strict static-analysis gate, the Makefile entrypoints, the dependency split, and a green CI pipeline with something real to lint and test.

## Progress

| Task | Description                                          | Status         | Commit |
|------|------------------------------------------------------|----------------|--------|
| 1    | Package skeleton, metadata & config module           | ✅ Done        | —      |
| 2    | Tooling config, dependency manifests & repo hygiene  | ✅ Done        | —      |
| 3    | Makefile + test scaffolding (first green check+test) | ✅ Done        | —      |
| 4    | Complete the directory tree (placeholders)           | ✅ Done        | —      |
| 5    | README skeleton (per-role "what this demonstrates")  | ⬜ Not started | —      |
| 6    | GitHub Actions CI                                    | ⬜ Not started | —      |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Tasks 1–2 are bootstrap (no Makefile yet — verified by running tools/imports directly). **From Task 3 onward, every commit must keep `make check` + `make test` green.**

**Out of scope (deferred to the epics that own them):** real data/feature/model/serving/demo code, actual notebooks, `docs/` deep-dives, the `joblib`-deserialization `bandit` config (Epic 05, when the first `joblib.load` lands), and the Pydantic mypy plugin (Epic 05, when schemas appear).

---

## Task 1 — Package skeleton, metadata & config module — ✅ Done

**Outcome.** Created the importable `src/matchodds` package (6 files):
- `src/matchodds/__metadata__.py` — `__version__` (`0.1.0`), `__author__` (Spyro Kantareli), `__license__` (`MIT`), `__repository__`.
- `src/matchodds/__init__.py` — re-exports `__version__` / `__author__` / `__license__` via `__all__` (explicit re-export, mypy-strict clean).
- `src/matchodds/config.py` — typed constants: `RANDOM_SEED: int`, repo-root-derived `REPO_ROOT` / `DATA_DIR` / `RAW_DIR` / `PROCESSED_DIR` / `MODELS_DIR` (`pathlib.Path`), `ROLLING_WINDOW_N: int`, `LEAGUES: tuple[str, ...]` (6: Greek Super League + EPL, La Liga, Serie A, Bundesliga, Ligue 1).
- `src/matchodds/data/__init__.py`, `features/__init__.py`, `modeling/__init__.py` — package markers (filled by Epics 02–04).

**Deviation (structural, recorded).** `__metadata__.py` was placed **inside the package** (`src/matchodds/__metadata__.py`) rather than at the repo root as originally drawn. Reason: under the src layout, a root-level module isn't importable from the installed package, so `matchodds.__version__` (this task's acceptance criterion) couldn't resolve it. The placement keeps a single source of truth; CLAUDE.md's directory tree was corrected to match.

**Verification (no Makefile yet).** `PYTHONPATH=src python3 -c "import matchodds, matchodds.config; ..."` — imports succeed, `__version__ == "0.1.0"`, every config constant present with its documented type, `REPO_ROOT` resolves to `…/match-odds`. ✅

> Acceptance criterion (kept for the record): `PYTHONPATH=src python -c "import matchodds, matchodds.config; print(matchodds.__version__)"` succeeds, and every config constant is present with the documented type.

---

## Task 2 — Tooling config, dependency manifests & repo hygiene — ✅ Done

**Outcome.** Created 9 declarative/config files:
- `pyproject.toml` — setuptools src-layout build (`package-dir = {"" = "src"}`, `packages.find` over `src`), **dynamic version** read from `matchodds.__metadata__.__version__`; 120-col `black` / `isort` (black profile) / `mypy` (strict) / `bandit` / `vulture`; a forward-looking `ignore_missing_imports` override for `xgboost`/`statsmodels`/`streamlit`/`lightgbm`.
- `setup.cfg` — `[flake8]` at 120 cols, ignoring the black-incompatible `E203`/`W503`.
- `pytest.ini` — `testpaths = tests`, `--strict-markers --strict-config`.
- `requirements.txt` / `requirements-dev.txt` / `requirements-test.txt` — pinned core / dev (gate linters + notebooks + modelling/viz) / test deps.
- `.gitignore` — data + model-artifact rules, `.venv/`, tooling caches, notebook checkpoints.
- `.env.template` — placeholder (no local secrets yet).
- `.vulture_allowlist.py` — whitelists the not-yet-consumed `config.py` constants + `__repository__`.

**Decisions / deviations (recorded).**
- **Python standardised on 3.14** (dev env is 3.14.5): `requires-python = ">=3.14"`, black `py314`, mypy `python_version = "3.14"`; CLAUDE.md and Task 6 updated to match.
- **Version caps corrected** to admit what actually resolves: `black <27` (26.5.1), `mypy <3` (2.1.0).
- **mypy extras** beyond the original scope: `mypy_path = "src"` + `explicit_package_bases = true` so `mypy src` resolves the top-level module as `matchodds` with no editable install. `warn_unused_configs` deliberately **not** enabled (it flags the forward-looking ML-lib overrides until Epic 04 imports them).
- **`readme` omitted** from `pyproject.toml` `[project]` until `README.md` lands (Task 5), so `pip install -e .` (Task 3) doesn't fail on a missing file.
- `packages.find` used instead of an explicit `packages = ["matchodds"]` list (equivalent, standard src-layout idiom).

**Verification (no Makefile yet).** All six gate commands run clean against `src` in `.venv` (black 26.5.1, isort, flake8 7.3.0, mypy 2.1.0 strict, bandit 1.9.4, vulture 2.16): `black --check src`, `isort --check-only src`, `flake8 src`, `mypy src`, `bandit -r src`, `vulture src .vulture_allowlist.py` — **all PASS**. ✅

> Note: no `joblib.load` exists yet, so `bandit` is clean; the deserialization-skip config is deferred to Epic 05.

---

## Task 3 — Makefile + test scaffolding (first green check + test) — ✅ Done

**Outcome.** The epic's exit-criteria triple is green on Python 3.14.5.
- `makefile` — `install-env` / `install` / `install-dev` / `install-test`, `format`, `check` (isort, black, flake8, mypy, bandit, vulture, nbqa), `test`, `test-report`, plus exit-0 stub targets for `data`/`features`/`train`/`repro`/`serve`/`up`/`down`/`demo`/`nb-run`. `nb-lint` no-ops cleanly when no notebooks exist. `check` runs mypy + bandit on `src` only; isort/black/flake8 also cover `tests/` and `.vulture_allowlist.py`. `install-dev` installs dev **and** test deps so the documented verify chain works.
- `tests/conftest.py` — prepends `src/` to `sys.path` so tests import `matchodds` even without an editable install.
- `tests/unit/test_metadata.py` — 3 smoke tests (version string, `RANDOM_SEED` int, `LEAGUES` non-empty tuple).
- `tests/fixtures/.gitkeep`.

**Decisions / deviations (recorded).**
- **Python 3.14 wheel gaps** (touches Task 2's requirements files): `pyarrow` deferred to **Epic 02** and `streamlit` to **Epic 06** — neither has a cp314 wheel yet (pyarrow fails to build from source; streamlit depends on it). Comments in the requirements files mark the deferrals.
- **`isort` cap `<7`→`<9`** to admit the current 8.0.1 (install-dev had been downgrading it to 6.1.0); consistent with the earlier black/mypy cap fixes.
- The editable build (dynamic version from `matchodds.__metadata__`) and the full dep stack (pandas 2.3.3, numpy 2.4.6, scikit-learn 1.8.0, xgboost 2.1.4, statsmodels 0.14.6, matplotlib 3.10.9, jupyter, papermill, nbqa, pytest) all install cleanly on 3.14.5.

**Verification.** `make install-dev` → rc 0; `make check` → rc 0 (gate clean, nb-lint skipped); `make test` → **3 passed**; `make format` → idempotent (zero diff); `make data` stub → rc 0. ✅

---

## Task 4 — Complete the directory tree (placeholders) — ✅ Done

**Outcome.** Added 8 `.gitkeep` markers so the full CLAUDE.md target layout is tracked from the start: `notebooks/`, `serving/`, `demo/`, `models/`, `docs/`, `docs/history/`, `data/raw/`, `data/processed/`.

**Verification.**
- All 8 placeholders are seen by git (untracked, will commit); the target tree matches CLAUDE.md.
- Gitignore exceptions confirmed via `git check-ignore`: `.gitkeep` and `models/v1.*` are **tracked**, while `data/*.csv`, `data/*.parquet`, and stray `models/*.joblib` are **ignored**.
- `make check` → PASS, `make test` → 3 passed. ✅

No deviations.

---

## Task 5 — README skeleton (per-role "what this demonstrates")

**Scope (files created):** `README.md` — title + one-line pitch; **"What this demonstrates per role"** section (ML Engineer primary, Data Engineer bonus); a quickstart stub referencing the `make` targets; a status line pointing at `MASTER_PLAN.md` for epic progress; license note; links to `MASTER_PLAN.md` and `CLAUDE.md`.

**Acceptance criteria.** Renders as valid Markdown; per-role section present; no broken relative links. Full polish (eval tables, demo GIF) is Epic 08 — this is the skeleton only.

**Verification.** `make check` + `make test` still green (README is not linted, but the commit must not regress them).

---

## Task 6 — GitHub Actions CI

**Scope (files created):** `.github/workflows/ci.yml` — triggers on push and pull_request; Python 3.14; steps: checkout → setup-python (pip cache) → `make install-dev` → `make check` → `make test` → `make nb-lint`.

**Acceptance criteria.**
- Valid workflow YAML mirroring the local `make` gate.
- Green on the scaffold from a fresh clone (`nb-lint` no-ops with no notebooks; `nb-run` is **not** wired into CI until Epic 02 adds a runnable notebook).

**Verification.** `make install-dev && make check && make test && make nb-lint` locally reproduces the CI steps.

---

## Definition of done (Epic 01)

`make install-dev && make check && make test` pass locally and in CI on a fresh clone; the directory layout matches `CLAUDE.md`; README carries the per-role section. On close-out: mark Epic 01 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-01-scaffold.md`.
