# PLAN — Epic 02: Data Acquisition & Cleaning

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-02-data`

**Goal.** Turn public multi-source CSVs into one clean, canonical **master matches table** — reproducibly, with no committed working data. This is the small ETL that carries the Data-Engineer signal and feeds Epic 03's feature pipeline.

## Progress

| Task | Description                                                    | Status         | Commit |
|------|----------------------------------------------------------------|----------------|--------|
| 1    | Data deps, `config` → pydantic `Settings`, parquet engine      | ✅ Done        | —      |
| 2    | `data.schema` — Pydantic `Match` model + validation            | ⬜ Not started | —      |
| 3    | `data.sources` — download clients + version manifest           | ⬜ Not started | —      |
| 4    | `data.teams` — canonical team-name normalization               | ⬜ Not started | —      |
| 5    | `data.matches` — build master table + `make data`              | ⬜ Not started | —      |
| 6    | `01_data.ipynb` + wire `make nb-run` into CI                   | ⬜ Not started | —      |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** The toolchain exists from Epic 01, so **every commit must keep `make check` + `make test` green** (no bootstrap phase this epic).

## Decisions baked in (flag at review to change)

1. **CI/tests are offline & deterministic.** Unit tests and the CI notebook run use **tiny committed fixtures** under `tests/fixtures/` (a few rows per source); **real downloads happen only via `make data` locally**. CI never depends on `football-data.co.uk` being up. (These fixtures are small test artifacts, not the working dataset — they live under `tests/`, not the gitignored `data/`.)
2. **Parquet engine on Python 3.14 — RESOLVED (Task 1):** `pyarrow` 24.0.0 ships a cp314 wheel (Epic 01's failure was the old `<21` cap excluding it). Pinned `pyarrow>=24,<25`; **`matches.parquet` stays** — no gzip-CSV fallback needed.
3. **`config.py` graduates to a pydantic-settings `Settings`** with a `MATCHODDS_` env prefix — this is what lets CI point the notebook at sample fixtures via env overrides.
4. **Scope default:** the 6 leagues already in `config` + the **last 10 completed seasons** each (`settings.seasons_back`, configurable).
5. **football-data.co.uk has two layouts** the source client must handle: the big-5 per-season files (`E0`, `D1`, `SP1`, `I1`, `F1`, …) and the combined "new/extra leagues" file that carries the **Greek Super League** (different columns / fewer odds).

**Out of scope (their own epics):** feature engineering (Epic 03), any modelling (Epic 04), the serving schemas (Epic 05). No features are computed here — this epic stops at the clean master matches table.

---

## Task 1 — Data deps, `config` → pydantic `Settings`, parquet engine — ✅ Done

**Outcome.**
- `requirements.txt`: added `pyarrow>=24,<25`, `pydantic>=2.9,<3`, `pydantic-settings>=2.5,<3`.
- `src/matchodds/config.py`: now a typed `Settings(BaseSettings)` (env prefix `MATCHODDS_`) — `random_seed`, `rolling_window_n`, `leagues`, `seasons_back`, source base URLs, `repo_root` / `data_dir` / `models_dir`, plus derived `raw_dir` / `processed_dir` properties; module-level `settings = Settings()`.
- `pyproject.toml`: `pydantic.mypy` plugin enabled.
- `tests/unit/test_metadata.py` + `.vulture_allowlist.py` moved to the `settings` API; added an env-override test.

**Decisions / deviations (recorded).**
- **Parquet stays — `pyarrow>=24`.** 24.0.0 ships a cp314 wheel; Epic 01's failure was the old `<21` cap. No gzip-CSV fallback; `matches.parquet` remains the output format.
- **Path model:** `data_dir` is the single env-overridable root; `raw_dir`/`processed_dir` are derived read-only properties (override `MATCHODDS_DATA_DIR` to redirect both — covers the CI need, keeps non-optional `Path` typing).
- **Dropped `env_file`** from the settings config to avoid a `python-dotenv` dependency — env-var overrides only.

**Verification.** `make install-dev` → OK (pyarrow 24.0.0, parquet round-trip works); `make check` → PASS (pydantic.mypy active, vulture clean); `make test` → **4 passed** (incl. env override). ✅

---

## Task 2 — `data.schema`: Pydantic `Match` model + validation

**Scope.**
- `src/matchodds/data/schema.py`: a Pydantic v2 `Match` model — `home` / `away` (canonical, str), `league`, `date` (date), `ft_home_goals` / `ft_away_goals` (int ≥ 0), `result` (`Literal["H","D","A"]`), optional closing odds, `source` provenance. Validators: goals non-negative; `result` consistent with the goals; date sane.
- `tests/unit/test_schema.py`: valid row passes; negative goals rejected; result/goals mismatch rejected; odds optional.

**Acceptance criteria.** Model validates/rejects exactly per the rules above; `make check` + `make test` green.

**Verification.** `pytest -k schema -v`.

---

## Task 3 — `data.sources`: download clients + version manifest

**Scope.**
- `src/matchodds/data/sources.py`: clients fetching football-data.co.uk (**both layouts**, Decision 5) and openfootball CSVs into `settings.raw_dir`. Pinned base URLs; cache (skip re-download unless `force=True`); polite timeout + retry. Write `raw/_manifest.json` recording source URLs, filenames, fetch timestamp, and per-file checksums (the data-versioning record).
- `tests/unit/test_sources.py`: URL/path construction per league+season; parsing of tiny **committed sample CSVs** for each layout (offline); manifest written correctly. The real network fetch is verified by a `tests/manual/` smoke I run during implementation (not part of `make test`).

**Acceptance criteria.** Offline tests pass; a real fetch of one league-season verified locally during implementation; `make check` + `make test` green.

**Verification.** `pytest -k sources -v`; one real `sources` fetch locally.

---

## Task 4 — `data.teams`: canonical team-name normalization

**Scope.**
- `src/matchodds/data/teams.py`: a canonical team-name map across sources for the in-scope leagues + `normalize(raw, league) -> canonical`. Built from the **real raw spellings** observed via Task 3. Unknown names raise a clear error — never silently pass through.
- `tests/fixtures/raw_team_names.txt`: every distinct raw home/away name seen across sources (generated during implementation), so the resolution test runs **offline**.
- `tests/unit/test_teams.py`: assert every name in the fixture resolves; an unmapped name raises.

**Acceptance criteria.** 100% of in-scope raw names resolve; unmapped names fail loudly; `make check` + `make test` green.

**Verification.** `pytest -k teams -v`.

---

## Task 5 — `data.matches`: build master table + `make data`

**Scope.**
- `src/matchodds/data/matches.py`: orchestrate raw → per-row `Match` validation (Task 2) → team normalization (Task 4) → assemble DataFrame → **dedupe** (one row per match) → sort by date → write `settings.processed_dir/matches.{parquet|csv.gz}` + a `matches.meta.json` sidecar (row counts, leagues/seasons covered, source-manifest reference, build timestamp).
- Wire `make data` to run `sources` (download) → `matches` (build).
- `tests/unit/test_matches.py`: on the committed sample raw set, the build yields the expected normalized/validated/deduped rows; FTHG/FTAG present; dedup works.

**Acceptance criteria.** `make data` produces the processed table **from scratch** (real run verified locally); the offline fixture build test passes; `make check` + `make test` green.

**Verification.** `make data` end-to-end locally (network); `pytest -k matches -v`.

---

## Task 6 — `01_data.ipynb` + wire `make nb-run` into CI

**Scope.**
- `notebooks/01_data.ipynb`: a thin narrative importing `matchodds.data` — load the processed table, show coverage per league/season, and sanity-check (row counts, date ranges, missing-odds rate, goal distributions). Cells import from the package; no heavy logic inline (keeps `nbqa` trivial).
- `tests/fixtures/sample/`: a tiny committed sample dataset; CI runs the notebook against it **offline** by overriding `MATCHODDS_*` paths (pydantic-settings env), so `make nb-run` is deterministic and network-free.
- Wire `make nb-run` into `.github/workflows/ci.yml` (the step Epic 01 deferred); `make nb-lint` now actively lints the notebook (must pass `nbqa` isort/black/flake8).

**Acceptance criteria.** `make nb-lint` passes on the notebook; `make nb-run` executes it headless against the sample (offline); CI runs both steps; `make check` + `make test` green.

**Verification.** `make nb-lint && make nb-run`; re-parse `ci.yml`.

---

## Definition of done (Epic 02)

`make data` produces the master matches table from scratch (network, locally); team-name normalization tests pass; `01_data.ipynb` lints and runs headless in CI against the committed sample. On close-out: mark Epic 02 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-02-data.md`.
