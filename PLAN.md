# PLAN — Epic 02: Data Acquisition & Cleaning

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-02-data`

**Goal.** Turn public football-data.co.uk CSVs into one clean, canonical **master matches table** — reproducibly, with no committed working data. This is the small ETL that carries the Data-Engineer signal and feeds Epic 03's feature pipeline.

## Progress

| Task | Description                                                    | Status         | Commit |
|------|----------------------------------------------------------------|----------------|--------|
| 1    | Data deps, `config` → pydantic `Settings`, parquet engine      | ✅ Done        | —      |
| 2    | `data.schema` — Pydantic `Match` model + validation            | ✅ Done        | —      |
| 3    | `data.sources` — download client + version manifest            | ✅ Done        | —      |
| 4    | `data.teams` — canonical team-name normalization               | ✅ Done        | —      |
| 5    | `data.matches` — build master table + `make data`              | ✅ Done        | —      |
| 6    | `01_data.ipynb` + wire `make nb-run` into CI                   | ⬜ Not started | —      |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** The toolchain exists from Epic 01, so **every commit must keep `make check` + `make test` green** (no bootstrap phase this epic).

## Decisions baked in (flag at review to change)

1. **CI/tests are offline & deterministic.** Unit tests and the CI notebook run use **tiny committed fixtures** under `tests/fixtures/` (a few rows per source); **real downloads happen only via `make data` locally**. CI never depends on `football-data.co.uk` being up. (These fixtures are small test artifacts, not the working dataset — they live under `tests/`, not the gitignored `data/`.)
2. **Parquet engine on Python 3.14 — RESOLVED (Task 1):** `pyarrow` 24.0.0 ships a cp314 wheel (Epic 01's failure was the old `<21` cap excluding it). Pinned `pyarrow>=24,<25`; **`matches.parquet` stays** — no gzip-CSV fallback needed.
3. **`config.py` graduates to a pydantic-settings `Settings`** with a `MATCHODDS_` env prefix — this is what lets CI point the notebook at sample fixtures via env overrides.
4. **Scope default:** the 6 leagues already in `config` + the **last 10 completed seasons** each (`settings.seasons_back`, configurable).
5. **Single source, one layout (revised after probing).** football-data.co.uk covers all 6 leagues — including the **Greek Super League (`G1`)** — in the *same* `mmz4281/{season}/{div}.csv` format with odds. openfootball was dropped: its CSVs are stale (end 2020-21), carry no Greek data, and have no odds. No dual-layout handling, no second source.

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

## Task 2 — `data.schema`: Pydantic `Match` model + validation — ✅ Done

**Outcome.**
- `src/matchodds/data/schema.py`: `Match` (`frozen=True`, `extra="forbid"`) — non-empty `home`/`away`/`league`, `date`, non-negative `ft_home_goals`/`ft_away_goals`, `result` ∈ {H,D,A}, optional decimal odds (`odds_home`/`draw`/`away`, each > 1), `source` ∈ {football-data, openfootball}. A single `model_validator(mode="after")` enforces: plausible date range (1990–2100), home ≠ away, and result consistent with the score.
- `tests/unit/test_schema.py`: 9 tests — valid, valid+odds, negative goals, result/score mismatch, same team, odds ≤ 1, unknown source, extra field, out-of-range date.

**Decisions / deviations (recorded).**
- **One `model_validator`, no `field_validator`** — vulture flagged the date `field_validator`'s unused `cls` (100%); folding the date check into the `self`-based validator removes it (no line-level suppression).
- **`league` is a non-empty str, not checked against `settings.leagues`** — league-membership enforcement is deferred to the matches builder (Task 5), where `settings` is in scope and names are already canonical (`data.teams`, Task 4).
- **Odds are generic optional fields**; mapping from source-specific columns (B365*, PSC*, …) happens in Tasks 3/5.

**Verification.** `make check` → PASS; `make test` → **13 passed** (4 metadata + 9 schema). ✅

---

## Task 3 — `data.sources`: download client + version manifest — ✅ Done

**Outcome.**
- `src/matchodds/data/sources.py`: `season_code` / `recent_seasons` (July season cutover) / `file_url`; a retrying `_download` that tolerates 404 (unpublished season → skip); `download_all` (caches into `raw_dir`, writes `raw/_manifest.json` with per-file sha256 + bytes); `load_raw`. `DIVISIONS` maps the 6 in-scope leagues incl. Greece (`G1`) to football-data.co.uk div codes.
- `tests/unit/test_sources.py`: 6 offline tests (season code, July cutover, URL, `load_raw` columns, `download_all` writes files+manifest with a mocked fetch, cache reuse) against a committed `tests/fixtures/footballdata_sample.csv`.
- Moved `httpx` to core `requirements.txt`; removed the orphaned `openfootball_base_url`.

**Decisions / deviations (recorded).**
- **Single-source football-data.co.uk** (probing showed openfootball is stale to 2020-21, has no Greece, and no odds). Greece is `G1` in the *same* main layout → **Decision 5's "dual layout" is void**; no second source. CLAUDE.md / MASTER_PLAN data-source sections updated.
- **Manifest is a plain dict → JSON** (loose versioning metadata); pydantic stays reserved for row-level validation (`Match`).
- `download_all` is **404-tolerant** (records only fetched files), so Greece's shorter history won't break a full run.

**Verification.** Real fetch of `E0` 2023-24 → 380 rows, core columns + `B365H` odds present; `make check` → PASS; `make test` → **19 passed** (4 metadata + 9 schema + 6 sources). ✅

---

## Task 4 — `data.teams`: canonical team-name normalization — ✅ Done

**Outcome.**
- `src/matchodds/data/teams.py`: `normalize(raw, league) -> canonical` + `canonical_names(league)`, backed by a generated registry; raises `UnknownTeamError` for unknown team **or** league (fail-loud — new/renamed teams must be added deliberately).
- `src/matchodds/data/canonical_teams.json`: **186 canonical names** across the 6 leagues (incl. 25 Greek), generated from the real downloads.
- `tests/manual/build_team_registry.py`: committed, reproducible generator (download → extract distinct names → write registry + fixture).
- `tests/fixtures/raw_team_names.txt`: offline corpus for the resolution test.
- `tests/unit/test_teams.py`: 5 tests (every league covered, every raw name resolves, unknown team raises, unknown league raises, whitespace stripped).

**Decisions / deviations (recorded).**
- **Aliases start empty** — football-data.co.uk spellings are internally consistent across seasons, so single-source normalization is *canonicalize + validate*, not cross-source reconciliation.
- **Registry is generated data** loaded by `teams.py`; regenerate via the committed `tests/manual/build_team_registry.py`.
- Building the registry warmed `data/raw/` (gitignored cache), so Task 5's `make data` starts warm.

**Verification.** `canonical_teams.json` + `raw_team_names.txt` tracked (the `data/` ignore only anchors to root); `make check` → PASS; `make test` → **24 passed** (4 metadata + 9 schema + 6 sources + 5 teams). ✅

---

## Task 5 — `data.matches`: build master table + `make data` — ✅ Done

**Outcome.**
- `src/matchodds/data/matches.py`: read each raw CSV → per-row `Match` validation (Task 2) + team normalization (Task 4) → dedupe on `(league, date, home, away)` → sort by date → write `processed_dir/matches.parquet` + `matches.meta.json` (rows, per-league counts, date range, source-manifest ref, build timestamp). `main()` wires download → build.
- `make data` → `python -m matchodds.data.matches`.
- `tests/unit/test_matches.py`: 3 offline builder tests (normalize/validate, dedupe, parquet+meta) against the committed sample CSV.

**Decisions / deviations (recorded).**
- **Parquet only** (Decision 2 resolved to pyarrow ≥24) — no `csv.gz` fallback.
- **`format="mixed"`** for dates (football-data mixes `DD/MM/YYYY` and `DD/MM/YY`) — explicit + silences the inference warning.
- **Postponements/abandonments skipped** (no goals/date/result); **unknown teams fail loud** (`UnknownTeamError` propagates).
- **Closing-odds priority** PSC → B365C → AvgC → PS → B365 → Avg.
- `matches.parquet` / `.meta.json` are gitignored build artifacts (reproduced via `make data`), not committed.

**Verification.** Real `make data` → **20,294 matches** across 6 leagues (2016→2026, 100% odds coverage), no warnings, ~3s; `make check` → PASS; `make test` → **27 passed** (4 metadata + 9 schema + 6 sources + 5 teams + 3 matches). ✅

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
