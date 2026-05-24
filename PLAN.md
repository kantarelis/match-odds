# PLAN — Epic 03: Feature Pipeline (leakage-free)

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-03-features`

**Goal.** Implement the `features(date_cutoff)` leakage contract — the heart of the ML-Engineering signal. Walk the master matches table in chronological order and, for every match, emit a feature row computed **strictly from data before kickoff**. The same accumulator code builds the training table (`make features`) and the single feature row served at inference time (Epic 05) — one code path, no train/serve skew.

## Progress

| Task | Description                                                          | Status         | Commit |
|------|----------------------------------------------------------------------|----------------|--------|
| 1    | Pipeline scaffold — accumulator contract, chronological driver, leakage harness, `season_of` | ✅ Done        | —      |
| 2    | `elo.py` — pre-match Elo snapshot                                    | ✅ Done        | —      |
| 3    | `form.py` — rolling last-N form                                     | ✅ Done        | —      |
| 4    | `head_to_head.py` — pairwise H2H history                            | ✅ Done        | —      |
| 5    | `strength.py` — home/away venue strength (season-scoped)            | ✅ Done        | —      |
| 6    | `season.py` — season-progress features                              | ⬜ Not started | —      |
| 7    | Materialize: `make features`, `features.parquet` + meta, end-to-end leakage + train/serve-equivalence tests | ⬜ Not started | —      |
| 8    | `docs/feature-pipeline.md` — every feature + its pre-match snapshot  | ⬜ Not started | —      |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** The toolchain exists from Epic 01, so **every commit must keep `make check` + `make test` green** (no bootstrap phase this epic). No notebooks are touched in Epic 03, so `make nb-lint` / `make nb-run` are not part of any task's gate.

**Consolidation note (flag at review).** Tasks 4 (H2H) and 6 (season-progress) are the two smallest accumulators; if you'd prefer fewer commits they can be merged into one task. The split below keeps one named feature family per commit (clean per-feature revert/blame), matching the Epic 02 style.

## Decisions baked in (flag at review to change)

1. **Single-pass chronological accumulator design — the leakage guarantee is structural.** Each feature family is a stateful *accumulator* exposing two methods: `pre_match(home, away, league, date) -> dict[str, float]` (reads state built only from earlier matches) and `update(match) -> None` (folds in this match's result). The pipeline sorts matches chronologically and **snapshots before it updates** — so a match's features can never see its own or any later result. Matches sharing a date are snapshotted as a *block* before any of them updates state, so a match's features never depend on another match played the same day (the master table has no kickoff times, so within-day order is unknown). Leakage-free *by construction*, not by a downstream check.
2. **Two entrypoints in `pipeline.py`, one shared engine.** `build_feature_table(matches)` is the full training sweep (one pre-match row per historical match); `features(matches, fixtures, *, date_cutoff)` is the serving / point-in-time call (replay history `date < date_cutoff`, then snapshot the given fixtures — no result needed). Both drive the *same* accumulators, so there is no train/serve skew. A test pins their **equivalence**: a row from the training sweep equals `features(date_cutoff=that match's date)` for that fixture.
3. **Cold-start policy.** A team/pair with no prior history gets neutral defaults: Elo = `elo_base`; rolling / venue / H2H features = `NaN` paired with an explicit count column (`*_n`, `h2h_matches`) so the model can learn the low-data regime. Imputation for estimators that can't take NaN (logreg) is **Epic 04's** job, not the pipeline's.
4. **Odds pass through untouched.** `odds_home/draw/away` are legitimate pre-kickoff info (closing odds set before kickoff) and are carried into the feature table **as passthrough columns** for Epic 04's bookmaker baseline and Epic 07's betting-edge analysis. They are **not** turned into engineered model features here — keeping the model's signal independent of the bookmaker is a modeling decision deferred to Epic 04.
5. **Season = football-data July cutover (Aug–May).** A pure calendar helper `season_of(date) -> str` lives in `features/base.py` (Task 1) and is shared by strength (venue records reset per season) and season-progress. It mirrors the cutover already used in `sources.season_code` — extracted, not duplicated.
6. **Output artifacts.** `make features` writes `processed_dir/features.parquet` + a `features.meta.json` sidecar (feature-column list, config params used — Elo K/base/home-adv, rolling N — row + per-league counts, source `matches.meta.json` reference, build timestamp). Both are **gitignored build artifacts** reproduced via `make features`, exactly like `matches.parquet`.
7. **Offline, deterministic tests.** A committed, human-readable `tests/fixtures/synthetic_matches.csv` (4 teams, one league, two seasons, full home/away double round-robin → ~24 matches with rematches and a season boundary) drives every value-pinning and leakage test. No network; pinned values are hand-computable.
8. **Config additions stay minimal.** Elo introduces `elo_base` (1500.0), `elo_k` (20.0), `elo_home_advantage` (65.0) on `settings` (with `MATCHODDS_` overrides); form reuses the existing `rolling_window_n`. Features are fully **deterministic** — `random_seed` is not consumed here.

**Out of scope (their own epics):** any modeling, training, or calibration and the time-ordered CV splitter (Epic 04); serving wiring (Epic 05); notebooks (`02_modeling.ipynb` is Epic 04; `01_data.ipynb` is unchanged); `make repro` (Epic 04, chains data → features → train). No new data sources or raw columns beyond the master matches table.

---

## Task 1 — Pipeline scaffold: accumulator contract, chronological driver, leakage harness — ✅ Done

**Outcome.**
- `src/matchodds/features/base.py`: the `FeatureAccumulator` `Protocol` (`feature_names`, `pre_match(home, away, league, date) -> dict[str, float]`, `update(match: MatchRow) -> None`), the frozen `MatchRow` record fed to `update`, and the pure `season_of(date) -> str` helper (reuses `sources.season_code` + the July cutover).
- `src/matchodds/features/pipeline.py`: the chronological engine — `build_feature_table(matches, accumulators=None)` (one pre-match row per match: identifiers + features + odds passthrough + `result` label) and `features(matches, fixtures, *, date_cutoff, accumulators=None)` (replay history `< cutoff`, then snapshot fixtures), both driven by `default_accumulators()` (empty until Tasks 2–6 wire the families in).
- `tests/fixtures/synthetic_matches.csv`: deterministic 24-match fixture (4 teams, one league, two seasons, full home/away double round-robin, **two matches per date** so day-batching is exercised).
- `tests/unit/test_pipeline.py`: 6 tests — chronological ordering, label/odds passthrough, the day-batched leakage meta-test, append-future invariance, the `features()` pre-cutoff contract, `season_of` cutover.
- `.vulture_allowlist.py`: forward-consumed entrypoints (`build_feature_table`, `features`, `season_of`, `MatchRow` goal fields).

**Decisions / deviations (recorded).**
- **Day-batched snapshot, not strict per-match.** A match snapshots after exactly the **strictly-earlier-dated** matches; same-day matches are excluded — the master table has no kickoff times, so within-day order is unknown and including them would be potential leakage. This makes `build_feature_table` and `features(date_cutoff)` provably equivalent (the Task 7 train/serve guarantee) and, since a team plays at most once per day, never changes a team-keyed feature value. The meta-test asserts this day-batched invariant rather than the spec's looser "i-th match sees exactly `i` updates" wording.
- **Cutover is July 1, not August.** `season_of` mirrors `sources.recent_seasons` (`month >= 7` → new season), so `2023-07-31` resolves to the *new* season — correcting the spec's `2023-07-31 → prior season` example. Tested across the boundary.
- **`accumulators=None` sentinel** (→ `default_accumulators()`) instead of a literal `()` default, so callers get the wired set by omission; the typed `MatchRow` (not a raw pandas row) is the `update` contract, decoupling accumulators from pandas.
- Dropped `runtime_checkable` from the Protocol (no `isinstance` use → unused import).

**Verification.** `make check` → PASS (isort, black, flake8, mypy strict on 12 files, bandit, vulture, nb-lint); `make test` → **33 passed** (27 prior + 6 pipeline). ✅

---

## Task 2 — `elo.py`: pre-match Elo snapshot — ✅ Done

**Outcome.**
- `src/matchodds/features/elo.py`: `EloAccumulator` keyed by `(league, team)`; `pre_match` emits `elo_home` / `elo_away` / `elo_diff` (diff carries the home-advantage bonus); `update` applies the standard logistic Elo with a 1 / 0.5 / 0 result score and a zero-sum delta. Cold start = `settings.elo_base`. The constructor takes optional `base` / `k` / `home_advantage` overrides (defaulting to `settings`) so tests pin hand-computable math.
- `src/matchodds/config.py`: added `elo_base` (1500.0), `elo_k` (20.0), `elo_home_advantage` (65.0).
- `src/matchodds/features/pipeline.py`: `default_accumulators()` returns `(EloAccumulator(),)`.
- `src/matchodds/features/base.py`: made the Protocol's `pre_match` / `update` params **positional-only** (see deviation).
- `tests/unit/test_elo.py`: 7 tests — cold start + home-advantage diff, K-factor / result-score update math, zero-sum conservation, sequential carry-forward, league isolation, pipeline integration (cold-start first row), and Elo-specific append-future leakage.
- `tests/unit/test_pipeline.py`: the two driver tests now pass `accumulators=[]`.

**Decisions / deviations (recorded).**
- **Protocol params made positional-only** (touches `base.py`). Elo's `pre_match` ignores `date`, and vulture flags unused arguments at 100% but ignores `_`-prefixed names (confirmed empirically); positional-only params let an accumulator rename the unused one to `_date` without breaking structural conformance — no line-level silencer. The pipeline already calls these positionally. This is the pattern the date-agnostic accumulators (form, H2H) reuse.
- **Two Task-1 driver tests updated** to pass `accumulators=[]`, since wiring Elo means the bare-default path now carries `elo_*` columns. The driver-in-isolation contract is unchanged.
- **New `elo_*` settings are consumed by `elo.py`**, so — unlike the earlier forward-declared config — they need no `.vulture_allowlist.py` entry.

**Verification.** `make check` → PASS (mypy strict on 13 files, vulture clean); `make test` → **40 passed** (33 prior + 7 Elo). ✅

---

## Task 3 — `form.py`: rolling last-N form — ✅ Done

**Outcome.**
- `src/matchodds/features/form.py`: `FormAccumulator` keeps a per-`(league, team)` deque of the last `settings.rolling_window_n` matches (any venue) holding `(points, goals_for, goals_against)`. `pre_match` emits rolling means for both sides — `form_{home,away}_{ppg,gf,ga}` — plus a `form_{home,away}_n` count; cold start is `NaN` with `n == 0`. Constructor takes an optional `window` override for tests.
- `src/matchodds/features/pipeline.py`: `default_accumulators()` returns `(EloAccumulator(), FormAccumulator())`.
- `.vulture_allowlist.py`: removed `settings.rolling_window_n` now that `form.py` consumes it (per the allowlist's "remove as consumers land" rule).
- `tests/unit/test_form.py`: 6 tests — cold-start NaN/zero-count, two-venue rolling means, window bound (oldest drops at `window=2`), away-perspective correctness, pipeline integration (cold-start first row), append-future leakage.

**Decisions / deviations (recorded).**
- None structural. Form is date-agnostic, so it reuses the positional-only `_date` pattern from Task 2.
- Allowlist hygiene: `settings.rolling_window_n` moved from forward-declared to consumed.

**Verification.** `make check` → PASS (mypy strict on 14 files, vulture clean); `make test` → **46 passed** (40 prior + 6 form). ✅

---

## Task 4 — `head_to_head.py`: pairwise H2H history — ✅ Done

**Outcome.**
- `src/matchodds/features/head_to_head.py`: `HeadToHeadAccumulator` keeps a per-`(league, team_lo, team_hi)` running tally (a small `_PairRecord` dataclass) of prior meetings. `pre_match` reports from the **current home team's perspective**: `h2h_matches`, `h2h_home_wins`, `h2h_draws`, `h2h_away_wins`, `h2h_home_goals_avg`, `h2h_away_goals_avg`. Cold start → zero counts / `NaN` averages. All prior meetings count (venue- and recency-agnostic).
- `src/matchodds/features/pipeline.py`: `default_accumulators()` returns `(EloAccumulator(), FormAccumulator(), HeadToHeadAccumulator())`.
- `tests/unit/test_head_to_head.py`: 6 tests — first-meeting zero/NaN, home-perspective counts + goal averages, perspective flip on the reverse fixture, pair isolation, pipeline integration (the second Alpha–Beta meeting correctly sees 2 priors), append-future leakage.

**Decisions / deviations (recorded).**
- None structural. Date-agnostic, so it reuses the positional-only `_date` pattern; no new settings → no allowlist change.
- Win attribution uses an explicit `winner = home if result == "H" else away; winner == lo` rather than boolean arithmetic, for readability.

**Verification.** `make check` → PASS (mypy strict on 15 files, vulture clean); `make test` → **52 passed** (46 prior + 6 H2H). ✅

---

## Task 5 — `strength.py`: home/away venue strength (season-scoped) — ✅ Done

**Outcome.**
- `src/matchodds/features/strength.py`: `StrengthAccumulator` keeps per-`(league, team, season, venue)` records (a `_VenueRecord` dataclass). `pre_match` reads the home team's **home-venue** record and the away team's **away-venue** record for the current season (`season_of(date)`), emitting `strength_{home,away}_{ppg,gf,ga,n}`. Records reset at each season boundary (season is part of the key); cold start / season start → `NaN` with `n == 0`. This is the first accumulator that reads the match `date`.
- `src/matchodds/features/pipeline.py`: `default_accumulators()` returns `(EloAccumulator(), FormAccumulator(), HeadToHeadAccumulator(), StrengthAccumulator())`.
- `.vulture_allowlist.py`: removed `season_of` and the two `MatchRow` goal-field entries (and the now-unused `base` import) — form / H2H / strength are their real consumers. The allowlist now only forward-declares `pipeline.build_feature_table` / `features` (consumed in Task 7).
- `tests/unit/test_strength.py`: 6 tests — cold-start NaN, home-record-uses-only-home-matches, away-record-uses-only-away-matches, season reset, pipeline integration (Alpha's home count accumulates through season one then resets to 0 at the season-two opener), append-future leakage.

**Decisions / deviations (recorded).**
- None structural. `pre_match` keeps the `date` argument (consumed by `season_of`), so no `_date`.
- Allowlist hygiene continued: `season_of` and `MatchRow.ft_*` moved from forward-declared to consumed.

**Verification.** `make check` → PASS (mypy strict on 16 files, vulture clean); `make test` → **58 passed** (52 prior + 6 strength). ✅

---

## Task 6 — `season.py`: season-progress features

**Scope (files to touch).**
- `src/matchodds/features/season.py` (new): `SeasonAccumulator` (uses `season_of` from `base.py`). `pre_match` emits `season_home_matchday`, `season_away_matchday` (count of that team's matches already played this season, +1), `season_fraction` (elapsed fraction of a 38-matchday season, capped at 1.0), and `days_since_last_match_home` / `_away` (rest days; `NaN` at season start).
- `src/matchodds/features/pipeline.py`: add `SeasonAccumulator` to `default_accumulators()`.
- `tests/unit/test_season.py` (new).

**Acceptance criteria.**
- Matchday counters increment per team within a season and **reset to 1** at the season boundary.
- `days_since_last_match_*` matches the synthetic dates; `NaN` on the season's first appearance.
- **Leakage:** append-future invariance holds for all `season_*` columns.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 7 — Materialize: `make features`, `features.parquet` + meta, end-to-end guards

**Scope (files to touch).**
- `src/matchodds/features/pipeline.py`: add `build(write=True)` (load via `matches.load()` → `build_feature_table(default_accumulators())` → write `processed_dir/features.parquet` + `features.meta.json`), a `main()` entrypoint, and the `if __name__ == "__main__"` guard — mirroring `data/matches.py`.
- `makefile`: replace the `features` stub with `python -m matchodds.features.pipeline`.
- `tests/unit/test_features_table.py` (new): end-to-end integration over the synthetic fixture.

**Acceptance criteria.**
- `build_feature_table(default_accumulators())` emits **all** feature columns (`elo_*`, `form_*`, `h2h_*`, `strength_*`, `season_*`) plus identifiers/label/odds, one row per match.
- **End-to-end leakage guard:** fabricate a future result, rebuild, and assert every earlier row is byte-identical across the full feature set (the canonical "future cannot influence the past" test from the epic scope).
- **Train/serve equivalence (Decision 2):** for a sampled match, the training-sweep row equals `features(date_cutoff=that match's date)` for that fixture — proving the shared code path.
- `features.meta.json` records the feature-column list and the config params used.
- `make features` succeeds against the real built `matches.parquet` locally (writes `features.parquet`).

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 8 — `docs/feature-pipeline.md`: every feature + its pre-match snapshot

**Scope (files to touch).**
- `docs/feature-pipeline.md` (new): the contract and the single-pass design (snapshot-before-update); a table of every feature column with the exact pre-match snapshot it reads, its cold-start value, and the config param it depends on; the train/serve-shared code path; and how each leakage guard enforces the contract. Satisfies CLAUDE.md's rule that *every* feature documents which pre-match snapshot it reads.

**Acceptance criteria.**
- Every column produced by `build_feature_table` appears in the doc with its snapshot + cold-start behavior.
- The leakage contract and the train/serve-equivalence guarantee are described.
- Docs-only commit — no code changes.

**Gate.** `make check` → PASS; `make test` → all green (unchanged).

---

## Definition of done (Epic 03)

`make features` emits `processed_dir/features.parquet` from the built master matches table; the leakage guards pass (snapshot-before-update meta-test, per-feature append-future invariance, the end-to-end fabricate-future guard, and the train/serve-equivalence test); and every feature is documented with its pre-match snapshot in `docs/feature-pipeline.md`. On close-out: mark Epic 03 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-03-features.md`.
