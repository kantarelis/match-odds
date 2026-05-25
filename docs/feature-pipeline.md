# Feature pipeline

The feature pipeline turns the master matches table (`data/processed/matches.parquet`, built by
`make data`) into a leakage-free feature table (`data/processed/features.parquet`, built by
`make features`). It lives in `src/matchodds/features/` and is the heart of the ML-Engineering
signal: **every feature for a match is computed only from information available strictly before that
match's kickoff.**

This document is also the contract required by `CLAUDE.md`: *every* feature states exactly which
pre-match snapshot it reads. If a feature here could not be computed without peeking at or after the
match it describes, that is a bug, not a feature.

## The leakage contract

A match's feature row may depend only on matches played on **strictly earlier dates**. Concretely:

- Same-day matches are **excluded**. The master table records a date but no kickoff time, so the
  order of matches within a day is unknown; treating any same-day match as "earlier" would risk
  leaking the future. (In practice a team plays at most once per day, so this never changes a
  team-keyed value — but it makes the contract exact.)
- A team's or pair's **first** appearance has no history, so the feature is a documented cold-start
  value (a neutral default or `NaN` with a companion count) rather than an error.

## The single-pass design

Each feature family is a stateful **accumulator** (`src/matchodds/features/base.py`,
`FeatureAccumulator`) exposing two positional-only methods:

- `pre_match(home, away, league, date)` → the pre-kickoff feature values, read from state built only
  from earlier matches.
- `update(match)` → folds a played match's result into the accumulator's state.

The driver (`pipeline.build_feature_table`) walks the matches in chronological order and, for each
**date as a block**, first takes every match's `pre_match` snapshot, *then* applies every match's
`update`. Because the whole day is snapshotted before any of it updates state, a match's features can
never see its own result or any same-day/later result. Leakage-free **by construction**, not by a
downstream check.

## Two entrypoints, one code path (no train/serve skew)

Both entrypoints in `pipeline.py` drive the *same* accumulators, so training and serving cannot
diverge:

- `build_feature_table(matches, accumulators)` — the full training sweep: one pre-match row per
  historical match. `make features` runs this over the master table.
- `features(matches, fixtures, *, date_cutoff, accumulators)` — the point-in-time / serving call:
  replay history with `date < date_cutoff`, then snapshot the upcoming fixtures (no result needed).
  Epic 05's service uses this to build the single feature row at request time.

The equivalence is tested: a training-sweep row for a match equals `features(date_cutoff=that
match's date)` for that fixture (`tests/unit/test_features_table.py`).

## Feature table schema

`build_feature_table` emits, in order: **identifiers** → **features** → **odds passthrough** →
**label**.

| Group       | Columns                                              | Notes                                                        |
|-------------|------------------------------------------------------|--------------------------------------------------------------|
| Identifiers | `league`, `date`, `home`, `away`                     | Canonical names; `date` is the kickoff date.                 |
| Features    | the 30 columns documented below                      | All `float`; leakage-free pre-match snapshots.               |
| Odds        | `odds_home`, `odds_draw`, `odds_away`                | Bookmaker closing odds, **passed through untouched**.        |
| Label       | `result`                                             | `H` / `D` / `A` (home perspective).                          |

**Odds are passthrough, not engineered.** Closing odds are set before kickoff, so they are
legitimate pre-match information and are carried through for Epic 04's bookmaker baseline and Epic
07's betting-edge analysis. They are deliberately **not** turned into model features here — keeping
the model's signal independent of the bookmaker is a modelling decision left to Epic 04.

## Feature reference

Every feature column, the pre-match snapshot it reads, its cold-start value, and the config field it
depends on. Config defaults are in `src/matchodds/config.py` (`settings`); all are overridable via
`MATCHODDS_*` environment variables. Features are fully deterministic — `random_seed` is **not**
used here.

### Elo — `src/matchodds/features/elo.py`

Per-`(league, team)` rating, updated by the standard logistic Elo rule (result score 1 / 0.5 / 0,
zero-sum delta). Config: `elo_base` (1500.0), `elo_k` (20.0), `elo_home_advantage` (65.0).

| Column      | Pre-match snapshot                                                   | Cold start            |
|-------------|----------------------------------------------------------------------|-----------------------|
| `elo_home`  | Home team's Elo rating going into the match.                         | `elo_base`            |
| `elo_away`  | Away team's Elo rating going into the match.                         | `elo_base`            |
| `elo_diff`  | `elo_home + elo_home_advantage − elo_away`.                          | `elo_home_advantage`  |

### Rolling form — `src/matchodds/features/form.py`

Per-`(league, team)` deque of the last `rolling_window_n` matches (either venue), holding points /
goals-for / goals-against. Config: `rolling_window_n` (5).

| Column          | Pre-match snapshot                                              | Cold start |
|-----------------|----------------------------------------------------------------|------------|
| `form_home_ppg` | Home team's mean points per game over its last ≤N matches.     | `NaN`      |
| `form_home_gf`  | Home team's mean goals scored over its last ≤N matches.        | `NaN`      |
| `form_home_ga`  | Home team's mean goals conceded over its last ≤N matches.      | `NaN`      |
| `form_home_n`   | Number of matches available in the window (0..N).              | `0`        |
| `form_away_ppg` | Away team's mean points per game over its last ≤N matches.     | `NaN`      |
| `form_away_gf`  | Away team's mean goals scored over its last ≤N matches.        | `NaN`      |
| `form_away_ga`  | Away team's mean goals conceded over its last ≤N matches.      | `NaN`      |
| `form_away_n`   | Number of matches available in the window (0..N).              | `0`        |

### Head-to-head — `src/matchodds/features/head_to_head.py`

Per unordered `(league, team_lo, team_hi)` pair, a tally of all prior meetings (venue- and
recency-agnostic), reported from the **current home team's** perspective. No config.

| Column                | Pre-match snapshot                                            | Cold start |
|-----------------------|---------------------------------------------------------------|------------|
| `h2h_matches`         | Number of prior meetings between the two clubs.               | `0`        |
| `h2h_home_wins`       | Prior meetings won by the current home team.                  | `0`        |
| `h2h_draws`           | Prior meetings drawn.                                         | `0`        |
| `h2h_away_wins`       | Prior meetings won by the current away team.                  | `0`        |
| `h2h_home_goals_avg`  | Mean goals scored by the current home team across meetings.   | `NaN`      |
| `h2h_away_goals_avg`  | Mean goals scored by the current away team across meetings.   | `NaN`      |

### Home/away strength — `src/matchodds/features/strength.py`

Per-`(league, team, season, venue)` record; **resets each season** (`season_of`). The home team's
**home** record and the away team's **away** record are read. No config (uses the season calendar).

| Column               | Pre-match snapshot                                                       | Cold start |
|----------------------|--------------------------------------------------------------------------|------------|
| `strength_home_ppg`  | Home team's points per game **at home** this season.                     | `NaN`      |
| `strength_home_gf`   | Home team's mean goals scored at home this season.                       | `NaN`      |
| `strength_home_ga`   | Home team's mean goals conceded at home this season.                     | `NaN`      |
| `strength_home_n`    | Home team's number of home matches this season.                          | `0`        |
| `strength_away_ppg`  | Away team's points per game **away** this season.                        | `NaN`      |
| `strength_away_gf`   | Away team's mean goals scored away this season.                          | `NaN`      |
| `strength_away_ga`   | Away team's mean goals conceded away this season.                        | `NaN`      |
| `strength_away_n`    | Away team's number of away matches this season.                          | `0`        |

### Season progress — `src/matchodds/features/season.py`

Per-`(league, team, season)` count of matches played and last-match date; **resets each season**.
The season runs Aug–May with a July cutover (`season_of`, mirroring `data.sources.season_code`).
The season length is approximated as 38 matchdays; the fraction is capped at 1.0.

| Column                       | Pre-match snapshot                                                      | Cold start |
|------------------------------|-------------------------------------------------------------------------|------------|
| `season_home_matchday`       | Home team's matches played this season + 1 (the matchday it now plays). | `1`        |
| `season_away_matchday`       | Away team's matches played this season + 1.                             | `1`        |
| `season_fraction`            | `min(1.0, max(home_matchday, away_matchday) / 38)`.                     | `1/38`     |
| `days_since_last_match_home` | Days since the home team's previous match this season.                  | `NaN`      |
| `days_since_last_match_away` | Days since the away team's previous match this season.                  | `NaN`      |

## Cold-start policy

A team or pair with no qualifying history gets a neutral default rather than a fabricated value:

- **Elo** starts at `elo_base` (a real, usable rating).
- **Rolling / venue / head-to-head** features are `NaN`, paired with an explicit count column
  (`form_*_n`, `strength_*_n`, `h2h_matches`) so the model can distinguish a genuine value from a
  cold start.

Imputation for estimators that cannot consume `NaN` (e.g. logistic regression) is **Epic 04's**
responsibility, not the pipeline's — the pipeline reports honest "no data yet" values.

## Leakage guards

The contract is enforced by tests, layered from the driver outward:

- **Snapshot-before-update (driver):** a probe accumulator asserts that at each snapshot it has seen
  exactly the matches on strictly-earlier dates — never a same-day or later match
  (`tests/unit/test_pipeline.py`).
- **Per-feature append-future invariance:** each family's test fabricates a future match, rebuilds,
  and asserts every earlier row is unchanged (`tests/unit/test_{elo,form,head_to_head,strength,
  season}.py`).
- **End-to-end guard:** the same fabricate-future check across the *full* feature set
  (`tests/unit/test_features_table.py`).
- **Train/serve equivalence:** a training-sweep row equals the point-in-time `features(date_cutoff)`
  call for that fixture, proving the shared code path (`tests/unit/test_features_table.py`).

All tests are offline and deterministic, driven by `tests/fixtures/synthetic_matches.csv` (4 teams,
one league, two seasons, full home/away double round-robin) with hand-computable expected values.

## Reproducing the feature table

```bash
make data       # build data/processed/matches.parquet (network, one-off)
make features   # build data/processed/features.parquet + features.meta.json
```

`features.meta.json` records the row/league counts, the full feature-column list, the config
parameters used (Elo base/K/home-advantage, rolling window), and a build timestamp. Both the parquet
and the meta are gitignored build artifacts, reproduced from the committed code and the pinned data
version.
