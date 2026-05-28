# Model card — `models/v1.joblib`

Structured per [Mitchell et al., 2019 — *Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03677)
so the sections are scannable by name. Numbers below are read directly from
[`models/v1.metadata.json`](../models/v1.metadata.json); re-check by running `cat models/v1.metadata.json`
or `make repro`.

## Model details

- **Artifact:** `models/v1.joblib` (committed; ~58 KB)
- **Model:** multinomial **logistic regression** (`sklearn.pipeline.Pipeline` — median impute →
  standard scale → `LogisticRegression`, `lbfgs`, `max_iter=1000`)
- **Calibration:** **sigmoid** (Platt), via the **hold-out (prefit)** scheme introduced in
  Epic 06.5 — see *Calibration* below.
- **Output:** calibrated 1X2 probability vector `[home_win, draw, away_win]`, rows sum to 1.
- **Trained:** `2026-05-28` (UTC) · **Seed:** `42` · **CV folds:** 5 (forward-chained temporal).
- **Library versions:** numpy `2.4.6`, pandas `2.3.3`, scikit-learn `1.8.0`, scipy `1.17.1`,
  xgboost `2.1.4`, joblib `1.5.3`, Python `3.14`.
- **Source code:** [`src/matchodds/modeling/logistic.py`](../src/matchodds/modeling/logistic.py),
  [`src/matchodds/modeling/calibration.py`](../src/matchodds/modeling/calibration.py),
  [`src/matchodds/modeling/train.py`](../src/matchodds/modeling/train.py).

## Intended use

- **Primary use case.** An educational portfolio piece demonstrating notebook-to-service ML
  engineering: leakage-free temporal CV, a head-to-head bake-off against domain-canonical and
  market-implied baselines, hold-out calibration with reliability checking, and a Dockerised
  inference service.
- **Secondary use case.** A 1X2 (home / draw / away) probability service for the six in-scope
  leagues listed in *Data*, exposed by `POST /predict` from the FastAPI service in
  [`serving/`](../serving/) and consumed by the [Streamlit demo](../demo/).
- **Users.** Recruiters and engineers reviewing the repo; the author's own retrospective evaluation.
- **Not intended for.** See *Limitations* and *Ethics / out of scope* below.

## Data

- **Single source:** [football-data.co.uk](https://www.football-data.co.uk/) — one CSV per
  league-season at `mmz4281/{season}/{div}.csv`. No scraping, no live feed, no authenticated APIs
  (`CLAUDE.md` → *Constraints*).
- **Volume:** 20,294 matches, `2016-08-12 → 2026-05-21`.
- **Leagues in scope (six):** English Premier League (3,790 matches) · Spanish La Liga (3,790) ·
  Italian Serie A (3,790) · French Ligue 1 (3,476) · German Bundesliga (3,060) ·
  Greek Super League (2,388).
- **Per-match columns kept:** identifiers (league / date / home / away), full-time result
  (`H` / `D` / `A`), full-time goals (used by the Dixon-Coles candidate; never a feature), and
  bookmaker closing odds where available (used by the *baseline* and the Epic 07 backtest; never an
  inference-time feature).
- **Provenance manifest:** `raw/_manifest.json` — pinned URLs and snapshot date, regenerable via
  `make data`.

## Features (30 columns)

Engineered by five stateful accumulators under
[`src/matchodds/features/`](../src/matchodds/features/), all read **strictly before kickoff**
(`features(date_cutoff)` contract — see [`docs/feature-pipeline.md`](feature-pipeline.md) for the
per-feature leakage proof):

| Family | Columns | What it captures |
|---|---|---|
| **Elo** | `elo_home`, `elo_away`, `elo_diff` | Per-league Elo with a home-advantage bonus; zero-sum updates. |
| **Form** | `form_{home,away}_{ppg,gf,ga,n}` (× 2) | Each side's rolling last-N record (points / goals for / goals against / sample size), either venue. |
| **Head-to-head** | `h2h_matches`, `h2h_{home_wins,draws,away_wins}`, `h2h_{home,away}_goals_avg` | Pairwise prior meetings from the current home team's perspective. |
| **Venue strength** | `strength_{home,away}_{ppg,gf,ga,n}` (× 2) | Season-scoped home-record-of-the-home-team + away-record-of-the-away-team; reset at season boundaries. |
| **Season** | `season_{home,away}_matchday`, `season_fraction`, `days_since_last_match_{home,away}` | Where the fixture sits in the season; rest days per side. |

Cold-start fixtures (a team / pair with no prior history) snapshot `NaN` with a companion `_n` count
of 0 — the discriminative model's median-imputation step then handles this without leaking a fake
average from the test window.

## Metrics

Headline numbers and the four-model bake-off table live in
[`docs/evaluation.md`](evaluation.md) — not duplicated here so they cannot drift.

Selection is on mean CV **log-loss** across the *deployable* models; the shipped logistic + sigmoid
artifact wins on both proper scoring rules (log-loss + Brier). Accuracy is reported but never
optimised (`CLAUDE.md` → *ML / Modeling Discipline*).

## Calibration

The shipped model is calibrated via the **hold-out (prefit)** scheme introduced in Epic 06.5: the
base estimator is fit on the final fold's expanding-window training slice, then wrapped in
`sklearn.frozen.FrozenEstimator` and handed to `CalibratedClassifierCV`, which fits the calibration
regressor on the strictly-later held-out tail. The calibrator therefore only ever sees data dated
**after** the base's training window — leakage-free.

The previous ensemble-over-folds construction over-flattened confident predictions and inverted the
real home advantage on near-even derbies (`AEK(H)` vs `PAOK`: 0.336 home-win deployed, 0.434 with
the fix). The fix also improved the shipped model's mean CV log-loss `0.9989 → 0.9945` and Brier
`0.5954 → 0.5930`. Reliability diagrams live in
[`notebooks/02_modeling.ipynb`](../notebooks/02_modeling.ipynb).

## Limitations

- **Out-of-scope leagues / teams.** The service rejects requests outside the six leagues above or
  whose team names are not in the canonical registry — by design (no fabricated predictions). A
  newly-promoted team with no prior history snapshots `NaN` form / Elo and the model leans on the
  base rate.
- **Cold-start fixtures.** Season-opener matches and pair-first meetings carry less signal because
  several feature families have empty state at that point.
- **Calendar-day granularity.** The master table has no kickoff times, so same-day matches are not
  ordered. The pipeline treats a day as one block and snapshots every fixture before any of them
  update accumulator state.
- **Upsets are upsets.** The output is a calibrated probability, not a prediction. A 65%
  home-favourite still loses one in three; do not interpret a confident output as a guaranteed one.
- **No in-play modelling.** Features are pre-kickoff only; the model has no notion of in-game state.
- **Six leagues, ten seasons.** Out-of-distribution leagues (women's football, lower divisions,
  cup competitions, friendlies) are not represented and the model should not be applied to them.

## Ethics / out of scope

- **No live betting, no real-money use, no advice.** This repo carries no live-odds ingestion, no
  bet-placement code, and no real-money framing anywhere (`CLAUDE.md` → *Constraints*). The output
  is calibrated probabilities for a portfolio demo, not a recommendation to act on.
- **The betting-edge notebook ([`notebooks/03_betting_edge.ipynb`](../notebooks/03_betting_edge.ipynb))
  is historical and illustrative only.** It measures the model against past closing odds; the
  observed result on the current artifact is a *negative* ROI at every tested threshold — the
  honest outcome that closing odds are a very efficient benchmark and this model targets honest
  probabilities, not betting profit.
- **No bias audit was performed.** This is a sports-results model on six men's first-division
  leagues — out of scope for the demographic-bias frameworks model cards usually flag, but worth
  noting that the data scope is itself a choice (no women's leagues, no lower divisions).
- **No PII.** Inputs are public team names + a date; outputs are three numbers summing to one.

## Maintainer & provenance

- **Maintainer:** Spyro Kantareli ([@kantarelis](https://github.com/kantarelis)).
- **Repository:** [`match-odds`](https://github.com/kantarelis/match-odds).
- **Refresh:** rebuild the artifact with `make repro` (`data → features → train`); deterministic
  from `settings.random_seed` (default 42) on a pinned data version.
- **Versioning:** the shipped artifact stays `v1.joblib`; calibration-only fixes overwrite in place
  (no version bump) — see the Epic 06.5 close-out in [`docs/history/`](history/).
- **Per-epic history:** [`docs/history/`](history/) carries the archived plan + outcome record for
  every epic that shaped this artifact.
