# PLAN — Epic 04: Modeling, Evaluation & Calibration

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `epic-04-modeling`

**Goal.** Train, compare, calibrate, and freeze the shipped match-outcome model. Four approaches — the bookmaker-implied **baseline**, multinomial **logistic regression**, **XGBoost**, and the domain-canonical **Dixon-Coles** bivariate-Poisson goals model — are evaluated head-to-head under **strictly temporal** cross-validation on **proper scoring rules** (log-loss + Brier; accuracy secondary). The best *deployable* model is calibrated and frozen to `models/v1.joblib` (+ metadata sidecar), and `make repro` reproduces its metrics deterministically from the pinned data version. All models share one `OutcomeModel` interface and one feature table, so the code that trains is the code Epic 05 serves — no train/serve skew.

## Progress

| Task | Description                                                              | Status         | Commit |
|------|--------------------------------------------------------------------------|----------------|--------|
| 1    | `metrics.py` — proper scoring (log-loss, Brier) + accuracy + reliability helper, `[H,D,A]` convention | ⬜ Not started | —      |
| 2    | `cv.py` — time-ordered (forward-chaining) CV splitter                    | ⬜ Not started | —      |
| 3    | `base.py` model interface + `baselines.py` bookmaker-implied baseline    | ⬜ Not started | —      |
| 4    | `logistic.py` — multinomial logistic regression (impute + scale)         | ⬜ Not started | —      |
| 5    | `xgboost_model.py` — gradient-boosted trees (native NaN)                 | ⬜ Not started | —      |
| 6    | Carry full-time goals into the feature table (generative-model target)   | ⬜ Not started | —      |
| 7    | `dixon_coles.py` — bivariate-Poisson goals model (MLE)                   | ⬜ Not started | —      |
| 8    | `calibration.py` — Platt/isotonic calibration of the discriminative models | ⬜ Not started | —      |
| 9    | `train.py` — temporal-CV bake-off, freeze `v1.joblib` + metadata; `make train` + `make repro` | ⬜ Not started | —      |
| 10   | `02_modeling.ipynb` — narrative + 4-way comparison + reliability diagrams; wire nb gates | ⬜ Not started | —      |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** The toolchain exists from Epic 01, so **every commit must keep `make check` + `make test` green**. Because `vulture` scans `src` only, a model/helper consumed *just by tests* until `train.py` (Task 9) lands gets a `.vulture_allowlist.py` entry, **removed when its real `src` consumer lands** — the epic-03 hygiene pattern. Notebooks are only touched in Task 10, so `make nb-lint` / `make nb-run` are part of that task's gate alone.

**Consolidation note (flag at review).** The split keeps one named concern per commit (clean per-model blame/revert), matching epic 02/03. If you prefer fewer commits: Tasks 1+2 (metrics + CV) are both small pure "evaluation harness" modules and can merge; Tasks 4+5 (LR + XGBoost) are the two discriminative models and can merge; Task 6 (goals passthrough) can fold into Task 7 (its only consumer).

## Decisions baked in (flag at review to change)

1. **One uniform interface, one training table.** Every model implements `OutcomeModel.predict_proba(table) -> ndarray (n, 3)` in the fixed class order **`[H, D, A]`**; a shared label helper maps `result` ↔ index. Discriminative models read the 30 feature columns; the baseline reads `odds_*`; Dixon-Coles reads identifiers (+ goals at fit). All four train/evaluate from the single feature table, so serving is uniform regardless of which model wins.
2. **Deployable vs. benchmark.** The bookmaker baseline needs closing odds, which the Epic-05 serving request (`{home, away, league, date}`) does **not** carry — so it is a **benchmark to beat, never the shipped artifact**. `v1.joblib` is selected among the **deployable** models (LR / XGBoost / Dixon-Coles), all of which predict from identifiers + reconstructed features alone. (If you want the baseline shippable we'd need request-time odds — out of current scope.)
3. **Proper scoring rules drive selection.** Headline = **log-loss** (primary) + **Brier**; accuracy is reported but never decisive. Selection minimizes mean CV log-loss; a model that wins accuracy while losing log-loss does not win.
4. **Temporal CV only.** `cv.py` is forward-chaining (expanding-window) over date-sorted matches — every test fold is strictly later than its training rows, and a single day is never split across the boundary. No `KFold`/`StratifiedKFold` on match rows, ever. Calibration uses these same temporal folds (never calibrate on data the model trained on).
5. **Calibration is part of the model — for the discriminative ones.** LR + XGBoost are wrapped with calibration (Platt/sigmoid vs isotonic, chosen by CV log-loss) over the temporal folds; the shipped artifact is the *calibrated* estimator. Dixon-Coles is naturally calibrated (generative) and gets a reliability **check**, not a wrapper.
6. **Cold-start NaNs, per model.** LR gets median imputation + scaling inside its sklearn `Pipeline`; XGBoost consumes NaN natively (no imputer); Dixon-Coles ignores the engineered features. The `*_n` / `h2h_matches` count columns stay, so the low-data regime remains learnable.
7. **Dixon-Coles is fit per league** by MLE (`scipy.optimize`): per-team attack/defense, a home-advantage term, and the low-score (τ) correction; optional exponential time-decay (`dc_half_life_days`). 1X2 from summing the score matrix to `dc_max_goals`. Unknown team at predict → league-average strength fallback. It needs full-time goals → **Task 6 carries `ft_home_goals`/`ft_away_goals` into the feature table** as label-side passthrough (a target, never a feature). Adds `scipy` to `requirements.txt` (currently transitive; the service may ship DC).
8. **Frozen artifact + metadata sidecar.** `make train` writes the calibrated winner to `models/v1.joblib` and `models/v1.metadata.json` (data version from `matches.meta.json`/`features.meta.json`, `random_seed`, per-model CV log-loss/Brier/accuracy, selected model + calibration method, the feature-column list, library versions). **Both files are committed** — the `.gitignore` already exempts `models/v1.*`. Epic 05's service loads this object and calls `predict_proba`; model classes therefore live in importable `matchodds.modeling` (not in `train.py`) so unpickling works.
9. **Reproduce metrics, not bytes.** Pickled bytes are not bit-reproducible; per CLAUDE.md the determinism contract is that `make repro` on the pinned data version reproduces the **metric scores** recorded in the metadata. `settings.random_seed` flows into every split, estimator, and calibrator; XGBoost runs single-threaded (or fixed `n_jobs`) so its scores are stable.
10. **Offline, deterministic tests.** Unit tests build a tiny feature table from the committed `tests/fixtures/synthetic_matches.csv` and train into a `tmp_path` models dir — no network, no touching the real `models/v1.joblib`. Real `make repro` (download → features → train) stays a local activity.
11. **Minimal config additions.** New `settings` fields land where first consumed: `cv_splits` (Task 2), `dc_max_goals` + `dc_half_life_days` (Task 7). Calibration method choice and model hyperparameters live in code (a small, fixed, seeded set — no expensive search), keeping `make repro` fast and deterministic.

**Out of scope (their own epics):** the serving layer (Epic 05), the demo (Epic 06), the betting-edge notebook (Epic 07), and the model-card / evaluation deep-dive docs (Epic 08 — only `02_modeling.ipynb` is built here). No live odds. No GPU. No hyperparameter search beyond a small fixed seeded set. No new data sources or raw columns beyond the master matches table (Task 6 only passes its existing goals through).

---

## Task 1 — `modeling/metrics.py`: proper scoring rules + reliability helper

**Scope (files to touch).**
- `src/matchodds/modeling/metrics.py` (new): the fixed `CLASSES = ("H", "D", "A")` order and a `result` ↔ index encoder; `log_loss`, `brier_score` (multiclass), and `accuracy` over `(y_true_idx, proba)`; `reliability_curve(proba, y_true, *, n_bins)` returning per-bin mean-predicted vs. observed-frequency arrays for the diagrams; a `score_summary(proba, y_true) -> dict` bundling the three headline metrics.
- `tests/unit/test_metrics.py` (new).
- `.vulture_allowlist.py`: add the new public symbols (consumed by tests now, by `train.py`/the notebook later; removed in Task 9).

**Acceptance criteria.**
- log-loss and Brier match hand-computed values on a tiny pinned example; perfect predictions → 0; the encoder round-trips and the class order is exactly `[H, D, A]` everywhere.
- `reliability_curve` bins a deterministic example correctly.
- Pure functions, no RNG.

**Gate.** `make check` → PASS; `make test` → all green (+ new metrics tests).

---

## Task 2 — `modeling/cv.py`: time-ordered cross-validation splitter

**Scope (files to touch).**
- `src/matchodds/modeling/cv.py` (new): a forward-chaining / expanding-window splitter yielding `(train_idx, test_idx)` over a **date-sorted** frame; each test fold is contiguous and strictly later than all its training rows, split on **date boundaries** so a day is never divided across train/test. Usable both directly and as a `cv` argument to scikit-learn's `CalibratedClassifierCV` (Task 8). A guard raises if the input isn't sorted by `date`.
- `src/matchodds/config.py`: add `cv_splits: int = 5`.
- `tests/unit/test_cv.py` (new).
- `.vulture_allowlist.py`: add the splitter (+ `cv_splits` if not yet read in `src`).

**Acceptance criteria.**
- For every fold `max(train.date) < min(test.date)` — strictly temporal, no leakage; folds cover the tail of the data and are deterministic given `n_splits`.
- A day is never split across the train/test boundary; unsorted input raises.
- No `sklearn.KFold`/`StratifiedKFold` on match rows anywhere.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 3 — `modeling/base.py` interface + `modeling/baselines.py` bookmaker baseline

**Scope (files to touch).**
- `src/matchodds/modeling/base.py` (new): `OutcomeModel` `Protocol` — `fit(table) -> OutcomeModel` and `predict_proba(table) -> ndarray (n, 3)` in `[H, D, A]`; the shared `feature_columns()` convention (derived from `pipeline.default_accumulators()`'s feature names) used by the discriminative models.
- `src/matchodds/modeling/baselines.py` (new): `BookmakerBaseline` — turns `odds_home/draw/away` into implied probabilities and removes the overround (normalise `1/odds`) → `[H, D, A]`; `fit` is a no-op (stateless). Documented benchmark-only (needs odds → not deployable).
- `tests/unit/test_baselines.py` (new).
- `.vulture_allowlist.py`: add `OutcomeModel`, `BookmakerBaseline`.

**Acceptance criteria.**
- Implied probs sum to 1 per row, overround removed, matching a hand-computed example; shape `(n, 3)`, order `[H, D, A]`.
- Missing-odds rows have documented behaviour (NaN row or raise — pick one and test it).
- `BookmakerBaseline` structurally satisfies `OutcomeModel` (mypy strict).

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 4 — `modeling/logistic.py`: multinomial logistic regression

**Scope (files to touch).**
- `src/matchodds/modeling/logistic.py` (new): `LogisticModel` wrapping a sklearn `Pipeline` = `SimpleImputer(strategy="median")` → `StandardScaler` → `LogisticRegression(multinomial, random_state=settings.random_seed, …)`. `fit` selects the feature columns + encodes labels; `predict_proba` returns `[H, D, A]` aligned to the encoder. Precise return types for mypy strict.
- `tests/unit/test_logistic.py` (new).
- `.vulture_allowlist.py`: add `LogisticModel`.

**Acceptance criteria.**
- Fits on the synthetic feature table; `predict_proba` shape `(n, 3)`, rows sum to 1, order `[H, D, A]`.
- NaN cold-start rows are imputed (no error).
- Deterministic: two fits with the same seed give identical probabilities.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 5 — `modeling/xgboost_model.py`: gradient-boosted trees

**Scope (files to touch).**
- `src/matchodds/modeling/xgboost_model.py` (new): `XGBoostModel` over `XGBClassifier(objective="multi:softprob", num_class=3, random_state=settings.random_seed, n_jobs=1, …)`; selects feature columns; **native NaN** handling (no imputer); `predict_proba` → `[H, D, A]`.
- `tests/unit/test_xgboost_model.py` (new).
- `.vulture_allowlist.py`: add `XGBoostModel`.

**Acceptance criteria.**
- Fits on synthetic; shape `(n, 3)`; rows sum to 1; order `[H, D, A]`; NaN inputs accepted natively.
- Deterministic with the seed (single-threaded so scores are stable).

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 6 — Carry full-time goals into the feature table (generative-model target)

**Scope (files to touch).**
- `src/matchodds/features/pipeline.py`: carry `ft_home_goals` / `ft_away_goals` through `build_feature_table` as **label-side passthrough** columns (alongside `result`), mirroring the odds passthrough. They are the generative model's targets — **never** read as features. `features()` (serving) is unchanged (no goals at request time).
- `docs/feature-pipeline.md`: add the two columns to the schema table with a note that they are generative-model targets, not features.
- `tests/unit/test_features_table.py`: update the column-order assertion; assert the goals carry the right values and are absent from `default_accumulators()` feature names.

**Acceptance criteria.**
- Feature-table column order becomes `identifiers → features → odds → ft_home_goals → ft_away_goals → result`, one row per match, goals matching the source.
- No feature name collides with the goal columns; all existing leakage + train/serve-equivalence guards stay green.

**Gate.** `make check` → PASS; `make test` → all green (this reopens Epic-03 pipeline code by design — flagged in Decision 7).

---

## Task 7 — `modeling/dixon_coles.py`: bivariate-Poisson goals model

**Scope (files to touch).**
- `src/matchodds/modeling/dixon_coles.py` (new): `DixonColesModel` — per-league MLE fit (`scipy.optimize.minimize`) of per-team attack/defense, a home-advantage term, and the Dixon-Coles low-score (τ) correction; optional exponential time-decay (`settings.dc_half_life_days`). `predict_proba` builds the score matrix to `settings.dc_max_goals` and sums to `[H, D, A]`. Reads goals (Task 6) at fit, identifiers (+date) at predict; unknown team → league-average strength fallback.
- `src/matchodds/config.py`: add `dc_max_goals: int = 10` and `dc_half_life_days: int | None = None`.
- `requirements.txt`: add `scipy` (currently transitive; DC makes it a direct runtime dep).
- `tests/unit/test_dixon_coles.py` (new).
- `.vulture_allowlist.py`: add `DixonColesModel` + the new config fields.

**Acceptance criteria.**
- On a synthetic league with a clearly stronger team, fitted attack/defense recover the ordering; `predict_proba` shape `(n, 3)`, rows sum to 1, order `[H, D, A]`.
- Deterministic (fixed optimiser init/seed); fits well under the test timeout; unknown team falls back without error.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 8 — `modeling/calibration.py`: calibrate the discriminative models

**Scope (files to touch).**
- `src/matchodds/modeling/calibration.py` (new): `calibrate(model, table, *, method, cv)` wrapping a discriminative `OutcomeModel`'s underlying sklearn estimator with `CalibratedClassifierCV` over the **temporal** folds from `cv.py`, returning a calibrated `OutcomeModel` that preserves `[H, D, A]`; choose `method ∈ {sigmoid, isotonic}` by CV log-loss. Dixon-Coles passes through unchanged (reliability-checked, not wrapped).
- `tests/unit/test_calibration.py` (new).
- `.vulture_allowlist.py`: add `calibrate`.

**Acceptance criteria.**
- Calibrated LR/XGB still output `(n, 3)` summing to 1; calibration uses only temporally-held-out folds.
- Calibrated CV log-loss ≤ uncalibrated on the synthetic set — or, if the fixture is too small to show improvement, assert the wrapper is applied + shape/seed determinism (document which).

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 9 — `modeling/train.py`: bake-off, select, freeze artifact; `make train` + `make repro`

**Scope (files to touch).**
- `src/matchodds/modeling/train.py` (new): load the feature table; run baseline + LR + XGBoost + Dixon-Coles through `cv.py`; compute `metrics.score_summary` per model per fold → mean; calibrate the discriminative models; select the best **deployable** model by mean CV log-loss; refit it (calibrated) on all data; `joblib.dump` → `models/v1.joblib`; write `models/v1.metadata.json` (data version, seed, per-model CV scores, selected model + calibration method, feature columns, library versions). `main()` + `__main__`.
- `makefile`: `train` → `python -m matchodds.modeling.train`; `repro` → `data` → `features` → `train`.
- `.vulture_allowlist.py`: **remove** the entries now consumed by `train.py` (metrics, cv, base, baselines, the three models, calibration).
- `tests/unit/test_train.py` (new): on a `tmp_path` models dir + synthetic feature table — artifact written + reloadable, the loaded object's `predict_proba` sums to 1; metadata has the required keys; selection prefers lower log-loss and never picks the baseline; identical metadata metrics across two runs.

**Acceptance criteria.**
- `make train` (real features — manual note) and the unit test both write a loadable `v1.joblib` + valid metadata.
- Selection is by log-loss; the baseline is scored but never shipped.
- Reproducibility: two runs on the same data produce identical metadata metrics.
- `make repro` chains data → features → train.

**Gate.** `make check` → PASS (vulture clean after the allowlist removals); `make test` → all green.

---

## Task 10 — `notebooks/02_modeling.ipynb`: narrative, 4-way comparison, reliability diagrams

**Scope (files to touch).**
- `notebooks/02_modeling.ipynb` (new): a thin narrative importing `matchodds.modeling` — load the feature table; baseline → LR → XGBoost → Dixon-Coles **temporal-CV comparison table** (log-loss / Brier / accuracy); reliability diagrams **before vs. after** calibration; show the selected model. Logic stays in the package (cells call `metrics`/`cv`/`train` helpers; no re-implemented math).
- `.github/workflows/ci.yml`: run the modeling notebook in the offline smoke. **Decision to flag:** the current committed sample (`tests/fixtures/sample/raw/E0_2324.csv`, ~2 rows) is too small to CV/train — either commit a larger offline sample (e.g. one real season CSV) or let the notebook read `cv_splits` / data size from env so the smoke runs a degenerate split. Pick one at review.

**Acceptance criteria.**
- `make nb-lint` PASS (nbqa isort/black/flake8); `make nb-run` executes `02_modeling.ipynb` headless on the offline sample.
- The comparison table + reliability diagrams render; cells stay thin (import-and-call), so notebook lint stays trivial.

**Gate.** `make check` (incl. `nb-lint`) → PASS; `make nb-run` → PASS; `make test` → all green.

---

## Definition of done (Epic 04)

`make repro` builds data → features → trains and freezes `models/v1.joblib` + `models/v1.metadata.json`, reproducing the metadata's metric scores deterministically from the pinned data version; the four approaches are compared head-to-head under temporal CV on log-loss + Brier (+ accuracy), with the best **deployable** model calibrated and shipped (the baseline scored but never selected); `02_modeling.ipynb` renders the comparison + reliability diagrams and runs headless in CI. On close-out: mark Epic 04 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-04-modeling.md`.
