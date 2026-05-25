# PLAN — Epic 04: Modeling, Evaluation & Calibration

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test`, then stops for the user to review/commit/push.

**Branch:** `implement-modeling-evaluation-and-calibration`

**Goal.** Train, compare, calibrate, and freeze the shipped match-outcome model. Four approaches — the bookmaker-implied **baseline**, multinomial **logistic regression**, **XGBoost**, and the domain-canonical **Dixon-Coles** bivariate-Poisson goals model — are evaluated head-to-head under **strictly temporal** cross-validation on **proper scoring rules** (log-loss + Brier; accuracy secondary). The best *deployable* model is calibrated and frozen to `models/v1.joblib` (+ metadata sidecar), and `make repro` reproduces its metrics deterministically from the pinned data version. All models share one `OutcomeModel` interface and one feature table, so the code that trains is the code Epic 05 serves — no train/serve skew.

## Progress

| Task | Description                                                              | Status         | Commit |
|------|--------------------------------------------------------------------------|----------------|--------|
| 1    | `metrics.py` — proper scoring (log-loss, Brier) + accuracy + reliability helper, `[H,D,A]` convention | ✅ Done        | `c7fcd31` |
| 2    | `cv.py` — time-ordered (forward-chaining) CV splitter                    | ✅ Done        | `e2de41b` |
| 3    | `base.py` model interface + `baselines.py` bookmaker-implied baseline    | ✅ Done        | `86ba5a9` |
| 4    | `logistic.py` — multinomial logistic regression (impute + scale)         | ✅ Done        | `3a6af05` |
| 5    | `xgboost_model.py` — gradient-boosted trees (native NaN)                 | ✅ Done        | `52f70d8` |
| 6    | Carry full-time goals into the feature table (generative-model target)   | ✅ Done        | `90bc06b` |
| 7    | `dixon_coles.py` — bivariate-Poisson goals model (MLE)                   | ✅ Done        | `683d159` |
| 8    | `calibration.py` — Platt/isotonic calibration of the discriminative models | ✅ Done        | `a320d7b` |
| 9    | `train.py` — temporal-CV bake-off, freeze `v1.joblib` + metadata; `make train` + `make repro` | ✅ Done        | `b274486` |
| 10   | `02_modeling.ipynb` — narrative + 4-way comparison + reliability diagrams; wire nb gates | ✅ Done        | `5ea5d91` |

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

## Task 1 — `modeling/metrics.py`: proper scoring rules + reliability helper — ✅ Done

**Outcome.**
- `src/matchodds/modeling/metrics.py`: the fixed `CLASSES = ("H", "D", "A")` order + `encode_labels` / `decode_labels`; the proper scoring rules `log_loss` (clipped cross-entropy on the true-class probability) and `brier_score` (multiclass mean squared error, range `[0, 2]`); `accuracy` (argmax); `score_summary` bundling the three; and `reliability_curve` returning per-bin `(mean_predicted, observed_frequency, count)`. Private `_as_proba` / `_as_labels` coerce inputs and validate the `(n, 3)` shape. Pure, no RNG.
- `tests/unit/test_metrics.py`: 10 tests — `[H,D,A]` order + label round-trip, hand-computed log-loss / Brier, perfect-prediction zeros, confident-miss-stays-finite, argmax accuracy, summary keys, deterministic binning, empty-bin `NaN`, and `(n, 3)` shape rejection.
- `.vulture_allowlist.py`: added `metrics.{encode_labels, decode_labels, reliability_curve, score_summary}` (`log_loss` / `brier_score` / `accuracy` need no entry — `score_summary` consumes them in `src`).

**Decisions / deviations (recorded).**
- **`reliability_curve` is a pooled one-vs-rest curve** — the full `(n, 3)` matrix flattened, each class-probability paired with its one-hot outcome, then binned into one calibration curve over all class-probabilities (what the Task 10 reliability diagrams want).
- Added defensive shape validation (`_as_proba` raises on a non-`(n, 3)` matrix) beyond the spec — documents the contract; tested.

**Verification.** `make check` → PASS (mypy strict on 18 files, vulture clean); `make test` → **78 passed** (68 prior + 10 metrics). ✅

---

## Task 2 — `modeling/cv.py`: time-ordered cross-validation splitter — ✅ Done

**Outcome.**
- `src/matchodds/modeling/cv.py`: `TimeOrderedSplit` — forward-chaining / expanding-window temporal CV. It partitions the distinct dates into `n_splits + 1` contiguous groups (the first train-only, each later one a test fold), cutting with `searchsorted(side="right")` on each group's last date so a day is never split across the boundary and every test fold lies on strictly later dates than its training rows. Constructed *with* the dates, so an instance doubles as a scikit-learn `cv` object (`split` / `get_n_splits` ignore their args).
- `src/matchodds/config.py`: added `cv_splits: int = 5` (consumed by `cv.py` → no allowlist entry).
- `tests/unit/test_cv.py`: 8 tests — strict temporal order + expanding window, day-never-split, unsorted-input raises, too-few-dates raises, `n_splits >= 2`, default-from-settings, determinism + reusable `split()`, and the real synthetic-fixture dates.
- `.vulture_allowlist.py`: added `cv.TimeOrderedSplit` + its `split` / `get_n_splits` (test-only consumers until Tasks 8/9).

**Decisions / deviations (recorded).**
- None structural. The **construct-with-dates** design (a `PredefinedSplit`-style splitter) is what lets the instance be passed straight to `CalibratedClassifierCV(cv=...)` in Task 8; the scikit-learn protocol args (`_x` / `_y` / `_groups`) are `_`-prefixed so vulture ignores them (the elo-task pattern). Dates are normalised via `pd.to_datetime`, so a `date` column loads cleanly whether parquet yields `Timestamp` or python `date`.

**Verification.** `make check` → PASS (mypy strict on 19 files, vulture clean); `make test` → **86 passed** (78 prior + 8 CV). ✅

---

## Task 3 — `modeling/base.py` interface + `modeling/baselines.py` bookmaker baseline — ✅ Done

**Outcome.**
- `src/matchodds/modeling/base.py`: the `OutcomeModel` `Protocol` (`fit(table) -> OutcomeModel`, `predict_proba(table) -> (n, 3)` in `[H, D, A]`; positional-only args, docstring + `...` stub bodies mirroring `FeatureAccumulator`) and `feature_columns()`, read from the live accumulator set so it tracks the feature table.
- `src/matchodds/modeling/baselines.py`: `BookmakerBaseline` — `1 / odds` normalised to drop the overround → `[H, D, A]`; stateless `fit` returns `self`; missing odds → all-`NaN` row. Documented benchmark-only (serving carries no odds, so it is never shipped).
- `tests/unit/test_base.py`: 1 test — `feature_columns()` equals the pipeline features, length 30, one column per family present, identifiers / odds / goals / label excluded.
- `tests/unit/test_baselines.py`: 4 tests — overround removal to `[3/7, 2/7, 2/7]`, rows sum to 1 in home/draw/away order, no-op `fit` returns self, missing-odds NaN row.
- `.vulture_allowlist.py`: added `base.OutcomeModel.fit` / `.predict_proba`, `base.feature_columns`, `baselines.BookmakerBaseline` (first src callers land in Tasks 4-5 / 9; `feature_columns` drops in Task 4).

**Decisions / deviations (recorded).**
- **`BookmakerBaseline` explicitly subclasses the `OutcomeModel` Protocol** rather than conforming only structurally — this makes mypy strict verify conformance now (the acceptance criterion) and sets a readable `class XModel(OutcomeModel)` convention for Tasks 4-7. Differs from the feature-accumulator structural pattern.
- **Added `tests/unit/test_base.py`** (the spec listed only `test_baselines.py`) — the natural home for the `base` module's `feature_columns` test.
- **Missing-odds → all-`NaN` row** (the documented + tested choice of the two the spec offered).

**Verification.** `make check` → PASS (mypy strict on 21 files, vulture clean); `make test` → **91 passed** (86 prior + 5: 1 base + 4 baselines). ✅

---

## Task 4 — `modeling/logistic.py`: multinomial logistic regression — ✅ Done

**Outcome.**
- `src/matchodds/modeling/logistic.py`: `LogisticModel(base.OutcomeModel)` wrapping a sklearn `Pipeline` = `SimpleImputer(strategy="median", keep_empty_features=True)` → `StandardScaler` → `LogisticRegression(max_iter=1000, random_state=settings.random_seed)`. `fit` selects `feature_columns()` + encodes the label; `predict_proba` scatters the fitted `classes_` back into a full `(n, 3)` `[H, D, A]` array (a fold missing an outcome still returns three columns).
- `pyproject.toml`: added `sklearn.*` to the mypy `ignore_missing_imports` override.
- `tests/unit/test_logistic.py`: 3 tests — `(n, 3)` shape + rows sum to 1 + non-negative; NaN cold-start rows imputed (no error, no NaN out); deterministic across two seeded fits.
- `.vulture_allowlist.py`: added `logistic.LogisticModel`; removed the now-consumed `base.*` entries (and the `base` import) — `logistic.py` calls `feature_columns()` and accesses `.fit` / `.predict_proba`.

**Decisions / deviations (recorded).**
- **Added `sklearn.*` to mypy `ignore_missing_imports`** — scikit-learn 1.8.0 ships no `py.typed`, so strict mypy cannot import it. The CLAUDE.md-sanctioned option-2 (project-wide config) path for a stub-less library.
- **No `multi_class="multinomial"` argument** — sklearn 1.8 *removed* that parameter; the default lbfgs solver already fits multinomial for multiclass targets, so multinomial behaviour comes from the default (the plan's intent) rather than the removed arg.
- **`SimpleImputer(keep_empty_features=True)`** beyond the spec — keeps an all-NaN column (a possible early-fold cold start in Task 9's CV) so the column layout stays fixed.

**Verification.** `make check` → PASS (mypy strict on 22 files, vulture clean); `make test` → **94 passed** (91 prior + 3 logistic). ✅

---

## Task 5 — `modeling/xgboost_model.py`: gradient-boosted trees — ✅ Done

**Outcome.**
- `src/matchodds/modeling/xgboost_model.py`: `XGBoostModel(base.OutcomeModel)` over `XGBClassifier(objective="multi:softprob", n_estimators=200, max_depth=3, learning_rate=0.1, n_jobs=1, random_state=settings.random_seed)`. No imputer — `NaN` cold-start features go straight to the booster (native default-direction handling); `predict_proba` uses the same `classes_`-scatter to a full `(n, 3)` `[H, D, A]` array as the logistic model.
- `tests/unit/test_xgboost_model.py`: 3 tests — `(n, 3)` shape + rows sum to 1 + non-negative; NaN features accepted natively (no NaN out); exact determinism across two seeded fits.
- `.vulture_allowlist.py`: added `xgboost_model.XGBoostModel`.

**Decisions / deviations (recorded).**
- **No explicit `num_class=3`** — `XGBClassifier` derives the class count from the labels at fit time and manages it internally; passing `num_class` conflicts with the sklearn wrapper. Class alignment is handled by the `classes_`-scatter, so it stays robust to a fold missing an outcome.
- **Fixed hyperparameters in code** (`n_estimators=200`, `max_depth=3`, `learning_rate=0.1`) — a small, seeded set per Decision 11; any real tuning is a notebook concern, not this epic's.

**Verification.** `make check` → PASS (mypy strict on 23 files, vulture clean); `make test` → **97 passed** (94 prior + 3 XGBoost). ✅

---

## Task 6 — Carry full-time goals into the feature table (generative-model target) — ✅ Done

**Outcome.**
- `src/matchodds/features/pipeline.py`: added `_GOALS = ["ft_home_goals", "ft_away_goals"]`, populated from the typed `MatchRow` in each `build_feature_table` row, between odds and label — column order is now `identifiers → features → odds → ft_home_goals → ft_away_goals → result`. `features()` (serving) is untouched (no goals at request time). Module docstring updated.
- `docs/feature-pipeline.md`: schema intro now reads `… → odds passthrough → goals passthrough → label`; added a Goals row to the schema table and a paragraph explaining goals are a generative-model target, never a feature, and absent from `features()`.
- `tests/unit/test_features_table.py`: updated the column-order assertion; new `test_full_time_goals_pass_through_on_the_label_side` (goals excluded from the features, values match the source in chronological order).
- `tests/unit/test_pipeline.py`: updated the no-accumulator driver's column assertion to include the goal columns.

**Decisions / deviations (recorded).**
- Goal values come from the typed `MatchRow` (`match.ft_home_goals`), not `record.get(...)` — already parsed to `int`, mirroring how the label uses `match.result`.
- Also had to update **`test_pipeline.py`**'s no-accumulator column assertion (the spec only named `test_features_table.py`) — an expected consequence of the schema change, caught by the gate.

**Verification.** `make check` → PASS (mypy strict on 23 files, vulture clean); `make test` → **98 passed** (97 prior + 1 goals passthrough); real-data `make features` → **20,294 rows, 40 columns** with the goals on the label side. ✅

---

## Task 7 — `modeling/dixon_coles.py`: bivariate-Poisson goals model — ✅ Done

**Outcome.**
- `src/matchodds/modeling/dixon_coles.py`: `DixonColesModel(base.OutcomeModel)` — per-league MLE (`scipy.optimize.minimize`, L-BFGS-B) of per-team attack/defense, a home-advantage term, and the low-score (τ/ρ) correction, with optional exponential time-decay (`dc_half_life_days`). `predict_proba` builds the Poisson score matrix to `dc_max_goals`, applies the τ correction to the four dependent cells, then sums the lower-triangle / diagonal / upper-triangle to a normalised `[H, D, A]` row. Reads goals at fit, identifiers only at predict. The private `_LeagueFit` dataclass holds a league's params + the averages used for fallback; `_tau` / `_poisson_pmf` / `_fit_league` / `_outcome_probabilities` are the module-level numeric helpers.
- `src/matchodds/config.py`: added `dc_max_goals: int = 10` and `dc_half_life_days: int | None = None` (consumed by `dixon_coles.py` → no allowlist entry, per the Task-2 convention).
- `pyproject.toml`: added `scipy.*` to the mypy `ignore_missing_imports` override (scipy ships no `py.typed` — the CLAUDE.md option-2 path, same as sklearn/xgboost).
- `requirements.txt`: `scipy>=1.13,<2` promoted from transitive to a direct runtime dep (the service may ship DC).
- `tests/unit/test_dixon_coles.py`: 5 tests — shape/sum-to-one on the real feature table; attack **and** defense ordering recovered on a hand-built hierarchy league; `[H, D, A]` order via predictions (strong-at-home → home win, strong-away → away win); unseen-team **and** unseen-league fallback without error; deterministic refit (`assert_array_equal`).
- `.vulture_allowlist.py`: added `dixon_coles.DixonColesModel` (test-only consumer until `train.py`, Task 9).

**Decisions / deviations (recorded).**
- **Config fields not added to the allowlist** (the spec said to). They are read in `dixon_coles.py`, so they have a real `src` consumer — the Task-2 `cv_splits` precedent (consumed in `src` → no entry). Only `DixonColesModel` needed one.
- **Two-sided fallback** beyond the spec's "unknown team": an unseen *team* falls back to its league's average strength, an unseen *league* to the global average across fitted leagues (`_global_fallback`), initialised to a neutral team so even a pre-`fit` predict can't error. Both are tested.
- **Attack ridge pinned by post-fit re-centring.** Adding `c` to every attack and subtracting it from every defense leaves the rates (and predictions) unchanged — a one-dimensional ridge. Rather than constrain the optimiser, attack is re-centred to mean 0 after the fit; predictions are invariant, parameters stay interpretable, and the league-average fallback becomes a true neutral team. Determinism comes from a fixed zero/`HOME_ADV_INIT` start (no RNG), so `random_seed` is unused here.
- **τ-positivity guarded numerically:** ρ bounded to `±0.99` in the optimiser and τ clipped to `1e-10` before the log, so the 0-0 cell can't drive the likelihood to `NaN`; the corrected score-matrix cells are clipped to `≥0` before summing.

**Verification.** `make check` → PASS (mypy strict on 24 files, vulture clean); `make test` → **103 passed** (98 prior + 5 Dixon-Coles). ✅

---

## Task 8 — `modeling/calibration.py`: calibrate the discriminative models — ✅ Done

**Outcome.**
- `src/matchodds/modeling/calibration.py`: `calibrate(model, table, *, method="auto", cv)` clones the discriminative model's underlying sklearn estimator, wraps it in `CalibratedClassifierCV` over the **temporal** folds from `cv.py`, and returns a calibrated `OutcomeModel` (`_CalibratedModel`) preserving `[H, D, A]` via the same `classes_`-scatter as the wrapped models. `method="auto"` picks sigmoid vs isotonic by **nested** temporal CV (`_select_method`): the inner calibration split is rebuilt from each outer training fold's own dates, so both levels stay forward-chaining. A `Discriminative` `Protocol` types the `estimator` handle so mypy verifies conformance without reaching into privates.
- `src/matchodds/modeling/logistic.py` + `xgboost_model.py`: added a one-line read-only `estimator` property exposing the underlying sklearn estimator (the calibration handle).
- `tests/unit/test_calibration.py`: 5 tests — calibrated LR and XGB output valid `(n, 3)` grids summing to 1; the shipped object wraps a `CalibratedClassifierCV` fit with our `TimeOrderedSplit` (never a random KFold); `auto` selects one of the two methods; deterministic across two calibrations.
- `.vulture_allowlist.py`: added `calibration.calibrate` + `calibration._CalibratedModel.method` (test-only consumers until `train.py`, Task 9). The `estimator` property has a real `src` consumer (calibration.py) → no entry.

**Decisions / deviations (recorded).**
- **Method-selection = "auto" inside `calibrate`** (chosen at review over deferring to Task 9). Selection runs both methods through nested temporal CV and picks the lower mean held-out log-loss.
- **Auto-selection hardened for small folds** (beyond the spec). `CalibratedClassifierCV` requires ≥ `n_splits` examples *per class*, which tiny inner folds violate (it raised on the synthetic). `_inner_cv` now caps the inner `n_splits` by the rarest class's count **and** the distinct-date count, skips any outer fold whose training slice can't form a valid inner split, and falls back to sigmoid if none are usable.
- **`estimator` property added to two model files** outside Task 8's stated scope (chosen at review over isinstance-dispatching into privates) — the clean public handle calibration wraps.
- **`_CalibratedModel.fit` is a no-op returning self** — `calibrate` clones + fits via CV, so the returned object is already fitted; `fit` exists only for `OutcomeModel` conformance (mirrors `BookmakerBaseline`).
- **Tests assert mechanics, not improvement** — the acceptance criterion's documented escape hatch, since the synthetic fixture is too small for calibration to lower log-loss.

**Verification.** `make check` → PASS (mypy strict on 25 files, vulture clean); `make test` → **108 passed** (103 prior + 5 calibration). ✅

---

## Task 9 — `modeling/train.py`: bake-off, select, freeze artifact; `make train` + `make repro` — ✅ Done

**Outcome.**
- `src/matchodds/modeling/train.py`: a `_Candidate` registry (name, factory, `deployable`, `calibrated`) drives the bake-off. `run(table, models_dir)` scores every candidate under one `TimeOrderedSplit` via `_cv_scores` (mean per-fold `score_summary`), selects the min-mean-CV-log-loss **deployable** model (`_BY_NAME` lookup), refits it on all data (`_fit_final`), `joblib.dump`s it to `models/v1.joblib`, and writes `models/v1.metadata.json`. Discriminative models are evaluated in their calibrated, deployable form (`_fit_for_eval` → `_fit_calibrated`, with a raw-estimator fallback when a cold-start fold is too small to calibrate). `_load_features` + `main()` + `__main__` wire the real-data path; metadata carries seed, cv_splits, data version (`_data_version` from the meta sidecars), per-model CV scores + `deployable` flags, selected model, calibration method, feature columns, and `_library_versions`.
- `src/matchodds/modeling/calibration.py`: renamed `_inner_cv` → public `safe_calibration_cv` (shared by `_select_method` and train.py); added the `CalibratedOutcome` `Protocol` so `calibrate`'s return surfaces the chosen `method` with full typing.
- `src/matchodds/modeling/dixon_coles.py`: **bugfix** — cap the Poisson rates at `max_goals` in `_outcome_probabilities`, so a degenerate MLE on a tiny CV fold can't underflow the truncated score matrix to all-zeros (a `ZeroDivisionError` train.py's small-fold CV surfaced). Real rates (~1-3) are far below the cap and untouched.
- `makefile`: `train` → `python -m matchodds.modeling.train`; `repro: data features train`.
- `pyproject.toml`: added `joblib.*` to the mypy `ignore_missing_imports` override (no `py.typed`).
- `.vulture_allowlist.py`: removed every entry now consumed by train.py (the three models, baseline, `calibrate`, `_CalibratedModel.method`, `encode_labels`/`score_summary`, `TimeOrderedSplit`/`split`); kept only `metrics.decode_labels`, `metrics.reliability_curve` (Task 10 notebook) and `cv.TimeOrderedSplit.get_n_splits` (sklearn-protocol only).
- `tests/unit/test_train.py`: 3 tests on a `tmp_path` dir + synthetic table — reloadable artifact whose `predict_proba` sums to 1 + required metadata keys; selection is the lowest-log-loss deployable and never the baseline; identical metadata metrics across two runs.

**Decisions / deviations (recorded).**
- **Per-fold calibration uses fixed sigmoid; the final shipped model uses `method="auto"`** (sigmoid vs isotonic chosen by CV log-loss on all data, recorded in metadata). Running auto's nested selection inside every bake-off fold would be triple-nested and slow; sigmoid is the robust per-fold default. (Fills an unspecified detail in the plan — flagged at review.)
- **`calibration.py` touched outside the stated scope** — exposed `safe_calibration_cv` and added the `CalibratedOutcome` protocol, so train.py builds valid per-fold/final calibration splits and reads the chosen method with static typing (which let `_CalibratedModel.method` leave the allowlist rather than stay).
- **Dixon-Coles rate cap** is a real latent-bug fix in Task-7 code, surfaced by train.py's small-fold CV (not a train-only concern).
- **`make train` not executed for real** — it needs `make features` (network + data) first; per the acceptance criterion real `make train` is a manual note, and the unit test covers the logic offline.
- **`base`/`cv` allowlist:** `base.*` was already removed (Task 4); `cv.get_n_splits` stays (never called by our `src`, only by sklearn at runtime).

**Verification.** `make check` → PASS (mypy strict on 26 files, vulture clean); `make test` → **111 passed** (108 prior + 3 train); `make -n repro` chains data → features → train. ✅

---

## Task 10 — `notebooks/02_modeling.ipynb`: narrative, 4-way comparison, reliability diagrams — ✅ Done

**Outcome.**
- `notebooks/02_modeling.ipynb`: a thin narrative over `matchodds.modeling` — builds the feature table in-process (`pipeline.build(write=False)`), runs the temporal-CV bake-off via `train.run()` and renders the 4-way comparison table (log-loss / Brier / accuracy + `deployable` flag, sorted by log-loss) with the selected model + calibration method, then plots **XGBoost reliability before vs. after** sigmoid calibration on the final temporal fold (`metrics.reliability_curve`). Cells import-and-call only — no re-implemented math.
- `tests/fixtures/sample/raw/E0_2324.csv`: replaced the 2-match stub with a **deterministic synthetic E0 season** — 10 real EPL teams (all in the canonical registry), full double round-robin = 90 matches over 18 match-days, goals drawn from strength-based Poisson rates (seed 42), odds derived from the true probabilities (so the baseline is a genuine benchmark), matches **date-shuffled** so results are class-balanced across the season. The processed parquet/meta stay gitignored and are regenerated by the CI smoke's `matches.build()`.
- `.github/workflows/ci.yml`: renamed the smoke step to "Notebook smoke (offline sample)" + comment; it builds the sample master table then `make nb-run` executes both notebooks (01_data + 02_modeling) headless.
- `src/matchodds/modeling/calibration.py`: **robustness fix** to `safe_calibration_cv` — also reject a split where any fold's *training* portion misses a class (an expanding-window fold over early-season dates would fit a single-class XGBoost/LR base estimator and raise); `train.py` already falls back to the raw estimator when it returns `None`.

**Decisions / deviations (recorded).**
- **Offline-data decision → synthetic-season fixture** (chosen at review over a real-season CSV or env-degraded split): no network, deterministic, repo stays self-contained; both notebook smokes run the real pipeline. On the sample the bake-off selects **xgboost (sigmoid)** and calibration improves held-out log-loss (1.37 → 1.02); the bookmaker baseline scores best overall (its odds were generated from the truth) but is correctly excluded as non-deployable — a clean illustration of the benchmark.
- **`calibration.py` touched again** (Task 8/9 code): the larger sample surfaced the single-class-training-subfold crash; the fold-coverage guard is the fix. Existing tests unaffected.
- **Cell `id` warning left as-is** — `01_data.ipynb` also omits cell ids; matched the existing convention rather than diverge (it is a warning, not an error).
- **Sample generator not committed** — matches the existing `synthetic_matches.csv` convention (static committed fixture, one-off seeded script).

**Verification.** `make check` → PASS (mypy strict on 26 files, vulture clean, `nb-lint` clean); `make nb-run` (sample env) → both notebooks execute 100% headless; `make test` → **111 passed**.

---

## Definition of done (Epic 04)

`make repro` builds data → features → trains and freezes `models/v1.joblib` + `models/v1.metadata.json`, reproducing the metadata's metric scores deterministically from the pinned data version; the four approaches are compared head-to-head under temporal CV on log-loss + Brier (+ accuracy), with the best **deployable** model calibrated and shipped (the baseline scored but never selected); `02_modeling.ipynb` renders the comparison + reliability diagrams and runs headless in CI. On close-out: mark Epic 04 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-04-modeling.md`.
