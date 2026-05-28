# PLAN — Epic 06.5: Calibration Fix & Model Re-selection

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test` (+ `make nb-lint` / `make nb-run` when notebooks change), then stops for the user to review/commit/push.

**Branch:** `implement-recalibration`

**Goal.** Fix the calibration step that over-flattens probabilities and **inverts home advantage** in near-even matchups (the reversed-derby behaviour found while exercising the Epic 06 demo). Replace the ensemble-over-temporal-folds `CalibratedClassifierCV` with a **leakage-free hold-out (prefit) calibration** that preserves sharpness, **re-run the proper-scoring model selection** across the recalibrated logistic + XGBoost and the naturally-calibrated Dixon-Coles, and **re-freeze `models/v1.joblib`** — judged on log-loss/Brier (primary) with a **home-advantage sanity guardrail**. Calibration-only: no feature-pipeline, serving, or demo changes.

## Diagnosis (why this epic exists)

The deployed logistic model is calibrated via `CalibratedClassifierCV` over forward-chaining temporal folds (`ensemble=True` — the only mode compatible with a non-partitioning temporal CV). That **averages fold-subset submodels and sigmoid-squashes**, flattening confident predictions toward the base rate — enough to reverse the real, data-confirmed home advantage (Greek big-4 derbies: **44% home / 25% away**) for elite-vs-elite fixtures. Measured on AEK(H) vs PAOK: raw logistic **0.472** home → deployed **0.336** (away favoured). A leakage-free **hold-out (prefit)** calibration restores the correct ordering in **both** mid-season and season-opener regimes (AEK home 0.45 / 0.43), with **no feature-pipeline change** needed. `predict_proba` column order, label encoding, and the Elo / strength / form / h2h features were all verified correct — the defect is solely in the calibration construction.

## Progress

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| 1 | Hold-out (prefit) calibration in `calibration.py` + updated calibration tests (incl. a home-advantage no-inversion guard) | ✅ Done | — |
| 2 | Re-train → re-select → re-freeze `models/v1.joblib` + `v1.metadata.json`; validate log-loss/Brier ≥ parity and the home-advantage sanity | ⬜ Not started | — |
| 3 | Refresh `notebooks/02_modeling.ipynb` calibration narrative + reliability diagrams | ⬜ Not started | — |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Task 1 changes calibration internals while preserving the `calibrate()` interface, so `train.py` and the (old) committed artifact stay valid and serving tests stay green; Task 2 regenerates the artifact so code and artifact re-converge; Task 3 keeps `nb-lint` / `nb-run` green.

**Consolidation note (flag at review).** Tasks 1–2 (the fix + the retrain it enables) could merge into one "recalibrate + re-freeze" commit, but splitting keeps the code change reviewable before the binary artifact churns. Task 3 (notebook narrative) is deferrable to Epic 08 (docs) if you'd rather keep this epic code-only — flag at review.

## Decisions baked in (flag at review to change)

1. **"Let metrics decide."** Re-run the existing proper-scoring selection (CV log-loss primary) across recalibrated logistic + XGBoost + Dixon-Coles; freeze whichever wins. Home advantage is a **guardrail / regression test**, never the selection objective (CLAUDE.md → *Proper scoring rules first*).
2. **Hold-out (prefit) calibration.** Use the **final fold** of the existing `TimeOrderedSplit` as a temporal hold-out: fit the base estimator on the fold's (expanding-window) training slice, calibrate it on the strictly-later held-out tail via `sklearn.frozen.FrozenEstimator` + `CalibratedClassifierCV`. Leakage-free (the calibrator only sees data after the base's training window). Keeps sigmoid-vs-isotonic auto-selection, the `[H, D, A]` output contract, determinism, and the small-data → raw-estimator fallback. The `calibrate(model, table, *, method, cv)` signature is preserved, so **`train.py` needs no change**.
3. **Artifact stays `v1.joblib`.** The re-frozen model overwrites `models/v1.joblib` + `v1.metadata.json` (same filename) so the **frozen serving layer is untouched** — no `v2`. Serving is model-agnostic (`predict_proba`), so a different *selected* model ships transparently.
4. **Calibration-only.** No feature-pipeline change (the season-scoped strength reset stays; hold-out calibration fixes the inversion in both mid-season and season-opener regimes — verified). No serving or demo change.
5. **Determinism preserved.** Every split / estimator / calibrator stays seeded; `make repro` on the current pinned data snapshot reproduces the re-frozen artifact's metric scores.

## Out of scope

Feature-pipeline changes; serving-layer changes (frozen after Epic 05); the demo (unchanged); a model-version bump. The betting-edge notebook (Epic 07) and the README / model-card / eval-doc polish (Epic 08). A latent `TimeOrderedSplit.split(groups=…)` gap (it raises under `cross_val_predict` / `ensemble=False`) is **not exercised** by the hold-out approach and is left for a future cleanup.

---

## Task 1 — Hold-out (prefit) calibration in `calibration.py` + tests ✅

**Outcome.** `calibrate()` rewritten to the hold-out (prefit) scheme of Decision 2: it pulls the **final** `(train_idx, holdout_idx)` from the `TimeOrderedSplit`, fits the cloned base on the train slice, then fits `CalibratedClassifierCV(FrozenEstimator(base), ...)` on the strictly-later holdout — calibrator sees only post-training-window data. `_select_method` ("auto") now scores sigmoid vs isotonic on a temporal sub-split of the **holdout itself** (last fold of `safe_calibration_cv(holdout)`), falling back to sigmoid if the holdout is too small to sub-split. `safe_calibration_cv` gained the held-out-classes guard (the final fold's test slice must carry all three outcomes, in addition to the existing per-fold-train check). `_CalibratedModel`'s `[H, D, A]` scatter, the `calibrate(...)` signature, determinism, and the small-data → raw fallback contract are all preserved — `train.py` unchanged, the committed (old) `v1.joblib` still loads, serving tests stay green.

Tests rewritten: the wrapper test now asserts the new structure (`CalibratedClassifierCV` whose inner estimator is a `FrozenEstimator`; the last fold's holdout dates are strictly after the train slice's). The valid-grid / auto-method / determinism tests are kept. Added the Epic-06.5 home-advantage no-inversion guard: on a strongly-home-biased fixture, the calibrated model keeps `P(home) > P(away)` for a perfectly balanced feature row (every feature at its training-set median).

**Deviations (with reason).**
1. **`cv=[(arange(n), arange(n))]` passed explicitly to `CalibratedClassifierCV`.** sklearn 1.8's `FrozenEstimator` + `CalibratedClassifierCV` path still routes through `cross_val_predict` internally (the docstring talks about "the prefit pattern", but `_ensemble="auto"` only flips `ensemble=False` — it does not skip CV). The default 5-fold `StratifiedKFold` then runs its per-fold class-diversity check and raises on tiny / class-imbalanced holdouts. A trivial single-fold cv (all indices in both train and test) bypasses that check; under a frozen base the fold-level fit is a no-op so the predictions are exact, and calibration proceeds as the prefit-style fit on the full holdout that the docstring promises. Not in PLAN's Decision 2 wording but a necessary implementation detail.
2. **`tests/fixtures/home_advantage_matches.csv` (new).** 90 matches over 30 dates (6 teams × 3 seasons, round-robin home/away, 51/22/17 H/D/A from a 60/20/20 outcome bag). Generated deterministically with random seed 2; seed 42 (the first try, matching `config.RANDOM_SEED`) left no Draws in the final fold's holdout, tripping `safe_calibration_cv`'s new all-classes guard inside the calibration test. Seed 2 is the first low-numbered seed that yields all three classes in the last fold with a healthy draw count.

**Gate.** `make check` → PASS · `make test` → 141 passed.

---

## Task 2 — Re-train, re-select, re-freeze the artifact

**Scope (files to touch).**
- Run `make train` (`matchodds.modeling.train`) on the current real feature table → regenerate `models/v1.joblib` + `models/v1.metadata.json` with hold-out calibration; the bake-off re-selects on CV log-loss across recalibrated logistic + XGBoost + Dixon-Coles. Any minimal `train.py` / metadata tweak only if required (none expected — Decision 2).
- Commit the regenerated `models/v1.joblib` + `v1.metadata.json`.

**Acceptance criteria.**
- `make train` is deterministic; `make repro` (data → features → train) reproduces the metadata metrics.
- **Metric parity:** the new artifact's mean CV **log-loss / Brier do not regress** versus the previous metadata (record before/after in the post-task summary; an improvement is expected from removing the over-flattening).
- **Home-advantage sanity:** the re-frozen shipped model gives `P(home) > P(away)` for clear home favourites and sensible derby orderings (spot-check AEK/PAOK, Olympiakos/PAOK via the serving `Inference` path) — reported in the summary.
- `make check` + `make test` green (`test_train.py` structural + determinism tests still pass).

**Gate.** `make check` → PASS; `make test` → all green; metric before/after + home-advantage spot-check in the post-task summary.

---

## Task 3 — Refresh `notebooks/02_modeling.ipynb`

**Scope (files to touch).**
- `notebooks/02_modeling.ipynb`: update the calibration narrative (now hold-out / prefit, not ensemble-over-folds) and re-render the reliability diagrams + bake-off table against the recalibrated models; add a one-line note on the home-advantage fix. Re-execute headless.

**Acceptance criteria.**
- `make nb-lint` + `make nb-run` green; the notebook's calibration story matches the shipped code; `make check` + `make test` stay green.

**Gate.** `make check` → PASS; `make test` → green; `make nb-lint` + `make nb-run` → green.

---

## Definition of done (Epic 06.5)

The shipped `models/v1.joblib` uses leakage-free hold-out calibration; the home-advantage inversion is gone (clear home favourites satisfy `P(home) > P(away)`, derbies read sensibly) with no regression in log-loss / Brier; `make repro` reproduces the artifact; the modeling notebook reflects the new calibration; `make check` + `make test` + `nb-lint` / `nb-run` are green. On close-out: mark Epic 06.5 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-06.5-recalibration.md`.
