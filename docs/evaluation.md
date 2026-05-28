# Evaluation

How `match-odds` measures its models, why those particular metrics, and where the headline numbers in
the README come from. The protocol lives in `src/matchodds/modeling/cv.py` + `src/matchodds/modeling/metrics.py`;
this document is the reader-facing companion.

## CV protocol — strictly temporal

Every model in the bake-off is scored under one forward-chaining
[`TimeOrderedSplit`](../src/matchodds/modeling/cv.py): each fold's training rows lie on **strictly
earlier** dates than its test rows, and a single calendar day is never split across the
train / test boundary. Five folds by default (`settings.cv_splits = 5`); a plain `KFold` or
`StratifiedKFold` would leak the future into the past and is **never** used on match data
(`CLAUDE.md` → *ML / Modeling Discipline*).

The same splitter feeds the calibration step (see *Calibration* below), so calibration data is
**always strictly later** than the base estimator's training window — leakage-free at both levels.

## Metrics — proper scoring rules first

A model that "predicts the right winner" is not enough — the probabilities must be honest. So:

- **Log-loss** (cross-entropy) — primary; the headline metric.
- **Brier score** (multiclass) — primary; cross-checks log-loss without the log's tail sensitivity.
- **Accuracy** — reported but **never optimised**. A model that improves accuracy while worsening
  log-loss is a regression, not progress (`CLAUDE.md` → *ML / Modeling Discipline*).

Selection minimises mean CV log-loss across the **deployable** candidates. The bookmaker baseline is
scored as a benchmark but is **never** the shipped artifact — the serving request carries no closing
odds, so the baseline cannot be evaluated at inference time.

Definitions and implementations:
[`matchodds.modeling.metrics`](../src/matchodds/modeling/metrics.py).

## Bake-off — current artifact

Snapshot from [`models/v1.metadata.json`](../models/v1.metadata.json), **current as of `2026-05-28`**
(re-check the date / numbers by running `cat models/v1.metadata.json` or running `make repro`):

- **Data version:** 20,294 matches across the six in-scope leagues, dates `2016-08-12 → 2026-05-21`.
- **Random seed:** 42 · **CV splits:** 5 · **Selected model:** `logistic` · **Calibration:** `sigmoid` (hold-out / prefit, see below).

| Model         | Log-loss ↓ | Brier ↓  | Accuracy ↑ | Deployable |
|---------------|-----------:|---------:|-----------:|:----------:|
| baseline      |   0.963294 | 0.572118 |   0.538249 |     no     |
| **logistic**  | **0.994531** | **0.593024** | **0.520419** | **yes (selected)** |
| xgboost       |   1.002992 | 0.598731 |   0.513558 |     yes    |
| dixon_coles   |   1.002473 | 0.598192 |   0.506645 |     yes    |

The bookmaker baseline beats every deployable model — this is the expected (and honest) result on
closing odds, which already encode the market's probability estimate. The shipped artifact is
chosen from the *deployable* set; `logistic + sigmoid` wins on both proper scoring rules.

## Calibration — hold-out (prefit), Epic 06.5

The discriminative models (`logistic`, `xgboost`) are scored in their **deployable, calibrated**
form. Calibration is built as a **leakage-free hold-out**: the final fold of `TimeOrderedSplit`
defines the split — the base estimator is fit on the fold's expanding-window training slice, then
wrapped in `sklearn.frozen.FrozenEstimator` and handed to `CalibratedClassifierCV`, which fits the
calibration regressor on the strictly-later held-out tail. The calibrator therefore only ever sees
data dated **after** the base's training window.

`method = "auto"` picks sigmoid (Platt) vs isotonic by held-out log-loss on a **temporal sub-split
of the holdout itself**; method-selection respects the same forward-chaining discipline at the
inner level. Implementation: [`matchodds.modeling.calibration`](../src/matchodds/modeling/calibration.py).

Dixon-Coles is generative and naturally calibrated, so it is reliability-*checked*, never wrapped.

> **Why this matters (the Epic 06.5 fix).** The previous ensemble-over-folds construction
> (`CalibratedClassifierCV(cv=temporal_split, ensemble=True)`) averaged per-fold submodels and
> sigmoid-squashed confident predictions enough to **invert** the real home advantage on near-even
> derbies. The hold-out scheme restored the ordering (`AEK(H)` vs `PAOK`: `0.336` broken → `0.434`
> fixed) and improved the shipped logistic model's mean CV log-loss `0.9989 → 0.9945` and
> Brier `0.5954 → 0.5930`.

Reliability diagrams (XGBoost raw vs hold-out-calibrated, plotted against observed frequency per
predicted-probability bin) live in [`notebooks/02_modeling.ipynb`](../notebooks/02_modeling.ipynb) —
re-rendered every time `make nb-run` executes the notebook, kept out of this document so they cannot
go stale relative to the shipped artifact.

## Reproducibility

`make repro` is the full chain — `data → features → train` — and is deterministic from the pinned
data version: every splitter, estimator, and calibrator takes `settings.random_seed` (default 42).
Two consecutive `make train` runs on the same data produce **identical** metrics across every
candidate (verified at the close of Epic 06.5, Task 2). A fresh-clone `make install-dev && make
repro` reproduces the metric scores recorded in `models/v1.metadata.json` exactly — not the pickled
bytes, but the metrics they imply.

For the chain of decisions behind these numbers, see the per-epic plans archived under
[`docs/history/`](history/) — Epic 04 (the original bake-off) and Epic 06.5 (the calibration fix).
