# PLAN — Epic 07: Betting-Edge Analysis (optional appendix)

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test` (+ `make nb-lint` / `make nb-run` when notebooks change), then stops for the user to review/commit/push.

**Branch:** `epic-07-betting`

**Goal.** A **historical, retrospective, illustrative-only** comparison of the shipped model's probabilities against bookmaker closing odds: extract implied probabilities (overround removed), measure per-match edge, and backtest a flat-stake positive-EV strategy on out-of-sample matches. The reusable math (overround removal, edge / EV, backtest summary) lives in `matchodds.modeling.betting` with tests; the narrative + plots live in `notebooks/03_betting_edge.ipynb`. **No live odds, no bet-placement code, no real-money framing** (CLAUDE.md → *Constraints*).

## Progress

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| 1 | `matchodds.modeling.betting`: overround removal + per-match edge / EV + unit tests | ✅ Done | — |
| 2 | `matchodds.modeling.betting`: flat-stake positive-EV backtest summary + unit tests | ⬜ Not started | — |
| 3 | `notebooks/03_betting_edge.ipynb` narrative + caveats; `make nb-lint` / `make nb-run` green | ⬜ Not started | — |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Tasks 1–2 are pure additions to a new file under `src/matchodds/modeling/` — they do not touch the frozen Epic 04 modeling code or the shipped `models/v1.joblib`. Task 3 adds a new notebook and keeps `make nb-lint` + `make nb-run` green.

**Consolidation note (flag at review).** Tasks 1–2 (the reusable math + the backtest summary on top of it) could merge into a single "betting math" commit, but splitting keeps the per-match helpers reviewable before the backtest scaffolding lands on top. The optional-appendix nature of this epic also makes it easy to ship just Tasks 1–2 (the importable code + tests) and defer Task 3 (the notebook) — flag at review.

## Decisions baked in (flag at review to change)

1. **Module location: `matchodds.modeling.betting` (single new file).** Matches the existing modeling-adjacent helpers (`metrics.py`, `calibration.py`, `baselines.py`) and the MASTER_PLAN note *"any reusable math lifted into `matchodds.modeling` (e.g. overround removal) with tests"*. Not a new top-level subpackage — the surface area for an optional appendix should stay small.
2. **No edits to `matchodds.modeling.baselines`.** `BookmakerBaseline` already turns odds into probabilities for the Epic 04 bake-off and is part of the frozen-artifact-adjacent code path; the new `betting.py` reimplements the (tiny) overround removal as its own pure helper rather than refactoring the baseline to share it. Any de-duplication can be a follow-up cleanup outside this epic.
3. **Out-of-sample backtest by forward-chained refit.** The notebook refits the **selected** model (currently logistic, per `models/v1.metadata.json`) on each `TimeOrderedSplit` fold's training slice and accumulates predictions on the strictly-later test slice, so the backtest is genuinely out-of-sample. The shipped `models/v1.joblib` is not used for backtesting (it was fit on all data — in-sample). Cheap because the selected model is logistic; if XGBoost ever wins the bake-off, the per-fold refit cost is still <1 minute on the full feature table.
4. **Bet selection: flat stake on EV > `edge_threshold` (default 0).** Plan-scope is *"a backtest of a flat-stake positive-EV strategy"* — no Kelly, no fractional sizing, no compounding. The notebook reports a small sensitivity table (thresholds 0% / 2% / 5%) on top of the helper; the helper itself takes a single threshold and stays simple.
5. **Closing odds (the columns Epic 02 already loaded).** No new ingestion. The backtest uses the `odds_home` / `odds_draw` / `odds_away` columns already in `data/processed/matches.parquet`; rows whose odds are missing or invalid (e.g. ≤ 1.0) are excluded with a per-fold count reported in the summary.
6. **Caveats stated up-front in the notebook.** A markdown block at the very top: historical-only, closing odds (not live), bookmaker hold ignored beyond the implied-probability removal, transaction costs / limits / line moves ignored, small-sample variance flagged, no real-money implication. Mirrors CLAUDE.md → *"No live betting / real-money framing"*.

## Out of scope

Live-odds ingestion (forbidden by CLAUDE.md); bet placement code; Kelly or fractional staking; in-play / streaming odds; a portfolio-level allocator; the `BookmakerBaseline` refactor noted in Decision 2; any change to the shipped `models/v1.joblib` or the serving layer. Epic 08 (docs / model card / README pass) is separately tracked.

---

## Task 1 — `matchodds.modeling.betting`: overround + edge / EV + tests ✅

**Outcome.** Added `src/matchodds/modeling/betting.py` with the three pure helpers exactly as scoped:

- `implied_probabilities(odds)` — `1 / odds` row-normalised; overround removed. Strict on invalid input: raises `ValueError` on a row that isn't all finite and `> 1.0`.
- `edge(model_prob, market_odds)` — `model_prob − implied_prob`, per outcome.
- `expected_value(model_prob, market_odds)` — `P · odds − 1` per unit stake; positive ⇒ favourable bet. This is the bet-selection criterion Task 2's backtest will key off, since the same edge can be favourable or unfavourable depending on the payout.

Pure numerics; no I/O, no model dependency, no global state; no edits to Epic 04 code or the shipped `models/v1.joblib`. Module docstring notes the strictness as a deliberate boundary against the *"no live betting / real-money framing"* constraint in CLAUDE.md.

Added `tests/unit/test_betting.py` with 8 pinned tests: overround removal on the same `(2.0, 3.0, 3.0) → (3/7, 2/7, 2/7)` worked example as `test_baselines.py`; a realistic 6.6% bookmaker hold on `(2.0, 3.3, 3.8)`; `ValueError` for each of `1.0 / 0.5 / −1.0 / NaN / inf`; determinism; `edge` matches manual calc; `EV = P·odds − 1` pinned on a break-even case (`(0.5, 0.25, 0.25)` at `(2.0, 3.0, 3.0) → (0, −0.25, −0.25)`) and a favourable-bet case (all-positive EVs); invalid-odds rejection inherited through `expected_value`.

**Deviations (with reason).**
1. **`.vulture_allowlist.py` updated.** Added `betting.edge` and `betting.expected_value` to the forward-reference list because they have no `src/` consumer until Task 2 (backtest) / Task 3 (notebook); without this, `vulture` (run inside `make check`) reports them as dead code. `implied_probabilities` doesn't need listing since `edge` already calls it. This is the project-wide allowlist convention already used for cross-epic forward references — not a line-level suppression. Not in PLAN's Task 1 scope but a necessary `make check`-keeping consequence of landing the helpers one commit before their consumers.

**Gate.** `make check` → PASS · `make test` → 149 passed (+8 new betting tests).

---

## Task 2 — `matchodds.modeling.betting`: flat-stake EV>0 backtest summary + tests

**Scope (files to touch).**
- `src/matchodds/modeling/betting.py`: add `backtest(model_probs, market_odds, results, *, edge_threshold=0.0, stake=1.0) -> dict[str, float | int]` returning at minimum `n_matches`, `n_bets`, `n_wins`, `total_staked`, `total_return`, `pnl`, `roi`, `win_rate`. Bet selection rule: for each match, pick the single outcome (H / D / A) with the largest positive EV above `edge_threshold` and stake `stake` on it (one bet per match max); a match with no qualifying outcome contributes to `n_matches` but not to `n_bets`. Pure function — takes arrays, returns the summary; no I/O, no model orchestration.
- `tests/unit/test_betting.py`: extend with a small synthetic 1X2 fixture (a handful of matches with known model probs, odds, and outcomes) where the expected PnL, ROI, n_bets, and win_rate are pinned exactly. Edge-case tests: zero bets when `edge_threshold` is unrealistically high; deterministic across two calls; handles a row where two outcomes share the maximum EV (deterministic tie-break — e.g. lowest index wins).

**Acceptance criteria.**
- `make check` + `make test` green.
- The helper is **summary-only** (numeric returns); per-bet ledger formatting belongs in the notebook.

**Gate.** `make check` → PASS; `make test` → all green.

---

## Task 3 — `notebooks/03_betting_edge.ipynb` narrative

**Scope (files to touch).**
- `notebooks/03_betting_edge.ipynb` (new): the historical/illustrative appendix. Cells, in order:
  1. **Caveats block** (top markdown) — historical-only, closing odds, no transaction costs / limits / line moves, no real-money implication.
  2. **Setup** — load `pipeline.build(write=False)`; show the rows that carry usable closing odds (sample fixtures committed under `tests/fixtures/sample/` keep CI green when real odds are missing).
  3. **Overround removal** — apply `betting.implied_probabilities` to a small sample; show that the raw `1/odds` exceeds 1 (the overround / bookmaker hold) and the normalised vector sums to 1.
  4. **Model vs book** — refit the selected model (per `models/v1.metadata.json`, currently logistic) on each `TimeOrderedSplit` fold's training slice, accumulate predictions on the strictly-later test slice; plot the distribution of `betting.edge` across out-of-sample matches.
  5. **Flat-stake EV>0 backtest** — call `betting.backtest` on the accumulated out-of-sample predictions vs market odds, at edge thresholds 0% / 2% / 5%; render a small summary table (ROI, n_bets, win_rate per threshold) and a cumulative-PnL line plot.
  6. **Takeaways + repeat caveats** — frame everything as historical / illustrative.
- Re-execute headless. Notebook stores no outputs (repo convention — see `notebooks/02_modeling.ipynb`).

**Acceptance criteria.**
- `make nb-lint` + `make nb-run` green; the notebook imports from `matchodds.modeling.betting` (no logic re-defined inline); the caveats block + the closing markdown both restate the historical-only / no-real-money framing.
- `make check` + `make test` stay green.

**Gate.** `make check` → PASS; `make test` → green; `make nb-lint` + `make nb-run` → green.

---

## Definition of done (Epic 07)

`matchodds.modeling.betting` exposes overround removal, per-match edge / EV, and a flat-stake EV>0 backtest summary, all with pinned unit tests; `notebooks/03_betting_edge.ipynb` runs headless and tells the historical-only model-vs-book story with explicit caveats; the shipped `models/v1.joblib` and the serving layer are untouched. On close-out: mark Epic 07 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-07-betting.md`.
