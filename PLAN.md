# PLAN — Epic 08: Documentation, Model Card & Reproducibility

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test` (+ `make nb-lint` / `make nb-run` when notebooks change), then stops for the user to review/commit/push.

**Branch:** `epic-08-docs`

**Goal.** Make the repo do the recruiter work in 30 seconds and prove it reproduces end-to-end. Ship: a small **demo polish** pass (drop Streamlit's Deploy button while keeping the hamburger menu, remove the "Made with Streamlit" footer, and restore the bar-chart's H / D / A order — the three extras called out for this epic), the three documentation deep-dives the README links to (`docs/evaluation.md`, `docs/model_card.md`, `docs/architecture.md`), a **README pass** that opens with the per-role "what this demonstrates" map and ends with a verified `make repro` quickstart + a committed demo screenshot, and a **fresh-clone reproducibility check** to certify the README's claims. Plan step 2.9 (pinning the repo on the GitHub profile) is the user's manual closeout — not a code task.

## Progress

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| 1 | **Demo polish** — drop the Deploy button (keep the hamburger menu) + remove the "Made with Streamlit" footer + restore the H / D / A bar-chart order | ✅ Done | — |
| 2 | `docs/evaluation.md` — temporal-CV protocol + metric tables (recomputable via `make train`) + a link to `notebooks/02_modeling.ipynb` for reliability diagrams | ✅ Done | — |
| 3 | `docs/model_card.md` — intended use, data sources, metrics, calibration story, limitations, ethics | ✅ Done | — |
| 4 | `docs/architecture.md` — notebook → package → service data-flow diagram (Mermaid) + the leakage / determinism / train-serve-skew invariants | ✅ Done | — |
| 5 | `README.md` polish pass + committed demo screenshot + fresh-clone `make repro` reproducibility verification | ⬜ Not started | — |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Tasks 2-4 are markdown-only under `docs/`. Task 1 touches only `demo/` (configuration + a small `st.markdown` CSS injection); demo tests stay green. Task 5 may add `docs/images/demo.png` and edit `README.md` only.

**Consolidation note (flag at review).** Tasks 2-4 are three separate docs by design (each is the deep-dive for one chapter of the model story), but if you'd rather collapse them into one "all docs" commit that's a reasonable bundle. Task 1 is genuinely isolated demo work and should stay its own commit. Task 5 (README + screenshot + verification) is the final pass and is best as the last commit so the reproducibility check certifies the final state.

## Decisions baked in (flag at review to change)

1. **Demo polish via `demo/.streamlit/config.toml` (`toolbarMode = "viewer"`) + a small `demo/app.py` patch.** Streamlit's `viewer` mode hides the **developer** options (Deploy button, Rerun, Clear cache) from both the toolbar and the hamburger menu while **keeping the menu itself** with the **viewer** options end users actually find useful (Print, record screencast, theme toggle, About). The alternative `minimal` would hide the menu entirely too — that strips functionality the demo is happy to keep. The "Made with Streamlit" badge is then removed by a one-line `st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)` immediately after `st.set_page_config(...)` — Streamlit has no first-class toggle for the page-bottom footer. The bar-chart order (currently "Away | Draw | Home" because `st.bar_chart` sorts the DataFrame's string index alphabetically — A, D, H) is restored to "Home | Draw | Away" by switching to `st.altair_chart` with an explicit categorical sort order on the x-axis. Config lives **inside `demo/`** so it travels through the Dockerfile and `make up` picks it up automatically (no extra mount).
2. **Reliability diagrams stay in `notebooks/02_modeling.ipynb`; not committed as PNGs.** `docs/evaluation.md` quotes the numbers (CV log-loss / Brier / accuracy, refreshable by `make train` + reading `models/v1.metadata.json`) and links the notebook for the visual story. Avoids stale committed images and keeps the notebook the single source of truth for charts.
3. **Architecture diagram is a Mermaid flowchart inline in `docs/architecture.md`.** GitHub renders Mermaid natively in markdown — no PNG, no extra tooling, fully diffable. Cheaper to keep accurate than a checked-in image.
4. **Demo capture is an animated GIF at `docs/assets/demo.gif` (committed).** Matches the `quake-feed` precedent (`docs/assets/demo.gif` — same path, same idiom) and captures the 5–10 s interaction loop (pick league → pick teams → hit Predict → calibrated probabilities render + plain-English read) that a static frame can't. Captured with `wf-recorder -g "$(slurp)" -f docs/assets/demo.mp4` on Hyprland (the mainstream wlroots screen-capture pair) and converted with the two-pass `ffmpeg` `palettegen` / `paletteuse` pipeline at 15 fps · 820 px wide. Sized ~3–4 MB. The README links it inline via the centered `<p align="center"><img …/></p>` idiom. (Earlier PLAN draft picked PNG for weight; reversed at user request — the motion signal is worth the extra MBs.)
5. **Reproducibility verification is a documented sequence + manual checklist, not CI.** A fresh-clone `make install-dev && make repro && make up` is destructive (rewrites `models/v1.joblib`, re-downloads data) and impractical to run in the standard CI workflow; the README documents the steps and Task 5 includes a one-time manual run with the outcome (metadata metrics matched, demo loads) reported in the post-task summary. CI continues to run `make check` + `make test` + `make nb-lint` + `make nb-run` as today.
6. **Betting-edge framing stays "historical / illustrative only" everywhere.** The README, model card, and any cross-link to Epic 07 must preserve CLAUDE.md's *"no live betting / real-money framing"* constraint. The model card explicitly states this in the *Limitations* / *Out-of-scope* section.
7. **The CV integration ticket (a `data/projects.yaml` entry in `~/dev/cv`) is noted in the post-Task-5 summary as an out-of-repo follow-up.** Not implemented in this repo (different repo entirely); flagged so it doesn't get forgotten when the user starts the pin pass.

## Out of scope

CI-side reproducibility automation (would need docker-in-docker or significant CI rework — see Decision 5); GitHub Pages or a docs site; auto-generated diagrams from CI; auto-captured screenshots; a model-version bump (`v1.joblib` stays the shipped artifact); any code change outside the demo polish in Task 1. Plan step 2.9 (the user pins the repo on the GitHub profile) is an out-of-repo manual action, not a code task.

---

## Task 1 — Demo polish (Deploy button / "Made with Streamlit" footer / H/D/A bar-chart order) ✅

**Outcome.** Three polish items landed, all in `demo/`:

- **`demo/.streamlit/config.toml` (new).** `[client] toolbarMode = "viewer"` hides Streamlit's *developer* options — the Deploy button, Rerun, Clear cache — from both the top-right toolbar and the hamburger menu, while **keeping the menu itself** with the *viewer* options end users find useful (Print, record screencast, theme toggle, About). The file's comment spells out the `viewer` vs `minimal` distinction explicitly so a future reader does not "simplify" it to `minimal` and accidentally hide the menu too. Travels through the Dockerfile's existing `COPY demo/ ./demo/` line — no Dockerfile change needed (verified).
- **`demo/app.py` — footer hide.** A one-line `st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)` immediately after `st.set_page_config(...)` removes the "Made with Streamlit v1.57.0" page-bottom badge. Streamlit ships no first-class toggle; the in-file comment warns *"Do not strip as 'cruft' — without it the footer comes back"*.
- **`demo/app.py` — bar-chart order.** Replaced `st.bar_chart(chart, y="probability")` with `st.altair_chart` over a long-format `chart_data` DataFrame, pinning `alt.X("outcome:N", sort=list(_OUTCOME_LABELS), title=None)`. The explicit categorical sort overrides Vega-Lite's default alphabetical ordering (which had produced **A**way | **D**raw | **H**ome), restoring **Home | Draw | Away**. Added `import altair as alt` at the top; verified `altair 6.1.0` is already a transitive Streamlit dep — no `requirements-demo.txt` edit.

**Tests.** `demo/tests/test_app.py` was checked first — the AppTest smoke asserts on metrics and info content, not on the chart object or chart type, so it stayed untouched. All 155 tests still pass.

**Design note (worth flagging at review).** The `toolbarMode` choice went through two rounds with the user: an initial draft picked `minimal` (which hides the menu entirely — too aggressive); a clarification then dropped the config altogether (kept the full toolbar including Deploy); a second clarification surfaced the right answer — `viewer` hides Deploy + the other developer options without losing the menu. The shipped state matches the final clarification. The `viewer` choice is unambiguously documented inline in `demo/.streamlit/config.toml` for the next reader.

**Deviations.** None — final implementation matches the (final) PLAN spec.

**Gate.** `make check` → PASS · `make test` → 155 passed.

**Manual browser smoke (pending — performed by the user via `make up` / `http://localhost:8501`).** Confirm: no Deploy button in the toolbar; hamburger menu (☰) still present with Print / screencast / theme / About; no "Made with Streamlit" footer; bar chart reads **Home win → Draw → Away win** left-to-right.

---

## Task 2 — `docs/evaluation.md`: CV protocol + metric tables ✅

**Outcome.** Added `docs/evaluation.md` (91 lines, markdown-only), matching the tone and structure of the existing `docs/feature-pipeline.md`. Five sections in plan order:

- **CV protocol — strictly temporal.** `TimeOrderedSplit` semantics, 5 folds, never random; calibration's holdout is strictly later than the base estimator's training window so leakage is excluded at both levels. Inline link to `src/matchodds/modeling/cv.py`.
- **Metrics — proper scoring rules first.** Log-loss + Brier primary; accuracy reported but **never optimised** (quoting CLAUDE.md → *ML / Modeling Discipline*); bookmaker baseline scored as benchmark, never shipped because the serving request carries no closing odds. Link to `metrics.py`.
- **Bake-off — current artifact.** Built-in honesty marker (*"current as of `2026-05-28` — re-check via `cat models/v1.metadata.json` or `make repro`"*). Data version (20,294 matches, 2016-08-12 → 2026-05-21, seed 42, 5 folds) plus the four-row table with exact numbers from the current metadata (baseline 0.963294 / logistic 0.994531 [bold, selected] / xgboost 1.002992 / dixon_coles 1.002473 log-loss). One-paragraph framing notes the baseline beats every deployable model — *"this is the expected (and honest) result on closing odds, which already encode the market's probability estimate"*.
- **Calibration — hold-out (prefit), Epic 06.5.** Describes the `FrozenEstimator` + `CalibratedClassifierCV` construction, the sigmoid-vs-isotonic `auto` path on a temporal sub-split of the holdout itself, Dixon-Coles being reliability-checked rather than wrapped. Boxquoted *"Why this matters (the Epic 06.5 fix)"* sub-section with the AEK(H)-vs-PAOK regression case (`0.336 → 0.434`) and the metric improvements (`log-loss 0.9989 → 0.9945`, `Brier 0.5954 → 0.5930`). Points readers to `notebooks/02_modeling.ipynb` for reliability diagrams — kept out of `docs/` so they cannot go stale relative to the shipped artifact.
- **Reproducibility.** `make repro` is deterministic from the pinned data version; two consecutive `make train` runs produce identical metrics (verified at Epic 06.5 Task 2). Cross-links `docs/history/` for the per-epic decision trail.

No PNGs committed — reliability charts stay in the notebook (PLAN Decision 2 holds).

**Deviations.** None — sections, table, and cross-links match the plan exactly.

**Gate.** `make check` → PASS · `make test` → 155 passed.

---

## Task 3 — `docs/model_card.md`: intended use + ethics ✅

**Outcome.** Added `docs/model_card.md` (134 lines, markdown-only). Structured per the Mitchell-et-al. (2019) standard so a recruiter can jump to a section by name. Numbers and library versions read straight from `models/v1.metadata.json` (re-checkable with `cat models/v1.metadata.json` / `make repro`).

Sections, in order:

- **Model details** — logistic + sigmoid (hold-out / prefit), trained `2026-05-28`, seed `42`, 5 CV folds, full library version list (numpy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.17.1, xgboost 2.1.4, joblib 1.5.3, Python 3.14), inline links to the relevant source modules.
- **Intended use** — primary (educational portfolio piece showing notebook-to-service ML engineering), secondary (1X2 probability service via the FastAPI service / Streamlit demo), explicit "not intended for" pointer to *Limitations* + *Ethics*.
- **Data** — single source (football-data.co.uk, pinned-URL manifest, no scraping); 20,294 matches across `2016-08-12 → 2026-05-21`; six leagues with per-league counts (EPL 3,790 · La Liga 3,790 · Serie A 3,790 · Ligue 1 3,476 · Bundesliga 3,060 · Greek SL 2,388); columns kept; explicit note that bookmaker odds are used by the baseline + Epic 07 only, never at inference time.
- **Features (30 columns)** — five-row table mapping each accumulator family (Elo / Form / H2H / Venue strength / Season) to its columns and what it captures; cold-start NaN convention noted; cross-link to `docs/feature-pipeline.md` for the per-feature leakage proof.
- **Metrics** — **does not duplicate** `docs/evaluation.md`'s table (drift avoidance per PLAN Decision 2-adjacent); summarises only the selection rule (log-loss on the deployable set) and the headline outcome.
- **Calibration** — one paragraph on the Epic 06.5 hold-out / prefit construction; quotes the regression case (AEK(H) vs PAOK `0.336 → 0.434`) and the metric improvements; links the notebook for reliability diagrams.
- **Limitations** — six bullets: out-of-scope leagues / teams; cold-start fixtures; calendar-day granularity (no kickoff times → same-day matches treated as one block); "upsets are upsets" (calibrated probability ≠ prediction); no in-play modelling; six leagues / ten seasons is itself a scope choice (no OOD applicability).
- **Ethics / out of scope** — four bullets restating *no live betting / no real-money use / no advice* in three different framings; **explicitly names `notebooks/03_betting_edge.ipynb` as historical / illustrative only** and quotes its negative-ROI outcome (per PLAN Decision 6 — the disclaimer now hits the reader three times across README → model card → notebook); no bias audit was performed (with the reason given); no PII.
- **Maintainer & provenance** — maintainer + repo link, refresh command (`make repro`), versioning policy (calibration-only fixes overwrite `v1.joblib` in place), pointer to `docs/history/`.

**Deviations.** None — sections and content match the plan exactly. No duplication of `docs/evaluation.md`'s bake-off table; no committed images.

**Gate.** `make check` → PASS · `make test` → 155 passed.

---

## Task 4 — `docs/architecture.md`: data-flow diagram + invariants ✅

**Outcome.** Added `docs/architecture.md` (128 lines, markdown-only). Three top-level sections matching the plan plus a short cross-link footer:

- **Data flow.** Inline Mermaid `flowchart TD` showing the full chain `football-data.co.uk → data/raw/ → matches.parquet → features.parquet → models/v1.joblib + metadata → serving.app (POST /predict) → demo.app (Streamlit)`. Edges labelled with the `make` target and the responsible `matchodds.*` module. **Dotted arrow** from `matches.parquet` into the serving box explicitly labels the train-serve equivalence (`matchodds.features.pipeline.features(date_cutoff)`). Notebooks shown as read-only consumers (dotted in) so the reader sees they narrate but never serve. Colour-classed stores / code / external nodes (renders on GitHub).
- **Layers — every box → real code.** Eight-row table mapping each diagram node to its module path and one-line role. Modeling row cites the four bake-off candidates separately (`logistic.py` / `xgboost_model.py` / `dixon_coles.py` / `baselines.py`) plus `cv.py` / `metrics.py` / `calibration.py` so a reader can click straight to any single piece. Every link target verified against the live tree.
- **Invariants.** Three subsections in plain language, each with the specific code-level enforcement points:
  - *No leakage* — `features(date_cutoff)`'s pre-match-before-update ordering + `TimeOrderedSplit` + the Epic-06.5 hold-out calibrator.
  - *No train-serve skew* — `Inference.predict` calling the same `pipeline.features(...)` that `train.run(...)` consumes via `pipeline.build(...)`, plus `Inference._guard_no_skew` fail-fast at startup if the feature columns drift.
  - *Determinism* — `settings.random_seed = 42` flowing into every estimator / splitter / calibrator; XGBoost `n_jobs=1`; two-run reproducibility verified at Epic 06.5 Task 2. Honest caveat boxquoted: `make repro` reproduces the **metric scores** in the metadata, not necessarily the pickled bytes (which can vary across joblib / scikit-learn pickle-format revisions).
- **Where this comes from.** Cross-links to `docs/history/`, `docs/model_card.md`, `docs/evaluation.md`, `docs/feature-pipeline.md`.

**Deviations.** None from the plan spec. One in-flight self-catch worth noting: an initial `_guard_no_skew` link mistakenly pointed at `../src/matchodds/app/inference.py` (a path that doesn't exist) instead of `../serving/app/inference.py`; caught on a re-read of the file and fixed before stopping. No other broken links — verified all targets against the live tree.

**Gate.** `make check` → PASS · `make test` → 155 passed.

---

## Task 5 — `README.md` polish + demo screenshot + fresh-clone reproducibility verification

**Scope (files to touch).**
- `README.md`: a polish pass on top of whatever is already there. Final shape (top → bottom): a one-paragraph elevator pitch (what this is, why it exists, the two recruiter signals); the per-role *What this demonstrates* section (ML Engineer primary + Data Engineer bonus, with the existing wording reused if accurate); a *Quickstart* block (`make install-dev && make repro && make up`, with a sentence on `MATCHODDS_DATA_DIR` for the sample fixture path); a small results table linking out to `docs/evaluation.md`; a demo screenshot inline (`![Demo](docs/images/demo.png)`); a *Documentation* section with three bullets (the three new docs files); a *Reproducibility* section that names the exact command sequence the verification ran and links the artifact's metadata; a *Constraints / out-of-scope* section restating the no-live-betting framing.
- `docs/assets/demo.gif` (new): an animated GIF captured by the user from the running `make up` demo (the interaction loop: pick a league + two teams, hit Predict, watch the calibrated probabilities + plain-English read render). Captured with `wf-recorder -g "$(slurp)"` (Hyprland) and converted with `ffmpeg` two-pass palette generation; sized ~3–4 MB at 820 px wide. Source `demo.mp4` also kept in `docs/assets/` so the GIF can be regenerated without re-recording.
- **Fresh-clone verification (manual, one-off).** The user performs the destructive sequence — clone the repo into a scratch directory, `make install-dev && make repro && make up`, confirm the demo loads and `models/v1.metadata.json` metrics match the committed metadata. Outcome reported in the post-task summary.

**Acceptance criteria.**
- `make check` + `make test` green.
- The screenshot file exists and renders inline on the GitHub-rendered README.
- The Quickstart block runs end-to-end on a fresh clone (verified manually by the user; outcome captured in the post-task summary).
- The Constraints / out-of-scope section explicitly restates *"no live betting / real-money use"* (CLAUDE.md).

**Gate.** `make check` → PASS · `make test` → green. Post-task summary includes (a) the verified fresh-clone command sequence + outcome, (b) the screenshot path and size, (c) a one-line reminder for the user about plan step 2.9 (pin the repo on the GitHub profile) and the out-of-repo CV ticket (`~/dev/cv` → `data/projects.yaml` entry).

---

## Definition of done (Epic 08)

The demo no longer ships Streamlit's Deploy button or "Made with Streamlit" footer; `docs/evaluation.md` + `docs/model_card.md` + `docs/architecture.md` exist and each carry the recruiter-readable deep-dive their name implies; the README opens with the per-role map and ends with a verified `make repro` quickstart + a committed demo screenshot; a fresh-clone reproducibility run has been performed once and is reported. `make check` + `make test` + `make nb-lint` + `make nb-run` green throughout. On close-out: mark Epic 08 Done in `MASTER_PLAN.md` with the commit range and archive this file to `docs/history/epic-08-docs.md`. Plan step 2.9 (pin the repo on the GitHub profile) is the user's manual closeout.
