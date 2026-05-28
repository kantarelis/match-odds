# PLAN — Epic 08: Documentation, Model Card & Reproducibility

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test` (+ `make nb-lint` / `make nb-run` when notebooks change), then stops for the user to review/commit/push.

**Branch:** `epic-08-docs`

**Goal.** Make the repo do the recruiter work in 30 seconds and prove it reproduces end-to-end. Ship: a small **demo polish** pass (drop Streamlit's Deploy button while keeping the hamburger menu, remove the "Made with Streamlit" footer, and restore the bar-chart's H / D / A order — the three extras called out for this epic), the three documentation deep-dives the README links to (`docs/evaluation.md`, `docs/model_card.md`, `docs/architecture.md`), a **README pass** that opens with the per-role "what this demonstrates" map and ends with a verified `make repro` quickstart + a committed demo screenshot, and a **fresh-clone reproducibility check** to certify the README's claims. Plan step 2.9 (pinning the repo on the GitHub profile) is the user's manual closeout — not a code task.

## Progress

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| 1 | **Demo polish** — drop the Deploy button (keep the hamburger menu) + remove the "Made with Streamlit" footer + restore the H / D / A bar-chart order | ✅ Done | — |
| 2 | `docs/evaluation.md` — temporal-CV protocol + metric tables (recomputable via `make train`) + a link to `notebooks/02_modeling.ipynb` for reliability diagrams | ✅ Done | — |
| 3 | `docs/model_card.md` — intended use, data sources, metrics, calibration story, limitations, ethics | ⬜ Not started | — |
| 4 | `docs/architecture.md` — notebook → package → service data-flow diagram (Mermaid) + the leakage / determinism / train-serve-skew invariants | ⬜ Not started | — |
| 5 | `README.md` polish pass + committed demo screenshot + fresh-clone `make repro` reproducibility verification | ⬜ Not started | — |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Tasks 2-4 are markdown-only under `docs/`. Task 1 touches only `demo/` (configuration + a small `st.markdown` CSS injection); demo tests stay green. Task 5 may add `docs/images/demo.png` and edit `README.md` only.

**Consolidation note (flag at review).** Tasks 2-4 are three separate docs by design (each is the deep-dive for one chapter of the model story), but if you'd rather collapse them into one "all docs" commit that's a reasonable bundle. Task 1 is genuinely isolated demo work and should stay its own commit. Task 5 (README + screenshot + verification) is the final pass and is best as the last commit so the reproducibility check certifies the final state.

## Decisions baked in (flag at review to change)

1. **Demo polish via `demo/.streamlit/config.toml` (`toolbarMode = "viewer"`) + a small `demo/app.py` patch.** Streamlit's `viewer` mode hides the **developer** options (Deploy button, Rerun, Clear cache) from both the toolbar and the hamburger menu while **keeping the menu itself** with the **viewer** options end users actually find useful (Print, record screencast, theme toggle, About). The alternative `minimal` would hide the menu entirely too — that strips functionality the demo is happy to keep. The "Made with Streamlit" badge is then removed by a one-line `st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)` immediately after `st.set_page_config(...)` — Streamlit has no first-class toggle for the page-bottom footer. The bar-chart order (currently "Away | Draw | Home" because `st.bar_chart` sorts the DataFrame's string index alphabetically — A, D, H) is restored to "Home | Draw | Away" by switching to `st.altair_chart` with an explicit categorical sort order on the x-axis. Config lives **inside `demo/`** so it travels through the Dockerfile and `make up` picks it up automatically (no extra mount).
2. **Reliability diagrams stay in `notebooks/02_modeling.ipynb`; not committed as PNGs.** `docs/evaluation.md` quotes the numbers (CV log-loss / Brier / accuracy, refreshable by `make train` + reading `models/v1.metadata.json`) and links the notebook for the visual story. Avoids stale committed images and keeps the notebook the single source of truth for charts.
3. **Architecture diagram is a Mermaid flowchart inline in `docs/architecture.md`.** GitHub renders Mermaid natively in markdown — no PNG, no extra tooling, fully diffable. Cheaper to keep accurate than a checked-in image.
4. **Demo screenshot is a single PNG at `docs/images/demo.png` (committed).** Manual capture by the user with the demo + serving stack up via `make up`; the README links it inline. Not an animated GIF — bigger files, harder to refresh, README rendering is still fine with a static PNG.
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

## Task 3 — `docs/model_card.md`: intended use + ethics

**Scope (files to touch).**
- `docs/model_card.md` (new): standard model-card sections — *Model details* (logistic + sigmoid calibration via hold-out / prefit, seed, train date from metadata, library versions); *Intended use* (educational portfolio piece + a 1X2 probability service on six leagues); *Data* (football-data.co.uk only — pinned URL manifest, seasons, leagues; no scraping); *Features* (the 30 engineered columns in `feature_columns`, summarising the five accumulator families from `src/matchodds/features/`); *Metrics* (link `docs/evaluation.md` for the table — do not duplicate); *Calibration* (hold-out / prefit, Epic 06.5 fixed the home-advantage inversion); *Limitations* (out-of-scope leagues / teams; cold-start fixtures; an upset is still an upset; no in-play modelling); *Ethics / out-of-scope* (no live betting, no real-money use, no advice — restates CLAUDE.md's constraint); *Maintainer + provenance* (the metadata's data version + seed + library versions).
- The *Out-of-scope* section explicitly names the betting-edge notebook (Epic 07) as historical / illustrative — so a reader who jumps from the README to the model card to the notebook sees the same disclaimer three times.

**Acceptance criteria.**
- `make check` + `make test` green.
- Sections match the canonical model-card structure (Mitchell et al., 2019) so a recruiter can scan them by name.
- The ethics section is unambiguous about no-real-money framing.

**Gate.** `make check` → PASS · `make test` → green.

---

## Task 4 — `docs/architecture.md`: data-flow diagram + invariants

**Scope (files to touch).**
- `docs/architecture.md` (new): one *Diagram* section with a Mermaid flowchart of `raw CSVs → matches.parquet → features.parquet → models/v1.joblib → serving API → demo`; one *Layers* section that names each Python module the diagram references (so a reader can click straight from a box to the code); one *Invariants* section listing the three load-bearing rules in plain language — *No leakage* (`features(date_cutoff)` only reads strictly-earlier matches; `TimeOrderedSplit`; the Epic-06.5 hold-out calibrator), *No train-serve skew* (serving rebuilds features through the same `matchodds.features.pipeline.features`; `Inference._guard_no_skew` fails fast at startup if the feature columns drift), *Determinism* (every estimator / splitter / calibrator takes `settings.random_seed`; `make repro` reproduces the metadata metrics).

**Acceptance criteria.**
- `make check` + `make test` green.
- The Mermaid diagram renders correctly when previewed on GitHub.
- Every box in the diagram maps to a real file path stated in the *Layers* section.

**Gate.** `make check` → PASS · `make test` → green.

---

## Task 5 — `README.md` polish + demo screenshot + fresh-clone reproducibility verification

**Scope (files to touch).**
- `README.md`: a polish pass on top of whatever is already there. Final shape (top → bottom): a one-paragraph elevator pitch (what this is, why it exists, the two recruiter signals); the per-role *What this demonstrates* section (ML Engineer primary + Data Engineer bonus, with the existing wording reused if accurate); a *Quickstart* block (`make install-dev && make repro && make up`, with a sentence on `MATCHODDS_DATA_DIR` for the sample fixture path); a small results table linking out to `docs/evaluation.md`; a demo screenshot inline (`![Demo](docs/images/demo.png)`); a *Documentation* section with three bullets (the three new docs files); a *Reproducibility* section that names the exact command sequence the verification ran and links the artifact's metadata; a *Constraints / out-of-scope* section restating the no-live-betting framing.
- `docs/images/demo.png` (new): a single landscape-oriented PNG captured by the user from the running `make up` demo. Task instructions explain how to capture (`make up` → browser → screenshot → save to `docs/images/demo.png`). Target ≤ 300 KB so the README loads briskly.
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
