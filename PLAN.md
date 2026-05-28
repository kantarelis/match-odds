# PLAN — Epic 08: Documentation, Model Card & Reproducibility

Active epic from [`MASTER_PLAN.md`](MASTER_PLAN.md). Workflow and the absolute git rule live in [`CLAUDE.md`](CLAUDE.md) → *Development Methodology*. **One task = one commit.** Claude implements exactly one task on request, runs `make check` + `make test` (+ `make nb-lint` / `make nb-run` when notebooks change), then stops for the user to review/commit/push.

**Branch:** `epic-08-docs`

**Goal.** Make the repo do the recruiter work in 30 seconds and prove it reproduces end-to-end. Ship: a small **demo polish** pass (drop Streamlit's "Deploy" button, remove the "Made with Streamlit" footer, and restore the bar-chart's H / D / A order — the three extras called out for this epic), the three documentation deep-dives the README links to (`docs/evaluation.md`, `docs/model_card.md`, `docs/architecture.md`), a **README pass** that opens with the per-role "what this demonstrates" map and ends with a verified `make repro` quickstart + a committed demo screenshot, and a **fresh-clone reproducibility check** to certify the README's claims. Plan step 2.9 (pinning the repo on the GitHub profile) is the user's manual closeout — not a code task.

## Progress

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| 1 | **Demo polish** — hide the Streamlit "Deploy" toolbar entry + remove the "Made with Streamlit" footer + restore the H / D / A bar-chart order | ⬜ Not started | — |
| 2 | `docs/evaluation.md` — temporal-CV protocol + metric tables (recomputable via `make train`) + a link to `notebooks/02_modeling.ipynb` for reliability diagrams | ⬜ Not started | — |
| 3 | `docs/model_card.md` — intended use, data sources, metrics, calibration story, limitations, ethics | ⬜ Not started | — |
| 4 | `docs/architecture.md` — notebook → package → service data-flow diagram (Mermaid) + the leakage / determinism / train-serve-skew invariants | ⬜ Not started | — |
| 5 | `README.md` polish pass + committed demo screenshot + fresh-clone `make repro` reproducibility verification | ⬜ Not started | — |

**Legend:** ✅ Done · 🔄 In progress · ⬜ Not started

**Green-state rule.** Every commit keeps `make check` + `make test` green. Tasks 2-4 are markdown-only under `docs/`. Task 1 touches only `demo/` (configuration + a small `st.markdown` CSS injection); demo tests stay green. Task 5 may add `docs/images/demo.png` and edit `README.md` only.

**Consolidation note (flag at review).** Tasks 2-4 are three separate docs by design (each is the deep-dive for one chapter of the model story), but if you'd rather collapse them into one "all docs" commit that's a reasonable bundle. Task 1 is genuinely isolated demo work and should stay its own commit. Task 5 (README + screenshot + verification) is the final pass and is best as the last commit so the reproducibility check certifies the final state.

## Decisions baked in (flag at review to change)

1. **Demo polish via `demo/.streamlit/config.toml` + a small `demo/app.py` patch.** The toolbar entry (the "Deploy" button) is removed by `[client] toolbarMode = "minimal"` in the config file — supported since Streamlit ≥ 1.18, so safe on the pinned 1.57.0. The "Made with Streamlit" badge is removed by a one-line `st.markdown("<style>footer{visibility:hidden;}</style>", unsafe_allow_html=True)` immediately after `st.set_page_config(...)` — Streamlit has no first-class toggle. The bar-chart order (currently "Away | Draw | Home" because `st.bar_chart` sorts the DataFrame's string index alphabetically — A, D, H) is restored to "Home | Draw | Away" by switching to `st.altair_chart` with an explicit categorical sort order on the x-axis. Config lives **inside `demo/`** so it travels through the Dockerfile and `make up` picks it up automatically (no extra mount).
2. **Reliability diagrams stay in `notebooks/02_modeling.ipynb`; not committed as PNGs.** `docs/evaluation.md` quotes the numbers (CV log-loss / Brier / accuracy, refreshable by `make train` + reading `models/v1.metadata.json`) and links the notebook for the visual story. Avoids stale committed images and keeps the notebook the single source of truth for charts.
3. **Architecture diagram is a Mermaid flowchart inline in `docs/architecture.md`.** GitHub renders Mermaid natively in markdown — no PNG, no extra tooling, fully diffable. Cheaper to keep accurate than a checked-in image.
4. **Demo screenshot is a single PNG at `docs/images/demo.png` (committed).** Manual capture by the user with the demo + serving stack up via `make up`; the README links it inline. Not an animated GIF — bigger files, harder to refresh, README rendering is still fine with a static PNG.
5. **Reproducibility verification is a documented sequence + manual checklist, not CI.** A fresh-clone `make install-dev && make repro && make up` is destructive (rewrites `models/v1.joblib`, re-downloads data) and impractical to run in the standard CI workflow; the README documents the steps and Task 5 includes a one-time manual run with the outcome (metadata metrics matched, demo loads) reported in the post-task summary. CI continues to run `make check` + `make test` + `make nb-lint` + `make nb-run` as today.
6. **Betting-edge framing stays "historical / illustrative only" everywhere.** The README, model card, and any cross-link to Epic 07 must preserve CLAUDE.md's *"no live betting / real-money framing"* constraint. The model card explicitly states this in the *Limitations* / *Out-of-scope* section.
7. **The CV integration ticket (a `data/projects.yaml` entry in `~/dev/cv`) is noted in the post-Task-5 summary as an out-of-repo follow-up.** Not implemented in this repo (different repo entirely); flagged so it doesn't get forgotten when the user starts the pin pass.

## Out of scope

CI-side reproducibility automation (would need docker-in-docker or significant CI rework — see Decision 5); GitHub Pages or a docs site; auto-generated diagrams from CI; auto-captured screenshots; a model-version bump (`v1.joblib` stays the shipped artifact); any code change outside the demo polish in Task 1. Plan step 2.9 (the user pins the repo on the GitHub profile) is an out-of-repo manual action, not a code task.

---

## Task 1 — Demo polish (Deploy button + "Made with Streamlit" footer + H/D/A bar-chart order)

**Scope (files to touch).**
- `demo/.streamlit/config.toml` (new): `[client]` section with `toolbarMode = "minimal"`. Removes the *Deploy* entry from Streamlit's top-right toolbar.
- `demo/app.py`:
  - Add a one-line `st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)` (or equivalent CSS-injection helper) immediately after `st.set_page_config(...)` to hide the "Made with Streamlit v1.57.0" footer. A short code comment notes *why* (Streamlit ships no first-class toggle for the footer) so a future reader does not strip it as cruft.
  - In `_render_prediction`, replace `st.bar_chart(chart, y="probability")` with an `st.altair_chart(...)` call that pins the x-axis sort to `list(_OUTCOME_LABELS)` (H → D → A) — e.g. `alt.X("outcome:N", sort=list(_OUTCOME_LABELS), title=None)`. The DataFrame already lists rows in H/D/A order; the explicit categorical sort overrides Vega-Lite's default alphabetical ordering (which produces A / D / H today). Keep the *y* metric the same. Adds `import altair as alt` at the top.
- `demo/Dockerfile`: confirm `demo/.streamlit/` is copied into the image (it should already be — the Dockerfile copies the `demo/` directory wholesale — but verify and tweak only if needed).
- `demo/tests/test_app.py`: if the existing AppTest smoke asserts on the chart object's data, update it for the new chart type; otherwise leave untouched (AppTest doesn't render chart visuals).

**Acceptance criteria.**
- `make check` + `make test` green. `altair` is already a transitive dependency of Streamlit (no `requirements-demo.txt` edit needed — verify before importing).
- Manual smoke: `make up`, open the demo in a browser → no Deploy button, no "Made with Streamlit" footer, bar chart reads **Home win → Draw → Away win** left-to-right.
- The CSS injection lives in `demo/app.py` with a comment explaining why (no first-class Streamlit toggle exists for the footer).

**Gate.** `make check` → PASS · `make test` → green. Post-task summary includes a one-line confirmation of the browser smoke check covering all three polish items.

---

## Task 2 — `docs/evaluation.md`: CV protocol + metric tables

**Scope (files to touch).**
- `docs/evaluation.md` (new): sections in this order — *CV protocol* (`TimeOrderedSplit`, forward-chained, never a random split, see `src/matchodds/modeling/cv.py`); *Metrics* (log-loss + Brier as primary proper scoring rules, accuracy reported but never optimised, per CLAUDE.md → *ML / Modeling Discipline*); *Bake-off* (a table of mean-CV log-loss / Brier / accuracy / `deployable` for baseline / logistic / xgboost / dixon_coles, exactly the four rows in `models/v1.metadata.json["models"]`); *Calibration* (one paragraph on the Epic 06.5 hold-out / prefit construction; reference the in-notebook reliability diagrams in `notebooks/02_modeling.ipynb`); *Reproducibility* (one paragraph: `make repro` regenerates the artifact and the metric scores deterministically from the pinned data version).
- Metric numbers in the bake-off table come from the current `models/v1.metadata.json` (Task-2 logistic 0.9945 log-loss / 0.5930 Brier etc.) and are explicitly marked as *"current as of `<created_at>` — re-check via `cat models/v1.metadata.json`"* so the doc has a built-in honesty marker against drift.

**Acceptance criteria.**
- `make check` + `make test` green (markdown-only).
- The bake-off table values match the current `v1.metadata.json` exactly (verify before writing).
- The doc references `notebooks/02_modeling.ipynb` for reliability diagrams rather than committing PNGs.

**Gate.** `make check` → PASS · `make test` → green.

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
