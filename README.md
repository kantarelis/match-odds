# match-odds

Probabilistic football match-outcome predictor — calibrated home / draw / away win
probabilities for the Greek Super League and the top-5 European leagues, served behind a
FastAPI endpoint with a small Streamlit demo.

[![CI](https://github.com/kantarelis/match-odds/actions/workflows/ci.yml/badge.svg)](https://github.com/kantarelis/match-odds/actions/workflows/ci.yml)
![Coverage](coverage.svg)

> 🚧 **Under active construction.** Built epic-by-epic — see
> [`MASTER_PLAN.md`](MASTER_PLAN.md) for the roadmap and current status.

## What this demonstrates (per role)

**ML Engineer (primary)**
- Leakage-free, time-ordered cross-validation — no future information at training time.
- Model comparison spanning discriminative (logistic regression, XGBoost) and the
  domain-canonical generative approach (Dixon-Coles bivariate-Poisson).
- Calibrated probabilities (Platt / isotonic) validated with reliability diagrams.
- Evaluation on **proper scoring rules** (log-loss, Brier) — not just accuracy.
- A reproducible training pipeline frozen to a versioned artifact and shipped as an
  inference service (same feature code trains and serves — no train/serve skew).

**Data Engineer (bonus)**
- The feature-engineering pipeline is a small, clean ETL: multi-source public CSVs →
  normalized team names → a canonical master matches table → leakage-free features.

## Stack

Python 3.14 · pandas / NumPy / scikit-learn / XGBoost · statsmodels (Dixon-Coles) ·
FastAPI · Streamlit · Docker · GitHub Actions. Static-analysis gate: isort, black, flake8,
mypy (strict), bandit, vulture, nbqa.

## Data sources (public, no scraping)

- [football-data.co.uk](https://www.football-data.co.uk/) — historical results + closing odds.
- [openfootball](https://github.com/openfootball) — multi-league CSVs.

## Quickstart

```bash
make install-dev   # create .venv, install the package + dev/test deps
make check         # static-analysis gate (isort, black, flake8, mypy, bandit, vulture, nbqa)
make test          # run the test suite
make repro         # data -> features -> train (lands across Epics 02-04)
make serve         # FastAPI inference API locally (uvicorn, hot reload)
make up            # build + run the service in Docker (run `make data` first — see below)
make down          # stop the Docker service
make demo          # Streamlit demo (Epic 06)
```

Run `make help` for the full target list.

The Docker image is **data-free**: it bakes in the frozen `models/v1.*` but mounts the master
matches table read-only from `./data`, so run `make data` (which writes
`data/processed/matches.parquet`) before `make up`. The published host port defaults to `8000`;
override it with `MATCHODDS_SERVE_PORT`. Once up, the contract lives at `POST /predict`, with
Swagger at `/docs`.

## Repository

Architecture, conventions, and the development workflow are documented in
[`CLAUDE.md`](CLAUDE.md). The epic-level roadmap lives in [`MASTER_PLAN.md`](MASTER_PLAN.md).

## License

[MIT](LICENSE). No warranty; for research and demonstration only — not betting advice.
