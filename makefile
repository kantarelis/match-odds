# match-odds — developer entrypoints. See CLAUDE.md → Quick Commands.

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip

# Paths formatted / linted by the gate (mypy and bandit run on src only; see check below).
CODE_PATHS := src tests .vulture_allowlist.py

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "match-odds make targets:"
	@echo "  install-env    Create .env from .env.template"
	@echo "  install        Editable package + runtime core deps"
	@echo "  install-dev    Editable package + dev + test deps (local development)"
	@echo "  install-test   Test deps only"
	@echo "  format         Auto-format: isort + black (incl. notebooks via nbqa)"
	@echo "  check          Gate: isort black flake8 mypy bandit vulture nbqa"
	@echo "  test           Run the test suite"
	@echo "  test-report    Tests with HTML coverage report"
	@echo "  nb-lint        Lint notebooks via nbqa (no-op if none)"
	@echo "  data features train repro serve up down demo nb-run   (stubs until later epics)"

# ----- environment / install -----
$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install-env
install-env:
	@test -f .env || cp .env.template .env
	@echo ".env ready"

.PHONY: install
install: $(VENV)
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: $(VENV)
	$(PIP) install -e .
	$(PIP) install -r requirements-dev.txt -r requirements-test.txt

.PHONY: install-test
install-test: $(VENV)
	$(PIP) install -r requirements-test.txt

# ----- quality gate -----
.PHONY: format
format:
	$(PY) -m isort $(CODE_PATHS)
	$(PY) -m black $(CODE_PATHS)
	@if ls notebooks/*.ipynb >/dev/null 2>&1; then \
		$(PY) -m nbqa isort notebooks; \
		$(PY) -m nbqa black notebooks; \
	else echo "format: no notebooks yet — skipping nbqa"; fi

.PHONY: check
check:
	$(PY) -m isort --check-only $(CODE_PATHS)
	$(PY) -m black --check $(CODE_PATHS)
	$(PY) -m flake8 $(CODE_PATHS)
	$(PY) -m mypy src
	$(PY) -m bandit -q -r src
	$(PY) -m vulture src .vulture_allowlist.py
	$(MAKE) nb-lint

.PHONY: nb-lint
nb-lint:
	@if ls notebooks/*.ipynb >/dev/null 2>&1; then \
		$(PY) -m nbqa isort --check-only notebooks && \
		$(PY) -m nbqa black --check notebooks && \
		$(PY) -m nbqa flake8 notebooks; \
	else echo "nb-lint: no notebooks yet — skipping"; fi

# ----- tests -----
.PHONY: test
test:
	$(PY) -m pytest

.PHONY: test-report
test-report:
	$(PY) -m pytest --cov=matchodds --cov-report=html --cov-report=term

# ----- pipeline / serving stubs (filled by later epics) -----
.PHONY: data features train repro serve up down demo nb-run
data:
	@echo "make data — not implemented until Epic 02 (data acquisition)."
features:
	@echo "make features — not implemented until Epic 03 (feature pipeline)."
train:
	@echo "make train — not implemented until Epic 04 (modelling)."
repro:
	@echo "make repro — not implemented until Epic 04 (data -> features -> train)."
serve:
	@echo "make serve — not implemented until Epic 05 (FastAPI service)."
up:
	@echo "make up — not implemented until Epic 05 (docker compose)."
down:
	@echo "make down — not implemented until Epic 05 (docker compose)."
demo:
	@echo "make demo — not implemented until Epic 06 (Streamlit demo)."
nb-run:
	@echo "make nb-run — not implemented until Epic 02 (first notebook)."
