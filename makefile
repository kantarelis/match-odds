# match-odds — developer entrypoints. See CLAUDE.md → Quick Commands.

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
COVERAGE_BADGE := $(VENV)/bin/coverage-badge

# Paths formatted / linted by the gate (mypy / bandit / vulture run on src + serving/app; see check).
CODE_PATHS := src serving tests .vulture_allowlist.py

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "match-odds make targets:"
	@echo "  install-env    Create .env from .env.template"
	@echo "  install        Editable package + runtime core deps"
	@echo "  install-dev    Editable package + dev + test deps (local development)"
	@echo "  install-test   Test deps only"
	@echo "  install-serve  Editable package + serving runtime deps (fastapi, uvicorn)"
	@echo "  format         Auto-format: isort + black (incl. notebooks via nbqa)"
	@echo "  check          Gate: isort black flake8 mypy bandit vulture nbqa"
	@echo "  test           Run the test suite"
	@echo "  test-report    Tests + HTML coverage report, opened in the browser"
	@echo "  coverage-badge Regenerate coverage.svg (README badge) from .coverage"
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
	$(PY) -m ipykernel install --sys-prefix --name python3 >/dev/null 2>&1 || true

.PHONY: install-test
install-test: $(VENV)
	$(PIP) install -r requirements-test.txt

.PHONY: install-serve
install-serve: $(VENV)
	$(PIP) install -e .
	$(PIP) install -r requirements-serving.txt

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
	$(PY) -m mypy src serving/app
	$(PY) -m bandit -q -r src serving/app
	$(PY) -m vulture src serving/app .vulture_allowlist.py
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

.PHONY: test-report coverage-badge
test-report:
	@rm -f .coverage; \
		$(PY) -m pytest --cov=matchodds --cov-report=term --cov-report=html; RC=$$?; \
		if [ -f htmlcov/index.html ] && command -v xdg-open >/dev/null 2>&1; then \
			echo "Opening htmlcov/index.html in the default browser..."; \
			xdg-open htmlcov/index.html >/dev/null 2>&1 & \
		fi; \
		exit $$RC

coverage-badge:
	$(COVERAGE_BADGE) -f -o coverage.svg

# ----- pipeline / serving stubs (filled by later epics) -----
.PHONY: data features train repro serve up down demo nb-run
data:
	$(PY) -m matchodds.data.matches
features:
	$(PY) -m matchodds.features.pipeline
train:
	$(PY) -m matchodds.modeling.train
repro: data features train
serve:
	$(PY) -m serving.app
up:
	docker compose up -d --build
down:
	docker compose down
demo:
	@echo "make demo — not implemented until Epic 06 (Streamlit demo)."
nb-run:
	@if ls notebooks/*.ipynb >/dev/null 2>&1; then \
		for nb in notebooks/*.ipynb; do \
			echo "executing $$nb"; \
			$(PY) -m papermill "$$nb" "/tmp/$$(basename $$nb)" -k python3; \
		done; \
	else echo "nb-run: no notebooks yet — skipping"; fi
