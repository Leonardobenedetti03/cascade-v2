# Development commands. They use the Python of the active environment (e.g. .venv).
PY ?= python3

.PHONY: help test cov lint format clean

help:
	@echo "make test       - pytest (CPU only, under 3 minutes)"
	@echo "make cov        - pytest with coverage of src/"
	@echo "make lint       - ruff + black --check"
	@echo "make format     - black + ruff --fix"
	@echo "make clean      - remove Python caches and the local synthetic run"

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=src --cov-report=term-missing:skip-covered --cov-report=xml

lint:
	ruff check src scripts tests
	black --check src scripts tests

format:
	black src scripts tests
	ruff check --fix src scripts tests

clean:
	find . -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage coverage.xml _synthetic_root
