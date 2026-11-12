# Comandi di sviluppo. Python dell'ambiente attivo (es. ~/envs/dl4cv).
PY ?= python3

.PHONY: help test cov lint format synthetic clean

help:
	@echo "make test       - pytest (solo CPU, < 3 minuti)"
	@echo "make cov        - pytest con copertura di src/"
	@echo "make lint       - ruff + black --check"
	@echo "make format     - black + ruff --fix"
	@echo "make synthetic  - pipeline completa su dati sintetici (00 -> 55)"
	@echo "make clean      - rimuove cache Python e la run sintetica locale"

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

synthetic:
	scripts/run_synthetic_pipeline.sh --clean

clean:
	find . -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage coverage.xml _synthetic_root
