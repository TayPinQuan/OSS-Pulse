.PHONY: setup candidates download enrich staging analytics survival forecast causal dashboard test lint format clean

PY ?= python
ARGS ?=

setup:
	$(PY) -m pip install -e ".[dev]"

candidates:
	$(PY) -m osspulse.github_search $(ARGS)

download:
	$(PY) -m osspulse.download $(ARGS)

enrich:
	$(PY) -m osspulse.enrich $(ARGS)

enrich-materialise:
	$(PY) -m osspulse.enrich --materialise-only

staging:
	$(PY) -m osspulse.staging $(ARGS)

analytics:
	$(PY) -m osspulse.analytics $(ARGS)

survival:
	$(PY) -m osspulse.survival

forecast:
	$(PY) -m osspulse.forecast

causal:
	$(PY) -m osspulse.causal

dashboard:
	streamlit run app/dashboard.py

test:
	pytest

lint:
	ruff check .
	mypy src

format:
	ruff format .
	ruff check --fix .

clean:
	rm -rf data/raw data/staging data/analytics
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

pypi:
	$(PY) -m osspulse.pypi $(ARGS)

survival-advanced:
	$(PY) -m osspulse.survival_advanced