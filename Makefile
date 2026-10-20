PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python

.PHONY: venv test test-props lint fmt run bench bench-quick docker clean

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e '.[dev,speed]'

venv: $(PY)

test: $(PY)
	$(PY) -m pytest

## Property-based tests with ~5x more examples (what CI runs).
test-props: $(PY)
	HYPOTHESIS_PROFILE=ci $(PY) -m pytest tests/test_kv_cache_properties.py -k Machine
	HYPOTHESIS_PROFILE=ci $(PY) -m pytest tests/test_engine.py -k property

lint: $(PY)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

fmt: $(PY)
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

run: $(PY)
	$(PY) -m tokenserve

## Static vs continuous batching, prefix cache off vs on (~5 min, simulated backend).
bench: $(PY)
	$(PY) -m bench.run

bench-quick: $(PY)
	$(PY) -m bench.run --quick

docker:
	docker build -t tokenserve:dev .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis build dist src/*.egg-info
