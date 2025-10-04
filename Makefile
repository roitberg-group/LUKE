PYTHON ?= python
VENV_BIN := .venv/bin

.PHONY: venv install torch cpu-torch lint type test ci build dist clean

venv:
	@if [ ! -d .venv ]; then $(PYTHON) -m venv .venv; fi
	@echo "Run: source .venv/bin/activate"

install: venv
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -e .[chem,dev]

lint:
	$(VENV_BIN)/ruff check luke tests

type:
	$(VENV_BIN)/mypy luke

test:
	$(VENV_BIN)/pytest --disable-warnings --cov=luke --cov-report=term-missing

build: install
	$(VENV_BIN)/python -m build --sdist --wheel --outdir dist

ci: install lint type test build
	@echo "All CI-like steps completed"

dist: build
	$(VENV_BIN)/twine check dist/*

clean:
	rm -rf .venv dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache