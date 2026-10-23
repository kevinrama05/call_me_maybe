UV          = uv
MODULE      = src

FUNCTIONS   = data/input/functions_definition.json
INPUT       = data/input/function_calling_tests.json
OUTPUT      = data/output/function_calling_results.json

all: run

install:
	$(UV) sync

run:
	$(UV) run python -m $(MODULE) \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	$(UV) run python -m pdb -m $(MODULE) \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

clean:
	find . -path ./.venv -prune -o -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	$(UV) run flake8 .
	$(UV) run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

.PHONY: all install run debug clean lint lint-strict
