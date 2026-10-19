-include .env
export

JUDGE ?= llm
LIMIT ?= 50
TURN ?= 1

.PHONY: setup test lint fixture download demo report-baselines report clean

setup:            ## Install the package and dev tools into the active virtualenv
	pip install -e ".[dev]"

test:             ## Unit tests (no network)
	pytest

lint:
	ruff check . && ruff format --check .

fixture:          ## Regenerate the synthetic test fixtures
	python scripts/make_fixture.py

download:         ## Cache both MT-Bench splits from Hugging Face (about 1.4 MB)
	evalbridge download

demo:             ## Baseline report cards on the synthetic fixture. No network.
	evalbridge report --judge longer --fixture --out reports/demo
	evalbridge report --judge first --fixture --out reports/demo

report-baselines: ## Free report cards on the real data: two baselines + MT-Bench's recorded GPT-4
	evalbridge report --judge longer --turn $(TURN) --out reports/
	evalbridge report --judge first --turn $(TURN) --out reports/
	evalbridge report --judge gpt4-recorded --turn $(TURN) --out reports/

report:           ## LLM judge from .env on a sample: make report LIMIT=100 (cached, 2 calls/item)
	evalbridge report --judge $(JUDGE) --limit $(LIMIT) --turn $(TURN) --out reports/

clean:
	rm -rf data .cache reports/demo
