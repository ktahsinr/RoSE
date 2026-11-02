# IRIS — chart question answering for low-vision readers.
# `make backend` serves the API (demo mode with the sample data unless the
# IRIS_* env vars say otherwise); `make frontend` serves the Next.js app;
# `make dev` runs both.

PYTHON ?= .venv/bin/python
# Absolute paths: the backend target runs with cwd iris-backend/, so
# root-relative values would silently resolve to nowhere (count: 0).
export IRIS_DATASET_JSON ?= $(CURDIR)/iris-backend/sample_data/dataset.json
export IRIS_IMAGE_DIR    ?= $(CURDIR)/iris-backend/sample_data/images
SAMPLE_RESULTS           := $(CURDIR)/iris-backend/sample_data/results
export IRIS_RESULTS_DIR  ?= $(SAMPLE_RESULTS)

.PHONY: backend backend-data datasets frontend dev test test-backend test-frontend sample-data check-hosts

backend:
	cd iris-backend && ../$(PYTHON) -m uvicorn iris_backend.main:app --port 8000

# Build the real datasets (CPU, HF Hub): 250 ChartQAPro factoid + 100
# FigureQA + 100 PlotQA questions. Skips a dataset that is already built.
datasets:
	test -f data/chartqapro/dataset.json || \
	  $(PYTHON) scripts/build_chartqapro_dataset.py --n 250 --out data/chartqapro
	test -f data/figureqa/dataset.json || \
	  $(PYTHON) scripts/build_cauldron_dataset.py figureqa --n 100 --out data/figureqa
	test -f data/plotqa/dataset.json || \
	  $(PYTHON) scripts/build_cauldron_dataset.py plotqa --n 100 --out data/plotqa

# Serve every dataset found under data/ (comma-separated lists; the id
# prefix routes each question to its frontend section). Real questions must
# not replay the SYNTHETIC sample results that share their ids, so the
# sample-results default is swapped for the real run-outputs dir.
DATA_JSONS := $(wildcard $(CURDIR)/data/*/dataset.json)
comma := ,
empty :=
space := $(empty) $(empty)
backend-data:
	@test -n "$(DATA_JSONS)" || { echo "no datasets under data/ — run 'make datasets' first"; exit 1; }
	IRIS_DATASET_JSON="$(subst $(space),$(comma),$(DATA_JSONS))" \
	IRIS_IMAGE_DIR="$(subst $(space),$(comma),$(patsubst %/dataset.json,%/images,$(DATA_JSONS)))" \
	IRIS_RESULTS_DIR="$(if $(filter $(SAMPLE_RESULTS),$(IRIS_RESULTS_DIR)),$(CURDIR)/chartqapro/results/decompose,$(IRIS_RESULTS_DIR))" \
	$(MAKE) backend

frontend:
	cd iris-frontend && npm run dev

dev:
	$(MAKE) backend & $(MAKE) frontend; wait

sample-data:
	cd iris-backend && ../$(PYTHON) -m iris_backend.sampledata sample_data

test-backend:
	cd iris-backend && ../$(PYTHON) -m pytest tests/ -q
	cd chartqapro && ../$(PYTHON) tests/test_decompose.py

test-frontend:
	cd iris-frontend && npm test

test: test-backend test-frontend

# Zero-cost constraint: no source file may reference an external hostname
# (allowlist: localhost, 127.0.0.1, the IRIS_API_URL variable, huggingface.co).
check-hosts:
	$(PYTHON) scripts/check_hostnames.py
