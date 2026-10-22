# IRIS — chart question answering for low-vision readers.
# `make backend` serves the API (demo mode with the sample data unless the
# IRIS_* env vars say otherwise); `make frontend` serves the Next.js app;
# `make dev` runs both.

PYTHON ?= .venv/bin/python
export IRIS_DATASET_JSON ?= iris-backend/sample_data/dataset.json
export IRIS_IMAGE_DIR    ?= iris-backend/sample_data/images
export IRIS_RESULTS_DIR  ?= iris-backend/sample_data/results

.PHONY: backend frontend dev test test-backend test-frontend sample-data check-hosts

backend:
	cd iris-backend && ../$(PYTHON) -m uvicorn iris_backend.main:app --port 8000

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
