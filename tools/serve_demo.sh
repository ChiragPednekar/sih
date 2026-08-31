#!/usr/bin/env bash
# Serve the dashboard against the SYNTHETIC demo dataset.
# Never points at data_store/ -- see tools/run_demo.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
export RAINFALL_STORE_DIR=sample_data/store
export RAINFALL_ARTIFACT_DIR=sample_artifacts
export RAINFALL_DISTRICTS_PATH=sample_data/districts.geojson
exec .venv/bin/python -m uvicorn rainfall_pipeline.api.main:app "$@"
