#!/usr/bin/env bash
# Regenerate the SYNTHETIC dataset and run the whole pipeline over it.
#
# This proves the machinery works. It proves nothing about real forecasts.
# Your real data goes in data_store/, and this script never touches it.
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"

export RAINFALL_STORE_DIR=sample_data/store
export RAINFALL_ARTIFACT_DIR=sample_artifacts
export RAINFALL_DISTRICTS_PATH=sample_data/districts.geojson

echo "==> 1/3  generating synthetic data (fake -- not ERA5, not IMD)"
"$PY" tools/make_synthetic_dataset.py

echo
echo "==> 2/3  training + verification (train 2020, calibrate 2021, test 2022)"
"$PY" -m rainfall_pipeline.training.run_full_training_pipeline \
  --era5-path      sample_data/era5.parquet \
  --observed-path  sample_data/observed_rainfall.parquet \
  --nwp-path       sample_data/raw_nwp_forecast.parquet \
  --artifact-dir   sample_artifacts \
  --train-end 2020-09-30 --val-end 2021-09-30 \
  --rebuild 2>&1 | grep -vE '^\[LightGBM\]'

# Mark the artifacts as synthetic too, so the dashboard stamps every map it
# draws from them as an illustrative mock-up.
cp sample_data/SYNTHETIC.marker sample_artifacts/SYNTHETIC.marker 2>/dev/null || true
cp sample_data/SYNTHETIC.marker sample_data/store/SYNTHETIC.marker 2>/dev/null || true

echo
echo "==> 3/3  done. Report: sample_artifacts/verification_report.md"
echo "    Dashboard: start the API below, then open http://127.0.0.1:8000/"
echo "    Start the API against this demo with:"
echo "      RAINFALL_STORE_DIR=sample_data/store \\"
echo "      RAINFALL_ARTIFACT_DIR=sample_artifacts \\"
echo "      RAINFALL_DISTRICTS_PATH=sample_data/districts.geojson \\"
echo "      $PY -m uvicorn rainfall_pipeline.api.main:app"
