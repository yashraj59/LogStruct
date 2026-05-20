#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" -m pytest test_model.py -q
"$PYTHON" experiments/00_smoke/run.py
"$PYTHON" -m compileall logstruct experiments

cat <<'MSG'
Smoke pipeline completed.

Full biological experiments are intentionally not run unless the exact raw
datasets listed in each experiment config are present under data/raw/.
Run the dataset-specific downloader/preprocessor before submitting real claims.
MSG
