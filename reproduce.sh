#!/usr/bin/env bash
# Runs the unit test suite and every experiment at its default (time-limited)
# configuration. Use `python -m exp.<name> --full` for the paper-scale runs.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
