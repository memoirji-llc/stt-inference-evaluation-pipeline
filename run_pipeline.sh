#!/usr/bin/env bash
set -euo pipefail

# List the run-configs you want to execute in this experiment
RUNS=(
  "configs/runs/test_w2v2.yaml"
  "configs/runs/test_whisper.yaml"
)

REF_TXT="data/gt/gt_first120s.txt"   # align this with your local slice

# Call the Python orchestrator
uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --ref "$REF_TXT"