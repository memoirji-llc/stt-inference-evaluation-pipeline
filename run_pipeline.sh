#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# STT Benchmarking Pipeline Runner
# =============================================================================
# This script runs the inference + evaluation pipeline for STT models.
#
# Two workflows are supported:
# 1. Demo: Single file quick test (local audio + text reference)
# 2. Production: Multiple files from Azure Blob + parquet ground truth
#
# Usage:
#   bash run_pipeline.sh                       # Run demo workflow (default)
#   WORKFLOW=demo bash run_pipeline.sh         # Single-file demo test
#   WORKFLOW=production bash run_pipeline.sh   # Production Azure blob run
# =============================================================================

# Workflow selection (default: demo for quick testing)
WORKFLOW=${WORKFLOW:-"demo"}

if [ "$WORKFLOW" == "demo" ]; then
  echo "=== DEMO WORKFLOW: Single-file quick test ==="

  RUNS=(
    "configs/runs/test_whisper.yaml"
    "configs/runs/test_w2v2.yaml"
  )

  REF_TXT="data/gt/gt_first120s.txt"

  # Call the Python orchestrator with demo args
  uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --ref "$REF_TXT"

elif [ "$WORKFLOW" == "production" ]; then
  echo "=== PRODUCTION WORKFLOW: Azure Blob + parquet ground truth ==="

  RUNS=(
    "configs/runs/vhp-whisper-azure-sample.yaml"
  )

  PARQUET="data/veterans_history_project_resources.parquet"

  # Call the Python orchestrator with parquet args
  uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --parquet "$PARQUET"

else
  echo "ERROR: Unknown WORKFLOW='$WORKFLOW'"
  echo "Valid options: demo, production"
  exit 1
fi
