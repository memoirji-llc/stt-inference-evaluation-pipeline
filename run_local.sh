#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Local Pipeline Runner (Mac M1/M2/M3 - CPU)
# =============================================================================
# Use this script on your local Mac for development and testing.
#
# Workflows:
#   - inference-single-test: Quick test with 1-2 local audio files
#   - inference-batch: Process multiple files from Azure blob (using CPU)
#
# Usage:
#   bash run_local.sh                                    # Default: single test
#   WORKFLOW=inference-single-test bash run_local.sh     # Single file test
#   WORKFLOW=inference-batch bash run_local.sh           # Batch from Azure
# =============================================================================

WORKFLOW=${WORKFLOW:-"inference-single-test"}

if [ "$WORKFLOW" == "inference-single-test" ]; then
  echo "=== LOCAL: Single File Test (CPU) ==="

  RUNS=(
    "configs/runs/test_whisper.yaml"
    "configs/runs/test_w2v2.yaml"
  )

  REF_TXT="data/gt/gt_first120s.txt"

  uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --ref "$REF_TXT"

elif [ "$WORKFLOW" == "inference-batch" ]; then
  echo "=== LOCAL: Batch Processing from Azure (CPU) ==="

  RUNS=(
    "configs/runs/vhp-whisper-azure-sample.yaml"
  )

  PARQUET="data/raw/loc/veterans_history_project_resources.parquet"

  uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --parquet "$PARQUET"

else
  echo "ERROR: Unknown WORKFLOW='$WORKFLOW'"
  echo "Valid options: inference-single-test, inference-batch"
  exit 1
fi
