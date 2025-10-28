#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# VM Pipeline Runner (Azure NVIDIA T4 GPU - CUDA)
# =============================================================================
# Use this script on Azure VM with GPU for production runs.
#
# Workflows:
#   - inference-single-test: Quick test with 1-2 local audio files
#   - inference-batch: Process multiple files from Azure blob (using GPU)
#
# Usage:
#   bash run_vm.sh                                    # Default: single test
#   WORKFLOW=inference-single-test bash run_vm.sh     # Single file test
#   WORKFLOW=inference-batch bash run_vm.sh           # Batch from Azure (GPU)
# =============================================================================

WORKFLOW=${WORKFLOW:-"inference-single-test"}

if [ "$WORKFLOW" == "inference-single-test" ]; then
  echo "=== VM: Single File Test (GPU) ==="

  RUNS=(
    "configs/runs/test_whisper.yaml"
    "configs/runs/test_w2v2.yaml"
  )

  REF_TXT="data/gt/gt_first120s.txt"

  uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --ref "$REF_TXT"

elif [ "$WORKFLOW" == "inference-batch" ]; then
  echo "=== VM: Batch Processing from Azure (GPU) ==="

  RUNS=(
    "configs/runs/vhp-whisper-azure-gpu.yaml"
  )

  PARQUET="data/raw/loc/veterans_history_project_resources.parquet"

  uv run python scripts/run_pipeline.py --configs "${RUNS[@]}" --parquet "$PARQUET"

else
  echo "ERROR: Unknown WORKFLOW='$WORKFLOW'"
  echo "Valid options: inference-single-test, inference-batch"
  exit 1
fi
