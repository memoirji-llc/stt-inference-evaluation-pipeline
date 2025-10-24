#!/usr/bin/env bash
set -euo pipefail
uv run python scripts/loc_vhp_2_az_blob.py \
  --parquet data/raw/loc/veterans_history_project_resources.parquet \
  --prefix loc_vhp