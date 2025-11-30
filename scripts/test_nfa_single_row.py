#!/usr/bin/env python3
"""
Test NFA segmentation fix on a single row (blob_index=9644).

Usage:
    uv run python scripts/test_nfa_single_row.py
"""

import sys
import logging
from pathlib import Path

# Load Azure credentials
from dotenv import load_dotenv
load_dotenv(dotenv_path='credentials/creds.env')

import pandas as pd

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from nfa_segmentation_utils import process_single_vhp_row

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration
NEMO_MODEL = "stt_en_conformer_ctc_medium"
MAX_DURATION = 30.0
BLOB_PREFIX = "loc_vhp"
TRANSCRIPT_FIELD = "transcript_raw_text_only"
TARGET_BLOB_INDEX = 9644

# Find which parquet file has this blob_index
DATA_DIR = Path("data/raw/loc")
TRAIN_INPUT = DATA_DIR / "veterans_history_project_resources_pre2010_train.parquet"
VAL_INPUT = DATA_DIR / "veterans_history_project_resources_pre2010_val.parquet"


def main():
    logger.info("="*60)
    logger.info(f"Testing NFA fix on blob_index={TARGET_BLOB_INDEX}")
    logger.info("="*60)

    # Try to find the row in train or val
    target_row = None
    target_idx = None
    source_file = None

    for parquet_path in [TRAIN_INPUT, VAL_INPUT]:
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        matches = df[df['azure_blob_index'] == TARGET_BLOB_INDEX]
        if len(matches) > 0:
            target_idx = matches.index[0]
            target_row = df.loc[target_idx]
            source_file = parquet_path
            break

    if target_row is None:
        logger.error(f"Could not find blob_index={TARGET_BLOB_INDEX} in train or val parquets")
        return 1

    logger.info(f"Found in: {source_file}")
    logger.info(f"Row index: {target_idx}")
    logger.info("")

    # Process this single row
    output_rows = process_single_vhp_row(
        row=target_row,
        row_idx=target_idx,
        blob_prefix=BLOB_PREFIX,
        model_name=NEMO_MODEL,
        max_duration=MAX_DURATION,
        transcript_field=TRANSCRIPT_FIELD,
        max_audio_duration=1800.0,
        cleanup_temp=True
    )

    if not output_rows:
        logger.error("No segments generated!")
        return 1

    # Display results for verification
    logger.info("")
    logger.info("="*60)
    logger.info(f"GENERATED {len(output_rows)} SEGMENTS")
    logger.info("="*60)
    logger.info("")
    logger.info("Verify alignment - segment_idx should match filename number:")
    logger.info("")

    for row in output_rows:
        seg_idx = row['segment_idx']
        audio_url = row['segmented_audio_url']
        transcript = row['segmented_audio_transcript'][:60] + "..." if len(row['segmented_audio_transcript']) > 60 else row['segmented_audio_transcript']

        # Extract filename number from URL
        filename = audio_url.split('/')[-1]  # e.g., "9644_015.wav"
        file_num = int(filename.split('_')[1].split('.')[0])  # e.g., 15

        match = "✅" if seg_idx == file_num else "❌ MISMATCH!"
        logger.info(f"  {match} segment_idx={seg_idx:3d} -> {filename} | {transcript}")

    logger.info("")
    logger.info("If all show ✅, the fix is working correctly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
