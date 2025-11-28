#!/usr/bin/env python3
"""
Run NFA segmentation for train and validation sets.

This script processes both train and val parquet files to create segmented
audio clips with aligned transcripts for Whisper fine-tuning.

Usage:
    uv run python scripts/run_nfa_segmentation.py

The script will:
1. Process train set (2,273 files) → veterans_history_project_resources_pre2010_train_nfa_segmented.parquet
2. Process val set (569 files) → veterans_history_project_resources_pre2010_val_nfa_segmented.parquet

The script supports resumption - if it finds an existing output file, it will skip that dataset.
To re-run from scratch, delete the output parquet files first.

Logs are saved to: logs/nfa_segmentation_YYYYMMDD_HHMMSS.log
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Load Azure credentials
from dotenv import load_dotenv
load_dotenv(dotenv_path='credentials/creds.env')

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from nfa_segmentation_utils import process_parquet_batch

# Setup logging
def setup_logging():
    """Setup logging to both file and console."""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"nfa_segmentation_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger

# Configuration
NEMO_MODEL = "stt_en_conformer_ctc_medium"  # Changed from "large" to reduce GPU memory
MAX_DURATION = 30.0  # Max segment duration in seconds
BLOB_PREFIX = "loc_vhp"
TRANSCRIPT_FIELD = "transcript_raw_text_only"  # Use pre-cleaned plain text

# Input/Output paths
DATA_DIR = Path("data/raw/loc")
TRAIN_INPUT = DATA_DIR / "veterans_history_project_resources_pre2010_train.parquet"
TRAIN_OUTPUT = DATA_DIR / "veterans_history_project_resources_pre2010_train_nfa_segmented.parquet"
VAL_INPUT = DATA_DIR / "veterans_history_project_resources_pre2010_val.parquet"
VAL_OUTPUT = DATA_DIR / "veterans_history_project_resources_pre2010_val_nfa_segmented.parquet"


def main():
    # Setup logging
    logger = setup_logging()

    logger.info("="*80)
    logger.info("NFA SEGMENTATION - PRODUCTION RUN")
    logger.info("="*80)
    logger.info("")
    logger.info(f"Model: {NEMO_MODEL}")
    logger.info(f"Max segment duration: {MAX_DURATION}s")
    logger.info(f"Max audio duration: 1800s (30 min)")
    logger.info(f"Transcript field: {TRANSCRIPT_FIELD}")
    logger.info("")

    # Check if input files exist
    if not TRAIN_INPUT.exists():
        logger.error(f"Train input not found: {TRAIN_INPUT}")
        return 1
    if not VAL_INPUT.exists():
        logger.error(f"Val input not found: {VAL_INPUT}")
        return 1

    # Process train set
    logger.info("="*80)
    logger.info("STEP 1/2: PROCESSING TRAIN SET")
    logger.info("="*80)
    logger.info(f"Input:  {TRAIN_INPUT}")
    logger.info(f"Output: {TRAIN_OUTPUT}")
    logger.info("")

    if TRAIN_OUTPUT.exists():
        logger.warning(f"Train output already exists: {TRAIN_OUTPUT}")
        logger.warning("Skipping train set processing.")
        logger.warning("To re-run, delete the output file first.")
        logger.info("")
    else:
        try:
            logger.info("Starting train set processing...")
            df_train_segmented = process_parquet_batch(
                parquet_path=str(TRAIN_INPUT),
                output_parquet_path=str(TRAIN_OUTPUT),
                model_name=NEMO_MODEL,
                sample_size=None,  # Process ALL files
                max_duration=MAX_DURATION,
                blob_prefix=BLOB_PREFIX,
                transcript_field=TRANSCRIPT_FIELD,
                max_audio_duration=1800.0  # Skip files >30 min
            )

            logger.info("")
            logger.info("="*80)
            logger.info(f"✅ TRAIN SET COMPLETE: {len(df_train_segmented)} segments generated")
            logger.info(f"   Output saved to: {TRAIN_OUTPUT}")
            logger.info("="*80)
            logger.info("")
        except Exception as e:
            logger.error("")
            logger.error("="*80)
            logger.error("❌ TRAIN SET FAILED")
            logger.error("="*80)
            logger.exception(f"Error: {e}")
            logger.info("")
            logger.info("The script will continue with validation set processing.")
            logger.info("")

    # Process validation set
    logger.info("="*80)
    logger.info("STEP 2/2: PROCESSING VALIDATION SET")
    logger.info("="*80)
    logger.info(f"Input:  {VAL_INPUT}")
    logger.info(f"Output: {VAL_OUTPUT}")
    logger.info("")

    if VAL_OUTPUT.exists():
        logger.warning(f"Val output already exists: {VAL_OUTPUT}")
        logger.warning("Skipping val set processing.")
        logger.warning("To re-run, delete the output file first.")
        logger.info("")
    else:
        try:
            logger.info("Starting validation set processing...")
            df_val_segmented = process_parquet_batch(
                parquet_path=str(VAL_INPUT),
                output_parquet_path=str(VAL_OUTPUT),
                model_name=NEMO_MODEL,
                sample_size=None,  # Process ALL files
                max_duration=MAX_DURATION,
                blob_prefix=BLOB_PREFIX,
                transcript_field=TRANSCRIPT_FIELD,
                max_audio_duration=1800.0  # Skip files >30 min
            )

            logger.info("")
            logger.info("="*80)
            logger.info(f"✅ VALIDATION SET COMPLETE: {len(df_val_segmented)} segments generated")
            logger.info(f"   Output saved to: {VAL_OUTPUT}")
            logger.info("="*80)
            logger.info("")
        except Exception as e:
            logger.error("")
            logger.error("="*80)
            logger.error("❌ VALIDATION SET FAILED")
            logger.error("="*80)
            logger.exception(f"Error: {e}")
            logger.info("")
            return 1

    # Final summary
    logger.info("")
    logger.info("="*80)
    logger.info("🎉 NFA SEGMENTATION COMPLETE")
    logger.info("="*80)
    logger.info("")
    logger.info("Output files:")
    if TRAIN_OUTPUT.exists():
        logger.info(f"  ✅ Train: {TRAIN_OUTPUT}")
    else:
        logger.error(f"  ❌ Train: Not generated (check errors above)")

    if VAL_OUTPUT.exists():
        logger.info(f"  ✅ Val:   {VAL_OUTPUT}")
    else:
        logger.error(f"  ❌ Val:   Not generated (check errors above)")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify output files have correct schema and clean transcripts")
    logger.info("  2. Use these files for Whisper fine-tuning with LoRA")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
