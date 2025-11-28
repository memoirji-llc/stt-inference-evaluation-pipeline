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
"""

import sys
from pathlib import Path

# Load Azure credentials
from dotenv import load_dotenv
load_dotenv(dotenv_path='credentials/creds.env')

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from nfa_segmentation_utils import process_parquet_batch

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
    print("="*80)
    print("NFA SEGMENTATION - PRODUCTION RUN")
    print("="*80)
    print()
    print(f"Model: {NEMO_MODEL}")
    print(f"Max segment duration: {MAX_DURATION}s")
    print(f"Max audio duration: 1800s (30 min)")
    print(f"Transcript field: {TRANSCRIPT_FIELD}")
    print()

    # Check if input files exist
    if not TRAIN_INPUT.exists():
        print(f"ERROR: Train input not found: {TRAIN_INPUT}")
        return 1
    if not VAL_INPUT.exists():
        print(f"ERROR: Val input not found: {VAL_INPUT}")
        return 1

    # Process train set
    print("="*80)
    print("STEP 1/2: PROCESSING TRAIN SET")
    print("="*80)
    print(f"Input:  {TRAIN_INPUT}")
    print(f"Output: {TRAIN_OUTPUT}")
    print()

    if TRAIN_OUTPUT.exists():
        print(f"⚠️  Train output already exists: {TRAIN_OUTPUT}")
        print("   Skipping train set processing.")
        print("   To re-run, delete the output file first.")
        print()
    else:
        try:
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

            print()
            print("="*80)
            print(f"✅ TRAIN SET COMPLETE: {len(df_train_segmented)} segments generated")
            print(f"   Output saved to: {TRAIN_OUTPUT}")
            print("="*80)
            print()
        except Exception as e:
            print()
            print("="*80)
            print(f"❌ TRAIN SET FAILED")
            print("="*80)
            print(f"Error: {e}")
            print()
            print("The script will continue with validation set processing.")
            print()

    # Process validation set
    print("="*80)
    print("STEP 2/2: PROCESSING VALIDATION SET")
    print("="*80)
    print(f"Input:  {VAL_INPUT}")
    print(f"Output: {VAL_OUTPUT}")
    print()

    if VAL_OUTPUT.exists():
        print(f"⚠️  Val output already exists: {VAL_OUTPUT}")
        print("   Skipping val set processing.")
        print("   To re-run, delete the output file first.")
        print()
    else:
        try:
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

            print()
            print("="*80)
            print(f"✅ VALIDATION SET COMPLETE: {len(df_val_segmented)} segments generated")
            print(f"   Output saved to: {VAL_OUTPUT}")
            print("="*80)
            print()
        except Exception as e:
            print()
            print("="*80)
            print(f"❌ VALIDATION SET FAILED")
            print("="*80)
            print(f"Error: {e}")
            print()
            return 1

    # Final summary
    print()
    print("="*80)
    print("🎉 NFA SEGMENTATION COMPLETE")
    print("="*80)
    print()
    print("Output files:")
    if TRAIN_OUTPUT.exists():
        print(f"  ✅ Train: {TRAIN_OUTPUT}")
    else:
        print(f"  ❌ Train: Not generated (check errors above)")

    if VAL_OUTPUT.exists():
        print(f"  ✅ Val:   {VAL_OUTPUT}")
    else:
        print(f"  ❌ Val:   Not generated (check errors above)")

    print()
    print("Next steps:")
    print("  1. Verify output files have correct schema and clean transcripts")
    print("  2. Use these files for Whisper fine-tuning with LoRA")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
