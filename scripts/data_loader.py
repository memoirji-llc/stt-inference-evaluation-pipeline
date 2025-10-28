"""
Data loader for Veterans History Project dataset.
Handles parquet reading, sampling, and mapping to Azure blob paths.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List


def load_vhp_dataset(
    parquet_path: str,
    sample_size: Optional[int] = None,
    filter_has_transcript: bool = True,
    filter_has_media: bool = True
) -> pd.DataFrame:
    """
    Load and optionally sample the VHP dataset from parquet.

    Args:
        parquet_path: Path to veterans_history_project_resources.parquet
        sample_size: Number of items to sample (None = all)
        filter_has_transcript: Only include rows with transcripts
        filter_has_media: Only include rows with audio or video URLs

    Returns:
        DataFrame with sampled data
    """
    df = pd.read_parquet(parquet_path)

    # Filter for items that have transcripts
    if filter_has_transcript and 'fulltext_file_str' in df.columns:
        df = df[df['fulltext_file_str'].notna()]
        print(f"Filtered to {len(df)} items with transcripts")

    # Filter for items that have media (audio or video)
    if filter_has_media:
        has_media = (df['audio_url'].notna()) | (df['video_url'].notna())
        df = df[has_media]
        print(f"Filtered to {len(df)} items with media")

    # Sort by index for deterministic order
    df = df.sort_index()

    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        # Store original indices before sampling (for debugging blob path issues)
        original_parquet_indices = df.index.tolist()

        # Now sample - this should be deterministic across machines
        df_sampled = df.sample(n=sample_size, random_state=42)
        selected_original_indices = sorted(df_sampled.index.tolist())

        # Reset index to 0,1,2... for blob path generation
        df = df_sampled.reset_index(drop=True)

        print(f"Sampled {sample_size} items")
        print(f"  Original parquet row indices selected: {selected_original_indices[:5]}{'...' if len(selected_original_indices) > 5 else ''}")
        print(f"  Will look for blob paths: loc_vhp/0/ through loc_vhp/{sample_size-1}/")
    else:
        # No sampling - use all rows
        # CRITICAL: Reset index to 0,1,2... to match upload script sequential indices
        # The upload script processed ALL rows sequentially (0-4600), so we need to match that
        df = df.reset_index(drop=True)
        print(f"Using all {len(df)} items (no sampling)")
        print(f"  Will look for blob paths: loc_vhp/0/ through loc_vhp/{len(df)-1}/")

    return df


def get_blob_path_for_row(row: pd.Series, idx: int, blob_prefix: str = "vhp") -> List[str]:
    """
    Construct possible Azure blob paths for a given row.
    Returns a list of candidate paths to try, in order of preference.

    Matches the naming convention from loc_vhp_2_az_blob.py:
    {prefix}/{idx}/{media_type}.{ext}

    Args:
        row: DataFrame row with video_url/audio_url
        idx: Row index (used in blob path)
        blob_prefix: Blob storage prefix (default: "vhp")

    Returns:
        List of blob path candidates (empty if no media available)
    """
    candidates = []

    # CRITICAL: After sampling + reset_index, the row index (0-9) doesn't match
    # the original parquet index that was used during upload.
    # The upload script used the ORIGINAL parquet indices (0-10432), but
    # we're using RESET indices (0-9) after sampling.
    # So we CANNOT trust the parquet URLs to predict what blob path exists!
    #
    # Solution: Always try BOTH video.mp4 AND audio.mp3 for each index.
    # This way we'll find whatever actually got uploaded.

    has_video = pd.notnull(row.get('video_url')) and str(row['video_url']).strip()
    has_audio = pd.notnull(row.get('audio_url')) and str(row['audio_url']).strip()

    # If row has any media URL, try both video and audio paths
    if has_video or has_audio:
        # Try video first (upload script prefers video)
        candidates.append(f"{blob_prefix}/{idx}/video.mp4")
        # Then try audio as fallback
        candidates.append(f"{blob_prefix}/{idx}/audio.mp3")

    return candidates


def prepare_inference_manifest(
    df: pd.DataFrame,
    blob_prefix: str = "vhp"
) -> List[Dict]:
    """
    Prepare a manifest for inference: list of items with paths and metadata.

    Args:
        df: DataFrame from load_vhp_dataset
        blob_prefix: Azure blob prefix

    Returns:
        List of dicts with keys:
        - file_id: unique identifier
        - blob_path: path in Azure blob storage
        - collection_number: VHP collection number
        - ground_truth: transcript text (if available)
        - title: item title
    """
    manifest = []

    for idx, row in df.iterrows():
        blob_path_candidates = get_blob_path_for_row(row, idx, blob_prefix)

        if not blob_path_candidates:
            continue

        # Extract ground truth transcript if available
        # Note: fulltext_file_str is the raw XML, not cleaned
        # If you have 'transcript_raw_text_only' column (from notebook), use that
        gt_text = None
        if 'transcript_raw_text_only' in df.columns and pd.notnull(row.get('transcript_raw_text_only')):
            gt_text = row['transcript_raw_text_only']
        elif 'fulltext_file_str' in df.columns and pd.notnull(row.get('fulltext_file_str')):
            # Could clean it here, but for now just store raw
            gt_text = row['fulltext_file_str']

        manifest.append({
            'file_id': idx,
            'blob_path_candidates': blob_path_candidates,  # Now a list of paths to try
            'collection_number': row.get('collection_number', f'unknown_{idx}'),
            'ground_truth': gt_text,
            'title': row.get('title', ''),
        })

    return manifest


def save_manifest_to_parquet(manifest: List[Dict], output_path: str):
    """
    Save inference manifest to parquet for later analysis.

    Args:
        manifest: List of dicts from prepare_inference_manifest
        output_path: Path to save parquet file
    """
    df = pd.DataFrame(manifest)
    df.to_parquet(output_path, index=False)
    print(f"Saved manifest to {output_path}")
