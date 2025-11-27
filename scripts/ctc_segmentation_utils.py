"""
CTC Segmentation Utilities for VHP Audio Processing

Uses NeMo CTC-based forced alignment to segment long audio files with transcripts
into training-ready chunks (<30s) with aligned text.

Dependencies:
    - nemo_toolkit[asr]
    - ctc-segmentation
    - librosa
    - pydub
"""

import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

import librosa
import soundfile as sf
import pandas as pd
from pydub import AudioSegment

# NeMo imports
try:
    import nemo.collections.asr as nemo_asr
    from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_text
except ImportError as e:
    warnings.warn(f"NeMo/CTC-segmentation not installed: {e}")

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
import azure_utils
from evaluate import clean_raw_transcript_str


def prepare_text_for_segmentation(text: str) -> List[str]:
    """
    Prepare transcript text for CTC segmentation.

    Splits text into sentences and normalizes for alignment.

    Args:
        text: Full transcript text

    Returns:
        List of sentences ready for alignment
    """
    # Clean the transcript first
    cleaned_text = clean_raw_transcript_str(text)

    # Split into sentences at punctuation
    # Simple splitting - can be improved with nltk/spacy if needed
    import re
    sentences = re.split(r'[.!?]+\s+', cleaned_text)

    # Filter empty sentences and normalize
    sentences = [s.strip() for s in sentences if s.strip()]

    # Lowercase for CTC (models expect lowercase)
    sentences = [s.lower() for s in sentences]

    return sentences


def segment_audio_with_ctc(
    audio_path: str,
    transcript_sentences: List[str],
    model_name: str = "stt_en_conformer_ctc_large",
    sample_rate: int = 16000,
    min_confidence: float = -2.0
) -> List[Dict]:
    """
    Use CTC segmentation to align sentences to audio timestamps.

    Args:
        audio_path: Path to audio file (local)
        transcript_sentences: List of transcript sentences
        model_name: NeMo ASR model for CTC alignment
        sample_rate: Target sample rate
        min_confidence: Minimum confidence score to accept segment

    Returns:
        List of segments with timestamps and text:
        [
            {
                "start": 0.0,
                "end": 28.5,
                "text": "sentence text",
                "confidence": 0.95
            },
            ...
        ]
    """
    # Load NeMo ASR model
    print(f"Loading NeMo model: {model_name}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr_model.eval()

    # Get CTC logits from model using return_hypotheses
    print("Computing CTC logits...")
    hypotheses = asr_model.transcribe(
        audio=[audio_path],
        batch_size=1,
        return_hypotheses=True
    )

    # Extract alignments (log probabilities) from hypothesis
    logits = hypotheses[0].alignments  # Shape: (time_steps, num_classes)
    print(f"Logits shape: {logits.shape}")

    # Prepare text for alignment (join sentences with space)
    text = " ".join(transcript_sentences)

    # CTC segmentation configuration
    config = CtcSegmentationParameters()
    config.char_list = asr_model.decoder.vocabulary  # Model's character set
    config.index_duration = 0.04  # 40ms per frame for Conformer CTC models

    # Run CTC segmentation
    print("Running CTC segmentation...")
    ground_truth_mat, utt_begin_indices = prepare_text(config, text)
    timings, char_probs, state_list = ctc_segmentation(
        config, logits, ground_truth_mat
    )

    # Extract segments with timestamps
    segments = []
    for i, sentence in enumerate(transcript_sentences):
        if i >= len(timings) - 1:
            break

        start_time = timings[i]
        end_time = timings[i + 1]
        confidence = char_probs[i] if i < len(char_probs) else 0.0

        # Filter by confidence
        if confidence >= min_confidence:
            segments.append({
                "start": float(start_time),
                "end": float(end_time),
                "text": sentence,
                "confidence": float(confidence)
            })
        else:
            print(f"  Skipping low-confidence segment: {confidence:.2f} < {min_confidence}")

    print(f"Generated {len(segments)} segments from {len(transcript_sentences)} sentences")
    return segments


def cut_audio_segments(
    audio_path: str,
    segments: List[Dict],
    output_dir: str,
    base_name: str,
    max_duration: float = 30.0
) -> List[str]:
    """
    Cut audio file into segments based on timestamps.

    Args:
        audio_path: Path to original audio file
        segments: List of segment dicts with start/end times
        output_dir: Directory to save audio chunks
        base_name: Base name for output files (e.g., "123")
        max_duration: Maximum segment duration in seconds

    Returns:
        List of paths to created audio segment files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    audio = AudioSegment.from_file(audio_path)

    segment_paths = []
    for i, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        duration_sec = (end_ms - start_ms) / 1000.0

        # Skip if segment too long
        if duration_sec > max_duration:
            print(f"  Warning: Segment {i} too long ({duration_sec:.1f}s), skipping")
            continue

        # Extract segment
        segment_audio = audio[start_ms:end_ms]

        # Save as WAV
        output_path = os.path.join(output_dir, f"{base_name}_{i:03d}.wav")
        segment_audio.export(output_path, format="wav")

        segment_paths.append(output_path)

    return segment_paths


def upload_segments_to_blob(
    segment_paths: List[str],
    blob_prefix: str,
    blob_index: int
) -> List[str]:
    """
    Upload audio segments to Azure blob storage.

    Args:
        segment_paths: List of local audio segment paths
        blob_prefix: Azure blob prefix (e.g., "loc_vhp")
        blob_index: Index for blob path (e.g., 123 -> "loc_vhp/123/123_000.wav")

    Returns:
        List of Azure blob paths for uploaded segments
    """
    blob_paths = []

    for seg_path in segment_paths:
        # Construct blob path: loc_vhp/123/123_000.wav
        filename = Path(seg_path).name
        blob_path = f"{blob_prefix}/{blob_index}/{filename}"

        # Upload to Azure
        with open(seg_path, 'rb') as f:
            audio_bytes = f.read()

        # Use existing azure_utils (assumes upload function exists)
        # TODO: Add azure_utils.upload_blob_from_memory if not exists
        print(f"  Uploading: {blob_path}")
        azure_utils.upload_blob(blob_path, audio_bytes)

        blob_paths.append(blob_path)

    return blob_paths


def process_single_vhp_row(
    row: pd.Series,
    row_idx: int,
    blob_prefix: str = "loc_vhp",
    model_name: str = "stt_en_conformer_ctc_large",
    max_duration: float = 30.0,
    min_confidence: float = -2.0,
    temp_dir: Optional[str] = None
) -> List[Dict]:
    """
    Process a single VHP parquet row: download audio, segment, upload chunks.

    Args:
        row: DataFrame row with VHP data
        row_idx: Row index in original parquet
        blob_prefix: Azure blob prefix
        model_name: NeMo ASR model for alignment
        max_duration: Maximum segment duration
        min_confidence: Minimum CTC confidence
        temp_dir: Temporary directory for processing (auto-created if None)

    Returns:
        List of segment dicts ready for new parquet:
        [
            {
                "source_row_idx": 0,
                "segment_idx": 0,
                "audio_url": "loc_vhp/123/123_000.wav",
                "fulltext_file_str": "sentence text",
                "start_time": 0.0,
                "end_time": 28.5,
                "confidence": 0.95,
                # ... other columns from original row
            },
            ...
        ]
    """
    # Create temp directory
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()

    blob_index = row.get('azure_blob_index', row_idx)

    print(f"\n[Row {row_idx}] Processing blob_index={blob_index}")

    # 1. Download original audio from Azure
    print("  Downloading audio from Azure...")
    import data_loader
    blob_paths = data_loader.get_blob_path_for_row(row, row_idx, blob_prefix)

    if not blob_paths:
        print("  No blob path found, skipping")
        return []

    # Try each blob path
    audio_bytes = None
    for blob_path in blob_paths:
        if azure_utils.blob_exists(blob_path):
            audio_bytes = azure_utils.download_blob_to_memory(blob_path)
            break

    if audio_bytes is None:
        print("  Could not download audio, skipping")
        return []

    # Save to temp file
    temp_audio_path = os.path.join(temp_dir, f"original_{blob_index}.mp4")
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_bytes)

    # Convert to WAV 16kHz
    print("  Converting to WAV...")
    audio_seg = AudioSegment.from_file(temp_audio_path)
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
    wav_path = os.path.join(temp_dir, f"original_{blob_index}.wav")
    audio_seg.export(wav_path, format="wav")

    # 2. Prepare transcript
    print("  Preparing transcript...")
    full_transcript = row.get('fulltext_file_str', '')
    if not full_transcript:
        print("  No transcript found, skipping")
        return []

    sentences = prepare_text_for_segmentation(full_transcript)
    print(f"  Split into {len(sentences)} sentences")

    # 3. CTC segmentation
    print("  Running CTC segmentation...")
    try:
        segments = segment_audio_with_ctc(
            wav_path,
            sentences,
            model_name=model_name,
            min_confidence=min_confidence
        )
    except Exception as e:
        print(f"  CTC segmentation failed: {e}")
        return []

    if not segments:
        print("  No valid segments generated, skipping")
        return []

    # 4. Cut audio into chunks
    print("  Cutting audio segments...")
    segment_dir = os.path.join(temp_dir, "segments")
    segment_paths = cut_audio_segments(
        wav_path,
        segments,
        segment_dir,
        base_name=str(blob_index),
        max_duration=max_duration
    )

    # 5. Upload segments to Azure
    print("  Uploading segments to Azure...")
    blob_paths = upload_segments_to_blob(
        segment_paths,
        blob_prefix,
        blob_index
    )

    # 6. Create output records
    output_rows = []
    for i, (seg, blob_path) in enumerate(zip(segments, blob_paths)):
        # Copy original row data
        new_row = row.to_dict()

        # Update with segment-specific data
        new_row.update({
            "source_row_idx": row_idx,
            "segment_idx": i,
            "audio_url": blob_path,
            "video_url": None,  # Segments are audio-only
            "fulltext_file_str": seg["text"],  # Plain text, no XML
            "start_time": seg["start"],
            "end_time": seg["end"],
            "confidence": seg["confidence"],
            "segment_duration": seg["end"] - seg["start"]
        })

        output_rows.append(new_row)

    print(f"  Generated {len(output_rows)} segment rows")

    # Cleanup temp files
    for path in segment_paths:
        if os.path.exists(path):
            os.unlink(path)

    return output_rows


def process_parquet_batch(
    parquet_path: str,
    output_parquet_path: str,
    model_name: str = "stt_en_conformer_ctc_large",
    sample_size: Optional[int] = None,
    max_duration: float = 30.0,
    min_confidence: float = -2.0,
    blob_prefix: str = "loc_vhp"
) -> pd.DataFrame:
    """
    Process multiple rows from a parquet file with CTC segmentation.

    Args:
        parquet_path: Path to input parquet (e.g., _train.parquet)
        output_parquet_path: Path to save segmented parquet
        model_name: NeMo ASR model name
        sample_size: Number of rows to process (None = all)
        max_duration: Maximum segment duration
        min_confidence: Minimum CTC confidence
        blob_prefix: Azure blob prefix

    Returns:
        DataFrame with segmented data
    """
    # Load parquet
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")

    # Sample if requested
    if sample_size is not None and sample_size < len(df):
        df = df.head(sample_size)
        print(f"Processing first {sample_size} rows")

    # Process each row
    all_segments = []
    for idx, row in df.iterrows():
        segments = process_single_vhp_row(
            row,
            idx,
            blob_prefix=blob_prefix,
            model_name=model_name,
            max_duration=max_duration,
            min_confidence=min_confidence
        )
        all_segments.extend(segments)

    # Create output DataFrame
    df_segments = pd.DataFrame(all_segments)

    print(f"\n{'='*60}")
    print(f"SEGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Input rows: {len(df)}")
    print(f"Output segments: {len(df_segments)}")
    print(f"Average segments per file: {len(df_segments) / len(df):.1f}")

    # Save to parquet
    df_segments.to_parquet(output_parquet_path, index=False)
    print(f"\nSaved to: {output_parquet_path}")

    return df_segments
