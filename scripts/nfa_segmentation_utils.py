"""
NeMo Forced Aligner (NFA) utilities for audio segmentation.

This module uses NeMo's official Forced Aligner tool to segment long-form
audio interviews into training-ready chunks with precise word-level timestamps.
"""

import os
import json
import tempfile
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from pydub import AudioSegment
import soundfile as sf

# Import existing utilities
import azure_utils
from evaluate import clean_raw_transcript_str


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        # Fallback to pydub if soundfile fails
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e2:
            raise RuntimeError(f"Could not get audio duration: {e}, {e2}")


def prepare_text_for_nfa(fulltext_file_str: str) -> str:
    """
    Prepare transcript text for NFA alignment.

    Extracts plain text from XML and returns as a single string.
    NFA will handle sentence splitting based on punctuation.

    Args:
        fulltext_file_str: Raw XML transcript string

    Returns:
        Cleaned plain text suitable for NFA
    """
    # Clean the transcript to extract plain text
    cleaned_text = clean_raw_transcript_str(fulltext_file_str)

    return cleaned_text


def run_nfa_alignment(
    audio_path: str,
    transcript_text: str,
    output_dir: str,
    model_name: str = "stt_en_conformer_ctc_medium",  # Changed from "large" to reduce GPU memory
    nfa_script_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Run NeMo Forced Aligner on audio file with transcript.

    Args:
        audio_path: Path to WAV audio file
        transcript_text: Plain text transcript
        output_dir: Directory to save NFA outputs
        model_name: NeMo ASR model to use for alignment
        nfa_script_path: Path to NFA align.py script (auto-detected if None)

    Returns:
        Dict with paths to generated CTM files:
        {
            "tokens": "path/to/tokens.ctm",
            "words": "path/to/words.ctm",
            "segments": "path/to/segments.ctm"
        }
    """
    # Find NFA script if not provided
    if nfa_script_path is None:
        possible_paths = []

        # Check environment variable first
        if 'NEMO_REPO_PATH' in os.environ:
            nemo_repo = Path(os.environ['NEMO_REPO_PATH'])
            possible_paths.append(nemo_repo / "tools" / "nemo_forced_aligner" / "align.py")

        # Try to find from nemo package installation
        try:
            import nemo
            # NeMo installed as package - look in site-packages or nearby
            nemo_pkg_path = Path(nemo.__file__).parent

            # Check if NeMo repo is cloned adjacent to site-packages
            possible_paths.extend([
                nemo_pkg_path.parent.parent.parent / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py",
                nemo_pkg_path.parent / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py",
            ])
        except ImportError:
            pass

        # Add common clone locations
        possible_paths.extend([
            Path.home() / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py",
            Path("/workspace/NeMo/tools/nemo_forced_aligner/align.py"),
            Path.cwd() / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py",
            Path.cwd().parent / "NeMo" / "tools" / "nemo_forced_aligner" / "align.py",
        ])

        # Try each path
        nfa_script_path = None
        for p in possible_paths:
            if p.exists():
                nfa_script_path = p
                print(f"  Found NFA script: {nfa_script_path}")
                break

        if nfa_script_path is None or not nfa_script_path.exists():
            raise FileNotFoundError(
                f"Could not find NFA align.py script. Tried paths:\n" +
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\n\nPlease either:\n" +
                "  1. Clone NeMo repo: git clone https://github.com/NVIDIA/NeMo\n" +
                "  2. Set NEMO_REPO_PATH environment variable to the NeMo repo directory\n" +
                "  3. Specify nfa_script_path parameter explicitly in the function call"
            )

    # Create manifest file
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")

    manifest_data = {
        "audio_filepath": os.path.abspath(audio_path),
        "text": transcript_text
    }

    with open(manifest_path, 'w') as f:
        f.write(json.dumps(manifest_data) + "\n")

    print(f"  Running NFA alignment...")
    print(f"    Audio: {audio_path}")
    print(f"    Text: {len(transcript_text)} chars")
    print(f"    Model: {model_name}")

    # Run NFA
    cmd = [
        "python", str(nfa_script_path),
        f"pretrained_name={model_name}",
        f"manifest_filepath={manifest_path}",
        f"output_dir={output_dir}",
        'additional_segment_grouping_separator=[".","?","!","..."]',
        "batch_size=1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  NFA completed successfully")

        # Return paths to CTM files
        audio_basename = Path(audio_path).stem
        return {
            "tokens": os.path.join(output_dir, "ctm", "tokens", f"{audio_basename}.ctm"),
            "words": os.path.join(output_dir, "ctm", "words", f"{audio_basename}.ctm"),
            "segments": os.path.join(output_dir, "ctm", "segments", f"{audio_basename}.ctm")
        }

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"ERROR: NFA subprocess failed during alignment")
        print(f"{'='*80}")
        print(f"Exit code: {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        print(f"\nThis means NFA could not align the transcript to audio.")
        print(f"Common causes:")
        print(f"  - Transcript contains unsupported characters")
        print(f"  - Audio quality too poor for alignment")
        print(f"  - Model tokenization error")
        print(f"\nNFA STDOUT:")
        print(e.stdout if e.stdout else "(empty)")
        print(f"\nNFA STDERR:")
        print(e.stderr if e.stderr else "(empty)")
        print(f"{'='*80}\n")
        raise


def parse_ctm_file(ctm_path: str) -> List[Dict]:
    """
    Parse NFA CTM output file.

    CTM format: <utt_id> 1 <start_time> <duration> <text>

    Args:
        ctm_path: Path to CTM file

    Returns:
        List of segment dicts with start, end, duration, text
    """
    segments = []

    # NFA's special tokens to filter out (non-lexical markers)
    SPECIAL_TOKENS = {'NA', 'lex', '<unk>', '[UNK]', '<eps>', 'ε'}

    with open(ctm_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=4)
            if len(parts) < 5:
                continue

            utt_id, channel, start_time, duration, text = parts

            # Filter out special tokens
            if text in SPECIAL_TOKENS:
                continue

            start_time = float(start_time)
            duration = float(duration)
            end_time = start_time + duration

            segments.append({
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "text": text,
                "confidence": 1.0  # NFA doesn't provide confidence scores
            })

    return segments


def segment_audio_with_nfa(
    audio_path: str,
    transcript_text: str,
    model_name: str = "stt_en_conformer_ctc_medium",  # Changed from "large" to reduce GPU memory
    max_duration: float = 30.0,
    use_words: bool = False
) -> List[Dict]:
    """
    Segment audio using NeMo Forced Aligner.

    Args:
        audio_path: Path to audio file
        transcript_text: Plain text transcript
        model_name: NeMo model for alignment
        max_duration: Maximum segment duration (seconds)
        use_words: If True, use word-level CTM; if False, use segment-level CTM (recommended)
            - segment-level: Uses natural sentence boundaries (., ?, !, ...)
            - word-level: Individual words, requires manual grouping

    Returns:
        List of segment dicts with start, end, duration, text
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run NFA
        ctm_paths = run_nfa_alignment(
            audio_path=audio_path,
            transcript_text=transcript_text,
            output_dir=temp_dir,
            model_name=model_name
        )

        # Parse appropriate CTM file
        ctm_file = ctm_paths["words"] if use_words else ctm_paths["segments"]
        print(f"  Parsing {Path(ctm_file).name}...")

        all_segments = parse_ctm_file(ctm_file)
        segment_type = "sentence-level segments" if not use_words else "word-level tokens"
        print(f"  Found {len(all_segments)} {segment_type}")

        # Group segments to fit within max_duration
        grouped_segments = []
        current_group = []
        current_start = None
        current_texts = []

        for seg in all_segments:
            if current_start is None:
                current_start = seg["start"]

            # Check if adding this segment would exceed max_duration
            potential_end = seg["end"]
            potential_duration = potential_end - current_start

            if potential_duration > max_duration and current_group:
                # Finalize current group
                grouped_segments.append({
                    "start": current_start,
                    "end": current_group[-1]["end"],
                    "duration": current_group[-1]["end"] - current_start,
                    "text": " ".join(current_texts),
                    "confidence": 1.0
                })

                # Start new group
                current_group = [seg]
                current_start = seg["start"]
                current_texts = [seg["text"]]
            else:
                # Add to current group
                current_group.append(seg)
                current_texts.append(seg["text"])

        # Don't forget last group
        if current_group:
            grouped_segments.append({
                "start": current_start,
                "end": current_group[-1]["end"],
                "duration": current_group[-1]["end"] - current_start,
                "text": " ".join(current_texts),
                "confidence": 1.0
            })

        print(f"  Grouped into {len(grouped_segments)} segments (max {max_duration}s each)")

        return grouped_segments


def cut_audio_segments(
    audio_path: str,
    segments: List[Dict],
    output_dir: str,
    base_name: str,
    max_duration: float = 30.0
) -> List[str]:
    """
    Cut audio file into segments based on NFA timestamps.

    Args:
        audio_path: Path to original audio file
        segments: List of segment dicts with start/end times
        output_dir: Directory to save audio chunks
        base_name: Base name for output files
        max_duration: Maximum segment duration

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
        duration_sec = seg["duration"]

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
        blob_index: Index for blob path

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

        print(f"  Uploading: {blob_path}")
        azure_utils.upload_blob(blob_path, audio_bytes)

        blob_paths.append(blob_path)

    return blob_paths


def process_single_vhp_row(
    row,
    row_idx: int,
    blob_prefix: str = "loc_vhp",
    model_name: str = "stt_en_conformer_ctc_medium",  # Changed from "large" to reduce GPU memory
    max_duration: float = 30.0,
    temp_dir: Optional[str] = None,
    transcript_field: str = "fulltext_file_str",
    max_audio_duration: float = 1800.0  # Skip files longer than 30 minutes (default)
) -> List[Dict]:
    """
    Process a single VHP parquet row with NFA segmentation.

    Args:
        row: Pandas DataFrame row
        row_idx: Row index in original parquet
        blob_prefix: Azure blob prefix
        model_name: NeMo model for alignment (default: medium for lower GPU memory)
        max_duration: Max segment duration (seconds)
        temp_dir: Temporary directory (created if None)
        transcript_field: Column name to use for transcript text
            - "fulltext_file_str": Raw XML transcript (may have encoding issues)
            - "transcript_raw_text_only": Pre-cleaned plain text (recommended)
        max_audio_duration: Skip files longer than this (seconds, default 1800 = 30 min)
            - Prevents CUDA OOM errors on very long files
            - Set to None to disable duration filtering

    Returns:
        List of output row dicts for segmented parquet
    """
    import data_loader

    # Extract blob_index
    blob_index = row.get('azure_blob_index', row_idx)

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp = True
    else:
        cleanup_temp = False

    print(f"\n[Row {row_idx}] Processing blob_index={blob_index}")

    try:
        # 1. Download audio
        print("  Downloading audio from Azure...")
        blob_paths = data_loader.get_blob_path_for_row(row, row_idx, blob_prefix)

        if not blob_paths:
            print("  No blob path found, skipping")
            return []

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

        # Check audio duration to avoid CUDA OOM on very long files
        if max_audio_duration is not None:
            audio_duration = get_audio_duration(wav_path)
            print(f"  Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} min)")

            if audio_duration > max_audio_duration:
                print(f"  ⚠️  SKIPPING: Audio too long ({audio_duration/60:.1f} min > {max_audio_duration/60:.1f} min limit)")
                print(f"      Reason: Prevents CUDA OOM errors on T4 GPU")
                print(f"      To process this file: increase max_audio_duration or use smaller model")
                return []

        # 2. Prepare transcript
        print("  Preparing transcript...")
        full_transcript = row.get(transcript_field, '')
        if not full_transcript:
            print(f"  No transcript found in '{transcript_field}', skipping")
            return []

        # If using fulltext_file_str, clean it; otherwise use as-is
        if transcript_field == 'fulltext_file_str':
            transcript_text = prepare_text_for_nfa(full_transcript)
        else:
            transcript_text = full_transcript

        # Debug: Show what text we're passing to NFA
        print(f"  Transcript field used: '{transcript_field}'")
        print(f"  Transcript length: {len(transcript_text)} chars, {len(transcript_text.split())} words")
        print(f"  First 150 chars: {transcript_text[:150]}...")

        # 3. Run NFA segmentation
        print("  Running NFA segmentation...")
        segments = segment_audio_with_nfa(
            wav_path,
            transcript_text,
            model_name=model_name,
            max_duration=max_duration,
            use_words=False  # Use segment-level (sentence boundaries) instead of word-level
        )

        if not segments:
            print("  No valid segments generated, skipping")
            return []

        # 4. Cut audio
        print("  Cutting audio segments...")
        segment_dir = os.path.join(temp_dir, "segments")
        segment_paths = cut_audio_segments(
            wav_path,
            segments,
            segment_dir,
            base_name=str(blob_index),
            max_duration=max_duration
        )

        # 5. Upload to Azure
        print("  Uploading segments to Azure...")
        blob_paths = upload_segments_to_blob(
            segment_paths,
            blob_prefix,
            blob_index
        )

        # 6. Create output records
        output_rows = []
        for i, (seg, blob_path) in enumerate(zip(segments, blob_paths)):
            # Clean the segment text:
            # 1. Replace <space> with actual spaces
            cleaned_text = seg["text"].replace("<space>", " ")

            # 2. Remove NFA special token PATTERNS only
            # The marker pattern "NA lex NA" appears when NFA detects non-speech sounds.
            # We remove the pattern itself, but preserve standalone "NA" or "lex" as real words
            # (e.g., "My friend Lex" or "NA forces" should be kept).
            import re

            # Remove the specific "NA lex NA" marker patterns
            cleaned_text = re.sub(r'\b(NA\s+lex|lex\s+NA)(\s+NA)*\b', ' ', cleaned_text)
            cleaned_text = re.sub(r'\b(NA\s+){2,}', ' ', cleaned_text)  # Multiple consecutive NAs

            # Remove technical markers that are never real speech
            technical_tokens = ['<unk>', '[UNK]', '<eps>', 'ε']
            for token in technical_tokens:
                cleaned_text = cleaned_text.replace(token, ' ')

            # 3. Clean up multiple spaces and trim
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            new_row = row.to_dict()
            new_row.update({
                "source_row_idx": row_idx,
                "segment_idx": i,
                # NEW columns for segmented data (don't overwrite originals)
                "segmented_audio_url": blob_path,
                "segmented_audio_transcript": cleaned_text,
                # Keep video_url as None for segments (they're audio-only WAV files)
                "video_url": None,
                # Segment metadata
                "start_time": seg["start"],
                "end_time": seg["end"],
                "confidence": seg["confidence"],
                "segment_duration": seg["duration"]
            })
            output_rows.append(new_row)

        print(f"  Generated {len(output_rows)} segment rows")

        # Cleanup temp files
        for path in segment_paths:
            if os.path.exists(path):
                os.unlink(path)

        # Clear CUDA cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"  Cleared CUDA cache after processing file")

        return output_rows

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Failed to process ENTIRE FILE for row {row_idx}")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nThis means the full interview audio could not be segmented.")
        print(f"Possible causes:")
        print(f"  - Audio download failed")
        print(f"  - NFA alignment failed (check transcript quality)")
        print(f"  - Audio/transcript mismatch (different durations)")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        print(f"{'='*80}\n")
        return []

    finally:
        if cleanup_temp and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def process_parquet_batch(
    parquet_path: str,
    output_parquet_path: str,
    model_name: str = "stt_en_conformer_ctc_medium",  # Changed from "large" to reduce GPU memory
    sample_size: Optional[int] = None,
    max_duration: float = 30.0,
    blob_prefix: str = "loc_vhp",
    transcript_field: str = "fulltext_file_str",
    max_audio_duration: float = 1800.0  # Skip files longer than 30 minutes
) -> pd.DataFrame:
    """
    Process multiple rows from parquet with NFA segmentation.

    Args:
        parquet_path: Input parquet file path
        output_parquet_path: Output parquet file path
        model_name: NeMo model for alignment (default: medium for lower GPU memory)
        sample_size: Number of rows to process (None for all)
        max_duration: Max segment duration (seconds)
        blob_prefix: Azure blob prefix
        transcript_field: Column name for transcript
            - "fulltext_file_str": Raw XML (default, may have bugs)
            - "transcript_raw_text_only": Pre-cleaned (recommended)
        max_audio_duration: Skip files longer than this (seconds, default 1800 = 30 min)
            - Prevents CUDA OOM errors on very long files
            - Set to None to disable duration filtering

    Returns:
        DataFrame with segmented data
    """
    # Load input parquet
    df = pd.read_parquet(parquet_path)

    if sample_size is not None:
        df = df.head(sample_size)

    print(f"Processing {len(df)} rows from {parquet_path}")
    print("="*60)

    # Process each row
    all_output_rows = []

    # DON'T pass temp_dir - let each file create and clean up its own temp directory
    # This prevents disk space issues when processing thousands of files
    for idx, row in df.iterrows():
        output_rows = process_single_vhp_row(
            row,
            idx,
            blob_prefix=blob_prefix,
            model_name=model_name,
            max_duration=max_duration,
            temp_dir=None,  # Let each file handle its own temp directory
            transcript_field=transcript_field,
            max_audio_duration=max_audio_duration
        )
        all_output_rows.extend(output_rows)

    # Create output DataFrame
    df_output = pd.DataFrame(all_output_rows)

    # Save to parquet
    df_output.to_parquet(output_parquet_path, index=False)

    print("\n" + "="*60)
    print("SEGMENTATION COMPLETE")
    print("="*60)
    print(f"Input rows: {len(df)}")
    print(f"Output segments: {len(df_output)}")
    print(f"Average segments per file: {len(df_output)/len(df):.1f}")
    print(f"\nSaved to: {output_parquet_path}")

    return df_output
