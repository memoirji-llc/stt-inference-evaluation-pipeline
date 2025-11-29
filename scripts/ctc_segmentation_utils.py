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
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root to path
from scripts.cloud import azure_utils
from scripts.eval.evaluate import clean_raw_transcript_str


def prepare_text_for_segmentation(text: str, verbose: bool = False) -> List[str]:
    """
    Prepare transcript text for CTC segmentation.

    Splits text into sentences and normalizes for alignment.

    Args:
        text: Full transcript text
        verbose: If True, print detailed logging

    Returns:
        List of sentences ready for alignment
    """
    if verbose:
        print(f"    Raw transcript length: {len(text)} chars")
        print(f"    Raw transcript first 300 chars: {text[:300]}")

    # Clean the transcript first
    cleaned_text = clean_raw_transcript_str(text)

    if verbose:
        print(f"    Cleaned transcript length: {len(cleaned_text)} chars")
        print(f"    Cleaned transcript first 300 chars: {cleaned_text[:300]}")

    # Split into sentences at punctuation
    # Simple splitting - can be improved with nltk/spacy if needed
    import re
    sentences = re.split(r'[.!?]+\s+', cleaned_text)

    # Filter empty sentences and normalize
    sentences = [s.strip() for s in sentences if s.strip()]

    if verbose:
        print(f"    After splitting: {len(sentences)} sentences")
        print(f"    First sentence: {sentences[0] if sentences else 'NONE'}")
        print(f"    Last sentence: {sentences[-1] if sentences else 'NONE'}")

    # Lowercase for CTC (models expect lowercase)
    sentences = [s.lower() for s in sentences]

    return sentences


def segment_audio_with_ctc(
    audio_path: str,
    transcript_sentences: List[str],
    model_name: str = "stt_en_conformer_ctc_large",
    sample_rate: int = 16000,
    min_confidence: float = -2.0,
    chunk_duration: float = 120.0  # Process audio in 2-minute chunks
) -> List[Dict]:
    """
    Use CTC segmentation to align sentences to audio timestamps.

    Processes long audio files by chunking to avoid GPU OOM errors.

    Args:
        audio_path: Path to audio file (local)
        transcript_sentences: List of transcript sentences
        model_name: NeMo ASR model for CTC alignment
        sample_rate: Target sample rate
        min_confidence: Minimum confidence score to accept segment
        chunk_duration: Duration in seconds for each audio chunk (to avoid OOM)

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
    import torch
    import numpy as np

    # Load NeMo ASR model
    print(f"Loading NeMo model: {model_name}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr_model.eval()

    # Clear CUDA cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  Cleared CUDA cache before processing")

    # Load audio to check duration
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    audio_duration = len(audio) / sample_rate
    audio_size_mb = audio.nbytes / 1e6
    print(f"  Audio duration: {audio_duration:.1f}s ({audio_duration/60:.1f} minutes)")
    print(f"  Audio array size: {audio_size_mb:.1f}MB ({len(audio)} samples)")

    # Get CTC logits from model
    # For long audio, chunk to avoid GPU OOM, then concatenate logits
    print(f"  Computing CTC logits...")

    # Try processing entire file first, fall back to chunking if OOM
    try:
        with torch.no_grad():
            hypotheses = asr_model.transcribe(
                audio=[audio_path],
                batch_size=1,
                return_hypotheses=True
            )

            # Extract alignments (log probabilities)
            log_probs = hypotheses[0].alignments  # Shape: (time_steps, num_classes)
            print(f"  Logits shape: {log_probs.shape}")

            # CRITICAL: Move blank column from last position to first position
            # NeMo models have blank at last position, but ctc-segmentation expects it at position 0
            blank_col = log_probs[:, -1].reshape((log_probs.shape[0], 1))
            log_probs = np.concatenate((blank_col, log_probs[:, :-1]), axis=1)
            print(f"  Moved blank token to position 0, new shape: {log_probs.shape}")

            logits = log_probs

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  GPU OOM detected - falling back to chunked processing")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Chunk processing for very long audio
            # Adjust chunk duration based on audio length
            if audio_duration > 1800:  # 30 minutes
                chunk_duration = 60.0  # 1-minute chunks
            else:
                chunk_duration = 120.0  # 2-minute chunks

            print(f"  Processing in {chunk_duration}s chunks...")

            chunk_samples = int(chunk_duration * sample_rate)
            num_chunks = int(np.ceil(len(audio) / chunk_samples))

            all_logits = []

            for i in range(num_chunks):
                start_idx = i * chunk_samples
                end_idx = min((i + 1) * chunk_samples, len(audio))
                audio_chunk = audio[start_idx:end_idx]
                chunk_duration_actual = len(audio_chunk) / sample_rate

                print(f"    Chunk {i+1}/{num_chunks}: {start_idx/sample_rate:.1f}s - {end_idx/sample_rate:.1f}s ({chunk_duration_actual:.1f}s)")

                # Save chunk to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    temp_chunk_path = tmp_file.name
                    sf.write(temp_chunk_path, audio_chunk, sample_rate)

                try:
                    with torch.no_grad():
                        hypotheses = asr_model.transcribe(
                            audio=[temp_chunk_path],
                            batch_size=1,
                            return_hypotheses=True
                        )

                        chunk_logits = hypotheses[0].alignments
                        print(f"      Chunk {i+1} logits shape: {chunk_logits.shape}")

                        # Move blank column for this chunk
                        blank_col = chunk_logits[:, -1].reshape((chunk_logits.shape[0], 1))
                        chunk_logits = np.concatenate((blank_col, chunk_logits[:, :-1]), axis=1)

                        all_logits.append(chunk_logits)

                    # Clean up temp file
                    import os
                    os.unlink(temp_chunk_path)

                    # Clear CUDA cache after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as chunk_err:
                    print(f"      ERROR: Chunk {i+1} failed: {chunk_err}")
                    import os
                    if os.path.exists(temp_chunk_path):
                        os.unlink(temp_chunk_path)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            if not all_logits:
                print("  ERROR: All chunks failed")
                return []

            # Concatenate all chunk logits
            logits = np.concatenate(all_logits, axis=0)
            print(f"  Combined logits shape: {logits.shape}")
        else:
            raise

    # Calculate index_duration (time per CTC frame) dynamically
    # Following NeMo's approach: len(signal) / log_probs.shape[0] / sample_rate
    index_duration = len(audio) / logits.shape[0] / sample_rate
    print(f"  Index duration (time per CTC frame): {index_duration:.4f}s")

    # Verify logits duration
    logits_duration = logits.shape[0] * index_duration
    print(f"  Logits represent {logits_duration:.1f}s of audio (should match {audio_duration:.1f}s)")

    # Prepare text for alignment
    # CRITICAL: Pass list of sentences, NOT joined string!
    # The ctc-segmentation library needs sentences as separate list items
    text_list = transcript_sentences  # Keep as list
    text_char_count = sum(len(s) for s in text_list)
    text_word_count = sum(len(s.split()) for s in text_list)

    print(f"  Transcript stats:")
    print(f"    - Sentences: {len(text_list)}")
    print(f"    - Total characters: {text_char_count}")
    print(f"    - Total words: {text_word_count}")
    print(f"    - First sentence: {text_list[0][:100] if text_list else 'NONE'}")
    print(f"    - Last sentence: {text_list[-1][:100] if text_list else 'NONE'}")

    # CTC segmentation configuration
    config = CtcSegmentationParameters()

    # Extract vocabulary from ASR model (following NeMo's approach)
    # Check if BPE or character-based model
    if hasattr(asr_model, 'tokenizer'):  # BPE-based model
        vocabulary = asr_model.tokenizer.vocab
        print(f"  Detected BPE-based model")
    elif hasattr(asr_model.decoder, "vocabulary"):  # Character-based model
        vocabulary = asr_model.cfg.decoder.vocabulary  # Use cfg to get the raw list
        print(f"  Detected character-based model")
    else:
        raise ValueError("Unexpected model type. Vocabulary list not found.")

    print(f"  Raw vocabulary type: {type(vocabulary)}")

    # CRITICAL: Convert to regular Python list (force conversion from OmegaConf)
    # Must use list() constructor to create a new Python list
    vocabulary = list(vocabulary)

    # Verify it's actually a list now
    print(f"  Converted vocabulary type: {type(vocabulary)}")
    print(f"  First 5 tokens: {vocabulary[:5] if len(vocabulary) >= 5 else vocabulary}")

    # Add blank symbol "ε" at the beginning (position 0)
    vocabulary = ["ε"] + vocabulary

    # Assign to config - must be plain Python list
    config.char_list = vocabulary
    config.index_duration = index_duration  # Use calculated value, not hardcoded
    config.blank = 0  # Blank is at position 0 after moving the column

    print(f"  Vocabulary size (with blank): {len(config.char_list)} tokens")
    print(f"  Config char_list type: {type(config.char_list)}")
    print(f"  Blank token at position: {config.blank}")

    # Run CTC segmentation
    print("  Running CTC segmentation...")

    # DETAILED LOGGING before calling prepare_text
    print(f"  DEBUG: About to call prepare_text()")
    print(f"  DEBUG: config.char_list type = {type(config.char_list)}")
    print(f"  DEBUG: config.char_list is list? {isinstance(config.char_list, list)}")
    print(f"  DEBUG: config.char_list length = {len(config.char_list)}")
    if len(config.char_list) > 0:
        print(f"  DEBUG: config.char_list[0] type = {type(config.char_list[0])}")
        print(f"  DEBUG: config.char_list[0] value = {config.char_list[0]}")
    print(f"  DEBUG: text_list length = {len(text_list)} sentences")
    print(f"  DEBUG: text_list type = {type(text_list)}")

    try:
        print(f"  Preparing ground truth matrix for {len(text_list)} sentences...")

        # Check if BPE model or character-based model
        is_bpe_model = hasattr(asr_model, 'tokenizer')

        if is_bpe_model:
            # For BPE models, use custom preparation (following NeMo's approach)
            print(f"  Using BPE tokenization...")
            space_idx = vocabulary.index("▁")
            ground_truth_mat = [[-1, -1]]
            utt_begin_indices = []

            for uttr in text_list:
                ground_truth_mat += [[0, space_idx]]  # blank_idx=0, space
                utt_begin_indices.append(len(ground_truth_mat))
                token_ids = asr_model.tokenizer.text_to_ids(uttr)
                # blank token is moved from the last to the first (0) position
                token_ids = [idx + 1 for idx in token_ids]
                ground_truth_mat += [[t, -1] for t in token_ids]

            utt_begin_indices.append(len(ground_truth_mat))
            ground_truth_mat += [[0, space_idx]]
            ground_truth_mat = np.array(ground_truth_mat, np.int64)
        else:
            # For character-based models, use standard prepare_text
            print(f"  Using character-based segmentation...")
            config.excluded_characters = ".,-?!:»«;'›‹()"
            config.blank = vocabulary.index(" ")
            ground_truth_mat, utt_begin_indices = prepare_text(config, text_list)
            # Reset blank to 0 after prepare_text
            config.blank = 0

        print(f"  Ground truth matrix shape: {ground_truth_mat.shape}")
        print(f"  Utterance begin indices: {len(utt_begin_indices)} utterances")

        print(f"  Calling ctc_segmentation()...")
        print(f"  DEBUG: logits type = {type(logits)}")
        print(f"  DEBUG: logits shape = {logits.shape}")
        print(f"  DEBUG: ground_truth_mat type = {type(ground_truth_mat)}")

        timings, char_probs, state_list = ctc_segmentation(
            config, logits, ground_truth_mat
        )
        print(f"  ctc_segmentation() succeeded!")
        print(f"  Generated {len(timings)} timing points")
    except Exception as e:
        import traceback
        print(f"  CTC segmentation FAILED!")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {str(e)}")
        print(f"  Full traceback:")
        traceback.print_exc()
        print(f"  ")
        print(f"  Diagnostic info:")
        print(f"    - Audio duration: {audio_duration:.1f}s")
        print(f"    - Logits duration: {logits_duration:.1f}s")
        print(f"    - Transcript words: {text_word_count} (~{text_word_count/150:.1f} min at 150 WPM)")
        print(f"    - Config char_list final type: {type(config.char_list)}")
        print(f"  Skipping this file...")
        return []

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
            print(f"    Skipping low-confidence segment: {confidence:.2f} < {min_confidence}")

    print(f"  Generated {len(segments)} segments from {len(transcript_sentences)} sentences")
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
    from scripts.data import data_loader
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

    print(f"  Raw transcript type: {type(full_transcript)}")
    print(f"  Raw transcript length: {len(full_transcript)} chars")

    sentences = prepare_text_for_segmentation(full_transcript, verbose=True)
    print(f"  Split into {len(sentences)} sentences")

    # Calculate expected speech duration
    total_words = sum(len(s.split()) for s in sentences)
    expected_duration_min = total_words / 150  # 150 words per minute
    print(f"  Transcript has {total_words} words, expecting ~{expected_duration_min:.1f} min of speech")

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

    # Clear CUDA cache after processing this file to free memory for next file
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  Cleared CUDA cache after processing file")

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
