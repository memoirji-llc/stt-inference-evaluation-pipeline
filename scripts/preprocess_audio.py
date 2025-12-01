#!/usr/bin/env python3
"""
Audio Preprocessing Script for VHP Archival Speech

Orchestrates: Download (audio-raw) → Preprocess → Upload (audio-processed)

Available preprocessing methods:
- highpass_filter: Remove low-frequency rumble (<80 Hz)
- loudness_normalization: Normalize to target LUFS
- noise_reduction: Reduce background noise
- eq_high_freq_boost: Boost high frequencies (TODO)

Usage:
    # Single preprocessing method (recommended)
    python scripts/preprocess_audio.py \\
        --parquet data/raw/loc/veterans_history_project_resources_pre2010_test.parquet \\
        --method highpass_filter \\
        --source_container audio-raw \\
        --source_prefix loc_vhp \\
        --dest_container audio-processed \\
        --sample_size 10

Output structure:
    audio-processed/{method_name}/loc_vhp/{audio_id}/audio_processed.wav
"""

import io
import numpy as np
import pyloudnorm as pyln
from scipy import signal


def load_audio_bytes(audio_bytes, target_sr=16000):
    """
    Load audio from bytes to numpy array.

    Args:
        audio_bytes: Audio file bytes (mp3, mp4, wav, etc.)
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        waveform (np.array, float32, [-1, 1]), sample_rate (int)
    """
    from pydub import AudioSegment

    # Load with pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(target_sr)  # Resample

    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Normalize to float32 [-1, 1]
    if audio.sample_width == 2:  # 16-bit PCM
        waveform = samples.astype(np.float32) / 32768.0
    elif audio.sample_width == 4:  # 32-bit PCM
        waveform = samples.astype(np.float32) / 2147483648.0
    else:
        waveform = samples.astype(np.float32)

    return waveform, target_sr


def save_audio_bytes(waveform, sr, format='wav'):
    """
    Save waveform to bytes.

    Args:
        waveform: Audio waveform (np.array, float32, [-1, 1])
        sr: Sample rate
        format: Output format ('wav' for lossless, 'mp3' for compressed)

    Returns:
        bytes
    """
    from pydub import AudioSegment

    # Convert float32 [-1, 1] to int16
    waveform_int16 = (waveform * 32767).astype(np.int16)

    # Create AudioSegment
    audio = AudioSegment(
        waveform_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit
        channels=1       # Mono
    )

    # Export to bytes
    buffer = io.BytesIO()
    audio.export(buffer, format=format)
    return buffer.getvalue()


def loudness_normalize(waveform, sr, target_lufs=-16.0):
    """
    Normalize audio loudness to target LUFS using pyloudnorm.

    Recommended for all audio to ensure consistent volume levels across samples.

    Args:
        waveform: Audio waveform (np.array, float32)
        sr: Sample rate
        target_lufs: Target loudness in LUFS (default: -16.0, broadcast standard)

    Returns:
        Normalized waveform (np.array, float32, clipped to [-1, 1])
    """
    # Ensure minimum length for loudness meter (0.4 seconds)
    min_length = int(0.4 * sr)
    if len(waveform) < min_length:
        waveform = np.pad(waveform, (0, min_length - len(waveform)), mode='constant')

    # Measure current loudness
    meter = pyln.Meter(sr)
    try:
        current_lufs = meter.integrated_loudness(waveform)
    except:
        # If measurement fails (e.g., silence), return original
        return waveform

    # Skip normalization if loudness is too low (likely silence)
    if current_lufs < -70:
        return waveform

    # Normalize to target loudness (positional args: audio, input_loudness, target_loudness)
    normalized = pyln.normalize.loudness(waveform, current_lufs, target_lufs)

    # Clip to prevent clipping artifacts
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def highpass_filter(waveform, sr, cutoff_hz=80, order=5):
    """
    Apply high-pass filter to remove low-frequency rumble.

    Recommended for audio with low-frequency energy ratio >0.15.

    Args:
        waveform: Audio waveform (np.array, float32)
        sr: Sample rate
        cutoff_hz: Cutoff frequency (default: 80 Hz, removes tape hum and rumble)
        order: Filter order (default: 5, good balance of rolloff vs. phase)

    Returns:
        Filtered waveform (np.array, float32)
    """
    # Design Butterworth high-pass filter
    nyquist = sr / 2.0
    normal_cutoff = cutoff_hz / nyquist

    # Ensure cutoff is in valid range
    if normal_cutoff >= 1.0:
        # Cutoff too high, return original
        return waveform

    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)

    # Apply zero-phase filter (filtfilt) to avoid phase distortion
    filtered = signal.filtfilt(b, a, waveform)

    return filtered.astype(np.float32)


def noise_reduce(waveform, sr, stationary=True):
    """
    Apply noise reduction using noisereduce library.

    Recommended for audio with high noise (SNR <15 dB or ZCR >0.05).

    Args:
        waveform: Audio waveform (np.array, float32)
        sr: Sample rate
        stationary: Assume stationary noise (True for archival audio)

    Returns:
        Denoised waveform (np.array, float32)
    """
    try:
        import noisereduce as nr

        # Apply noise reduction
        # For archival audio, use first 0.5 seconds as noise profile
        reduced = nr.reduce_noise(
            y=waveform,
            sr=sr,
            stationary=stationary,
            prop_decrease=1.0  # Full noise reduction
        )

        return reduced.astype(np.float32)

    except ImportError:
        print("[WARNING] noisereduce not installed. Install with: pip install noisereduce")
        return waveform


def eq_high_freq_boost(waveform, sr, rolloff_hz=1000, boost_db=6.0):
    """
    Apply high-frequency boost for bandwidth-limited audio.

    Recommended for audio with spectral rolloff <4000 Hz.

    TODO: Implement high-shelf filter using scipy or torchaudio.

    Args:
        waveform: Audio waveform (np.array, float32)
        sr: Sample rate
        rolloff_hz: Frequency where boost starts (from audio analysis)
        boost_db: Amount of boost in dB (default: 6.0)

    Returns:
        Boosted waveform (np.array, float32)
    """
    # TODO: Implement high-shelf EQ filter
    # Options:
    # 1. scipy.signal.iirfilter with 'highshelf' btype (if available)
    # 2. Manual biquad filter design
    # 3. torchaudio.transforms.Equalizer

    print(f"[TODO] EQ high-freq boost above {rolloff_hz} Hz by {boost_db} dB")
    print("       Implement high-shelf filter in this function")

    return waveform


def preprocess_audio_single(audio_bytes, method, audio_id=None, metrics=None):
    """
    Apply single preprocessing method to audio.

    Args:
        audio_bytes: Original audio bytes (mp3, mp4, wav, etc.)
        method: Preprocessing method name
                Options: 'loudness_normalization', 'highpass_filter',
                        'noise_reduction', 'eq_high_freq_boost'
        audio_id: Audio ID for logging (optional)
        metrics: Dict of metrics from audio analysis (for adaptive processing)

    Returns:
        Preprocessed audio bytes (WAV format for lossless storage)

    Example:
        >>> audio_bytes_preprocessed = preprocess_audio_single(
        ...     audio_bytes_raw,
        ...     method='highpass_filter',
        ...     audio_id='1234'
        ... )
    """
    # Load audio
    waveform, sr = load_audio_bytes(audio_bytes, target_sr=16000)

    # Apply the single method
    if method == 'loudness_normalization':
        waveform = loudness_normalize(waveform, sr, target_lufs=-16.0)

    elif method == 'eq_high_freq_boost':
        # Use rolloff from metrics if available
        rolloff = metrics.get('spectral_rolloff_hz', 1000) if metrics else 1000
        waveform = eq_high_freq_boost(waveform, sr, rolloff_hz=rolloff)

    elif method == 'noise_reduction':
        waveform = noise_reduce(waveform, sr, stationary=True)

    elif method == 'highpass_filter':
        waveform = highpass_filter(waveform, sr, cutoff_hz=80)

    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    # Convert back to bytes (WAV for lossless storage)
    processed_bytes = save_audio_bytes(waveform, sr, format='wav')

    return processed_bytes


# Available preprocessing methods
AVAILABLE_METHODS = [
    'highpass_filter',
    'loudness_normalization',
    'noise_reduction',
    'eq_high_freq_boost'
]


def main():
    """
    CLI for batch preprocessing with Azure orchestration.

    Usage:
        python scripts/preprocess_audio.py \\
            --parquet data/raw/loc/veterans_history_project_resources_pre2010_test.parquet \\
            --method highpass_filter \\
            --source_container audio-raw \\
            --source_prefix loc_vhp \\
            --dest_container audio-processed \\
            --sample_size 10
    """
    import argparse
    import pandas as pd
    import sys
    import logging
    from pathlib import Path
    from tqdm import tqdm
    from dotenv import load_dotenv

    # Load credentials
    load_dotenv(dotenv_path='credentials/creds.env')

    from cloud.azure_utils import download_blob_to_memory, upload_blob

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    # Suppress Azure SDK verbose logging
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Batch audio preprocessing with Azure orchestration')
    parser.add_argument('--parquet', required=True,
                       help='Path to parquet file with audio metadata')
    parser.add_argument('--method', required=True, choices=AVAILABLE_METHODS,
                       help='Preprocessing method to apply (single method only)')
    parser.add_argument('--source_container', default='audio-raw',
                       help='Azure blob container for source audio (default: audio-raw)')
    parser.add_argument('--source_prefix', default='loc_vhp',
                       help='Blob prefix for source audio (default: loc_vhp)')
    parser.add_argument('--dest_container', default='audio-processed',
                       help='Azure blob container for output (default: audio-processed)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Number of files to process before checkpoint (default: 100)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Limit number of files to process (for testing)')

    args = parser.parse_args()

    # Load parquet
    logger.info(f"Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    if args.sample_size:
        df = df.head(args.sample_size)
        logger.info(f"Limited to {args.sample_size} samples for testing")

    logger.info("=" * 80)
    logger.info("AUDIO PREPROCESSING - AZURE ORCHESTRATION")
    logger.info("=" * 80)
    logger.info(f"Files to process:    {len(df)}")
    logger.info(f"Method:              {args.method}")
    logger.info(f"Source container:    {args.source_container}")
    logger.info(f"Source prefix:       {args.source_prefix}")
    logger.info(f"Dest container:      {args.dest_container}")
    logger.info(f"Output structure:    {args.dest_container}/{args.method}/{args.source_prefix}/{{audio_id}}/audio_processed.wav")
    logger.info("=" * 80)
    logger.info("")

    # Process files
    processed_count = 0
    error_count = 0
    skipped_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        audio_id = row.get('azure_blob_index', idx)

        try:
            # Step 1: Download from source container
            source_path = f"{args.source_prefix}/{audio_id}/audio.mp3"
            try:
                logger.debug(f"Downloading: {args.source_container}/{source_path}")
                audio_bytes = download_blob_to_memory(
                    blob_path=source_path,
                    container=args.source_container
                )
            except Exception as e:
                # Try video.mp4 if audio.mp3 fails
                source_path = f"{args.source_prefix}/{audio_id}/video.mp4"
                logger.debug(f"audio.mp3 not found, trying: {args.source_container}/{source_path}")
                audio_bytes = download_blob_to_memory(
                    blob_path=source_path,
                    container=args.source_container
                )

            # Step 2: Apply preprocessing
            logger.debug(f"Preprocessing with method: {args.method}")
            audio_bytes_processed = preprocess_audio_single(
                audio_bytes,
                method=args.method,
                audio_id=audio_id
            )

            # Step 3: Upload to destination container
            dest_path = f"{args.method}/{args.source_prefix}/{audio_id}/audio_processed.wav"
            logger.debug(f"Uploading: {args.dest_container}/{dest_path}")
            upload_blob(
                blob_path=dest_path,
                data=audio_bytes_processed,
                container=args.dest_container
            )

            processed_count += 1

        except FileNotFoundError as e:
            logger.warning(f"[SKIP] Audio not found for {audio_id}: {e}")
            skipped_count += 1

        except Exception as e:
            logger.error(f"[ERROR] Failed to process {audio_id}: {e}")
            error_count += 1

        # Checkpoint
        if (processed_count + error_count + skipped_count) % args.batch_size == 0:
            logger.info("")
            logger.info(f"Checkpoint: {processed_count} processed, {error_count} errors, {skipped_count} skipped")
            logger.info("")

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed:  {processed_count}")
    logger.info(f"Errors:     {error_count}")
    logger.info(f"Skipped:    {skipped_count}")
    logger.info(f"Total:      {len(df)}")
    logger.info("")
    logger.info(f"Output location: {args.dest_container}/{args.method}/{args.source_prefix}/")
    logger.info("")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
