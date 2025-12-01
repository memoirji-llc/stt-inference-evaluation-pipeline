"""
Audio Preprocessing Functions for VHP Archival Speech

Based on audio quality analysis results, provides functions to:
- Normalize loudness to target LUFS
- Remove low-frequency rumble with high-pass filter
- Reduce background noise
- Boost high frequencies for bandwidth-limited audio (TODO)

Usage:
    # In notebooks:
    from scripts.preprocess_audio import preprocess_audio

    # In CLI (for batch processing):
    python scripts/preprocess_audio.py --parquet data/test.parquet --variant full_pipeline
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


def preprocess_audio(audio_bytes, methods, audio_id=None, metrics=None):
    """
    Apply preprocessing pipeline to audio.

    Args:
        audio_bytes: Original audio bytes (mp3, mp4, wav, etc.)
        methods: List of method names to apply (in order)
                 Options: 'loudness_normalization', 'highpass_filter',
                         'noise_reduction', 'eq_high_freq_boost'
        audio_id: Audio ID for logging (optional)
        metrics: Dict of metrics from audio analysis (for adaptive processing)

    Returns:
        Preprocessed audio bytes (WAV format for lossless storage)

    Example:
        >>> audio_bytes_preprocessed = preprocess_audio(
        ...     audio_bytes_raw,
        ...     methods=['highpass_filter', 'noise_reduction', 'loudness_normalization'],
        ...     audio_id='1234',
        ...     metrics={'spectral_rolloff_hz': 850}
        ... )
    """
    # Load audio
    waveform, sr = load_audio_bytes(audio_bytes, target_sr=16000)

    # Apply methods in order
    for method in methods:
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
            print(f"[WARNING] Unknown preprocessing method: {method}")

    # Convert back to bytes (WAV for lossless storage)
    processed_bytes = save_audio_bytes(waveform, sr, format='wav')

    return processed_bytes


# Preprocessing variant presets
PREPROCESSING_VARIANTS = {
    'raw': [],  # No preprocessing (baseline)
    'normalized': ['loudness_normalization'],
    'normalized_eq': ['loudness_normalization', 'eq_high_freq_boost'],
    'normalized_denoised': ['loudness_normalization', 'noise_reduction'],
    'full_pipeline': ['highpass_filter', 'noise_reduction', 'eq_high_freq_boost', 'loudness_normalization']
}


def main():
    """
    CLI for batch preprocessing.

    Usage:
        python scripts/preprocess_audio.py \\
            --parquet data/raw/loc/veterans_history_project_resources_pre2010_test.parquet \\
            --variant full_pipeline \\
            --output_prefix loc_vhp_preprocessed/full_pipeline \\
            --batch_size 100
    """
    import argparse
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from cloud.azure_utils import download_blob_to_memory, upload_blob_from_memory

    parser = argparse.ArgumentParser(description='Batch audio preprocessing for VHP dataset')
    parser.add_argument('--parquet', required=True, help='Path to parquet file with audio metadata')
    parser.add_argument('--variant', required=True, choices=list(PREPROCESSING_VARIANTS.keys()),
                       help='Preprocessing variant to apply')
    parser.add_argument('--output_prefix', required=True,
                       help='Azure blob prefix for output (e.g., loc_vhp_preprocessed/full_pipeline)')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Number of files to process before checkpoint')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Limit number of files to process (for testing)')

    args = parser.parse_args()

    # Load parquet
    df = pd.read_parquet(args.parquet)
    if args.sample_size:
        df = df.head(args.sample_size)

    print(f"Processing {len(df)} files with variant: {args.variant}")
    print(f"Methods: {PREPROCESSING_VARIANTS[args.variant]}")
    print(f"Output prefix: {args.output_prefix}")
    print("=" * 70)

    # Load analysis results if available for adaptive processing
    analysis_path = Path('learnings/audio_quality_analysis/audio_quality_analysis.parquet')
    if analysis_path.exists():
        df_analysis = pd.read_parquet(analysis_path)
        print(f"✓ Loaded audio quality analysis for {len(df_analysis)} files")
    else:
        df_analysis = None
        print("⚠ No audio quality analysis found, proceeding without metrics")

    # Process files
    methods = PREPROCESSING_VARIANTS[args.variant]
    processed_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        audio_id = row.get('azure_blob_index', idx)

        try:
            # Download original
            blob_path = f"loc_vhp/{audio_id}/audio.mp3"
            try:
                audio_bytes = download_blob_to_memory(blob_path)
            except:
                blob_path = f"loc_vhp/{audio_id}/video.mp4"
                audio_bytes = download_blob_to_memory(blob_path)

            # Get metrics for adaptive processing
            metrics = None
            if df_analysis is not None:
                metrics_row = df_analysis[df_analysis['audio_id'] == str(audio_id)]
                if len(metrics_row) > 0:
                    metrics = metrics_row.iloc[0].to_dict()

            # Preprocess
            if methods:  # Skip if raw variant
                audio_bytes = preprocess_audio(audio_bytes, methods, audio_id, metrics)

            # Upload
            output_path = f"{args.output_prefix}/{audio_id}/audio.wav"
            upload_blob_from_memory(audio_bytes, output_path)

            processed_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to process {audio_id}: {e}")
            error_count += 1

        # Checkpoint
        if (processed_count + error_count) % args.batch_size == 0:
            print(f"\nCheckpoint: {processed_count} processed, {error_count} errors")

    print("\n" + "=" * 70)
    print(f"✓ COMPLETE")
    print(f"  Processed: {processed_count}")
    print(f"  Errors: {error_count}")


if __name__ == '__main__':
    main()
