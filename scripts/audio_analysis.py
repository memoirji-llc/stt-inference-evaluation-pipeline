"""
Audio Quality Analysis - Sequential Processing

Calculates audio quality metrics for all VHP files from Azure Blob Storage.
Output: audio_quality_analysis.parquet

Usage:
    uv run scripts/audio_analysis.py
"""

import sys
import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import pyloudnorm as pyln
from tqdm import tqdm
import io
from pydub import AudioSegment
from collections import Counter

# Azure blob utilities
from azure_utils import list_blobs, download_blob_to_memory


# Set Azure authentication environment variables
os.environ['AZURE_STORAGE_ACCOUNT'] = 'stgamiadata26828'
os.environ['AZURE_STORAGE_CONTAINER'] = 'audio-raw'
os.environ['AZURE_AUTH'] = 'connection_string'
os.environ['AZURE_STORAGE_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=stgamiadata26828;AccountKey=Ol7WsOhceB+UxH5x33nfL6dZZLG4coJBgaWAqsbuzMMZLZKnjCS8BCbeinEIdN/h8437NQosRiAI+AStBeJqdw==;EndpointSuffix=core.windows.net'

print("✓ Azure credentials set in environment")


# ==============================================================================
# Audio Quality Metric Functions
# ==============================================================================

def snr_cal(wv, sr) -> np.float32:
    """Calculate Signal-to-Noise Ratio (dB)"""
    rms_full = librosa.feature.rms(y=wv)
    noise_rms = np.mean(rms_full[:, :int(0.5*sr/512)])
    signal_rms = np.mean(rms_full)
    snr_db = 20 * np.log10(signal_rms / noise_rms)
    return snr_db


def spectral_rolloff_cal(wv, sr) -> np.float64:
    """Calculate spectral roll-off (85th percentile frequency)"""
    rolloff = librosa.feature.spectral_rolloff(y=wv, sr=sr, roll_percent=0.85)
    return np.median(rolloff)


def spectral_flatness_cal(wv) -> np.float64:
    """Calculate spectral flatness (0=tonal, 1=noisy)"""
    flatness = librosa.feature.spectral_flatness(y=wv)
    return np.mean(flatness)


def spectral_centroid_cal(wv, sr) -> np.float64:
    """Calculate spectral centroid (brightness)"""
    centroid = librosa.feature.spectral_centroid(y=wv, sr=sr)
    return np.median(centroid)


def zcr_cal(wv):
    """Calculate zero crossing rate (mean and variance)"""
    zcr = librosa.feature.zero_crossing_rate(wv)
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)
    return zcr_mean, zcr_var


def loudness_cal(wv, sr) -> np.float64:
    """Calculate loudness (LUFS - ITU-R BS.1770-4)"""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wv)
    return loudness


def low_frequency_energy(wv, sr, cutoff_hz=80) -> np.float64:
    """Calculate energy below cutoff frequency (for rumble detection)"""
    # Compute STFT
    D = np.abs(librosa.stft(wv))
    freqs = librosa.fft_frequencies(sr=sr)

    # Energy below cutoff
    low_freq_mask = freqs < cutoff_hz
    low_energy = np.sum(D[low_freq_mask, :])
    total_energy = np.sum(D)

    # Return ratio
    return low_energy / total_energy if total_energy > 0 else 0.0


# ==============================================================================
# Audio Analysis Function
# ==============================================================================

def analyze_audio_file(audio_bytes, audio_id: str) -> dict:
    """
    Analyze audio file and return quality metrics.

    Handles both audio (MP3, WAV) and video (MP4) files using pydub.

    Args:
        audio_bytes: Audio/video file as bytes
        audio_id: Identifier for the audio file

    Returns:
        Dictionary with audio_id and quality metrics
    """
    try:
        # Use pydub to load audio (handles MP3, MP4, M4A, WAV, etc.)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # Convert to mono and 16kHz (consistent with pipeline)
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(16000)

        # Convert to numpy array
        samples = np.array(audio_segment.get_array_of_samples())

        # Normalize to float32 [-1, 1]
        if audio_segment.sample_width == 2:  # 16-bit
            wv = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            wv = samples.astype(np.float32) / 2147483648.0
        else:
            wv = samples.astype(np.float32)

        sr = 16000  # Sample rate after conversion

        # Calculate metrics
        snr = float(snr_cal(wv, sr))
        rolloff = float(spectral_rolloff_cal(wv, sr))
        flatness = float(spectral_flatness_cal(wv))
        centroid = float(spectral_centroid_cal(wv, sr))
        zcr_mean, zcr_var = zcr_cal(wv)
        zcr_mean = float(zcr_mean)
        zcr_var = float(zcr_var)
        loudness = float(loudness_cal(wv, sr))
        low_freq_energy_ratio = float(low_frequency_energy(wv, sr, cutoff_hz=80))

        # Duration
        duration_sec = len(wv) / sr

        return {
            'audio_id': audio_id,
            'sample_rate': sr,
            'duration_sec': duration_sec,
            'snr_db': snr,
            'spectral_rolloff_hz': rolloff,
            'spectral_flatness': flatness,
            'spectral_centroid_hz': centroid,
            'zcr_mean': zcr_mean,
            'zcr_var': zcr_var,
            'loudness_lufs': loudness,
            'low_freq_energy_ratio': low_freq_energy_ratio,
            'status': 'success'
        }

    except Exception as e:
        return {
            'audio_id': audio_id,
            'status': 'error',
            'error_message': str(e)
        }


# ==============================================================================
# Issue Detection Functions
# ==============================================================================

def detect_audio_issues(row: pd.Series) -> list:
    """
    Detect audio issues based on quality metrics.

    Args:
        row: DataFrame row with audio metrics

    Returns:
        List of detected issues
    """
    issues = []

    if row['status'] != 'success':
        return issues

    # 1. Bandwidth-limited (lacking high frequencies)
    if row['spectral_rolloff_hz'] < 1000:
        issues.append('bandwidth_limited_severe')
    elif row['spectral_rolloff_hz'] < 4000:
        issues.append('bandwidth_limited_moderate')

    # 2. High noise
    if row['zcr_mean'] > 0.05:
        issues.append('high_noise_zcr')
    if row['snr_db'] < 15:
        issues.append('high_noise_snr')

    # 3. Low-frequency rumble
    if row['low_freq_energy_ratio'] > 0.15:
        issues.append('low_frequency_rumble')

    # 4. Low loudness
    if row['loudness_lufs'] < -30:
        issues.append('low_loudness')

    # 5. Very flat spectrum
    if row['spectral_flatness'] > 0.8:
        issues.append('very_flat_spectrum')

    return issues


def recommend_preprocessing(issues: list) -> list:
    """
    Recommend preprocessing methods based on detected issues.

    Args:
        issues: List of detected issues

    Returns:
        List of recommended preprocessing methods
    """
    recommendations = []

    # Always normalize loudness
    recommendations.append('loudness_normalization')

    # Bandwidth-limited → EQ boost high frequencies
    if 'bandwidth_limited_severe' in issues or 'bandwidth_limited_moderate' in issues:
        recommendations.append('eq_high_freq_boost')

    # High noise → Noise reduction
    if 'high_noise_zcr' in issues or 'high_noise_snr' in issues:
        recommendations.append('noise_reduction')

    # Low-frequency rumble → High-pass filter
    if 'low_frequency_rumble' in issues:
        recommendations.append('highpass_filter')

    return recommendations


# ==============================================================================
# Main Processing
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("Audio Quality Analysis - Sequential Processing")
    print("="*60 + "\n")

    # List all audio/video blobs from Azure
    blob_prefix = "loc_vhp/"

    print(f"Listing blobs with prefix: {blob_prefix}")
    audio_blobs = list_blobs(blob_prefix)

    # Filter for audio AND video files
    media_extensions = ('.mp3', '.mp4', '.wav', '.m4a', '.flac', '.ogg')
    audio_blobs = [b for b in audio_blobs if b.lower().endswith(media_extensions)]

    print(f"Found {len(audio_blobs)} media files")
    print(f"\nFirst 5 files:")
    for blob in audio_blobs[:5]:
        print(f"  - {blob}")

    # TESTING: Uncomment to limit processing
    # SAMPLE_SIZE = 100
    # audio_blobs = audio_blobs[:SAMPLE_SIZE]
    # print(f"\n🧪 TEST MODE: Processing only first {SAMPLE_SIZE} files")

    # Process all audio files (streaming - one at a time)
    results = []

    print(f"\nProcessing {len(audio_blobs)} audio files (streaming)...")
    print("Memory usage: Only stores metrics (~2-3 KB per file), not audio\n")

    for blob_path in tqdm(audio_blobs, desc="Processing"):
        audio_id = Path(blob_path).stem

        try:
            # Download ONE file, analyze, discard audio, keep metrics
            audio_bytes = download_blob_to_memory(blob_path)
            result = analyze_audio_file(audio_bytes, audio_id)
            results.append(result)

        except Exception as e:
            print(f"Error processing {blob_path}: {e}")
            results.append({
                'audio_id': audio_id,
                'status': 'error',
                'error_message': str(e)
            })

    print(f"\nProcessed {len(results)} files")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    print(f"Total files: {len(df)}")
    print(f"Successful: {(df['status'] == 'success').sum()}")
    print(f"Errors: {(df['status'] == 'error').sum()}")

    # Detect issues
    print("\nDetecting audio issues...")
    df['issues'] = df.apply(detect_audio_issues, axis=1)
    df['recommended_preprocessing'] = df['issues'].apply(recommend_preprocessing)

    df_with_issues = df[df['issues'].str.len() > 0]

    print(f"Files with issues: {len(df_with_issues)} / {len(df)}")

    all_issues = [issue for issues in df['issues'] for issue in issues]
    issue_counts = Counter(all_issues)

    print("\nIssue breakdown:")
    for issue, count in issue_counts.most_common():
        print(f"  {issue}: {count} files ({count/len(df)*100:.1f}%)")

    # Summary statistics
    df_success = df[df['status'] == 'success']

    if len(df_success) > 0:
        print("\nOverall Statistics:")
        print(df_success[['snr_db', 'spectral_rolloff_hz', 'spectral_flatness',
                          'zcr_mean', 'loudness_lufs']].describe())

        print("\nPreprocessing Recommendations:")
        all_recs = [rec for recs in df['recommended_preprocessing'] for rec in recs]
        rec_counts = Counter(all_recs)

        for rec, count in rec_counts.most_common():
            print(f"{rec:30s}: {count:4d} files ({count/len(df)*100:5.1f}%)")

    # Save results
    output_path = Path("data/audio_quality_analysis.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\n✓ Saved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\n" + "="*60)
    print("✓ All done!")
    print("="*60)


if __name__ == "__main__":
    main()
