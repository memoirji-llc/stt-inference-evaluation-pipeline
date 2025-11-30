#!/usr/bin/env python3
"""
Audio Quality Analysis - GPU-Accelerated Batch Processing

Standalone script version of notebooks/audio_analysis_fast_gpu.ipynb
Run in a screen session to prevent disconnection issues.

Usage:
    screen -S audio_analysis
    cd /workspace/amia2025-stt-benchmarking
    python scripts/run_audio_analysis.py

Output: data/audio_quality_analysis.parquet
"""

import sys
sys.path.append("scripts")

import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
import pyloudnorm as pyln
from tqdm import tqdm
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import os
import psutil
from collections import Counter

# Azure blob utilities
from cloud.azure_utils import list_blobs, download_blob_to_memory

# Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path='credentials/creds.env')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # A6000/Ampere optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("Enabled: TF32 matmul, TF32 cudnn, cudnn benchmark")

# Processing configuration - CONSERVATIVE to avoid OOM
# The VHP files are large (some 30+ min audio = 100MB+ each)
BATCH_SIZE = 32           # GPU batch - reduced from 128
DOWNLOAD_WORKERS = 50     # Parallel downloads - reduced from 100
DOWNLOAD_BATCH_SIZE = 100 # Files in RAM at once - reduced from 400
TARGET_SR = 16000

# Memory threshold - trigger GC if above this (in GB)
RAM_THRESHOLD_GB = 40

print(f"\nConfiguration:")
print(f"  Batch size: {BATCH_SIZE} files")
print(f"  Download workers: {DOWNLOAD_WORKERS}")
print(f"  Download batch size: {DOWNLOAD_BATCH_SIZE}")
print(f"  RAM threshold: {RAM_THRESHOLD_GB} GB")


# ============================================================================
# Audio Processing Functions
# ============================================================================

def load_audio_to_tensor(audio_bytes, target_sr=16000):
    """Load audio bytes to torch tensor and resample."""
    from pydub import AudioSegment

    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio_segment = audio_segment.set_channels(1)
    audio_segment = audio_segment.set_frame_rate(target_sr)

    samples = np.array(audio_segment.get_array_of_samples())

    if audio_segment.sample_width == 2:
        waveform = samples.astype(np.float32) / 32768.0
    elif audio_segment.sample_width == 4:
        waveform = samples.astype(np.float32) / 2147483648.0
    else:
        waveform = samples.astype(np.float32)

    return torch.from_numpy(waveform).float(), target_sr


def batch_compute_spectrogram(waveforms, sr, n_fft=2048, hop_length=512):
    """Compute spectrograms for batch of waveforms on GPU."""
    results = []
    freqs = torch.linspace(0, sr/2, n_fft//2 + 1, device=device)

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window_fn=torch.hann_window,
        power=None,
        center=True,
        pad_mode='reflect',
        normalized=False
    ).to(device)

    for wv in waveforms:
        wv_gpu = wv.to(device)
        min_length = n_fft
        if len(wv_gpu) < min_length:
            pad_needed = min_length - len(wv_gpu)
            wv_gpu = torch.nn.functional.pad(wv_gpu, (0, pad_needed), mode='constant', value=0)

        spec = spec_transform(wv_gpu)
        mag = torch.abs(spec)
        results.append((mag, freqs))

    return results


def snr_cal_batch(waveform, sr):
    """Calculate SNR matching librosa.feature.rms()."""
    frame_length = 2048
    hop_length = 512

    if len(waveform) < frame_length:
        pad_needed = frame_length - len(waveform)
        waveform = torch.nn.functional.pad(waveform, (0, pad_needed), mode='constant', value=0)

    pad_length = frame_length // 2
    waveform_2d = waveform.unsqueeze(0)
    waveform_padded = torch.nn.functional.pad(waveform_2d, (pad_length, pad_length), mode='reflect').squeeze(0)

    num_frames = 1 + (len(waveform_padded) - frame_length) // hop_length

    if num_frames <= 0:
        return 0.0

    rms_values = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(waveform_padded):
            break
        frame = waveform_padded[start:end]
        rms = torch.sqrt(torch.mean(frame ** 2))
        rms_values.append(rms)

    if len(rms_values) == 0:
        return 0.0

    rms_tensor = torch.stack(rms_values)
    noise_frames = int(0.5 * sr / hop_length)

    if noise_frames > 0 and noise_frames < len(rms_tensor):
        noise_rms = torch.mean(rms_tensor[:noise_frames])
        signal_rms = torch.mean(rms_tensor)

        if noise_rms > 0:
            snr_db = 20 * torch.log10(signal_rms / noise_rms)
            return snr_db.item()

    return 0.0


def spectral_rolloff_batch(spec_mag, freqs, sr, roll_percent=0.85):
    """Calculate spectral rolloff."""
    cumsum = torch.cumsum(spec_mag, dim=0)
    total_energy = cumsum[-1, :]
    threshold = roll_percent * total_energy
    rolloff_bins = torch.argmax((cumsum >= threshold.unsqueeze(0)).float(), dim=0)
    rolloff_hz = freqs[rolloff_bins]
    return torch.median(rolloff_hz).item()


def spectral_centroid_batch(spec_mag, freqs, sr):
    """Calculate spectral centroid."""
    freqs_2d = freqs.unsqueeze(1)
    centroid = torch.sum(freqs_2d * spec_mag, dim=0) / (torch.sum(spec_mag, dim=0) + 1e-8)
    return torch.median(centroid).item()


def spectral_flatness_batch(spec_mag):
    """Calculate spectral flatness."""
    spec_safe = spec_mag + 1e-10
    log_spec = torch.log(spec_safe)
    geometric_mean = torch.exp(torch.mean(log_spec, dim=0))
    arithmetic_mean = torch.mean(spec_mag, dim=0)
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    return torch.mean(flatness).item()


def zcr_batch(waveform):
    """Calculate zero crossing rate."""
    frame_length = 2048
    hop_length = 512

    if len(waveform) < frame_length:
        pad_needed = frame_length - len(waveform)
        waveform = torch.nn.functional.pad(waveform, (0, pad_needed), mode='constant', value=0)

    pad_length = frame_length // 2
    waveform_padded = torch.nn.functional.pad(waveform, (pad_length, pad_length), mode='constant', value=0)

    signs = torch.sign(waveform_padded)
    signs[signs == 0] = 0
    sign_changes = torch.abs(torch.diff(signs)) > 0

    num_frames = 1 + (len(waveform_padded) - frame_length) // hop_length

    if num_frames <= 0:
        return 0.0, 0.0

    zcr_values = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length - 1
        if end > len(sign_changes):
            break
        frame = sign_changes[start:end]
        zcr = torch.sum(frame.float()) / (frame_length - 1)
        zcr_values.append(zcr)

    if len(zcr_values) == 0:
        return 0.0, 0.0

    zcr_tensor = torch.stack(zcr_values)
    return torch.mean(zcr_tensor).item(), torch.var(zcr_tensor).item()


def loudness_cal(waveform, sr):
    """Calculate loudness (LUFS)."""
    wv_cpu = waveform.cpu().numpy()
    min_length = int(0.4 * sr)
    if len(wv_cpu) < min_length:
        wv_cpu = np.pad(wv_cpu, (0, min_length - len(wv_cpu)), mode='constant', constant_values=0)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wv_cpu)
    return float(loudness)


def low_frequency_energy_batch(spec_mag, freqs, sr, cutoff_hz=80):
    """Calculate low frequency energy ratio."""
    low_freq_mask = freqs < cutoff_hz
    low_energy = torch.sum(spec_mag[low_freq_mask, :])
    total_energy = torch.sum(spec_mag)

    if total_energy > 0:
        return (low_energy / total_energy).item()
    return 0.0


# ============================================================================
# Batch Processing
# ============================================================================

def analyze_audio_batch_gpu(audio_data_list):
    """Analyze a batch of audio files on GPU."""
    results = []
    waveforms = []
    audio_ids = []

    for audio_id, audio_bytes in audio_data_list:
        try:
            wv, sr = load_audio_to_tensor(audio_bytes, target_sr=TARGET_SR)
            waveforms.append(wv)
            audio_ids.append(audio_id)
        except Exception as e:
            results.append({
                'audio_id': audio_id,
                'status': 'error',
                'error_message': f"Load error: {str(e)}"
            })

    if not waveforms:
        return results

    spec_results = batch_compute_spectrogram(waveforms, TARGET_SR)

    for i, (audio_id, wv) in enumerate(zip(audio_ids, waveforms)):
        try:
            spec_mag, freqs = spec_results[i]
            wv_gpu = wv.to(device)

            duration_sec = len(wv) / TARGET_SR
            snr = snr_cal_batch(wv_gpu, TARGET_SR)
            rolloff = spectral_rolloff_batch(spec_mag, freqs, TARGET_SR)
            centroid = spectral_centroid_batch(spec_mag, freqs, TARGET_SR)
            flatness = spectral_flatness_batch(spec_mag)
            zcr_mean, zcr_var = zcr_batch(wv_gpu)
            low_freq_ratio = low_frequency_energy_batch(spec_mag, freqs, TARGET_SR)
            loudness = loudness_cal(wv, TARGET_SR)

            results.append({
                'audio_id': audio_id,
                'sample_rate': TARGET_SR,
                'duration_sec': float(duration_sec),
                'snr_db': float(snr),
                'spectral_rolloff_hz': float(rolloff),
                'spectral_flatness': float(flatness),
                'spectral_centroid_hz': float(centroid),
                'zcr_mean': float(zcr_mean),
                'zcr_var': float(zcr_var),
                'loudness_lufs': float(loudness),
                'low_freq_energy_ratio': float(low_freq_ratio),
                'status': 'success'
            })

        except Exception as e:
            results.append({
                'audio_id': audio_id,
                'status': 'error',
                'error_message': f"Processing error: {str(e)}"
            })

    del waveforms, spec_results
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return results


def download_single_blob(blob_path):
    """Download a single blob."""
    audio_id = Path(blob_path).parent.name
    try:
        audio_bytes = download_blob_to_memory(blob_path)
        return (audio_id, audio_bytes, None)
    except Exception as e:
        return (audio_id, None, str(e))


def download_batch_parallel(blob_paths, max_workers=100):
    """Download multiple blobs in parallel."""
    audio_data = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_blob, path): path for path in blob_paths}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            audio_id, audio_bytes, error = future.result()

            if error:
                errors.append({
                    'audio_id': audio_id,
                    'status': 'error',
                    'error_message': f"Download error: {error}"
                })
            else:
                audio_data.append((audio_id, audio_bytes))

    return audio_data, errors


# ============================================================================
# Issue Detection
# ============================================================================

def detect_audio_issues(row):
    """Detect audio issues based on quality metrics."""
    issues = []

    if row['status'] != 'success':
        return issues

    if row['spectral_rolloff_hz'] < 1000:
        issues.append('bandwidth_limited_severe')
    elif row['spectral_rolloff_hz'] < 4000:
        issues.append('bandwidth_limited_moderate')

    if row['zcr_mean'] > 0.05:
        issues.append('high_noise_zcr')
    if row['snr_db'] < 15:
        issues.append('high_noise_snr')

    if row['low_freq_energy_ratio'] > 0.15:
        issues.append('low_frequency_rumble')

    if row['loudness_lufs'] < -30:
        issues.append('low_loudness')

    if row['spectral_flatness'] > 0.8:
        issues.append('very_flat_spectrum')

    return issues


def recommend_preprocessing(issues):
    """Recommend preprocessing methods based on detected issues."""
    recommendations = ['loudness_normalization']

    if 'bandwidth_limited_severe' in issues or 'bandwidth_limited_moderate' in issues:
        recommendations.append('eq_high_freq_boost')

    if 'high_noise_zcr' in issues or 'high_noise_snr' in issues:
        recommendations.append('noise_reduction')

    if 'low_frequency_rumble' in issues:
        recommendations.append('highpass_filter')

    return recommendations


# ============================================================================
# Main
# ============================================================================

def get_ram_usage_gb():
    """Get current RAM usage in GB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9


def force_gc():
    """Force aggressive garbage collection."""
    gc.collect()
    gc.collect()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    import time
    import psutil

    print("\n" + "="*60)
    print("AUDIO QUALITY ANALYSIS")
    print("="*60)

    # System info
    total_ram = psutil.virtual_memory().total / 1e9
    print(f"System RAM: {total_ram:.1f} GB")
    print(f"RAM threshold for GC: {RAM_THRESHOLD_GB} GB")

    # List blobs
    blob_prefix = "loc_vhp/"
    print(f"\nListing blobs with prefix: {blob_prefix}")
    audio_blobs = list_blobs(blob_prefix)

    # Filter for original VHP files only
    audio_blobs = [b for b in audio_blobs if b.endswith('/audio.mp3') or b.endswith('/video.mp4')]
    print(f"Found {len(audio_blobs)} original VHP media files")

    # Process
    all_results = []
    total_files = len(audio_blobs)

    print(f"\nProcessing {total_files} files...")
    print(f"Download batch size: {DOWNLOAD_BATCH_SIZE}")
    print(f"GPU batch size: {BATCH_SIZE}")
    print("="*60)

    start_time = time.time()

    for i in range(0, total_files, DOWNLOAD_BATCH_SIZE):
        batch_blobs = audio_blobs[i:i + DOWNLOAD_BATCH_SIZE]

        print(f"\n[Batch {i//DOWNLOAD_BATCH_SIZE + 1}] Downloading {len(batch_blobs)} files...")

        audio_data, download_errors = download_batch_parallel(batch_blobs, max_workers=DOWNLOAD_WORKERS)
        all_results.extend(download_errors)

        print(f"Downloaded: {len(audio_data)} files, Errors: {len(download_errors)}")

        # Check RAM before processing
        ram_before = get_ram_usage_gb()
        print(f"RAM usage: {ram_before:.1f} GB")

        print(f"Processing on GPU...")
        for j in range(0, len(audio_data), BATCH_SIZE):
            gpu_batch = audio_data[j:j + BATCH_SIZE]
            batch_results = analyze_audio_batch_gpu(gpu_batch)
            all_results.extend(batch_results)

            # Clear processed batch from memory immediately
            for k in range(len(gpu_batch)):
                audio_data[j + k] = None  # Release reference

        # Aggressive cleanup after each download batch
        del audio_data
        force_gc()

        # Check RAM after processing
        ram_after = get_ram_usage_gb()
        print(f"RAM after GC: {ram_after:.1f} GB")

        # Extra GC if above threshold
        if ram_after > RAM_THRESHOLD_GB:
            print(f"⚠️  RAM above {RAM_THRESHOLD_GB}GB, forcing extra GC...")
            force_gc()
            import time as t
            t.sleep(1)  # Give OS time to reclaim
            force_gc()
            print(f"RAM after extra GC: {get_ram_usage_gb():.1f} GB")

        elapsed = time.time() - start_time
        processed = min(i + DOWNLOAD_BATCH_SIZE, total_files)
        rate = processed / elapsed
        remaining = (total_files - processed) / rate if rate > 0 else 0

        print(f"Progress: {processed}/{total_files} ({processed/total_files*100:.1f}%)")
        print(f"Elapsed: {elapsed/60:.1f} min, Rate: {rate:.1f} files/sec, ETA: {remaining/60:.1f} min")

    total_time = time.time() - start_time

    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"Total files: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average rate: {len(all_results)/total_time:.2f} files/second")
    print(f"Success: {sum(1 for r in all_results if r.get('status') == 'success')}")
    print(f"Errors: {sum(1 for r in all_results if r.get('status') == 'error')}")

    # Create DataFrame and detect issues
    df = pd.DataFrame(all_results)
    df['issues'] = df.apply(detect_audio_issues, axis=1)
    df['recommended_preprocessing'] = df['issues'].apply(recommend_preprocessing)

    print(f"\nFiles with issues: {(df['issues'].str.len() > 0).sum()} / {len(df)}")

    all_issues = [issue for issues in df['issues'] for issue in issues]
    issue_counts = Counter(all_issues)

    print("\nIssue breakdown:")
    for issue, count in issue_counts.most_common():
        print(f"  {issue}: {count} files ({count/len(df)*100:.1f}%)")

    # Save
    output_path = Path("data/audio_quality_analysis.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Clear GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
