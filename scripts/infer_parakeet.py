# NVIDIA Parakeet-TDT-0.6B-v3 Inference Script
# Fast-Conformer TDT (Transducer) model - fastest multilingual ASR on HF leaderboard
# 600M params, 25 European languages, auto language detection
# RTFx: 3332.74 (4.4x faster than Canary-1B-v2)

# config/ io
import os
import sys
import glob
import time
from pathlib import Path
import yaml
from tempfile import NamedTemporaryFile
from typing import Optional, Dict, List
# audio processing
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
# models
import torch
import nemo.collections.asr as nemo_asr
# experiment tracking
import wandb

# Import local modules (same directory)
# Add scripts directory to path for imports
_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))
from file_logger import log, init_logger
import azure_utils
import data_loader


def run(cfg):
    """
    Run NVIDIA Parakeet-TDT-0.6B-v3 inference on dataset from config.

    Key features:
    - Fastest multilingual ASR (RTFx: 3332.74 on HF leaderboard)
    - Auto language detection (no lang params needed)
    - Word/segment/char-level timestamps
    - Auto punctuation and capitalization
    - 25 European languages
    - Supports up to 24 min audio (full attention) or 3 hrs (local attention)

    Supports:
    - Local files via glob pattern
    - Azure blob storage via parquet manifest
    - Flexible duration (full file or sliced)
    - Per-file outputs and metrics
    - File logging (timestamped logs saved to output directory)
    """
    # Track total experiment wall-clock time
    experiment_start_time = time.time()

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize file logger FIRST (before any log() calls)
    init_logger(out_dir, prefix="parakeet")

    # Initialize wandb if enabled
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    if use_wandb:
        wandb.init(project=cfg.get("experiment_id", "parakeet-inference"), config=cfg)
        log("Wandb logging enabled")
    else:
        log("Wandb logging disabled")

    # Load model
    model_name = cfg["model"]["dir"]  # Can be HuggingFace ID like "nvidia/parakeet-tdt-0.6b-v3"
    device = cfg["model"].get("device", "cuda")

    log(f"Loading Parakeet-TDT-0.6B-v3 model from: {model_name}")
    log(f"Device: {device}")

    # Load Parakeet model (simpler API than Canary)
    # ASRModel.from_pretrained() auto-downloads from HuggingFace
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    # Log model info
    log(f"Model type: {type(model)}")
    log(f"Model tokenizer type: {type(model.tokenizer)}")

    # Move model to device if needed
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        log(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Model loaded on CPU")

    # Check if we should use local attention for very long audio
    use_local_attention = cfg["model"].get("use_local_attention", False)
    att_context_size = cfg["model"].get("att_context_size", [256, 256])

    # Check if we should enable subsampling conv chunking (for very long audio >3 hrs)
    enable_subsampling_chunking = cfg["model"].get("enable_subsampling_chunking", False)
    subsampling_factor = cfg["model"].get("subsampling_chunking_factor", 1)

    if use_local_attention:
        log(f"Enabling local attention with context size: {att_context_size}")
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=att_context_size
        )

        if enable_subsampling_chunking:
            log(f"Enabling subsampling conv chunking (factor: {subsampling_factor})")
            log("This allows processing up to 13 hours of audio (bulletproof mode)")
            model.change_subsampling_conv_chunking_factor(subsampling_factor)
        else:
            log("Subsampling chunking disabled (handles up to 3 hours)")
    else:
        log("Using full attention (default)")
        log("Max audio length: ~24 minutes on A100 80GB")

    log(f"Model ready for inference (auto language detection, no lang params needed)")

    # Determine input source
    source_type = cfg["input"].get("source", "local")
    duration_sec = cfg["input"].get("duration_sec")  # None = full duration
    sample_rate = cfg["input"].get("sample_rate", 16000)  # Parakeet expects 16kHz

    # Prepare file manifest
    if source_type == "azure_blob":
        # Load from parquet + Azure blob
        parquet_path = cfg["input"]["parquet_path"]
        sample_size = cfg["input"].get("sample_size")
        blob_prefix = cfg["input"].get("blob_prefix", "vhp")

        df = data_loader.load_vhp_dataset(parquet_path, sample_size=sample_size)
        manifest = data_loader.prepare_inference_manifest(df, blob_prefix=blob_prefix)
        log(f"Prepared manifest with {len(manifest)} items from Azure blob")
    else:
        # Local files via glob
        audio_glob = cfg["input"]["audio_glob"]
        file_paths = glob.glob(audio_glob)
        manifest = [
            {"file_id": i, "blob_path": p, "collection_number": Path(p).stem, "ground_truth": None, "title": ""}
            for i, p in enumerate(file_paths)
        ]
        log(f"Found {len(manifest)} local files")

    # Run inference on all files
    results = []
    hyp_path = out_dir / "hyp_parakeet.txt"

    # Enable timestamps if requested
    enable_timestamps = cfg["model"].get("enable_timestamps", False)
    log(f"Timestamps enabled: {enable_timestamps}")

    log(f"Starting inference on {len(manifest)} files...")
    with open(hyp_path, "w") as hout:
        for item in tqdm(manifest, desc="Transcribing"):
            file_id = item["file_id"]
            collection_num = item["collection_number"]

            # Handle both old format (single blob_path) and new format (blob_path_candidates list)
            if "blob_path_candidates" in item:
                source_paths = item["blob_path_candidates"]
            elif "blob_path" in item:
                source_paths = [item["blob_path"]]
            else:
                log(f"\n[{file_id}] ERROR: No blob path found in manifest")
                continue

            # Log attempt
            log(f"\n[{file_id}] Processing: {collection_num}")

            try:
                # Load audio
                start_time = time.time()

                # Download and check file size if Azure blob
                if source_type == "azure_blob":
                    # Try each candidate path until one succeeds
                    audio_bytes = None
                    successful_path = None

                    for source_path in source_paths:
                        try:
                            print(f"  Trying: {source_path}")
                            audio_bytes = azure_utils.download_blob_to_memory(source_path)
                            successful_path = source_path
                            print(f"  ✓ Found: {source_path}")
                            break
                        except Exception as e:
                            print(f"    × Not found: {source_path}")
                            continue

                    if audio_bytes is None:
                        raise FileNotFoundError(f"None of the candidate paths exist: {source_paths}")
                    file_size_mb = len(audio_bytes) / (1024 * 1024)
                    log(f"  Downloaded: {file_size_mb:.2f} MB")

                    # Check if file looks valid
                    if len(audio_bytes) < 100:
                        raise ValueError(f"File too small ({len(audio_bytes)} bytes) - likely empty or corrupted")

                    # Use pydub to handle both MP3 and MP4 (video) files
                    import io
                    log(f"  Normalizing audio (MP3/MP4 → 16kHz mono WAV)...")

                    # Load with pydub (handles MP3, MP4, M4A, WAV, etc.)
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

                    # Normalize: convert to mono, resample to target rate
                    audio_segment = audio_segment.set_channels(1)  # Mono
                    audio_segment = audio_segment.set_frame_rate(sample_rate)  # 16kHz

                    # Trim to desired duration if specified
                    if duration_sec is not None:
                        duration_ms = duration_sec * 1000
                        audio_segment = audio_segment[:duration_ms]

                    # Convert to numpy array
                    samples = np.array(audio_segment.get_array_of_samples())
                    # Normalize to float32 [-1, 1]
                    if audio_segment.sample_width == 2:  # 16-bit
                        wave = samples.astype(np.float32) / 32768.0
                    elif audio_segment.sample_width == 4:  # 32-bit
                        wave = samples.astype(np.float32) / 2147483648.0
                    else:
                        wave = samples.astype(np.float32)

                    actual_duration = len(wave) / sample_rate
                else:
                    # Load from local file
                    source_path = source_paths[0]
                    log(f"  File: {source_path}")
                    log(f"  Normalizing audio (MP3/MP4 → 16kHz mono WAV)...")
                    audio_segment = AudioSegment.from_file(source_path)
                    audio_segment = audio_segment.set_channels(1)
                    audio_segment = audio_segment.set_frame_rate(sample_rate)

                    if duration_sec is not None:
                        duration_ms = duration_sec * 1000
                        audio_segment = audio_segment[:duration_ms]

                    samples = np.array(audio_segment.get_array_of_samples())
                    if audio_segment.sample_width == 2:
                        wave = samples.astype(np.float32) / 32768.0
                    elif audio_segment.sample_width == 4:
                        wave = samples.astype(np.float32) / 2147483648.0
                    else:
                        wave = samples.astype(np.float32)

                    actual_duration = len(wave) / sample_rate

                load_time = time.time() - start_time
                log(f"  Audio duration: {actual_duration:.1f}s ({actual_duration/60:.1f} min)")

                # Check GPU memory if using GPU
                if device == "cuda" and torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_mem_free = gpu_mem_total - gpu_mem_allocated
                    log(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_free:.2f}GB free (total: {gpu_mem_total:.2f}GB)")

                # Transcribe
                log(f"  Starting transcription...")
                transcribe_start = time.time()

                # Save to temp file (Parakeet expects file path)
                with NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                    sf.write(tmp_wav.name, wave, sample_rate)
                    log(f"  Created temp WAV file: {tmp_wav.name}")

                    # Parakeet API: simple transcribe() call
                    # Auto-detects language, no source_lang/target_lang params needed!
                    log(f"  Calling model.transcribe()...")

                    output = model.transcribe(
                        [tmp_wav.name],
                        timestamps=enable_timestamps  # Enable word/segment/char timestamps if requested
                    )

                    log(f"  transcribe() returned. Type: {type(output)}, Length: {len(output)}")

                    # Extract text from output
                    if hasattr(output[0], 'text'):
                        log(f"  Has .text attribute")
                        hyp_text = output[0].text
                    else:
                        log(f"  No .text attribute, using str()")
                        hyp_text = str(output[0])

                    log(f"  Extracted text length: {len(hyp_text)} chars")

                    # Extract timestamps if enabled
                    word_timestamps = None
                    if enable_timestamps and hasattr(output[0], 'timestamp'):
                        word_timestamps = output[0].timestamp.get('word', None)
                        log(f"  Extracted {len(word_timestamps) if word_timestamps else 0} word timestamps")

                    total_time = time.time() - start_time
                    transcribe_time = time.time() - transcribe_start
                    speed_factor = actual_duration / total_time if total_time > 0 else 0

                    log(f"  Transcription complete: {len(hyp_text)} chars in {transcribe_time:.1f}s")
                    log(f"  Preview: {hyp_text[:100]}...")

                    # Write to output file (one line per file)
                    hout.write(hyp_text + "\n")

                    # Save per-file result
                    if cfg["output"].get("save_per_file", False):
                        per_file_path = out_dir / f"hyp_{file_id}.txt"
                        with open(per_file_path, "w") as f:
                            f.write(hyp_text)

                    # Log to wandb
                    if use_wandb:
                        wandb.log({
                            "file_id": file_id,
                            "collection_number": collection_num,
                            "duration_sec": actual_duration,
                            "processing_time_sec": total_time,
                            "hypothesis_length": len(hyp_text),
                            "speed_factor": speed_factor,
                        })

                    results.append({
                        "file_id": file_id,
                        "collection_number": collection_num,
                        "blob_path": successful_path if source_type == "azure_blob" else source_path,
                        "hypothesis": hyp_text,
                        "ground_truth": item.get("ground_truth"),
                        "title": item.get("title", ""),
                        "duration_sec": actual_duration,
                        "processing_time_sec": total_time,
                        "model_name": "parakeet-tdt-0.6b-v3",
                        "status": "success",
                        "error_message": None,
                    })

                    # Improved logging
                    log(f"  ✓ SUCCESS")
                    log(f"    - Audio duration: {actual_duration:.1f}s")
                    log(f"    - Processing time: {total_time:.1f}s (load: {load_time:.1f}s, transcribe: {transcribe_time:.1f}s)")
                    log(f"    - Speed: {speed_factor:.1f}x realtime")

            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)

                log(f"  ✗ FAILED: {error_type}")
                log(f"    Error: {error_msg}")

                # Additional context for common errors
                if "Format not recognised" in error_msg or "NoBackendError" in error_type:
                    log(f"    → This file may be corrupted, empty, or in an unsupported format")
                elif "BlobNotFound" in error_type or "ResourceNotFoundError" in error_type:
                    log(f"    → Blob does not exist in Azure storage")
                elif len(error_msg) < 200:
                    # Short error, show full traceback for debugging
                    log(f"    Traceback: {traceback.format_exc()}")

                if use_wandb:
                    wandb.log({"file_id": file_id, "error": error_msg, "error_type": error_type})

                results.append({
                    "file_id": file_id,
                    "collection_number": collection_num,
                    "blob_path": source_paths[0] if source_paths else "",
                    "hypothesis": "",
                    "ground_truth": item.get("ground_truth"),
                    "title": item.get("title", ""),
                    "duration_sec": 0,
                    "processing_time_sec": 0,
                    "model_name": "parakeet-tdt-0.6b-v3",
                    "status": f"failed: {error_type}",
                    "error_message": error_msg,
                })
                hout.write("\n")  # Empty line for failed files

    # Save results DataFrame
    df_results = pd.DataFrame(results)
    results_path = out_dir / "inference_results.parquet"
    df_results.to_parquet(results_path, index=False)
    log(f"\nSaved results to: {results_path}")

    # Calculate total experiment time
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time

    # Log aggregate metrics
    successful = df_results[df_results["status"] == "success"]
    total_audio_duration = successful["duration_sec"].sum()
    total_processing_time = successful["processing_time_sec"].sum()
    avg_speedup = successful["duration_sec"].sum() / successful["processing_time_sec"].sum() if len(successful) > 0 else 0

    if use_wandb:
        wandb.log({
            "total_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(results) - len(successful),
            "total_audio_duration_sec": total_audio_duration,
            "total_processing_time_sec": total_processing_time,
            "total_experiment_time_sec": total_experiment_time,
            "avg_processing_time_sec": successful["processing_time_sec"].mean(),
            "avg_speedup": avg_speedup,
        })

    # Print summary
    log(f"\n{'='*60}")
    log(f"EXPERIMENT SUMMARY")
    log(f"{'='*60}")
    log(f"Total files: {len(results)}")
    log(f"Successful: {len(successful)}")
    log(f"Failed: {len(results) - len(successful)}")
    log(f"Total audio duration: {total_audio_duration/60:.1f} minutes ({total_audio_duration/3600:.2f} hours)")
    log(f"Total experiment time: {total_experiment_time/60:.1f} minutes ({total_experiment_time/3600:.2f} hours)")
    log(f"Speedup: {avg_speedup:.2f}x realtime")
    log(f"{'='*60}")

    # Upload artifacts to wandb
    if use_wandb:
        wandb.save(str(hyp_path))
        wandb.save(str(results_path))

    log("Inference complete!")

    return {
        "hyp_path": str(hyp_path),
        "results_path": str(results_path),
        "output_dir": str(out_dir),
        "summary": {
            "total_files": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "avg_speedup": avg_speedup,
        }
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Parakeet-TDT-0.6B-v3 inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)
