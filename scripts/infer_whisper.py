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
from faster_whisper import WhisperModel, BatchedInferencePipeline
# experiment tracking
import wandb

# Import local modules (same directory)
# Add scripts directory to path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
from scripts.cloud import azure_utils
from scripts.data import data_loader
from scripts.file_logger import log, init_logger


def run(cfg):
    """
    Run Whisper inference on dataset from config.

    Supports:
    - Local files via glob pattern
    - Azure blob storage via parquet manifest
    - Batch processing (faster-whisper BatchedInferencePipeline)
    - Flexible duration (full file or sliced)
    - Per-file outputs and metrics
    - File logging (timestamped logs saved to output directory)
    """
    # Track total experiment wall-clock time
    experiment_start_time = time.time()

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize file logger FIRST (before any log() calls)
    init_logger(out_dir, prefix="whisper")

    # Initialize wandb if enabled
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    if use_wandb:
        wandb.init(project=cfg.get("experiment_id", "whisper-inference"), config=cfg)
        log("Wandb logging enabled")
    else:
        log("Wandb logging disabled")

    # Load model
    model_root = Path(cfg["model"]["dir"])
    model_name = cfg["model"]["name"]

    # Handle both HuggingFace format (with snapshots/) and CTranslate2 format (flat structure)
    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        # HuggingFace format: use latest snapshot
        model_snap = max(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        model_file_dir = str(model_snap)
        log(f"Loading model: {model_name} from HuggingFace snapshot: {model_file_dir}")
    else:
        # CTranslate2 format: use model_root directly
        model_file_dir = str(model_root)
        log(f"Loading model: {model_name} from CTranslate2 directory: {model_file_dir}")

    # Get device and compute type from config
    device = cfg["model"].get("device", "auto")
    compute_type = cfg["model"].get("compute_type", "default")

    log(f"Device: {device}, Compute type: {compute_type}")
    
    condition_on_previous_text = cfg["model"].get("condition_on_previous_text")
    if not condition_on_previous_text:
        condition_on_previous_text = True
        log(f"condition_on_previous_text not specified, auto-set to True")
    log(f"Context passing (condition_on_previous_text): {condition_on_previous_text}")
    
    base_model = WhisperModel(model_file_dir, device=device, compute_type=compute_type)

    # Use batched pipeline if batch_size specified
    batch_size = int(cfg["model"].get("batch_size"))
    if not batch_size:
        batch_size = 1
        log(f"batch_size not specified, auto-set to 1")
    
    if batch_size > 1:
        model = BatchedInferencePipeline(model=base_model)
        log(f"Using BatchedInferencePipeline with batch_size={batch_size}")
    
    else:
        model = base_model
        log("Batch size <= 1, using standard WhisperModel (no batching)")

    # Determine input source
    source_type = cfg["input"].get("source")
    if not source_type:
        source_type = "local"
        log(f"source_type not specified, auto-set to local")    
    log(f"source_type: {source_type}")
    
    duration_sec = cfg["input"].get("duration_sec")
    if not duration_sec:
        duration_sec = None
        log(f"duration_sec not specified, auto-set to None (full duration)") 
    log(f"duration_sec: {str(duration_sec)}")
        
    sample_rate = cfg["input"].get("sample_rate")
    if not sample_rate:
        sample_rate = 16000
        log(f"sample_rate not specified, auto-set to 16000")
    log(f"sample_rate: {str(sample_rate)}")

    # Prepare file manifest
    if source_type == "azure_blob":
        # Load from parquet + Azure blob
        parquet_path = cfg["input"]["parquet_path"]
        sample_size = cfg["input"].get("sample_size")
        blob_prefix = cfg["input"].get("blob_prefix", "vhp")

        df = data_loader.load_vhp_dataset(parquet_path, sample_size=sample_size)
        manifest = data_loader.prepare_inference_manifest(df, blob_prefix=blob_prefix)
        log(f"Prepared manifest with {len(manifest)} items from Azure blob")
    elif source_type == "local":
        # Local files via glob
        log(f"Preparing manifest from local (requires audio files location)")
        audio_glob = cfg["input"]["audio_glob"]
        
        if not audio_glob:
            log(f"audio_glob not specified, cannot locate local files")
            log("Ending inference process- Reason: failed to identify input source")
            raise ValueError("audio_glob must be specified for local source_type")
        
        file_paths = glob.glob(audio_glob)
        manifest = [
            {"file_id": i, "blob_path": p, "collection_number": Path(p).stem, "ground_truth": None, "title": ""}
            for i, p in enumerate(file_paths)
        ]
        log(f"Found {len(manifest)} local files")

    # Run inference on all files
    results = []
    hyp_path = out_dir / "hyp_whisper.txt"

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
            log(f"\nfile_id: [{file_id}] from collection: {collection_num}")
            try:
                # Load audio
                start_time = time.time()

                # Download and check file size if Azure blob
                if source_type == "azure_blob":
                    # Try each candidate path until one succeeds
                    audio_bytes = None
                    successful_path = None
                    log(f"Number of source_paths to try: {len(source_paths)}")
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

                    # Convert to numpy array for Whisper
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
                    # Load from local file (also use pydub for consistency)
                    # For local files, just use the first path (should only be one)
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
                log(f"  Audio loaded: {actual_duration:.1f}s ({actual_duration/60:.1f} min) in {load_time:.1f}s")

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

                with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, wave, sample_rate)

                    # Get context passing setting from config (default: True for fair comparison)
                    condition_on_previous_text = cfg["model"].get("condition_on_previous_text")

                    beam_size = cfg["model"].get("beam_size", 1)
                    temperature = cfg["model"].get("temperature", 0.0)
                    vad_filter = cfg["model"].get("vad_filter", True)
                    no_speech_threshold = cfg["model"].get("no_speech_threshold", 0.6)
                    word_timestamps = cfg["model"].get("word_timestamps", False)
                    initial_prompt = cfg["model"].get("initial_prompt", None)
                    suppress_tokens = cfg["model"].get("suppress_tokens", [-1])
                    
                    
                    transcribe_args = {
                        "beam_size": beam_size,
                        "temperature": temperature,
                        "vad_filter": vad_filter,
                        "no_speech_threshold": no_speech_threshold,
                        "word_timestamps": word_timestamps,
                        "initial_prompt": initial_prompt,
                        "suppress_tokens": suppress_tokens,
                        "condition_on_previous_text": condition_on_previous_text,
                    }
                    # specify batch_size
                    if batch_size > 1:
                        transcribe_args["batch_size"] = batch_size

                    segments, info = model.transcribe(tmp.name, **transcribe_args)

                    # Join segments
                    hyp_text = " ".join(s.text.strip() for s in segments)
                    total_time = time.time() - start_time
                    transcribe_time = time.time() - transcribe_start
                    speed_factor = actual_duration / total_time if total_time > 0 else 0
                    log(f"  Transcription took {transcribe_time:.1f}s")

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
                            "detected_language": info.language,
                            "hypothesis_length": len(hyp_text),
                        })

                    results.append({
                        "file_id": file_id,
                        "collection_number": collection_num,
                        "blob_path": source_path,
                        "hypothesis": hyp_text,
                        "duration_sec": actual_duration,
                        "processing_time_sec": total_time,
                        "language": info.language,
                        "status": "success",
                    })

                    # Improved logging
                    log(f"  Transcription complete!")
                    log(f"    - Audio duration: {actual_duration:.1f}s")
                    log(f"    - Processing time: {total_time:.1f}s (load: {load_time:.1f}s, transcribe: {transcribe_time:.1f}s)")
                    log(f"    - Speed: {speed_factor:.1f}x realtime")
                    log(f"    - Language: {info.language}")
                    log(f"    - Preview: {hyp_text[:80]}...")
                    log(f"  ✓ SUCCESS")

            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)

                log(f"  ✗ FAILED: {error_type}")
                log(f"    Error: {error_msg}")

                # Additional context for common errors
                if "Format not recognised" in error_msg or "NoBackendError" in error_type:
                    log(f"    → This file may be corrupted, empty, or in an unsupported format")
                    log(f"    → Check the blob in Azure storage: {source_path}")
                elif "BlobNotFound" in error_type or "ResourceNotFoundError" in error_type:
                    log(f"    → Blob does not exist in Azure storage")
                    log(f"    → Expected path: {source_path}")
                elif len(error_msg) < 200:
                    # Short error, show full traceback for debugging
                    log(f"    Traceback: {traceback.format_exc()}")

                if use_wandb:
                    wandb.log({"file_id": file_id, "error": error_msg, "error_type": error_type})
                results.append({
                    "file_id": file_id,
                    "collection_number": collection_num,
                    "blob_path": source_path,
                    "hypothesis": "",
                    "duration_sec": 0,
                    "processing_time_sec": 0,
                    "language": "",
                    "status": f"failed: {error_type}",
                    "error_message": error_msg,
                })
                hout.write("\n")  # Empty line for failed files

    # Save results DataFrame
    df_results = pd.DataFrame(results)
    results_path = out_dir / "inference_results.parquet"
    df_results.to_parquet(results_path, index=False)
    log(f"\nSaved inference results to {results_path}")

    # Calculate total experiment time
    experiment_end_time = time.time()
    total_experiment_time = experiment_end_time - experiment_start_time

    # Log aggregate metrics
    successful = df_results[df_results["status"] == "success"]
    total_audio_duration = successful["duration_sec"].sum()
    total_processing_time = successful["processing_time_sec"].sum()

    if use_wandb:
        wandb.log({
            "total_files": len(results),
            "successful_files": len(successful),
            "failed_files": len(results) - len(successful),
            "total_audio_duration_sec": total_audio_duration,
            "total_processing_time_sec": total_processing_time,
            "total_experiment_time_sec": total_experiment_time,  # Wall-clock time (includes overhead)
            "avg_processing_time_sec": successful["processing_time_sec"].mean(),
            "speedup_factor": total_audio_duration / total_experiment_time if total_experiment_time > 0 else 0,  # How many x realtime
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
    log(f"Speedup: {total_audio_duration/total_experiment_time:.2f}x realtime")
    log(f"{'='*60}")

    # Upload artifacts to wandb
    if use_wandb:
        wandb.save(str(hyp_path))
        wandb.save(str(results_path))

    log("Inference complete!")
    return {"hyp_path": str(hyp_path), "results_path": str(results_path)}
