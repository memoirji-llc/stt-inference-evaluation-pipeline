# NVIDIA Canary-1B Inference Script
# Pure ASR model (not LLM-based) with built-in timestamp support
# Uses NeMo's EncDecMultiTaskModel for transcription

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
from nemo.collections.asr.models import EncDecMultiTaskModel
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
    Run NVIDIA Canary-1B inference on dataset from config.

    Key differences from Canary-Qwen:
    - Uses EncDecMultiTaskModel (pure ASR, not SALM/LLM)
    - Can generate timestamps (unlike Canary-Qwen)
    - No max_new_tokens parameter (not LLM-based)
    - Uses transcribe() method instead of generate()

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
    init_logger(out_dir, prefix="canary1b")

    # Initialize wandb if enabled
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    if use_wandb:
        wandb.init(project=cfg.get("experiment_id", "canary1b-inference"), config=cfg)
        log("Wandb logging enabled")
    else:
        log("Wandb logging disabled")

    # Load model
    model_dir = cfg["model"]["dir"]
    device = cfg["model"].get("device", "cuda")

    log(f"Loading Canary-1B model from: {model_dir}")
    log(f"Device: {device}")

    # Load Canary-1B model (can be HuggingFace ID or local path)
    model = EncDecMultiTaskModel.from_pretrained(model_dir)

    # Configure decoding strategy
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1  # Greedy decoding (faster)
    model.change_decoding_strategy(decode_cfg)

    # Move model to device if needed
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        log(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Model loaded on CPU")

    # Determine input source
    source_type = cfg["input"].get("source", "local")
    duration_sec = cfg["input"].get("duration_sec")  # None = full duration
    sample_rate = cfg["input"].get("sample_rate", 16000)  # Canary expects 16kHz

    # Prepare file manifest
    if source_type == "azure_blob":
        # Load from parquet + Azure blob
        parquet_path = cfg["input"]["parquet_path"]
        sample_size = cfg["input"].get("sample_size")
        blob_prefix = cfg["input"].get("blob_prefix", "vhp")

        df = data_loader.load_vhp_dataset(parquet_path, sample_size=sample_size)
        files = df.to_dict("records")

        # Initialize Azure blob client
        blob_service = azure_utils.get_blob_service_client()
        container_client = blob_service.get_container_client(os.getenv("AZURE_CONTAINER_NAME"))

        log(f"Loaded {len(files)} files from {parquet_path}")
        log(f"Will process audio from Azure blob storage")

    else:
        # Load from local filesystem
        pattern = cfg["input"]["pattern"]
        filepaths = glob.glob(pattern, recursive=True)
        files = [{"local_path": str(p)} for p in filepaths]
        log(f"Found {len(files)} local files matching pattern: {pattern}")

    # Initialize results storage
    results = []
    total_audio_duration = 0.0  # Track total audio duration across all files

    # Process each file
    log(f"Starting inference on {len(files)} files...")
    for idx, file_info in enumerate(tqdm(files, desc="Processing files")):
        file_id = file_info.get("file_id", idx)

        try:
            # Download/load audio
            if source_type == "azure_blob":
                blob_path = file_info["blob_path"]
                collection_number = file_info.get("collection_number", "unknown")
                log(f"\n[{idx+1}/{len(files)}] Processing file_id={file_id}, collection={collection_number}")
                log(f"  Blob path: {blob_path}")

                # Download from Azure to temp file
                with NamedTemporaryFile(suffix=Path(blob_path).suffix, delete=True) as tmp_audio:
                    blob_client = container_client.get_blob_client(blob_path)
                    download_stream = blob_client.download_blob()
                    tmp_audio.write(download_stream.readall())
                    tmp_audio.flush()
                    audio_path = tmp_audio.name

                    # Load audio
                    wave, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

            else:
                # Load from local path
                audio_path = file_info["local_path"]
                log(f"\n[{idx+1}/{len(files)}] Processing: {audio_path}")
                wave, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

            # Optionally slice to specified duration
            if duration_sec:
                max_samples = int(duration_sec * sample_rate)
                if len(wave) > max_samples:
                    wave = wave[:max_samples]
                    log(f"  Sliced audio to {duration_sec}s")

            actual_duration = len(wave) / sample_rate
            total_audio_duration += actual_duration
            log(f"  Audio duration: {actual_duration:.1f}s")

            # Check GPU memory if available
            if device == "cuda" and torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                gpu_mem_free = gpu_mem_total - gpu_mem_allocated
                log(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_free:.2f}GB free (total: {gpu_mem_total:.2f}GB)")

            # Transcribe using Canary-1B's native transcribe() method
            log(f"  Starting transcription...")
            transcribe_start = time.time()

            with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, wave, sample_rate)

                # Canary-1B transcription (pure ASR, no prompting)
                # Default behavior: English ASR with punctuation
                predicted_text = model.transcribe(
                    paths2audio_files=[tmp.name],
                    batch_size=1,
                )

                # Extract transcription from result
                hyp_text = predicted_text[0].text if hasattr(predicted_text[0], 'text') else str(predicted_text[0])

            transcribe_time = time.time() - transcribe_start
            log(f"  Transcription complete: {len(hyp_text)} chars in {transcribe_time:.1f}s")
            log(f"  Preview: {hyp_text[:100]}...")

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save per-file output if requested
            if cfg["output"].get("save_per_file", False):
                hypothesis_path = out_dir / f"{file_id:05d}_hypothesis.txt"
                hypothesis_path.write_text(hyp_text, encoding='utf-8')

            # Store result
            result = {
                "file_id": file_id,
                "collection_number": file_info.get("collection_number"),
                "blob_path": file_info.get("blob_path"),
                "hypothesis": hyp_text,
                "duration_sec": actual_duration,
                "processing_time_sec": transcribe_time,
                "status": "success",
                "error_message": None
            }

            if use_wandb:
                wandb.log({
                    "file_processed": file_id,
                    "duration": actual_duration,
                    "processing_time": transcribe_time,
                    "speedup": actual_duration / transcribe_time if transcribe_time > 0 else 0,
                    "status": "success"
                })

        except Exception as e:
            log(f"  ERROR processing file_id={file_id}: {str(e)}")

            result = {
                "file_id": file_id,
                "collection_number": file_info.get("collection_number"),
                "blob_path": file_info.get("blob_path"),
                "hypothesis": None,
                "duration_sec": 0,
                "processing_time_sec": 0,
                "status": f"failed: {type(e).__name__}",
                "error_message": str(e)
            }

            if use_wandb:
                wandb.log({
                    "file_processed": file_id,
                    "status": "failed",
                    "error": str(e)
                })

        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = out_dir / "inference_results.parquet"
    results_df.to_parquet(results_path, index=False)
    log(f"\nSaved results to: {results_path}")

    # Calculate summary statistics
    total_experiment_time = time.time() - experiment_start_time
    successful_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - successful_count

    avg_speedup = 0
    if successful_count > 0:
        successful_results = [r for r in results if r["status"] == "success"]
        total_processing_time = sum(r["processing_time_sec"] for r in successful_results)
        avg_speedup = total_audio_duration / total_processing_time if total_processing_time > 0 else 0

    # Log summary
    log("\n" + "="*60)
    log(f"Total files: {len(results)}")
    log(f"Successful: {successful_count}")
    log(f"Failed: {failed_count}")
    log(f"Total audio duration: {total_audio_duration/60:.1f} minutes ({total_audio_duration/3600:.2f} hours)")
    log(f"Total experiment time: {total_experiment_time/60:.1f} minutes ({total_experiment_time/3600:.2f} hours)")
    log(f"Speedup: {avg_speedup:.2f}x realtime")
    log("="*60)
    log("Inference complete!")

    if use_wandb:
        wandb.log({
            "total_files": len(results),
            "successful": successful_count,
            "failed": failed_count,
            "total_audio_minutes": total_audio_duration / 60,
            "total_experiment_minutes": total_experiment_time / 60,
            "avg_speedup": avg_speedup
        })

    return {
        "results_path": results_path,
        "output_dir": out_dir,
        "summary": {
            "total_files": len(results),
            "successful": successful_count,
            "failed": failed_count,
            "avg_speedup": avg_speedup
        }
    }
