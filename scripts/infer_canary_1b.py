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

    # Log tokenizer info
    log(f"Model tokenizer type: {type(model.tokenizer)}")

    # Move model to device if needed
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        log(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Model loaded on CPU")

    log(f"Model ready for inference (using official Canary API with source_lang/target_lang params)")

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

    # Initialize results storage
    results = []
    total_audio_duration = 0.0  # Track total audio duration across all files

    # Process each file
    log(f"Starting inference on {len(manifest)} files...")
    hyp_path = out_dir / "hyp_canary1b.txt"

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
                    # Load from local file (also use pydub for consistency)
                    # For local files, just use the first path (should only be one)
                    source_path = source_paths[0]
                    log(f"  File: {source_path}")
                    log(f"  Normalizing audio (MP3/MP4 → 16kHz mono WAV)...")

                    # Load with pydub (handles MP3, MP4, M4A, WAV, etc.)
                    audio_segment = AudioSegment.from_file(source_path)

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

                total_audio_duration += actual_duration
                log(f"  Audio duration: {actual_duration:.1f}s")

                # Check GPU memory if available
                if device == "cuda" and torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_mem_free = gpu_mem_total - gpu_mem_allocated
                    log(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_free:.2f}GB free (total: {gpu_mem_total:.2f}GB)")

                # Transcribe using Canary-1B's native transcribe() method
                # Using the official API from NeMo tutorial (pass audio path + lang params directly)
                log(f"  Starting transcription...")
                transcribe_start = time.time()

                with NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                    sf.write(tmp_wav.name, wave, sample_rate)
                    log(f"  Created temp WAV file: {tmp_wav.name}")
                    log(f"  Calling model.transcribe() with audio path and language params...")

                    # Use official Canary API (from NeMo tutorial)
                    # Pass audio path directly with source_lang/target_lang parameters
                    try:
                        predicted_text = model.transcribe(
                            audio=[tmp_wav.name],
                            batch_size=1,
                            source_lang='en',  # Input audio language
                            target_lang='en',  # Output text language
                            timestamps=False,  # We don't need timestamps for WER evaluation
                        )
                        log(f"  transcribe() returned. Type: {type(predicted_text)}, Length: {len(predicted_text) if isinstance(predicted_text, list) else 'N/A'}")

                        if isinstance(predicted_text, list) and len(predicted_text) > 0:
                            log(f"  First result type: {type(predicted_text[0])}")
                            if hasattr(predicted_text[0], 'text'):
                                log(f"  Has .text attribute")
                            else:
                                log(f"  WARNING: No .text attribute found. Available attributes: {[a for a in dir(predicted_text[0]) if not a.startswith('_')]}")

                    except Exception as e:
                        log(f"  ERROR in model.transcribe(): {type(e).__name__}: {str(e)}")
                        import traceback
                        log(f"  Traceback:\n{traceback.format_exc()}")
                        raise

                    # Extract transcription from result
                    # Returns list of Hypothesis objects with .text attribute
                    log(f"  Extracting text from result...")
                    hyp_text = predicted_text[0].text if hasattr(predicted_text[0], 'text') else str(predicted_text[0])
                    log(f"  Extracted text length: {len(hyp_text)} chars")

                transcribe_time = time.time() - transcribe_start
                log(f"  Transcription complete: {len(hyp_text)} chars in {transcribe_time:.1f}s")
                log(f"  Preview: {hyp_text[:100]}...")

                # Write to combined hypothesis file
                hout.write(hyp_text + "\n")
                hout.flush()

                # Clear GPU cache between files (prevent memory accumulation)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                # Save per-file output if requested
                if cfg["output"].get("save_per_file", False):
                    hypothesis_path = out_dir / f"hyp_{file_id}.txt"
                    hypothesis_path.write_text(hyp_text, encoding='utf-8')

                # Store result
                ground_truth = item.get("ground_truth", None)
                result = {
                    "file_id": file_id,
                    "collection_number": collection_num,
                    "blob_path": successful_path if source_type == "azure_blob" else source_path,
                    "hypothesis": hyp_text,
                    "ground_truth": ground_truth,
                    "title": item.get("title", ""),
                    "duration_sec": actual_duration,
                    "processing_time_sec": transcribe_time,
                    "model_name": "canary-1b",
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
                    "collection_number": collection_num,
                    "blob_path": item.get("blob_path", ""),
                    "hypothesis": None,
                    "ground_truth": item.get("ground_truth", None),
                    "title": item.get("title", ""),
                    "duration_sec": 0,
                    "processing_time_sec": 0,
                    "model_name": "canary-1b",
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
        "hyp_path": str(hyp_path),
        "results_path": str(results_path),
        "output_dir": str(out_dir),
        "summary": {
            "total_files": len(results),
            "successful": successful_count,
            "failed": failed_count,
            "avg_speedup": avg_speedup
        }
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python infer_canary_1b.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run(cfg)
