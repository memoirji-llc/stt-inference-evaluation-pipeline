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
from nemo.collections.speechlm2.models import SALM
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
    Run NVIDIA Canary-Qwen inference on dataset from config.

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
    init_logger(out_dir, prefix="canary")

    # Initialize wandb if enabled
    use_wandb = os.getenv("WANDB_MODE") != "disabled"
    if use_wandb:
        wandb.init(project=cfg.get("experiment_id", "canary-inference"), config=cfg)
        log("Wandb logging enabled")
    else:
        log("Wandb logging disabled")

    # Set HuggingFace cache to use local models directory (avoid re-downloading)
    # This uses the project's models/canary cache instead of ~/.cache/huggingface
    project_root = Path(__file__).parent.parent
    models_cache = project_root / "models" / "canary"
    if models_cache.exists():
        os.environ['HF_HOME'] = str(models_cache)
        os.environ['TRANSFORMERS_CACHE'] = str(models_cache)
        log(f"Using local model cache: {models_cache}")
    else:
        log(f"Local cache not found at {models_cache}, will use default HuggingFace cache")

    # Load model
    model_dir = cfg["model"]["dir"]
    device = cfg["model"].get("device", "cuda")

    log(f"Loading Canary-Qwen model from: {model_dir}")
    log(f"Device: {device}")

    # Load Canary model (can be HuggingFace ID or local path)
    model = SALM.from_pretrained(model_dir)

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
    hyp_path = out_dir / "hyp_canary.txt"

    # User prompt for Canary (can be customized via config)
    user_prompt = cfg["model"].get("prompt", "Transcribe the following:")

    # Max new tokens is REQUIRED for Canary (LLM-based)
    # Default to 512 if not specified (enough for short audio clips)
    max_new_tokens = cfg["model"].get("max_new_tokens", 512)
    if max_new_tokens is None:
        log("WARNING: max_new_tokens not set in config. Using default of 512 tokens.")
        log("         For longer audio, increase max_new_tokens in config (e.g., 2048 for 30min audio)")
        max_new_tokens = 512

    # Chunking configuration (for long audio on limited GPU memory)
    chunk_duration_sec = cfg["model"].get("chunk_duration_sec", None)
    if chunk_duration_sec:
        log(f"Chunking enabled: {chunk_duration_sec}s chunks (for GPU memory management)")
    else:
        log("Chunking disabled: processing full audio (may fail on long files with limited GPU)")

    # Context passing configuration (for coherence across chunk boundaries)
    # Try new name first, fall back to old name for backwards compatibility
    custom_context_passing = cfg["model"].get("custom_context_passing", cfg["model"].get("use_context_passing", True))
    if chunk_duration_sec and custom_context_passing:
        log(f"Custom context passing enabled: previous chunk output → LLM prompt (like Whisper's condition_on_previous_text=True)")
    elif chunk_duration_sec and not custom_context_passing:
        log(f"Custom context passing disabled: each chunk processed independently (faster, lower coherence)")
    else:
        log(f"Custom context passing N/A: no chunking (full audio processed at once)")

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

                # Check GPU memory before transcription
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                    gpu_mem_free = gpu_mem_total - gpu_mem_allocated
                    log(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_free:.2f}GB free (total: {gpu_mem_total:.2f}GB)")

                # Decide if chunking is needed
                if chunk_duration_sec and actual_duration > chunk_duration_sec:
                    # Chunk the audio for GPU memory management
                    num_chunks = int(np.ceil(actual_duration / chunk_duration_sec))
                    log(f"  Audio duration {actual_duration:.1f}s > chunk size {chunk_duration_sec}s")
                    log(f"  → Creating {num_chunks} chunks for processing")
                    log(f"  → Using context passing (previous chunk output as LLM prompt)")

                    chunk_transcripts = []
                    previous_context = None  # Track previous chunk for context passing
                    transcribe_start = time.time()

                    for chunk_idx in range(num_chunks):
                        chunk_start_sec = chunk_idx * chunk_duration_sec
                        chunk_end_sec = min((chunk_idx + 1) * chunk_duration_sec, actual_duration)
                        chunk_start_sample = int(chunk_start_sec * sample_rate)
                        chunk_end_sample = int(chunk_end_sec * sample_rate)
                        chunk_wave = wave[chunk_start_sample:chunk_end_sample]
                        chunk_duration = len(chunk_wave) / sample_rate

                        log(f"  Chunk {chunk_idx+1}/{num_chunks} ({chunk_start_sec:.1f}-{chunk_end_sec:.1f}s, {chunk_duration:.1f}s): Transcribing...")

                        # Transcribe this chunk (WITH or WITHOUT context based on config)
                        with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                            sf.write(tmp.name, chunk_wave, sample_rate)

                            # Build prompt with context (if enabled and available)
                            if custom_context_passing and previous_context:
                                # Pass previous chunk as context (like Whisper's condition_on_previous_text=True)
                                prompt_content = f"Previous context: {previous_context}\n\n{user_prompt} {model.audio_locator_tag}"
                                log(f"  Chunk {chunk_idx+1}/{num_chunks}: Using {len(previous_context)} chars of context from previous chunk")
                            else:
                                # First chunk OR context passing disabled
                                prompt_content = f"{user_prompt} {model.audio_locator_tag}"
                                if chunk_idx == 0:
                                    log(f"  Chunk {chunk_idx+1}/{num_chunks}: First chunk (no context)")
                                elif not custom_context_passing:
                                    log(f"  Chunk {chunk_idx+1}/{num_chunks}: Independent (custom context passing disabled)")

                            prompts = [[{
                                "role": "user",
                                "content": prompt_content,
                                "audio": [tmp.name]
                            }]]

                            answer_ids = model.generate(prompts=prompts, max_new_tokens=max_new_tokens)
                            chunk_text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()

                            chunk_transcripts.append(chunk_text)
                            if custom_context_passing:
                                previous_context = chunk_text  # Save for next chunk (only if custom context passing enabled)
                            log(f"  Chunk {chunk_idx+1}/{num_chunks}: Complete ({len(chunk_text)} chars)")
                            log(f"  Chunk {chunk_idx+1}/{num_chunks}: Preview: {chunk_text[:50]}...")

                            # Clear GPU cache after each chunk
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    # Merge all chunks
                    log(f"  Merging {num_chunks} chunks → Final transcript")
                    hyp_text = " ".join(chunk_transcripts)
                    transcribe_time = time.time() - transcribe_start
                    log(f"  Transcription complete: {len(hyp_text)} total chars from {num_chunks} chunks")

                else:
                    # Process full audio (no chunking)
                    log(f"  Starting transcription on GPU...")
                    transcribe_start = time.time()

                    with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        sf.write(tmp.name, wave, sample_rate)

                        # Canary inference using the SALM generate API
                        # Format: list of conversations, each conversation is a list of message dicts
                        # IMPORTANT: Must include audio_locator_tag placeholder in prompt
                        prompts = [
                            [
                                {
                                    "role": "user",
                                    "content": f"{user_prompt} {model.audio_locator_tag}",
                                    "audio": [tmp.name]
                                }
                            ]
                        ]

                        # Generate transcription (max_new_tokens is required for Canary)
                        answer_ids = model.generate(prompts=prompts, max_new_tokens=max_new_tokens)

                        # Decode the generated tokens to text
                        hyp_text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()

                    transcribe_time = time.time() - transcribe_start
                    log(f"  Transcription took {transcribe_time:.1f}s")

                total_time = time.time() - start_time
                speed_factor = actual_duration / total_time if total_time > 0 else 0

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
                        "speed_factor": speed_factor,
                        "hypothesis_length": len(hyp_text),
                        "status": "success"
                    })

                results.append({
                    "file_id": file_id,
                    "collection_number": collection_num,
                    "blob_path": source_path,
                    "hypothesis": hyp_text,
                    "duration_sec": actual_duration,
                    "processing_time_sec": total_time,
                    "status": "success",
                })

                # Improved logging
                log(f"  Transcription complete!")
                log(f"    - Audio duration: {actual_duration:.1f}s")
                log(f"    - Processing time: {total_time:.1f}s (load: {load_time:.1f}s, transcribe: {transcribe_time:.1f}s)")
                log(f"    - Speed: {speed_factor:.1f}x realtime")
                log(f"    - Preview: {hyp_text[:80]}...")
                log(f"  ✓ SUCCESS")

                # Clear GPU cache after each file to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_mem_allocated_after = torch.cuda.memory_allocated() / 1024**3
                    log(f"  GPU cache cleared (now using {gpu_mem_allocated_after:.2f}GB)")

            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)

                log(f"  ✗ FAILED: {error_type}")
                log(f"    Error: {error_msg}")

                # Additional context for common errors
                if "out of memory" in error_msg.lower() or "OutOfMemoryError" in error_type:
                    log(f"    → GPU ran out of memory during transcription")
                    # Try to get audio duration if it was loaded
                    try:
                        if 'actual_duration' in locals():
                            log(f"    → Audio duration was {actual_duration:.1f}s ({actual_duration/60:.1f} min)")
                    except:
                        pass
                    if torch.cuda.is_available():
                        gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3
                        log(f"    → GPU memory at failure: {gpu_mem_allocated:.2f}GB allocated")
                    log(f"    → Try: 1) Reduce audio length with duration_sec config, or 2) Use larger GPU")
                elif "Format not recognised" in error_msg or "NoBackendError" in error_type:
                    log(f"    → This file may be corrupted, empty, or in an unsupported format")
                    log(f"    → Check the blob in Azure storage: {source_path}")
                elif "BlobNotFound" in error_type or "ResourceNotFoundError" in error_type:
                    log(f"    → Blob does not exist in Azure storage")
                    log(f"    → Expected path: {source_path}")
                elif len(error_msg) < 200:
                    # Short error, show full traceback for debugging
                    log(f"    Traceback: {traceback.format_exc()}")

                # Clear GPU cache even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log(f"    GPU cache cleared")

                if use_wandb:
                    wandb.log({"file_id": file_id, "error": error_msg, "error_type": error_type})
                results.append({
                    "file_id": file_id,
                    "collection_number": collection_num,
                    "blob_path": source_path,
                    "hypothesis": "",
                    "duration_sec": 0,
                    "processing_time_sec": 0,
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
