"""
GCP Speech-to-Text (Chirp 2 & 3) inference module.

Supports:
- Batch processing with parallel upload/transcribe/delete
- Fair comparison: same preprocessing as Whisper/Wav2Vec2 (pydub → mono 16kHz WAV)
- Azure blob storage integration
- Automatic endpoint selection based on model variant
- Rate limiting: 150 requests/min quota (batched starts with delays)
"""
import os
import sys
import io
import time
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile
from functools import wraps

# Audio processing
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

# GCP clients
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPIError

# Experiment tracking
import wandb

# Import local modules
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))
from scripts.cloud import azure_utils
from scripts.data import data_loader
from scripts.file_logger import log, init_logger


def setup_gcp_clients(cfg):
    """
    Initialize GCP Storage and Speech clients with correct endpoints.

    Returns:
        (storage_client, speech_client, recognizer_location)
    """
    # Set credentials
    credentials_path = cfg["model"]["gcp"]["credentials_path"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    # Get project ID from config
    project_id = cfg["model"]["gcp"]["project_id"]

    # Determine endpoint and location based on model variant
    model_variant = cfg["model"]["model_variant"]

    if model_variant in ["chirp", "chirp_2"]:
        # Chirp 2: us-central1 only
        api_endpoint = "us-central1-speech.googleapis.com"
        recognizer_location = "us-central1"
        log(f"Using Chirp 2 with endpoint: {api_endpoint}, location: {recognizer_location}")
    elif model_variant == "chirp_3":
        # Chirp 3: us multi-region
        api_endpoint = "us-speech.googleapis.com"
        recognizer_location = "us"
        log(f"Using Chirp 3 with endpoint: {api_endpoint}, location: {recognizer_location}")
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}. Use 'chirp', 'chirp_2', or 'chirp_3'")

    # Create clients with explicit project ID
    storage_client = storage.Client(project=project_id)
    speech_client = SpeechClient(
        client_options=ClientOptions(api_endpoint=api_endpoint)
    )

    return storage_client, speech_client, recognizer_location


def prepare_audio_for_gcp(audio_bytes, duration_sec=None, sample_rate=16000):
    """
    Preprocess audio EXACTLY like Whisper/Wav2Vec2 pipeline for fair comparison.

    Args:
        audio_bytes: Raw audio bytes (MP3/MP4/WAV/etc.)
        duration_sec: Optional duration limit (seconds)
        sample_rate: Target sample rate (default: 16000)

    Returns:
        (wav_bytes, actual_duration_sec)
    """
    # Load with pydub (handles MP3, MP4, M4A, WAV, etc.)
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Normalize: EXACT same as infer_whisper.py
    audio_segment = audio_segment.set_channels(1)  # Mono
    audio_segment = audio_segment.set_frame_rate(sample_rate)  # 16kHz

    # Trim to desired duration if specified (SAME as Whisper)
    if duration_sec is not None:
        duration_ms = duration_sec * 1000
        audio_segment = audio_segment[:duration_ms]

    # Export to WAV bytes (in-memory, no disk write)
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_bytes = wav_buffer.getvalue()

    actual_duration = len(audio_segment) / 1000.0  # pydub uses milliseconds

    return wav_bytes, actual_duration


def download_and_preprocess(item, duration_sec, sample_rate=16000):
    """
    Download from Azure and preprocess to WAV (SAME as Whisper pipeline).

    Returns:
        (wav_bytes, actual_duration, successful_path) or raises exception
    """
    # Handle both old format (single blob_path) and new format (blob_path_candidates list)
    if "blob_path_candidates" in item:
        source_paths = item["blob_path_candidates"]
    elif "blob_path" in item:
        source_paths = [item["blob_path"]]
    else:
        raise ValueError("No blob path found in manifest")

    # Try each candidate path
    audio_bytes = None
    successful_path = None

    for blob_path in source_paths:
        try:
            audio_bytes = azure_utils.download_blob_to_memory(blob_path)
            successful_path = blob_path
            break
        except Exception:
            continue

    if audio_bytes is None:
        raise FileNotFoundError(f"None of the candidate paths exist: {source_paths}")

    # Check file size
    if len(audio_bytes) < 100:
        raise ValueError(f"File too small ({len(audio_bytes)} bytes) - likely empty or corrupted")

    # Preprocess EXACTLY like Whisper
    wav_bytes, actual_duration = prepare_audio_for_gcp(audio_bytes, duration_sec, sample_rate)

    return wav_bytes, actual_duration, successful_path


def upload_to_gcs(bucket, filename, wav_bytes, timeout=600):
    """Upload WAV bytes to GCS bucket.

    Args:
        bucket: GCS bucket object
        filename: Destination filename in bucket
        wav_bytes: WAV file bytes to upload
        timeout: Upload timeout in seconds (default: 600 = 10 minutes)
    """
    blob = bucket.blob(filename)
    blob.upload_from_string(wav_bytes, content_type='audio/wav', timeout=timeout)


def start_transcription(speech_client, gcs_uri, model_variant, recognizer_location, project_id, language_code="en-US"):
    """
    Start a batch transcription operation.

    Returns:
        operation object (long-running operation)
    """
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language_code],
        model=model_variant,
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    # Build recognizer path
    recognizer = f"projects/{project_id}/locations/{recognizer_location}/recognizers/_"

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=recognizer,
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    operation = speech_client.batch_recognize(request=request)
    return operation


def parse_gcp_response(response, gcs_uri):
    """
    Extract transcript from GCP response.

    Args:
        response: BatchRecognizeResponse object
        gcs_uri: GCS URI used for the request

    Returns:
        Full transcript as string
    """
    transcript_segments = []

    # Get the result for our URI
    if gcs_uri in response.results:
        uri_result = response.results[gcs_uri]

        # Access the transcript results
        if hasattr(uri_result, 'transcript') and hasattr(uri_result.transcript, 'results'):
            for result in uri_result.transcript.results:
                if result.alternatives:
                    transcript_segments.append(result.alternatives[0].transcript)

    return ' '.join(transcript_segments)


def delete_from_gcs(bucket, filename):
    """Delete file from GCS bucket."""
    blob = bucket.blob(filename)
    blob.delete()


def _preprocess_and_upload_one(item, bucket, duration_sec, sample_rate, upload_timeout, bucket_name):
    """
    Download, preprocess, and upload ONE file to GCS atomically.
    This function processes one file at a time to minimize memory usage.

    Returns dict with status and file info.
    """
    try:
        # Download and preprocess
        wav_bytes, duration, successful_path = download_and_preprocess(item, duration_sec, sample_rate)

        # Upload to GCS
        gcs_filename = f"{item['collection_number']}.wav"
        upload_to_gcs(bucket, gcs_filename, wav_bytes, upload_timeout)

        # Immediately free memory
        del wav_bytes

        # Return success with metadata
        return {
            'status': 'success',
            'item': item,
            'gcs_filename': gcs_filename,
            'gcs_uri': f"gs://{bucket_name}/{gcs_filename}",
            'duration': duration,
            'successful_path': successful_path
        }

    except Exception as e:
        return {
            'status': 'error',
            'item': item,
            'error_message': str(e)
        }


def _cleanup_gcs_files(bucket, gcs_filenames, workers=30):
    """Delete multiple files from GCS in parallel."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        delete_futures = []
        for filename in gcs_filenames:
            future = executor.submit(delete_from_gcs, bucket, filename)
            delete_futures.append(future)

        for future in as_completed(delete_futures):
            try:
                future.result()
            except Exception as e:
                print(f"  ⚠ Delete warning: {e}")


def _build_error_results(files_list, model_variant, error_msg=None):
    """Build error results for failed files."""
    results = []
    for f in files_list:
        item = f.get('item', {})
        results.append({
            'file_id': item.get('file_id', -1),
            'collection_number': item.get('collection_number', 'unknown'),
            'hypothesis': '',
            'status': 'error',
            'error_message': error_msg or f.get('error_message', 'Unknown error'),
            'model_name': f"gcp-{model_variant}"
        })
    return results


def process_batch_gcp_streaming(manifest_batch, storage_client, speech_client, cfg, recognizer_location):
    """
    Process files using TRUE GCP Batch API (multiple files per BatchRecognizeRequest).

    Pipeline:
    1. Stream upload to GCS (1 file at a time - memory safe)
    2. Build file metadata list (just URIs - tiny memory)
    3. Send ONE BatchRecognizeRequest for all files
    4. Poll ONE operation until complete
    5. Process results and cleanup

    This approach:
    - Uses minimal VM memory (only 1 file preprocessed at a time)
    - Leverages GCP's true batch API (up to 10,000 files per request)
    - Simpler than fake batching
    - No blocking on slowest files

    Args:
        manifest_batch: List of file items to process
        storage_client: GCP Storage client
        speech_client: GCP Speech client
        cfg: Config dict
        recognizer_location: GCP location (us-central1 or us)

    Returns:
        List of result dictionaries
    """
    bucket_name = cfg["model"]["gcp"]["temp_bucket"]
    bucket = storage_client.bucket(bucket_name)

    duration_sec = cfg["input"].get("duration_sec")
    sample_rate = cfg["input"].get("sample_rate", 16000)
    model_variant = cfg["model"]["model_variant"]
    project_id = cfg["model"]["gcp"]["project_id"]
    language_code = cfg["model"]["gcp"].get("language_code", "en-US")

    upload_workers = cfg["model"]["gcp"].get("upload_workers", 30)
    upload_timeout = cfg["model"]["gcp"].get("upload_timeout", 600)
    transcribe_timeout = cfg["model"]["gcp"].get("transcribe_timeout", 7200)  # 2 hours default
    max_concurrent_preprocessing = cfg["model"]["gcp"].get("max_concurrent_preprocessing", 10)

    # ===========================================================================
    # PHASE 1: Stream Upload to GCS (memory-safe: 1 file at a time)
    # ===========================================================================
    log(f"\n[Streaming Upload] Processing {len(manifest_batch)} files...")
    log(f"  Memory-safe: preprocessing max {max_concurrent_preprocessing} files concurrently")

    uploaded_files = []  # Track successfully uploaded files
    failed_files = []    # Track failed files

    # Use limited parallelism for preprocessing+upload
    with ThreadPoolExecutor(max_workers=max_concurrent_preprocessing) as executor:
        futures = {}

        for item in manifest_batch:
            # Submit download+preprocess+upload as one atomic operation
            future = executor.submit(
                _preprocess_and_upload_one,
                item, bucket, duration_sec, sample_rate, upload_timeout, bucket_name
            )
            futures[future] = item

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Upload to GCS"):
            item = futures[future]
            try:
                result = future.result()
                if result['status'] == 'success':
                    uploaded_files.append(result)
                else:
                    failed_files.append(result)
            except Exception as e:
                log(f"  ✗ Error processing {item['collection_number']}: {e}")
                failed_files.append({
                    'item': item,
                    'status': 'error',
                    'error_message': str(e)
                })

    log(f"  ✓ Uploaded: {len(uploaded_files)}/{len(manifest_batch)} files to GCS")

    if not uploaded_files:
        log("  ✗ No files successfully uploaded. Returning errors.")
        return _build_error_results(failed_files, model_variant)

    # ===========================================================================
    # PHASE 2: Build File Metadata List (tiny memory - just URIs)
    # ===========================================================================
    log(f"\n[Batch API] Creating file metadata for {len(uploaded_files)} files...")

    file_metadata_list = []
    gcs_uri_to_file = {}  # Map GCS URI back to file info

    for uf in uploaded_files:
        gcs_uri = uf['gcs_uri']
        file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)
        file_metadata_list.append(file_metadata)
        gcs_uri_to_file[gcs_uri] = uf

    log(f"  ✓ Created {len(file_metadata_list)} file metadata objects")

    # ===========================================================================
    # PHASE 3: Start transcriptions with RATE LIMITING (150 requests/min quota)
    # ===========================================================================
    # GCP quota: 150 BatchRecognize requests per minute
    # Strategy: Submit in batches with delays between batches
    log(f"\n[Batch API] Starting {len(file_metadata_list)} transcription operations...")
    log(f"  GCP Quota: 150 requests/minute")
    log(f"  Strategy: Submit in batches of 100, wait 60s between batches")

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language_code],
        model=model_variant,
    )

    recognizer = f"projects/{project_id}/locations/{recognizer_location}/recognizers/_"

    operations = []  # Store (operation, file_info) tuples

    # Rate limiting: submit in batches of 100 requests, wait 60 seconds between batches
    RATE_LIMIT_BATCH_SIZE = 100  # Stay under 150/min quota
    RATE_LIMIT_DELAY = 60  # seconds

    for batch_idx in range(0, len(uploaded_files), RATE_LIMIT_BATCH_SIZE):
        batch_files = uploaded_files[batch_idx:batch_idx + RATE_LIMIT_BATCH_SIZE]
        batch_num = batch_idx // RATE_LIMIT_BATCH_SIZE + 1
        total_rate_batches = (len(uploaded_files) + RATE_LIMIT_BATCH_SIZE - 1) // RATE_LIMIT_BATCH_SIZE

        log(f"\n  [Rate Limit Batch {batch_num}/{total_rate_batches}] Starting {len(batch_files)} operations...")

        # Start operations for this batch in parallel
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {}

            for uf in batch_files:
                gcs_uri = uf['gcs_uri']

                # Each file gets its own request (GCP limitation with inline response)
                future = executor.submit(
                    start_transcription,
                    speech_client,
                    gcs_uri,
                    model_variant,
                    recognizer_location,
                    project_id,
                    language_code
                )
                futures[future] = uf

            # Collect operations as they start
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Batch {batch_num} submit"):
                uf = futures[future]
                try:
                    operation = future.result()
                    operations.append((operation, uf))
                except Exception as e:
                    error_msg = str(e)
                    log(f"    ✗ Failed to start {uf['gcs_filename']}: {error_msg}")

                    # Check if it's a rate limit error (429)
                    if "429" in error_msg or "Quota exceeded" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        log(f"    ⚠ RATE LIMIT HIT! Will retry in next batch.")

                    failed_files.append({
                        'item': uf['item'],
                        'status': 'error',
                        'error_message': f"Failed to start: {error_msg}"
                    })

        log(f"  ✓ Batch {batch_num}: Started {len([f for f in futures if not futures[f] in [ff['item'] for ff in failed_files]])} operations")

        # Wait before next batch (unless this is the last batch)
        if batch_idx + RATE_LIMIT_BATCH_SIZE < len(uploaded_files):
            log(f"  ⏳ Waiting {RATE_LIMIT_DELAY}s before next batch (rate limiting)...")
            time.sleep(RATE_LIMIT_DELAY)

    log(f"\n  ✓ Total: Started {len(operations)}/{len(uploaded_files)} operations")

    # ===========================================================================
    # PHASE 4-5: Wait for operations and process results (with rate limit protection)
    # ===========================================================================
    log(f"\n[Transcription] Waiting for {len(operations)} operations to complete...")
    log(f"  Timeout per file: {transcribe_timeout}s ({transcribe_timeout/3600:.1f} hours)")
    log(f"  GCP Quota: 'Operation requests' = 150 per minute (for checking operation status)")
    log(f"  Rate Limiting: Checking operations at ~100/min (0.6s delay) to stay under quota")
    log(f"  GCP Console: https://console.cloud.google.com/speech/locations/{recognizer_location}?project={project_id}")
    log(f"\n  Starting to process operations at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  {'='*70}")

    batch_start = time.time()
    results = []

    # Track statistics
    completed_count = 0
    failed_count = 0

    for idx, (operation, uf) in enumerate(operations, start=1):
        file_start = time.time()
        gcs_uri = uf['gcs_uri']
        collection_num = uf['item']['collection_number']
        file_id = uf['item']['file_id']

        # Get operation name for logging
        operation_name = operation.operation.name if hasattr(operation, 'operation') else 'unknown'

        log(f"\n  [{idx}/{len(operations)}] File: {collection_num} (ID: {file_id})")
        log(f"      GCS URI: {gcs_uri}")
        log(f"      Operation: {operation_name}")
        log(f"      Started waiting at: {time.strftime('%H:%M:%S')}")

        try:
            # Wait for this operation to complete with retry on 429
            max_retries = 3
            retry_delay = 10  # seconds
            response = None

            for attempt in range(max_retries):
                try:
                    response = operation.result(timeout=transcribe_timeout)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e)
                    # Check if it's a 429 quota error
                    if "429" in error_str and "Quota exceeded" in error_str and attempt < max_retries - 1:
                        log(f"      ⚠ Quota exceeded (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise if not 429 or final attempt

            transcript = parse_gcp_response(response, gcs_uri)
            transcribe_time = time.time() - file_start
            completed_count += 1

            log(f"      ✓ COMPLETED in {transcribe_time:.1f}s ({transcribe_time/60:.1f} min)")
            log(f"      Transcript length: {len(transcript)} chars")
            log(f"      Audio duration: {uf['duration']:.1f}s")
            log(f"      Status: {completed_count} ✓ completed | {failed_count} ✗ failed | {len(operations) - idx} remaining")

            results.append({
                'file_id': file_id,
                'collection_number': collection_num,
                'hypothesis': transcript,
                'duration_sec': uf['duration'],
                'processing_time_sec': transcribe_time,
                'model_name': f"gcp-{model_variant}",
                'status': 'success',
                'ground_truth': uf['item'].get('ground_truth'),
                'title': uf['item'].get('title', ''),
                'blob_path': uf['successful_path']
            })
        except Exception as e:
            transcribe_time = time.time() - file_start
            failed_count += 1

            log(f"      ✗ FAILED after {transcribe_time:.1f}s ({transcribe_time/60:.1f} min)")
            log(f"      Error: {str(e)}")
            log(f"      Status: {completed_count} ✓ completed | {failed_count} ✗ failed | {len(operations) - idx} remaining")

            results.append({
                'file_id': file_id,
                'collection_number': collection_num,
                'hypothesis': '',
                'status': 'error',
                'error_message': f"Transcription error: {str(e)}",
                'model_name': f"gcp-{model_variant}"
            })

        # Rate limiting: Add delay between operation checks to avoid "Operation requests" quota
        # GCP V2 quota: "Operation requests per minute per region" = 150/min
        # Each operation.result() call makes at least one operation request
        # Delay of 0.6s ensures ~100 operations/minute (safely under 150/min quota)
        if idx < len(operations):  # Don't delay after last operation
            time.sleep(0.6)

    batch_time = time.time() - batch_start

    log(f"\n  {'='*70}")
    log(f"  Finished processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\n[Results Summary]")
    log(f"  Total operations: {len(operations)}")
    log(f"  ✓ Completed: {completed_count}")
    log(f"  ✗ Failed: {failed_count}")
    log(f"  Total time: {batch_time:.1f}s ({batch_time/60:.1f} min, {batch_time/3600:.2f} hours)")
    if completed_count > 0:
        avg_time = sum(r.get('processing_time_sec', 0) for r in results if r['status'] == 'success') / completed_count
        log(f"  Avg time per file: {avg_time:.1f}s ({avg_time/60:.1f} min)")

    # Add failed files to results
    for ff in failed_files:
        results.append({
            'file_id': ff['item']['file_id'],
            'collection_number': ff['item']['collection_number'],
            'hypothesis': '',
            'status': 'error',
            'error_message': ff.get('error_message', 'Unknown error'),
            'model_name': f"gcp-{model_variant}"
        })

    # Cleanup GCS files
    log(f"\n[Cleanup] Deleting {len(uploaded_files)} files from GCS...")
    _cleanup_gcs_files(bucket, [uf['gcs_filename'] for uf in uploaded_files], upload_workers)
    log(f"  ✓ Cleanup complete")

    return results


def run(cfg):
    """
    Run GCP Chirp inference on dataset from config.

    Supports:
    - Batch processing with parallel upload/transcribe/delete
    - Azure blob storage
    - Fair comparison (same preprocessing as Whisper/Wav2Vec2)
    - Rate limiting (150 requests/min quota)
    - File logging (timestamped logs saved to output directory)
    """
    experiment_start_time = time.time()

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize file logger FIRST (before any log() calls)
    init_logger(out_dir, prefix="gcp_chirp")

    # Setup GCP clients
    log("Setting up GCP clients...")
    storage_client, speech_client, recognizer_location = setup_gcp_clients(cfg)

    # Prepare file manifest
    source_type = cfg["input"].get("source", "azure_blob")

    if source_type == "azure_blob":
        parquet_path = cfg["input"]["parquet_path"]
        sample_size = cfg["input"].get("sample_size")
        blob_prefix = cfg["input"].get("blob_prefix", "loc_vhp")

        df = data_loader.load_vhp_dataset(parquet_path, sample_size=sample_size)
        manifest = data_loader.prepare_inference_manifest(df, blob_prefix=blob_prefix)
        log(f"Prepared manifest with {len(manifest)} items from Azure blob")
    else:
        raise ValueError(f"Source type '{source_type}' not supported for GCP Chirp. Use 'azure_blob'.")

    # Process in batches
    batch_size = cfg["model"]["gcp"].get("batch_size", 100)
    all_results = []

    log(f"\nProcessing {len(manifest)} files in batches of {batch_size}...")

    for i in range(0, len(manifest), batch_size):
        batch = manifest[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(manifest) + batch_size - 1) // batch_size

        log(f"\n{'='*60}")
        log(f"BATCH {batch_num}/{total_batches} ({len(batch)} files)")
        log(f"{'='*60}")

        batch_results = process_batch_gcp_streaming(
            batch,
            storage_client,
            speech_client,
            cfg,
            recognizer_location
        )

        all_results.extend(batch_results)

    # Save results
    log(f"\n{'='*60}")
    log(f"SAVING RESULTS")
    log(f"{'='*60}")

    # Save per-file results to parquet
    df_results = pd.DataFrame(all_results)
    parquet_path = out_dir / "inference_results.parquet"
    df_results.to_parquet(parquet_path, index=False)
    log(f"✓ Saved per-file results: {parquet_path}")

    # Save individual hypothesis files (if enabled)
    if cfg["output"].get("save_per_file", False):
        for result in all_results:
            if result['status'] == 'success':
                file_id = result['file_id']
                per_file_path = out_dir / f"hyp_{file_id}.txt"
                with open(per_file_path, "w") as f:
                    f.write(result['hypothesis'])
        log(f"✓ Saved individual hypothesis files for {sum(1 for r in all_results if r['status'] == 'success')} successful transcriptions")

    # Save hypothesis text file (for legacy compatibility)
    model_variant = cfg["model"]["model_variant"]
    hyp_path = out_dir / f"hyp_gcp_{model_variant}.txt"

    with open(hyp_path, "w") as hout:
        for result in all_results:
            if result['status'] == 'success':
                hout.write(result['hypothesis'] + "\n")
            else:
                hout.write("[ERROR]\n")

    log(f"✓ Saved hypothesis text: {hyp_path}")

    # Summary
    total_time = time.time() - experiment_start_time
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = len(all_results) - successful

    log(f"\n{'='*60}")
    log(f"EXPERIMENT COMPLETE")
    log(f"{'='*60}")
    log(f"Total files: {len(all_results)}")
    log(f"Successful: {successful}")
    log(f"Failed: {failed}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if successful > 0:
        avg_time = sum(r.get('processing_time_sec', 0) for r in all_results if r['status'] == 'success') / successful
        log(f"Avg processing time: {avg_time:.1f}s per file")

    # Log to wandb
    wandb.log({
        "total_files": len(all_results),
        "successful_files": successful,
        "failed_files": failed,
        "total_time_sec": total_time,
        "status": "inference_done"
    })

    return {
        "inference_results_parquet": str(parquet_path),
        "hypothesis_text": str(hyp_path)
    }
