"""
GCP Speech-to-Text (Chirp 2 & 3) inference module.

Supports:
- Batch processing with parallel upload/transcribe/delete
- Fair comparison: same preprocessing as Whisper/Wav2Vec2 (pydub → mono 16kHz WAV)
- Azure blob storage integration
- Automatic endpoint selection based on model variant
"""
import os
import sys
import io
import time
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile

# Audio processing
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

# GCP clients
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

# Experiment tracking
import wandb

# Import local modules
_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))
import azure_utils
import data_loader


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
        print(f"Using Chirp 2 with endpoint: {api_endpoint}, location: {recognizer_location}")
    elif model_variant == "chirp_3":
        # Chirp 3: us multi-region
        api_endpoint = "us-speech.googleapis.com"
        recognizer_location = "us"
        print(f"Using Chirp 3 with endpoint: {api_endpoint}, location: {recognizer_location}")
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


def process_batch_gcp(manifest_batch, storage_client, speech_client, cfg, recognizer_location):
    """
    Process a batch of files with GCP Chirp using parallel upload/transcribe/delete.

    Pipeline:
    1. Download from Azure + Preprocess to WAV (parallel)
    2. Upload WAV to GCS (parallel)
    3. Start transcriptions (parallel, returns operations)
    4. Wait for all operations to complete
    5. Delete from GCS (parallel)

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

    upload_workers = cfg["model"]["gcp"].get("upload_workers", 50)
    transcribe_workers = cfg["model"]["gcp"].get("transcribe_workers", 30)
    upload_timeout = cfg["model"]["gcp"].get("upload_timeout", 600)
    transcribe_timeout = cfg["model"]["gcp"].get("transcribe_timeout", 7200)  # 2 hours default for long audio

    # IMPORTANT: Limit concurrent preprocessing to avoid OOM
    # Even if upload_workers is 50, only preprocess a few files at a time
    max_concurrent_preprocessing = cfg["model"]["gcp"].get("max_concurrent_preprocessing", 10)
    actual_preprocessing_workers = min(upload_workers, max_concurrent_preprocessing)

    # Phase 1: Download from Azure + Preprocess (parallel, but limited)
    print(f"\n[Batch] Phase 1: Downloading and preprocessing {len(manifest_batch)} files...")
    print(f"  Using {actual_preprocessing_workers} concurrent workers (max_concurrent_preprocessing={max_concurrent_preprocessing})")

    prepared_files = []
    with ThreadPoolExecutor(max_workers=actual_preprocessing_workers) as executor:
        futures = {}
        for item in manifest_batch:
            future = executor.submit(download_and_preprocess, item, duration_sec, sample_rate)
            futures[future] = item

        for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            item = futures[future]
            try:
                wav_bytes, duration, successful_path = future.result()
                prepared_files.append({
                    'item': item,
                    'wav_bytes': wav_bytes,
                    'duration': duration,
                    'successful_path': successful_path,
                    'gcs_filename': f"{item['collection_number']}.wav"
                })
            except Exception as e:
                print(f"  ✗ Error preprocessing {item['collection_number']}: {e}")
                prepared_files.append({
                    'item': item,
                    'error': str(e),
                    'gcs_filename': None
                })

    # Filter out failed preprocessing
    valid_files = [pf for pf in prepared_files if 'error' not in pf]
    failed_preprocessing = [pf for pf in prepared_files if 'error' in pf]

    print(f"  ✓ Preprocessed: {len(valid_files)}/{len(manifest_batch)} files")

    if not valid_files:
        print("  ✗ No files successfully preprocessed. Skipping batch.")
        return [{'file_id': pf['item']['file_id'], 'hypothesis': '', 'status': 'error',
                 'error_message': pf['error']} for pf in failed_preprocessing]

    # Phase 2: Upload WAV to GCS (parallel, with memory management)
    print(f"[Batch] Phase 2: Uploading {len(valid_files)} WAV files to GCS...")
    print(f"  Using {upload_workers} upload workers")

    with ThreadPoolExecutor(max_workers=upload_workers) as executor:
        # Create mapping of future -> preprocessed_file
        future_to_pf = {}
        for pf in valid_files:
            future = executor.submit(
                upload_to_gcs,
                bucket,
                pf['gcs_filename'],
                pf['wav_bytes'],
                upload_timeout
            )
            future_to_pf[future] = pf

        # Wait for uploads to complete
        for future in tqdm(as_completed(future_to_pf.keys()),
                          total=len(future_to_pf), desc="Uploading"):
            pf = future_to_pf[future]
            try:
                future.result()
            except Exception as e:
                print(f"  ✗ Upload failed for {pf['gcs_filename']}: {e}")
                pf['upload_error'] = str(e)

    # Filter out failed uploads
    uploaded_files = [pf for pf in valid_files if 'upload_error' not in pf]
    print(f"  ✓ Uploaded: {len(uploaded_files)}/{len(valid_files)} files")

    # Free up memory: delete WAV bytes after upload (keep only metadata)
    for pf in valid_files:
        if 'wav_bytes' in pf:
            del pf['wav_bytes']  # Free large WAV data from memory
    print(f"  ✓ Freed WAV data from memory")

    # Phase 3: Start transcriptions (parallel)
    print(f"[Batch] Phase 3: Starting {len(uploaded_files)} transcriptions...")

    operations = []
    with ThreadPoolExecutor(max_workers=transcribe_workers) as executor:
        transcribe_futures = []
        for pf in uploaded_files:
            gcs_uri = f"gs://{bucket_name}/{pf['gcs_filename']}"
            future = executor.submit(
                start_transcription,
                speech_client,
                gcs_uri,
                model_variant,
                recognizer_location,
                project_id,
                language_code
            )
            transcribe_futures.append((future, pf, gcs_uri))

        for future, pf, gcs_uri in tqdm(transcribe_futures, desc="Starting operations"):
            try:
                operation = future.result()
                operations.append((operation, pf, gcs_uri))
            except Exception as e:
                print(f"  ✗ Failed to start transcription for {pf['gcs_filename']}: {e}")
                pf['transcribe_error'] = str(e)

    print(f"  ✓ Started: {len(operations)}/{len(uploaded_files)} transcriptions")

    # Phase 4: Wait for operations to complete
    print(f"[Batch] Phase 4: Waiting for {len(operations)} transcriptions to complete...")

    results = []
    transcribe_start = time.time()

    for operation, pf, gcs_uri in tqdm(operations, desc="Transcribing"):
        file_start = time.time()
        try:
            response = operation.result(timeout=transcribe_timeout)
            transcript = parse_gcp_response(response, gcs_uri)
            transcribe_time = time.time() - file_start

            results.append({
                'file_id': pf['item']['file_id'],
                'collection_number': pf['item']['collection_number'],
                'hypothesis': transcript,
                'duration_sec': pf['duration'],
                'processing_time_sec': transcribe_time,
                'model_name': f"gcp-{model_variant}",
                'status': 'success',
                'ground_truth': pf['item'].get('ground_truth'),
                'title': pf['item'].get('title', ''),
                'blob_path': pf['successful_path']
            })
        except Exception as e:
            results.append({
                'file_id': pf['item']['file_id'],
                'collection_number': pf['item']['collection_number'],
                'hypothesis': '',
                'status': 'error',
                'error_message': f"Transcription error: {str(e)}",
                'model_name': f"gcp-{model_variant}"
            })

    total_transcribe_time = time.time() - transcribe_start
    print(f"  ✓ Completed {len(results)} transcriptions in {total_transcribe_time:.1f}s")

    # Phase 5: Delete from GCS (parallel)
    print(f"[Batch] Phase 5: Cleaning up {len(uploaded_files)} files from GCS...")

    with ThreadPoolExecutor(max_workers=upload_workers) as executor:
        delete_futures = []
        for pf in uploaded_files:
            future = executor.submit(
                delete_from_gcs,
                bucket,
                pf['gcs_filename']
            )
            delete_futures.append(future)

        for future in as_completed(delete_futures):
            try:
                future.result()
            except Exception as e:
                print(f"  ⚠ Delete warning: {e}")

    print(f"  ✓ Cleanup complete")

    # Add errors from failed preprocessing/upload
    for pf in failed_preprocessing:
        results.append({
            'file_id': pf['item']['file_id'],
            'collection_number': pf['item']['collection_number'],
            'hypothesis': '',
            'status': 'error',
            'error_message': pf['error'],
            'model_name': f"gcp-{model_variant}"
        })

    for pf in valid_files:
        if 'upload_error' in pf:
            results.append({
                'file_id': pf['item']['file_id'],
                'collection_number': pf['item']['collection_number'],
                'hypothesis': '',
                'status': 'error',
                'error_message': f"Upload error: {pf['upload_error']}",
                'model_name': f"gcp-{model_variant}"
            })
        elif 'transcribe_error' in pf:
            results.append({
                'file_id': pf['item']['file_id'],
                'collection_number': pf['item']['collection_number'],
                'hypothesis': '',
                'status': 'error',
                'error_message': pf['transcribe_error'],
                'model_name': f"gcp-{model_variant}"
            })

    return results


def run(cfg):
    """
    Run GCP Chirp inference on dataset from config.

    Supports:
    - Batch processing with parallel upload/transcribe/delete
    - Azure blob storage
    - Fair comparison (same preprocessing as Whisper/Wav2Vec2)
    """
    experiment_start_time = time.time()

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup GCP clients
    print("Setting up GCP clients...")
    storage_client, speech_client, recognizer_location = setup_gcp_clients(cfg)

    # Prepare file manifest
    source_type = cfg["input"].get("source", "azure_blob")

    if source_type == "azure_blob":
        parquet_path = cfg["input"]["parquet_path"]
        sample_size = cfg["input"].get("sample_size")
        blob_prefix = cfg["input"].get("blob_prefix", "loc_vhp")

        df = data_loader.load_vhp_dataset(parquet_path, sample_size=sample_size)
        manifest = data_loader.prepare_inference_manifest(df, blob_prefix=blob_prefix)
        print(f"Prepared manifest with {len(manifest)} items from Azure blob")
    else:
        raise ValueError(f"Source type '{source_type}' not supported for GCP Chirp. Use 'azure_blob'.")

    # Process in batches
    batch_size = cfg["model"]["gcp"].get("batch_size", 100)
    all_results = []

    print(f"\nProcessing {len(manifest)} files in batches of {batch_size}...")

    for i in range(0, len(manifest), batch_size):
        batch = manifest[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(manifest) + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch)} files)")
        print(f"{'='*60}")

        batch_results = process_batch_gcp(
            batch,
            storage_client,
            speech_client,
            cfg,
            recognizer_location
        )

        all_results.extend(batch_results)

    # Save results
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS")
    print(f"{'='*60}")

    # Save per-file results to parquet
    df_results = pd.DataFrame(all_results)
    parquet_path = out_dir / "inference_results.parquet"
    df_results.to_parquet(parquet_path, index=False)
    print(f"✓ Saved per-file results: {parquet_path}")

    # Save individual hypothesis files (if enabled)
    if cfg["output"].get("save_per_file", False):
        for result in all_results:
            if result['status'] == 'success':
                file_id = result['file_id']
                per_file_path = out_dir / f"hyp_{file_id}.txt"
                with open(per_file_path, "w") as f:
                    f.write(result['hypothesis'])
        print(f"✓ Saved individual hypothesis files for {sum(1 for r in all_results if r['status'] == 'success')} successful transcriptions")

    # Save hypothesis text file (for legacy compatibility)
    model_variant = cfg["model"]["model_variant"]
    hyp_path = out_dir / f"hyp_gcp_{model_variant}.txt"

    with open(hyp_path, "w") as hout:
        for result in all_results:
            if result['status'] == 'success':
                hout.write(result['hypothesis'] + "\n")
            else:
                hout.write("[ERROR]\n")

    print(f"✓ Saved hypothesis text: {hyp_path}")

    # Summary
    total_time = time.time() - experiment_start_time
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = len(all_results) - successful

    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total files: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if successful > 0:
        avg_time = sum(r.get('processing_time_sec', 0) for r in all_results if r['status'] == 'success') / successful
        print(f"Avg processing time: {avg_time:.1f}s per file")

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
