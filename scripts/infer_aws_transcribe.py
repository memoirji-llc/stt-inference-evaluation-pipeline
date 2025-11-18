"""
AWS Transcribe inference module.

Supports:
- Batch processing with parallel upload/transcribe/delete
- Fair comparison: same preprocessing as Whisper/Wav2Vec2/GCP (pydub → mono 16kHz WAV)
- Azure blob storage integration
- AWS automatic job queueing (up to 100 concurrent jobs)
- File logging (timestamped logs saved to output directory)
"""
import os
import sys
import io
import time
import urllib.request
import json
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Audio processing
from pydub import AudioSegment
import pandas as pd
from tqdm import tqdm

# AWS clients
import boto3
from botocore.exceptions import ClientError

# Experiment tracking
import wandb

# Import local modules
_scripts_dir = Path(__file__).parent
sys.path.insert(0, str(_scripts_dir))
import azure_utils
import data_loader
from file_logger import log, init_logger


def setup_aws_clients(cfg):
    """
    Initialize AWS S3 and Transcribe clients.

    If credentials_path is specified in config, load from that .env file.
    Otherwise, boto3 will use default credential chain (environment vars, ~/.aws/credentials, IAM role).

    Returns:
        (s3_client, transcribe_client, region)
    """
    region = cfg["model"]["aws"]["region"]

    # Optional: Load credentials from .env file if specified in config
    # This allows running without manual export, similar to GCP
    credentials_path = cfg["model"]["aws"].get("credentials_path")
    if credentials_path:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=credentials_path)
        log(f"Loaded credentials from: {credentials_path}")

    # Get credentials from environment (either loaded above or already set)
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Create clients with explicit credentials (same pattern as notebook)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

    transcribe_client = boto3.client(
        'transcribe',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

    log(f"AWS clients initialized (region: {region})")
    return s3_client, transcribe_client, region


def prepare_audio_for_aws(audio_bytes, duration_sec=None, sample_rate=16000):
    """
    Preprocess audio EXACTLY like Whisper/Wav2Vec2/GCP pipeline for fair comparison.

    Args:
        audio_bytes: Raw audio bytes (MP3/MP4/WAV/etc.)
        duration_sec: Optional duration limit (seconds)
        sample_rate: Target sample rate (default: 16000)

    Returns:
        (wav_bytes, actual_duration_sec)
    """
    # Load with pydub (handles MP3, MP4, M4A, WAV, etc.)
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

    # Normalize: EXACT same as infer_whisper.py and infer_gcp_chirp.py
    audio_segment = audio_segment.set_channels(1)  # Mono
    audio_segment = audio_segment.set_frame_rate(sample_rate)  # 16kHz

    # Trim to desired duration if specified (SAME as Whisper/GCP)
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
    Download from Azure and preprocess to WAV (SAME as Whisper/GCP pipeline).

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

    # Preprocess EXACTLY like Whisper/GCP
    wav_bytes, actual_duration = prepare_audio_for_aws(audio_bytes, duration_sec, sample_rate)

    return wav_bytes, actual_duration, successful_path


def upload_to_s3(s3_client, bucket, key, wav_bytes, timeout=600):
    """Upload WAV bytes to S3 bucket.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        key: Object key (path in bucket)
        wav_bytes: WAV file bytes to upload
        timeout: Upload timeout in seconds (default: 600 = 10 minutes)
    """
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=wav_bytes,
        ContentType='audio/wav'
    )


def delete_from_s3(s3_client, bucket, key):
    """Delete file from S3 bucket."""
    s3_client.delete_object(Bucket=bucket, Key=key)


def delete_transcription_job(transcribe_client, job_name):
    """Delete transcription job after completion."""
    try:
        transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
    except Exception as e:
        # Non-critical - job will auto-expire
        pass


def _preprocess_and_upload_one(item, s3_client, bucket, duration_sec, sample_rate, bucket_name):
    """
    Download, preprocess, and upload ONE file to S3 atomically.
    This function processes one file at a time to minimize memory usage.

    Returns dict with status and file info.
    """
    try:
        # Download and preprocess
        wav_bytes, duration, successful_path = download_and_preprocess(item, duration_sec, sample_rate)

        # Check AWS limits BEFORE uploading
        # AWS Transcribe limits: 4 hours OR 2 GB (whichever comes first)
        if duration > 14400:  # 4 hours = 14400 seconds
            return {
                'status': 'error',
                'item': item,
                'error_message': f'Audio duration {duration:.1f}s exceeds AWS 4-hour limit'
            }

        if len(wav_bytes) > 2 * 1024**3:  # 2 GB
            return {
                'status': 'error',
                'item': item,
                'error_message': f'File size {len(wav_bytes)/1024**3:.2f}GB exceeds AWS 2GB limit'
            }

        # Upload to S3
        # s3 key should align with az blob path of the media file
        ori_path = Path(successful_path)
        ori_path_wav = str(ori_path.with_suffix(".wav"))
        s3_key = str(ori_path_wav)
        upload_to_s3(s3_client, bucket, s3_key, wav_bytes)

        # Immediately free memory
        del wav_bytes

        # Return success with metadata
        return {
            'status': 'success',
            'item': item,
            's3_key': s3_key,
            's3_uri': f"s3://{bucket_name}/{s3_key}",
            'duration': duration,
            'successful_path': successful_path
        }

    except Exception as e:
        return {
            'status': 'error',
            'item': item,
            'error_message': str(e)
        }


def _cleanup_s3_files(s3_client, bucket, s3_keys, workers=30):
    """Delete multiple files from S3 in parallel."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        delete_futures = []
        for key in s3_keys:
            future = executor.submit(delete_from_s3, s3_client, bucket, key)
            delete_futures.append(future)

        for future in as_completed(delete_futures):
            try:
                future.result()
            except Exception as e:
                log(f"  ⚠ Delete warning: {e}")


def _cleanup_transcription_jobs(transcribe_client, job_names, workers=30):
    """Delete multiple transcription jobs in parallel."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        delete_futures = []
        for job_name in job_names:
            future = executor.submit(delete_transcription_job, transcribe_client, job_name)
            delete_futures.append(future)

        for future in as_completed(delete_futures):
            try:
                future.result()
            except Exception:
                pass  # Non-critical


def start_transcription_job(transcribe_client, job_name, s3_uri, language_code, region):
    """
    Start a transcription job for one file.

    Args:
        transcribe_client: boto3 Transcribe client
        job_name: Unique job name
        s3_uri: S3 URI (s3://bucket/key.wav)
        language_code: Language code (e.g., 'en-US')
        region: AWS region

    Returns:
        job_name (for polling)
    """
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        LanguageCode=language_code
    )
    return job_name


def poll_job_status(transcribe_client, job_name, timeout=14400, poll_interval=10):
    """
    Poll job until COMPLETED or FAILED.

    Args:
        transcribe_client: boto3 Transcribe client
        job_name: Job name to poll
        timeout: Max time to wait (default: 4 hours)
        poll_interval: Time between polls in seconds

    Returns:
        (status, transcript_or_error)
        status: 'success' or 'error'
        transcript_or_error: transcript text if success, error message if error
    """
    start_time = time.time()

    while True:
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            return ('error', f'Timeout after {timeout}s')

        # Get job status
        try:
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job = response['TranscriptionJob']
            status = job['TranscriptionJobStatus']

            if status == 'COMPLETED':
                # Fetch transcript from URI
                transcript_uri = job['Transcript']['TranscriptFileUri']
                with urllib.request.urlopen(transcript_uri) as r:
                    data = json.loads(r.read())
                transcript = data['results']['transcripts'][0]['transcript']
                return ('success', transcript)

            elif status == 'FAILED':
                error = job.get('FailureReason', 'Unknown error')
                return ('error', f'Job failed: {error}')

            # Still in progress
            time.sleep(poll_interval)

        except Exception as e:
            return ('error', f'Polling error: {str(e)}')


def process_batch_aws_streaming(manifest_batch, s3_client, transcribe_client, cfg, region):
    """
    Process files using AWS Transcribe batch API.

    Pipeline:
    1. Stream upload to S3 (1 file at a time - memory safe)
    2. Start transcription jobs (let AWS handle queueing beyond 100 concurrent)
    3. Poll each job until complete (sequential with detailed logging)
    4. Process results and cleanup (delete S3 files + jobs)

    This approach:
    - Uses minimal VM memory (only 1 file preprocessed at a time)
    - Leverages AWS automatic job queueing (up to 100 concurrent)
    - Provides detailed per-job logging for visibility

    Args:
        manifest_batch: List of file items to process
        s3_client: boto3 S3 client
        transcribe_client: boto3 Transcribe client
        cfg: Config dict
        region: AWS region

    Returns:
        List of result dictionaries
    """
    bucket_name = cfg["model"]["aws"]["temp_bucket"]

    duration_sec = cfg["input"].get("duration_sec")
    sample_rate = cfg["input"].get("sample_rate", 16000)
    language_code = cfg["model"]["aws"].get("language_code", "en-US")

    upload_timeout = cfg["model"]["aws"].get("upload_timeout", 600)
    transcribe_timeout = cfg["model"]["aws"].get("transcribe_timeout", 14400)  # 4 hours default
    poll_interval = cfg["model"]["aws"].get("poll_interval", 10)
    max_concurrent_preprocessing = cfg["model"]["aws"].get("max_concurrent_preprocessing", 3)

    # ===========================================================================
    # PHASE 1: Stream Upload to S3 (memory-safe: max N files at a time)
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
                item, s3_client, bucket_name, duration_sec, sample_rate, bucket_name
            )
            futures[future] = item

        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Upload to S3"):
            item = futures[future]
            
            try:
                result = future.result()
                if result['status'] == 'success':
                    uploaded_files.append(result)
                else:
                    failed_files.append(result)
                    print(f"  ⚠ Skipped {item['collection_number']}: {result['error_message']}")
            except Exception as e:
                log(f"  ✗ Error processing {item['collection_number']}: {e}")
                failed_files.append({
                    'item': item,
                    'status': 'error',
                    'error_message': str(e)
                })

    log(f"  ✓ Uploaded: {len(uploaded_files)}/{len(manifest_batch)} files to S3")

    if not uploaded_files:
        log("  ✗ No files successfully uploaded. Returning errors.")
        return _build_error_results(failed_files)

    # ===========================================================================
    # PHASE 2: Start transcription jobs with throttling (AWS 100 concurrent limit)
    # ===========================================================================
    log(f"\n[Job Submission] Starting {len(uploaded_files)} transcription jobs...")
    log(f"  AWS Concurrent Limit: 100 jobs")
    log(f"  Strategy: Submit in batches of 80, wait for some to complete before submitting more")

    jobs = []  # Store (job_name, file_info) tuples
    pending_uploads = list(uploaded_files)  # Files waiting to be submitted
    timestamp = int(time.time())

    # Initial batch: submit first 80 jobs
    BATCH_SIZE = 80  # Stay under 100 limit with safety margin
    initial_batch = pending_uploads[:BATCH_SIZE]
    pending_uploads = pending_uploads[BATCH_SIZE:]

    log(f"\n  [Initial Batch] Submitting {len(initial_batch)} jobs...")
    for idx, uf in enumerate(initial_batch, start=1):
        s3_uri = uf['s3_uri']
        collection_num = uf['item']['collection_number']
        file_id = uf['item']['file_id']
        safe_collection_num = collection_num.replace('/', '_')
        job_name = f"{safe_collection_num}_{timestamp}_{file_id}"

        try:
            start_transcription_job(transcribe_client, job_name, s3_uri, language_code, region)
            jobs.append((job_name, uf))

            if idx <= 5 or idx % 20 == 0:
                log(f"    [{idx}/{len(initial_batch)}] Submitted: {collection_num}")
        except Exception as e:
            log(f"    ✗ Failed to start {collection_num}: {e}")
            failed_files.append({
                'item': uf['item'],
                'status': 'error',
                'error_message': f"Failed to start job: {str(e)}"
            })

    log(f"  ✓ Initial batch: {len(jobs)} jobs submitted")

    # Now submit remaining jobs dynamically as others complete
    if pending_uploads:
        log(f"\n  [Dynamic Submission] {len(pending_uploads)} jobs remaining, will submit as slots free up...")
        pending_idx = 0
        running_jobs = set(job_name for job_name, _ in jobs)  # Track running job names

        while pending_uploads:
            # Wait a bit for some jobs to complete
            time.sleep(15)  # Check every 15 seconds

            # Check how many jobs are still running
            still_running = set()
            for job_name in running_jobs:
                try:
                    response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                    status = response['TranscriptionJob']['TranscriptionJobStatus']
                    if status == 'IN_PROGRESS':
                        still_running.add(job_name)
                except:
                    pass  # Job might be done/deleted

            running_jobs = still_running
            available_slots = 100 - len(running_jobs)

            if available_slots > 20:  # Submit more if we have at least 20 free slots
                to_submit = min(available_slots - 10, len(pending_uploads))  # Keep 10 slot buffer
                batch = pending_uploads[:to_submit]
                pending_uploads = pending_uploads[to_submit:]

                log(f"    [{len(running_jobs)} running, {available_slots} slots] Submitting {to_submit} more jobs...")

                for uf in batch:
                    s3_uri = uf['s3_uri']
                    collection_num = uf['item']['collection_number']
                    file_id = uf['item']['file_id']
                    safe_collection_num = collection_num.replace('/', '_')
                    job_name = f"{safe_collection_num}_{timestamp}_{file_id}"

                    try:
                        start_transcription_job(transcribe_client, job_name, s3_uri, language_code, region)
                        jobs.append((job_name, uf))
                        running_jobs.add(job_name)
                        pending_idx += 1
                    except Exception as e:
                        log(f"    ✗ Failed to start {collection_num}: {e}")
                        failed_files.append({
                            'item': uf['item'],
                            'status': 'error',
                            'error_message': f"Failed to start job: {str(e)}"
                        })

    log(f"\n  ✓ All jobs submitted: {len(jobs)}/{len(uploaded_files)} total")

    # ===========================================================================
    # PHASE 3: Poll jobs with detailed logging
    # ===========================================================================
    log(f"\n[Transcription] Polling {len(jobs)} jobs...")
    log(f"  Timeout per job: {transcribe_timeout}s ({transcribe_timeout/3600:.1f} hours)")
    log(f"  Poll interval: {poll_interval}s")
    log(f"  AWS Console: https://{region}.console.aws.amazon.com/transcribe/home?region={region}#jobs")
    log(f"\n  Starting to poll at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  {'='*70}")

    batch_start = time.time()
    results = []

    # Track statistics
    completed_count = 0
    failed_count = 0

    for idx, (job_name, uf) in enumerate(jobs, start=1):
        file_start = time.time()
        collection_num = uf['item']['collection_number']
        file_id = uf['item']['file_id']
        s3_uri = uf['s3_uri']

        log(f"\n  [{idx}/{len(jobs)}] File: {collection_num} (ID: {file_id})")
        log(f"      Job Name: {job_name}")
        log(f"      S3 URI: {s3_uri}")
        log(f"      Started polling at: {time.strftime('%H:%M:%S')}")

        try:
            # Poll until complete
            status, result = poll_job_status(transcribe_client, job_name, transcribe_timeout, poll_interval)
            transcribe_time = time.time() - file_start

            if status == 'success':
                completed_count += 1
                transcript = result

                log(f"      ✓ COMPLETED in {transcribe_time:.1f}s ({transcribe_time/60:.1f} min)")
                log(f"      Transcript length: {len(transcript)} chars")
                log(f"      Audio duration: {uf['duration']:.1f}s")
                log(f"      Status: {completed_count} ✓ completed | {failed_count} ✗ failed | {len(jobs) - idx} remaining")

                results.append({
                    'file_id': file_id,
                    'collection_number': collection_num,
                    'hypothesis': transcript,
                    'duration_sec': uf['duration'],
                    'processing_time_sec': transcribe_time,
                    'model_name': 'aws-transcribe',
                    'status': 'success',
                    'ground_truth': uf['item'].get('ground_truth'),
                    'title': uf['item'].get('title', ''),
                    'blob_path': uf['successful_path']
                })
            else:
                failed_count += 1
                error_msg = result

                log(f"      ✗ FAILED after {transcribe_time:.1f}s ({transcribe_time/60:.1f} min)")
                log(f"      Error: {error_msg}")
                log(f"      Status: {completed_count} ✓ completed | {failed_count} ✗ failed | {len(jobs) - idx} remaining")

                results.append({
                    'file_id': file_id,
                    'collection_number': collection_num,
                    'hypothesis': '',
                    'status': 'error',
                    'error_message': error_msg,
                    'model_name': 'aws-transcribe'
                })

        except Exception as e:
            transcribe_time = time.time() - file_start
            failed_count += 1

            log(f"      ✗ EXCEPTION after {transcribe_time:.1f}s ({transcribe_time/60:.1f} min)")
            log(f"      Error: {str(e)}")
            log(f"      Status: {completed_count} ✓ completed | {failed_count} ✗ failed | {len(jobs) - idx} remaining")

            results.append({
                'file_id': file_id,
                'collection_number': collection_num,
                'hypothesis': '',
                'status': 'error',
                'error_message': f'Exception: {str(e)}',
                'model_name': 'aws-transcribe'
            })

    batch_time = time.time() - batch_start

    log(f"\n  {'='*70}")
    log(f"  Finished polling at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\n[Results Summary]")
    log(f"  Total jobs: {len(jobs)}")
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
            'model_name': 'aws-transcribe'
        })

    # ===========================================================================
    # PHASE 4: Cleanup (delete S3 files + transcription jobs)
    # ===========================================================================
    log(f"\n[Cleanup] Deleting {len(uploaded_files)} S3 files and {len(jobs)} transcription jobs...")

    # Delete S3 files
    _cleanup_s3_files(s3_client, bucket_name, [uf['s3_key'] for uf in uploaded_files])

    # Delete transcription jobs
    _cleanup_transcription_jobs(transcribe_client, [job_name for job_name, _ in jobs])

    log(f"  ✓ Cleanup complete")

    return results


def _build_error_results(files_list):
    """Build error results for failed files."""
    results = []
    for f in files_list:
        item = f.get('item', {})
        results.append({
            'file_id': item.get('file_id', -1),
            'collection_number': item.get('collection_number', 'unknown'),
            'hypothesis': '',
            'status': 'error',
            'error_message': f.get('error_message', 'Unknown error'),
            'model_name': 'aws-transcribe'
        })
    return results


def run(cfg):
    """
    Run AWS Transcribe inference on dataset from config.

    Supports:
    - Batch processing with parallel upload/transcribe/delete
    - Azure blob storage
    - Fair comparison (same preprocessing as Whisper/Wav2Vec2/GCP)
    - File logging (timestamped logs saved to output directory)
    """
    experiment_start_time = time.time()

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize file logger FIRST (before any log() calls)
    init_logger(out_dir, prefix="aws_transcribe")

    # Setup AWS clients
    log("Setting up AWS clients...")
    s3_client, transcribe_client, region = setup_aws_clients(cfg)

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
        raise ValueError(f"Source type '{source_type}' not supported for AWS Transcribe. Use 'azure_blob'.")

    # Process in batches (for code consistency, but AWS handles queueing)
    batch_size = cfg["model"]["aws"].get("batch_size", 1000)
    all_results = []

    log(f"\nProcessing {len(manifest)} files in batches of {batch_size}...")

    for i in range(0, len(manifest), batch_size):
        batch = manifest[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(manifest) + batch_size - 1) // batch_size

        log(f"\n{'='*60}")
        log(f"BATCH {batch_num}/{total_batches} ({len(batch)} files)")
        log(f"{'='*60}")

        batch_results = process_batch_aws_streaming(
            batch,
            s3_client,
            transcribe_client,
            cfg,
            region
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
    hyp_path = out_dir / "hyp_aws_transcribe.txt"

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
