```
export AZURE_STORAGE_ACCOUNT=stgamiadata26828
export AZURE_STORAGE_CONTAINER=audio-raw
export AZURE_AUTH=managed_identity
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=stgamiadata26828;AccountKey=nNB/yPIgD2OK4paO6iUTYessNkiQ2PVne0EqTYAehDWo+qxXhVtNQPqlKvVdkL+polFXvA6gl433+AStxIs4Pg==;EndpointSuffix=core.windows.net"
export AZURE_AUTH=connection_string
```
az storage blob list \
  --account-name $AZURE_STORAGE_ACCOUNT \
  --container-name $AZURE_STORAGE_CONTAINER \
  --prefix loc_vhp/ \
  -o table

---

## GCP Chirp Pipeline Commands (VM)

### Test Run: 100 files (memory + eval fix validation)
```bash
cd ~/projects/amia2025-stt-benchmarking
uv run python scripts/run_pipeline.py \
    --configs configs/runs/test-gcp-chirp3-sample100-vhp-pre2010.yaml \
    --parquet data/raw/loc/veterans_history_project_resources_pre2010.parquet
```

### Production Run: ALL pre-2010 files (~3,800 files, inference only)
```bash
cd ~/projects/amia2025-stt-benchmarking
screen -S gcp-full
uv run python scripts/run_pipeline.py \
    --configs configs/runs/vhp-pre2010-chirp3-full-gpu.yaml
# Ctrl+A, D to detach
# screen -r gcp-full to reattach
```

### Re-evaluate Existing Experiments (Fix WER Scores)

#### Re-eval: 10-file test
```bash
cd ~/projects/amia2025-stt-benchmarking
uv run python scripts/evaluate.py \
    --config configs/runs/test-gcp-chirp3-sample5-vhp-pre2010.yaml \
    --inference_results outputs/test-gcp-chirp3-sample5-vhp-pre2010/inference_results.parquet \
    --parquet data/raw/loc/veterans_history_project_resources_pre2010.parquet
```

#### Re-eval: distil-whisper full pre-2010
```bash
cd ~/projects/amia2025-stt-benchmarking
uv run python scripts/evaluate.py \
    --config configs/runs/vhp-pre2010-distil-large-v3-full-gpu.yaml \
    --inference_results outputs/vhp-pre2010-distil-large-v3-full-gpu/inference_results.parquet \
    --parquet data/raw/loc/veterans_history_project_resources_pre2010.parquet
```

#### Template: Re-eval any experiment
```bash
cd ~/projects/amia2025-stt-benchmarking
uv run python scripts/evaluate.py \
    --config configs/runs/EXPERIMENT_NAME.yaml \
    --inference_results outputs/EXPERIMENT_NAME/inference_results.parquet \
    --parquet data/raw/loc/PARQUET_FILE.parquet
```

### Monitor Running Experiments

#### Monitor GCP Operations (Real-Time Logging)
```bash
# The pipeline now logs each file as it's processed:
# - Timestamps for each operation
# - GCS URIs and operation names
# - Real-time progress (X ✓ completed | Y ✗ failed | Z remaining)
# - Direct link to GCP Console for monitoring

# Example output:
# [1/98] File: AFC1991001_00001_ms01 (ID: 42)
#     GCS URI: gs://memoirji-amia-2025-temp/AFC1991001_00001_ms01.wav
#     Operation: projects/123/locations/us/operations/456789
#     Started waiting at: 14:23:45
#     ✓ COMPLETED in 342.5s (5.7 min)
#     Status: 45 ✓ completed | 2 ✗ failed | 51 remaining
```

#### Monitor Memory Usage
```bash
# Watch memory in real-time
watch -n 2 'free -h && echo "---" && ps aux | grep python | grep -v grep | head -5'

# Or use htop
htop
```

#### Monitor GCP Console
```bash
# Link is printed in logs, format:
# https://console.cloud.google.com/speech/locations/us?project=memoirji-amia-2025
# (replace 'us' with 'us-central1' for Chirp 2)
```

### Check Experiment Status
```bash
# List all output directories
ls -lh outputs/

# Check specific experiment results
ls -lh outputs/test-gcp-chirp3-sample100-vhp-pre2010/

# Quick WER check from CSV
head -20 outputs/test-gcp-chirp3-sample100-vhp-pre2010/evaluation_results.csv
```

### Download Results to Local (from local machine)
```bash
# Download evaluation results
scp amia-gpu:~/projects/amia2025-stt-benchmarking/outputs/EXPERIMENT_NAME/evaluation_results.parquet ./

# Download full output directory
rsync -avz --progress amia-gpu:~/projects/amia2025-stt-benchmarking/outputs/EXPERIMENT_NAME/ ./outputs/EXPERIMENT_NAME/
```


!! important when setting up new "pod" (vm): 
- on runpod, when specify public key, take the one from local - theres a key already (~/.ssh/id_ed25519), then they will use that later when we try to auth
- mind the uv sync since some packages are not included due to machine diff- torch, nemo, 
- need to install `screen`, htop