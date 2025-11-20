# amia2025-stt-benchmarking

## Introduction
Benchmarking commercial and open-source speech-to-text models on degraded archival audio from the Library of Congress Veterans History Project (pre-2010 oral histories with analog-era degradation).

**AMIA 2025 Presentation:** "Transcribing a Broken Record: Benchmarking STT for Archival Oral Histories"
- Session 12205 | Dec 5, 2024, 3:45-4:15pm | Baltimore Marriott Waterfront
- Focus: Practical infrastructure challenges for archivists building transcription pipelines

---

## Quick Status

### ✅ Completed
- 500-sample benchmarks: Google Chirp 2/3, AWS Transcribe (full-duration audio)
- Production-grade pipeline: Rate limiting, concurrent job management, memory optimization
- Tutorial notebooks: Whisper, Wav2Vec2, Canary-Qwen (1-sample quickstart)

### ⏳ In Progress
- Open-source baselines: Whisper, Wav2Vec2 (pending)
- Canary-Qwen: **BLOCKED** by Azure VM RAM constraints (needs 110GB, have 28GB)

### 📋 Key Documents
- [ABSTRACTS.md](ABSTRACTS.md) - AMIA abstract, goals, timeline, lessons learned
- [CURRENT_STATUS.md](CURRENT_STATUS.md) - Detailed project status
- [docs/](docs/) - Reference guides (CONFIG_GUIDE, QUICK_START, etc.)
- [communications/](communications/) - AMIA emails, speaker bio, UMSI invitations

---

## 📁 Project Structure

data/           # Raw + processed audio data (not tracked in Git)

models/         # Fine-tuned or checkpointed models (e.g., Whisper FT)

scripts/        # Scripts for running inference, training, evaluation

results/        # Transcripts, metrics, WER breakdowns

configs/        # YAML or JSON config files for model and data

tests/          # Pytest-based unit tests for reproducibility

logs/           # System + training logs (auto-generated)

notebooks/      # Jupyter playgrounds (for exploration only)

docs/           # MkDocs or other docs

## Current VMs (Azure)
| Alias | Azure Name | Purpose | Specs | State |
|--------|-------------|----------|--------|--------|
| `vm-amia-preproc` | `vm-amia-t4-cpu` | CPU preprocessing | D4s_v5 | Deallocated |
| `vm-amia-gpu` | `vm-amia-t4` | GPU inference/fine-tuning | NC4as_T4_v3 | On demand |

## Connect to VMs via Remote-SSH
VS Code: *Remote-SSH → Connect to Host → amia-gpu*; open
```
/home/arthur/projects/amia2025-stt-benchmarking
```
Terminal:
```
ssh arthur@172.191.111.144
```
GPU VM: 
```
ssh amia-gpu
```
CPU VM ("backup" VM): 
```
ssh amia-preproc
```
(*ssh alias in `~/.ssh/config`)
<!-- - Streamlit/Gradio: open http://localhost:8501 (tunneled via `LocalForward 8501 127.0.0.1:8501`) -->

## Azure CLI - VM
check current VM status:
`az vm list -d -o table`
deallocate VM when done:
`az vm deallocate -g rg-amia-ml -n vm-amia-t4`
list all vms: 
`az vm list-sizes --location eastus -o table`
confirm rg status:
`az group show -n rg-amia-ml -o table`

## Azure CLI - Storage Access
On VM:

`az login --identity --allow-no-subscriptions`
`export STG=stgamiadata26828`
`az storage blob list --account-name "$STG" --container-name audio-raw --auth-mode login -o table`

Grant VM access from Local:

1.Vars
```
SUB=$(az account show --query id -o tsv)
```
```
SCOPE="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$STG"
```
```
PRINCIPAL_ID=$(az vm show -g "$RG" -n "vm-amia-t4" --query "identity.principalId" -o tsv)
```
2.Allow the VM’s identity to read/write blobs
```az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" \
  --scope "$SCOPE"
```
3.(optional) If you also want the identity to “see” the subscription in az account show:
```
az role assignment create \
  --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Reader" \
  --scope "/subscriptions/$SUB"
```

## Points to note:
- “RBAC = identity-based, needs a role.
Connection string = key-based, works anywhere.”
- CUDA (Compute Unified Device Architecture) is NVIDIA’s programming interface that lets your Python libraries (like PyTorch or Whisper) talk directly to the GPU.


## Environment setup with `uv`:
To activate virtual environment in terminal:
`source .venv/bin/activate`
Make sure notebook can be run:
`uv add --dev ipykernel`

Testing PyTorch + MPS + torchaudio:
1.
```
uv pip install "torchvision==0.24.*"
```
2. 
```
uv run python - <<'PY'
import torch, torchaudio
print("torch:", torch.__version__, "| mps:", torch.backends.mps.is_available())
waveform = torch.randn(1, 16000)  # 1s fake audio
resamp = torchaudio.transforms.Resample(16000, 8000)
print("resampled shape:", resamp(waveform).shape)
PY
```

Run jupyter lab:
`uv run --with jupyter jupyter lab`

Download model
`uv run hf download facebook/wav2vec2-base-960h --local-dir ./models/local/facebook--wav2vec2-base-960h`

---

## Pipeline Execution

### Important: GCP Chirp Memory Optimization

**Issue:** When running GCP Chirp on large datasets (100+ files), the VM may crash silently during preprocessing due to out-of-memory (OOM) errors.

**Solution:** Use the `max_concurrent_preprocessing` parameter to limit how many files are preprocessed simultaneously, regardless of batch size.

**Recommended settings for T4 VM (16 GB RAM):**
```yaml
model:
  gcp:
    max_concurrent_preprocessing: 10  # Only preprocess 10 files at a time
    upload_workers: 30                # Upload can be higher (network-bound)
    transcribe_workers: 20
    batch_size: 50                    # Process 50 files before checkpoint
```

See [docs/GCP_MEMORY_OPTIMIZATION.md](docs/GCP_MEMORY_OPTIMIZATION.md) for detailed explanation and troubleshooting.

### Important: Azure Blob Index Mapping

**Context:** Audio files are stored in Azure Blob Storage with paths like `loc_vhp/{index}/video.mp4`, where the index comes from the **full** `veterans_history_project_resources.parquet` file.

**Issue:** When you filter or slice the parquet (e.g., pre-2010, post-2010), the row indices change, breaking the blob path lookups.

**Solution:** The pipeline uses the `azure_blob_index` column to map filtered parquet rows back to their original blob paths. This column is:
- **Automatically added** when you use [notebooks/vhp_data_slicing.ipynb](notebooks/vhp_data_slicing.ipynb) to create filtered datasets
- **Handled automatically** by [scripts/data_loader.py](scripts/data_loader.py) with this priority:
  1. `azure_blob_index` (for filtered datasets like pre-2010, post-2010)
  2. `original_parquet_index` (for sampled datasets)
  3. `idx` (for full parquet without filtering/sampling)

**When creating new filtered parquet files**, always preserve the original index as `azure_blob_index`:
```python
# After filtering
df_filtered['azure_blob_index'] = df_filtered.index
df_filtered.to_parquet('path/to/output.parquet', index=False)
```

### Full Pipeline (Inference + Evaluation)

Run both inference and evaluation in one command:

```bash
uv run python scripts/run_pipeline.py \
    --configs configs/runs/your-experiment.yaml \
    --parquet data/veterans_history_project_resources.parquet
```

The pipeline will:
1. Run inference (generate transcripts)
2. Automatically evaluate against ground truth
3. Save results to `outputs/your-experiment/`

### Standalone Evaluation

If you already have inference results and want to re-evaluate with different normalization:

```bash
uv run python scripts/evaluate.py \
    --config configs/runs/your-experiment.yaml \
    --inference_results outputs/your-experiment/inference_results.parquet \
    --parquet data/veterans_history_project_resources.parquet
```

**Outputs:**
- `evaluation_results.parquet` - Per-file WER metrics
- `evaluation_results.csv` - Same as above, CSV format for inspection
- Console output showing mean/median WER and error breakdown

### Text Normalization Options

Normalization is controlled via the config YAML file under the `evaluation` section.

**Option 1: Default (jiwer with contraction expansion)** - Recommended
```yaml
# No evaluation section needed - this is the default
# OR explicitly:
evaluation:
  use_whisper_normalizer: false
```

Applies:
- Contraction expansion (we're → we are, can't → can not)
- Lowercase conversion
- Punctuation removal
- Whitespace normalization

**Option 2: Whisper normalizer** - For comprehensive normalization
```yaml
evaluation:
  use_whisper_normalizer: true
```

Additionally handles:
- Number normalization (10 → ten, 1st → first)
- Date/currency formatting
- Abbreviation expansion (Dr. → doctor)

**Note:** Requires `openai-whisper` package. Install with:
```bash
uv pip install openai-whisper
```

### Understanding WER Metrics

**WER Formula:**
```
WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words
```

- **Substitutions (S)**: Wrong word predicted
- **Deletions (D)**: Reference word missing from hypothesis
- **Insertions (I)**: Extra word added by model

**Example:**
```
Reference:  "I went to the store"
Hypothesis: "I go to store"

S=1 (went→go), D=1 (the missing), I=0
WER = (1 + 1 + 0) / 5 = 0.4 (40%)
```

**Learning Resources:**
- [WER Normalization Guide](learnings/wer-normalization-guide.md) - Comprehensive guide to WER calculation and normalization
- [Evaluation Learning Notebook](notebooks/evals_learn.ipynb) - Step-by-step WER calculation examples