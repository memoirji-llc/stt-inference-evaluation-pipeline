# amia2025-stt-benchmarking

## introduction
Benchmarking open-source speech-to-text models (Whisper, Wav2Vec2, MMS) on historical audio such as radio shows, cassette digitizations, and oral history recordings.

This project supports a paper accepted to AMIA 2025 and aims to create a reproducible, professional-grade benchmark pipeline for degraded analog-era audio.

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