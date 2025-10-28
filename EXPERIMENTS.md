# Experiment Configurations

This document lists all available experiment configurations and how to run them.

## Available Configurations

### Whisper Base Model (3 CPU + 9 GPU configs)

**CPU (Local Mac):**
- `vhp-whisper-sample10-300s-cpu.yaml` - 10 files, 5min
- `vhp-whisper-sample10-600s-cpu.yaml` - 10 files, 10min
- `vhp-whisper-sample10-full-cpu.yaml` - 10 files, full audio

**GPU (Azure VM):**
- `vhp-whisper-sample10-300s-gpu.yaml` - 10 files, 5min
- `vhp-whisper-sample10-600s-gpu.yaml` - 10 files, 10min
- `vhp-whisper-sample10-full-gpu.yaml` - 10 files, full audio
- `vhp-whisper-sample50-300s-gpu.yaml` - 50 files, 5min
- `vhp-whisper-sample50-600s-gpu.yaml` - 50 files, 10min
- `vhp-whisper-sample50-full-gpu.yaml` - 50 files, full audio
- `vhp-whisper-sample100-300s-gpu.yaml` - 100 files, 5min
- `vhp-whisper-sample100-600s-gpu.yaml` - 100 files, 10min
- `vhp-whisper-sample100-full-gpu.yaml` - 100 files, full audio

### Whisper Large-v3 Model (6 GPU configs)

**GPU (Azure VM) - batch_size: 8 for larger model:**
- `vhp-large-v3-sample10-300s-gpu.yaml` - 10 files, 5min
- `vhp-large-v3-sample50-600s-gpu.yaml` - 50 files, 10min
- `vhp-large-v3-sample50-full-gpu.yaml` - 50 files, full audio
- `vhp-large-v3-sample100-300s-gpu.yaml` - 100 files, 5min
- `vhp-large-v3-sample100-600s-gpu.yaml` - 100 files, 10min
- `vhp-large-v3-sample100-full-gpu.yaml` - 100 files, full audio ⭐ MOST COMPREHENSIVE

### Distil-Whisper Large-v3 Model (6 GPU configs)

**GPU (Azure VM) - batch_size: 12 for distilled model (faster):**
- `vhp-distil-large-v3-sample10-300s-gpu.yaml` - 10 files, 5min
- `vhp-distil-large-v3-sample50-600s-gpu.yaml` - 50 files, 10min
- `vhp-distil-large-v3-sample50-full-gpu.yaml` - 50 files, full audio
- `vhp-distil-large-v3-sample100-300s-gpu.yaml` - 100 files, 5min
- `vhp-distil-large-v3-sample100-600s-gpu.yaml` - 100 files, 10min
- `vhp-distil-large-v3-sample100-full-gpu.yaml` - 100 files, full audio

---

## How to Run Experiments

### Single Experiment

**Local (CPU):**
```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-whisper-sample10-300s-cpu.yaml \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

**VM (GPU):**
```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-large-v3-sample100-full-gpu.yaml \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

### Multiple Experiments (Sequential)

Run multiple configs in one command - they will execute sequentially:

```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-whisper-sample10-300s-gpu.yaml \
            configs/runs/vhp-large-v3-sample10-300s-gpu.yaml \
            configs/runs/vhp-distil-large-v3-sample10-300s-gpu.yaml \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

### Using Model Override (Testing Other Models)

To quickly test a model without creating a YAML:

```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-whisper-sample10-300s-gpu.yaml \
  --model-dir "./models/faster-whisper/models--Systran--faster-whisper-medium" \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

This will:
- Use all settings from the base config (sample size, duration, batch_size, etc.)
- Override only the model directory
- Update the experiment_id automatically (adds `-override-whisper-medium`)
- Track everything in wandb

**Available models on VM:**
- `models--Systran--faster-whisper-base`
- `models--Systran--faster-whisper-base.en`
- `models--Systran--faster-whisper-small`
- `models--Systran--faster-whisper-small.en`
- `models--Systran--faster-whisper-medium`
- `models--Systran--faster-whisper-medium.en`
- `models--Systran--faster-whisper-large-v1`
- `models--Systran--faster-whisper-large-v2`
- `models--Systran--faster-whisper-large-v3`
- `models--Systran--faster-distil-whisper-small.en`
- `models--Systran--faster-distil-whisper-medium.en`
- `models--Systran--faster-distil-whisper-large-v2`
- `models--Systran--faster-distil-whisper-large-v3`

---

## Recommended Experiment Workflow

### Phase 1: Quick Model Comparison (10 files, 5min)
Compare different models on a small sample:
```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-whisper-sample10-300s-gpu.yaml \
            configs/runs/vhp-large-v3-sample10-300s-gpu.yaml \
            configs/runs/vhp-distil-large-v3-sample10-300s-gpu.yaml \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

### Phase 2: Medium-Scale Validation (50 files, 10min)
Validate top performers on larger sample:
```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-large-v3-sample50-600s-gpu.yaml \
            configs/runs/vhp-distil-large-v3-sample50-600s-gpu.yaml \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

### Phase 3: Full Benchmark (100 files, full audio)
Final comprehensive benchmark:
```bash
uv run python scripts/run_pipeline.py \
  --configs configs/runs/vhp-large-v3-sample100-full-gpu.yaml \
            configs/runs/vhp-distil-large-v3-sample100-full-gpu.yaml \
  --parquet data/raw/loc/veterans_history_project_resources.parquet
```

---

## Tracking Results

All experiments are automatically logged to wandb:
- **Project:** `amia-stt`
- **Groups:** `experiments-cpu`, `experiments-gpu`, `experiments-gpu-large-v3`, `experiments-gpu-distil-large-v3`
- **Tags:** Model name, sample size, duration, device type

Filter experiments in wandb by:
- `sample10`, `sample50`, `sample100`
- `duration-300s`, `duration-600s`, `full-duration`
- `whisper-base`, `whisper-large-v3`, `distil-whisper-large-v3`
- `cpu`, `gpu`, `t4`

---

## Estimating Runtime

Approximate runtimes on Azure VM with T4 GPU:

| Model | Sample Size | Duration | Estimated Time |
|-------|-------------|----------|----------------|
| base | 10 | 300s | 5-10 min |
| base | 50 | 600s | 30-45 min |
| base | 100 | full | 2-4 hours |
| large-v3 | 10 | 300s | 10-15 min |
| large-v3 | 50 | 600s | 1-2 hours |
| large-v3 | 100 | full | 4-8 hours |
| distil-large-v3 | 10 | 300s | 7-12 min |
| distil-large-v3 | 50 | 600s | 45-90 min |
| distil-large-v3 | 100 | full | 3-6 hours |

*Actual times vary based on audio file lengths and quality.*
