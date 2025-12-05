# AMIA 2025 STT Benchmarking

Benchmarking commercial and open-source speech-to-text models on bandwidth-limited archival audio from the Library of Congress Veterans History Project (pre-2010 oral histories with analog-era equipment limitations).

**AMIA 2025 Presentation:** "Transcribing a Broken Record: Benchmarking STT for Archival Oral Histories"
- Session 12205 | Dec 5, 2024, 3:45-4:15pm | Baltimore Marriott Waterfront
- Focus: Practical infrastructure challenges for archivists building transcription pipelines

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/amia2025-stt-benchmarking.git
cd amia2025-stt-benchmarking

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Optional: Install dependency groups

```bash
# For development (testing, linting)
uv sync --group dev

# For training (GPU acceleration)
uv sync --group train

# For data processing
uv sync --group data
```

### Running Jupyter notebooks

```bash
uv run --with jupyter jupyter lab
```

---

## Project Structure

```
amia2025-stt-benchmarking/
├── configs/           # YAML config files for experiments
│   └── runs/          # Experiment-specific configurations
│
├── data/              # Audio data and parquet metadata (not tracked in Git)
│   ├── raw/           # Original audio files
│   └── processed/     # Preprocessed audio segments
│
├── docs/              # Reference documentation
│
├── learnings/         # Research notes and analysis findings
│
├── models/            # Fine-tuned or downloaded model checkpoints
│   └── faster-whisper/  # CTranslate2 format models
│
├── notebooks/         # Jupyter notebooks for exploration
│   ├── 1_get_resources.ipynb      # Data collection
│   ├── 2_filter_records.ipynb     # Dataset filtering
│   ├── 3_add_audio_features.ipynb # Audio analysis
│   ├── 4_create_splits.ipynb      # Train/val/test splits
│   ├── 5_aggregate_results.ipynb  # Results aggregation
│   └── tutorial_*.ipynb           # Tutorial notebooks
│
├── outputs/           # Inference results, metrics, transcripts
│
├── scripts/           # Python scripts for pipeline execution
│   ├── data/          # Data loading and preprocessing
│   ├── eval/          # Evaluation and WER calculation
│   └── run_*.py       # Main pipeline scripts
│
├── tests/             # Pytest unit tests
│
└── wandb/             # Weights & Biases experiment logs
```

---

## Models Evaluated

### Commercial APIs
- **Google Chirp 2 & 3**: Google Cloud Speech-to-Text
- **AWS Transcribe**: Amazon's STT service

### Open-Source Models
- **Whisper Large-v3**: OpenAI's multilingual STT model (1.55B parameters)
- **Whisper Base**: Smaller variant for baseline comparison (74M parameters)
- **Parakeet-TDT 1.1B**: NVIDIA NeMo model optimized for conversational audio

---

## Running Experiments

### Inference Only

```bash
uv run python scripts/run_inference.py --config configs/runs/your-experiment.yaml
```

### Full Pipeline (Inference + Evaluation)

```bash
uv run python scripts/run_pipeline.py \
    --configs configs/runs/your-experiment.yaml \
    --parquet data/veterans_history_project_resources.parquet
```

### Standalone Evaluation

```bash
uv run python scripts/eval/evaluate.py \
    --config configs/runs/your-experiment.yaml \
    --inference_results outputs/your-experiment/inference_results.parquet \
    --parquet data/veterans_history_project_resources.parquet
```

---

## Key Findings

### Audio Characteristics

The VHP pre-2010 collection exhibits **bandwidth-limited but clean** audio:
- **SNR: 33 dB** - Excellent signal-to-noise ratio (clean recording)
- **Spectral Roll-off: 562 Hz** - Severe bandwidth limitation (analog equipment)
- **ZCR: 0.014** - Low zero-crossing rate (not noisy)

This is a distinct challenge from noisy audio: the recordings are clean within a limited frequency range due to analog-era equipment, not corrupted by noise.

### WER Metrics

Word Error Rate (WER) is calculated as:
```
WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words
```

See [learnings/wer-normalization-guide.md](learnings/wer-normalization-guide.md) for detailed explanation.

---

## Hardware Requirements

| Task | Minimum GPU | VRAM Required |
|------|-------------|---------------|
| Whisper Base inference | CPU or any GPU | 2-4 GB |
| Whisper Large-v3 inference | T4 or better | 10-12 GB |
| Parakeet-TDT 1.1B inference | A6000 recommended | ~7 GB (but needs headroom) |
| Whisper Large-v3 fine-tuning | A6000 or better | 40+ GB |

---

## Contributing

Contributions welcome! Areas of interest:
- Additional STT model benchmarks
- Audio preprocessing techniques for bandwidth-limited audio
- Fine-tuning experiments on archival datasets

---

## Acknowledgments

- **Library of Congress Veterans History Project** for providing the oral history collection
- **Greg Palumbo** (Library of Congress) for VHP data access and guidance
- AMIA 2025 for the presentation opportunity

---

## License

This project is for research and educational purposes. The VHP audio data is subject to Library of Congress terms of use.
