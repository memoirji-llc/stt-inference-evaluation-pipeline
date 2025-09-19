# amia2025-stt-benchmarking

# amia2025-stt-benchmarking

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