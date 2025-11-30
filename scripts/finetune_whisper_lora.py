#!/usr/bin/env python
"""
Fine-tuning Whisper Large-v3 with LoRA/PEFT

Target Hardware: RunPod A6000 (48GB VRAM)

This script fine-tunes Whisper large-v3 on VHP oral history audio using LoRA (Low-Rank Adaptation).
Converted from notebooks/finetune_whisper_lora.ipynb for running in screen sessions.

USAGE:
    # On RunPod/GPU instance, navigate to project directory
    cd /workspace/amia2025-stt-benchmarking

    # Run in screen session (recommended for long-running jobs)
    screen -S whisper-ft
    python scripts/finetune_whisper_lora.py
    # Ctrl+A, D to detach; screen -r whisper-ft to reattach

    # Or run directly
    python scripts/finetune_whisper_lora.py

REQUIREMENTS:
    - Edit CONFIG section (line ~103) to set train/val parquet paths and hyperparameters
    - Ensure credentials/creds.env exists with Azure blob storage credentials
    - Install dependencies: peft, accelerate, torch (see cell at line ~42)

OUTPUT:
    Training artifacts saved to CONFIG["output_dir"]:
    - checkpoint-*/: Training checkpoints
    - lora-weights/: Final LoRA adapters (~60MB)
    - merged-model/: Full model with LoRA merged (HF format, ~3GB)
    - runs/: Tensorboard logs

NEXT STEPS:
    After training, convert to CTranslate2 format and run production pipeline.
    See "Model Outputs & Next Steps" section at end of script.
"""

# %% [markdown]
# # Fine-tuning Whisper Large-v3 with LoRA/PEFT
#
# **Target Hardware**: RunPod A6000 (48GB VRAM)
#
# This script fine-tunes Whisper large-v3 on VHP oral history audio using LoRA (Low-Rank Adaptation).
# 
# ## Key Configuration
# - **Model**: openai/whisper-large-v3 (via HuggingFace transformers)
# - **Method**: LoRA with r=32, alpha=64
# - **Learning Rate**: 1e-5 (CRITICAL: not 1e-3)
# - **Precision**: fp16 (NOT int8 for V3)
# - **Batch Size**: 4 with gradient accumulation 4 (effective: 16)
# 
# ## Data Requirements
# - Parquet files: `veterans_history_project_resources_pre2010_train.parquet` and `_val.parquet`
# - Azure blob storage connection for audio files
# 
# ### How Many Training Samples Do You Need?
# 
# **Audio constraint**: ≤30 seconds per sample (Whisper's max context). VHP interviews are 30-60+ minutes, so we use **NeMo Forced Aligner** to create shorter segments.
# 
# **Projected sample count from NFA segmentation** (500-file run):
# - Success rate: ~15% (files >30min skipped for CUDA OOM prevention)
# - Segments per file: 30-40
# - **Total: 2,250-3,000 segments** (sufficient for LoRA)
# 
# **Is this enough?** Yes. LoRA trains only ~1% of parameters for domain adaptation. Industry benchmarks show 1,000-5,000 samples work well for acoustic adaptation without retraining language understanding.
# 
# **References:**
# - [NeMo Forced Aligner](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speech_data_processor.html)
# - [Whisper LoRA Fine-tuning (HuggingFace)](https://huggingface.co/blog/fine-tune-whisper)
# 
# See [learnings/whisper-lora-finetuning.md](../learnings/whisper-lora-finetuning.md) for gotchas.

# %% [markdown]
# ## 1. Setup Dependencies
# 
# Fine-tuning requires additional packages not in the base project. Add them via uv:

# %%
# Add fine-tuning dependencies to pyproject.toml (run once)
# uv add peft accelerate
#
# For A6000 with CUDA 11.8:
# uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# %%
import sys
import os
from pathlib import Path

# Get script location for resolving paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Load Azure credentials from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=_PROJECT_ROOT / 'credentials/creds.env')

import torch
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset, Audio

# Use script location defined at top of file
SCRIPT_DIR = _SCRIPT_DIR
PROJECT_ROOT = _PROJECT_ROOT

# Add project root to path for package imports
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules using package structure (no name conflicts)
from scripts.data import data_loader
from scripts.cloud import azure_utils
from scripts.eval.evaluate import clean_raw_transcript_str

# Import HuggingFace evaluate (no conflict with scripts.eval.evaluate)
import evaluate as hf_evaluate

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Configuration

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data paths - VHP parquet files (use PROJECT_ROOT for absolute paths)
    "train_parquet": str(PROJECT_ROOT / "data/raw/loc/veterans_history_project_resources_pre2010_train_nfa_segmented.parquet"),
    "val_parquet": str(PROJECT_ROOT / "data/raw/loc/veterans_history_project_resources_pre2010_val_nfa_segmented.parquet"),

    # Azure blob settings (same as inference configs)
    "blob_prefix": "loc_vhp",

    # Sampling (set to None to use all data, or small number for testing)
    "train_sample_size": 2400,  # 80/20 split with val
    "val_sample_size": 600,  # 80/20 split with train
    "random_seed": 42,

    # Output directory - follows convention: {dataset}-{model}-{task}-{infra}
    "output_dir": str(PROJECT_ROOT / "outputs/vhp-pre2010-whisper-large-v3-lora-ft-a6000"),
    
    # Model - using HuggingFace transformers (not faster-whisper, which is inference-only)
    # Note: For inference we use faster-whisper, but for fine-tuning we need the original HF model
    "model_name": "openai/whisper-large-v3",
    
    # LoRA configuration
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    
    # Training hyperparameters
    "learning_rate": 1e-5,           # CRITICAL: Use 1e-5 for V3 (not 1e-3)
    "batch_size": 4,                  # Per device
    "gradient_accumulation": 4,       # Effective batch = 16
    "warmup_steps": 500,
    "max_steps": 5000,               # Adjust based on data size
    "eval_steps": 500,
    "save_steps": 500,
    
    # Precision
    "fp16": True,                    # Use fp16 for V3
    "bf16": False,

    # Token limit - Whisper's decoder has a hard limit of 448 tokens
    # Samples with transcripts exceeding this will be skipped during preprocessing
    "max_label_length": 448,
}

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"Output directory: {CONFIG['output_dir']}")

# %% [markdown]
# ## 3. Load Data
# 
# Using existing `data_loader.py` and `azure_utils.py` from scripts/.
# 
# Ground truth is extracted from `fulltext_file_str` column using `clean_raw_transcript_str()` from evaluate.py (see [notebooks/evals_learn.ipynb](./evals_learn.ipynb) for details on how this works).

# %%
def load_finetune_dataset(parquet_path: str, sample_size: int = None, random_seed: int = 42):
    """
    Load dataset for fine-tuning.
    
    Handles both:
    - Segmented parquets (with segmented_audio_url column) - loads directly
    - Original parquets (without segmented_audio_url) - uses data_loader filtering
    """
    # Load parquet to check if it's segmented
    df = pd.read_parquet(parquet_path)
    
    is_segmented = 'segmented_audio_url' in df.columns
    
    if is_segmented:
        # For segmented parquets, don't use data_loader filtering
        # (it filters by video_url/audio_url which segmented parquets don't have)
        print(f"Detected segmented parquet with {len(df)} segments")
        
        # Filter to rows with valid segmented_audio_url and transcript
        df = df[df['segmented_audio_url'].notna() & (df['segmented_audio_url'] != '')]
        print(f"Filtered to {len(df)} segments with segmented_audio_url")
        
        df = df[df['segmented_audio_transcript'].notna() & (df['segmented_audio_transcript'] != '')]
        print(f"Filtered to {len(df)} segments with segmented_audio_transcript")
        
        # Apply sampling if requested
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_seed)
            print(f"Sampled {sample_size} segments")
        else:
            print(f"Using all {len(df)} segments (no sampling)")
    else:
        # For original parquets, use existing data_loader
        df = data_loader.load_vhp_dataset(
            parquet_path=parquet_path,
            sample_size=sample_size,
            filter_has_transcript=True,
            filter_has_media=True
        )
    
    print(f"Loaded {len(df)} samples")
    return df

# Load train and validation sets
print("Loading training data...")
df_train = load_finetune_dataset(
    CONFIG["train_parquet"], 
    sample_size=CONFIG["train_sample_size"],
    random_seed=CONFIG["random_seed"]
)

print("\nLoading validation data...")
df_val = load_finetune_dataset(
    CONFIG["val_parquet"],
    sample_size=CONFIG["val_sample_size"],
    random_seed=CONFIG["random_seed"]
)

print(f"\nTrain: {len(df_train)} samples")
print(f"Val: {len(df_val)} samples")

# %%
import gc
from datasets import concatenate_datasets

def process_single_audio(row, idx, blob_prefix, is_segmented, max_duration_sec=30):
    """
    Process a single audio sample. Returns dict with audio array and transcript, or None if failed.
    """
    from tempfile import NamedTemporaryFile
    from pydub import AudioSegment
    import librosa

    # Get transcript
    if is_segmented:
        transcript = row.get('segmented_audio_transcript', '')
    else:
        raw_transcript = row.get('fulltext_file_str', '')
        transcript = clean_raw_transcript_str(raw_transcript)

    if not transcript.strip():
        return None

    # Get blob path
    if is_segmented:
        blob_path = row.get('segmented_audio_url', '')
        if not blob_path:
            return None
        blob_path_candidates = [blob_path]
    else:
        blob_path_candidates = data_loader.get_blob_path_for_row(row, idx, blob_prefix)
        if not blob_path_candidates:
            return None

    # Download and process audio
    for blob_path in blob_path_candidates:
        try:
            if not azure_utils.blob_exists(blob_path):
                continue

            audio_bytes = azure_utils.download_blob_to_memory(blob_path)

            # Convert to WAV 16kHz mono
            with NamedTemporaryFile(suffix=Path(blob_path).suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            audio_seg = AudioSegment.from_file(tmp_path)
            audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)

            wav_path = tmp_path.replace(Path(blob_path).suffix, '.wav')
            audio_seg.export(wav_path, format='wav')

            # Load as numpy array
            audio_data, sr = librosa.load(wav_path, sr=16000)

            # Cleanup temp files immediately
            os.unlink(tmp_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)

            # Free audio_seg memory
            del audio_seg, audio_bytes

            # For non-segmented data, skip if too long
            if not is_segmented:
                if len(audio_data) / 16000 > max_duration_sec:
                    del audio_data
                    return None

            return {
                "audio": {"array": audio_data, "sampling_rate": 16000},
                "sentence": transcript
            }
        except Exception as e:
            print(f"  ERROR processing {blob_path}: {e}")
            continue

    return None


def prepare_hf_dataset_batched(df: pd.DataFrame, blob_prefix: str, save_path: str, batch_size: int = 100):
    """
    Memory-efficient dataset preparation: process in batches and save to disk.

    Args:
        df: DataFrame with VHP data
        blob_prefix: Azure blob prefix
        save_path: Path to save the processed dataset
        batch_size: Number of samples to process before saving to disk
    """
    import shutil

    is_segmented = 'segmented_audio_url' in df.columns
    print(f"[MEMORY-EFFICIENT] Processing {len(df)} samples in batches of {batch_size}")
    print(f"[MEMORY-EFFICIENT] is_segmented={is_segmented}")
    print(f"[MEMORY-EFFICIENT] Will save to: {save_path}")

    # Clean up any existing partial dataset
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    total_processed = 0
    total_skipped = 0
    batch_datasets = []
    current_batch = []

    for i, (idx, row) in enumerate(df.iterrows()):
        result = process_single_audio(row, idx, blob_prefix, is_segmented)

        if result is not None:
            current_batch.append(result)
            total_processed += 1
        else:
            total_skipped += 1

        # Progress update every 50 samples
        if (i + 1) % 50 == 0:
            print(f"[MEMORY-EFFICIENT] Progress: {i + 1}/{len(df)} (processed: {total_processed}, skipped: {total_skipped})")

        # Save batch to disk when full
        if len(current_batch) >= batch_size:
            print(f"[MEMORY-EFFICIENT] Saving batch of {len(current_batch)} samples...")
            batch_dataset = Dataset.from_dict({
                "audio": [r["audio"] for r in current_batch],
                "sentence": [r["sentence"] for r in current_batch]
            })
            batch_dataset = batch_dataset.cast_column("audio", Audio(sampling_rate=16000))
            batch_datasets.append(batch_dataset)

            # Free memory
            del current_batch
            gc.collect()
            current_batch = []
            print(f"[MEMORY-EFFICIENT] Batch saved, memory freed")

    # Save remaining samples
    if current_batch:
        print(f"[MEMORY-EFFICIENT] Saving final batch of {len(current_batch)} samples...")
        batch_dataset = Dataset.from_dict({
            "audio": [r["audio"] for r in current_batch],
            "sentence": [r["sentence"] for r in current_batch]
        })
        batch_dataset = batch_dataset.cast_column("audio", Audio(sampling_rate=16000))
        batch_datasets.append(batch_dataset)
        del current_batch
        gc.collect()

    if not batch_datasets:
        raise ValueError("No valid samples found!")

    # Concatenate all batches
    print(f"[MEMORY-EFFICIENT] Concatenating {len(batch_datasets)} batches...")
    final_dataset = concatenate_datasets(batch_datasets)

    # Free batch datasets
    del batch_datasets
    gc.collect()

    # Save to disk
    print(f"[MEMORY-EFFICIENT] Saving final dataset to disk: {save_path}")
    final_dataset.save_to_disk(save_path)

    # Free the in-memory dataset
    del final_dataset
    gc.collect()

    print(f"[MEMORY-EFFICIENT] Complete! Processed: {total_processed}, Skipped: {total_skipped}")

    # Load from disk (memory-mapped)
    from datasets import load_from_disk
    return load_from_disk(save_path)


# %%
# Prepare datasets with memory-efficient batch processing
CACHE_DIR = PROJECT_ROOT / "outputs" / "dataset_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PREPARING TRAINING DATASET (memory-efficient batch processing)")
print("=" * 60)
train_dataset = prepare_hf_dataset_batched(
    df_train,
    CONFIG["blob_prefix"],
    save_path=str(CACHE_DIR / "train_dataset"),
    batch_size=100
)

# Free df_train memory
del df_train
gc.collect()

print("\n" + "=" * 60)
print("PREPARING VALIDATION DATASET")
print("=" * 60)
val_dataset = prepare_hf_dataset_batched(
    df_val,
    CONFIG["blob_prefix"],
    save_path=str(CACHE_DIR / "val_dataset"),
    batch_size=100
)
# Sample a few items from validation set for quick inference test
df_val_sample = df_val.sample(n=min(10, len(df_val)), random_state=CONFIG["random_seed"])
print(f"Validation samples for testing: {len(df_val_sample)}")
# Free df_val memory
del df_val
gc.collect()

print(f"\nFinal dataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val: {len(val_dataset)}")

# %%
# Preview a random sample
import random
random.seed(CONFIG["random_seed"])

sample_idx = random.randint(0, len(train_dataset) - 1)
sample = train_dataset[sample_idx]

print(f"Sample {sample_idx}:")
print(f"  Audio duration: {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.1f}s")
print(f"  Transcript preview: {sample['sentence'][:200]}...")

# %% [markdown]
# ## 4. Initialize Model
# 
# **Note on model choice**: For fine-tuning we use `openai/whisper-large-v3` from HuggingFace transformers. This is different from inference where we use `faster-whisper` (CTranslate2 optimized). The fine-tuned weights can later be converted to faster-whisper format for inference.

# %%
# Load processor
processor = WhisperProcessor.from_pretrained(CONFIG["model_name"])

# Load model in fp16
print(f"Loading model: {CONFIG['model_name']}")
model = WhisperForConditionalGeneration.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,  # IMPORTANT: Don't use 8-bit for V3 (causes hallucinations)
)

# Clear forced decoder IDs (important for fine-tuning)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Check model dtype
# print(f"[DTYPE DEBUG] Model dtype: {model.dtype}")
# print(f"[DTYPE DEBUG] Encoder conv1 weight dtype: {model.model.encoder.conv1.weight.dtype}")
# print(f"[DTYPE DEBUG] Encoder conv1 bias dtype: {model.model.encoder.conv1.bias.dtype if model.model.encoder.conv1.bias is not None else 'None'}")

# %% [markdown]
# ## 5. Apply LoRA
# 
# ### Why LoRA for Whisper Fine-tuning?
# 
# **LoRA (Low-Rank Adaptation)** is chosen over full fine-tuning for several reasons:
# 
# 1. **Memory Efficiency**: Full fine-tuning of Whisper large-v3 (1.5B params) requires 30-40GB VRAM. LoRA reduces this to ~20GB by only training adapter weights.
# 
# 2. **Catastrophic Forgetting Prevention**: LoRA preserves the base model's general ASR capabilities while adapting to domain-specific audio. Full fine-tuning risks losing pre-trained knowledge.
# 
# 3. **Faster Training**: Only ~1% of parameters are trainable (15.7M vs 1.5B), significantly reducing training time.
# 
# 4. **Easy Model Merging**: LoRA weights can be merged with base model for deployment, or kept separate for A/B testing.
# 
# **References:**
# - [HuggingFace PEFT Whisper Training](https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb)
# - [LoRA Paper](https://arxiv.org/abs/2106.09685)

# %%
# Configure LoRA
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    # NOTE: task_type removed - SEQ_2_SEQ_LM expects input_ids but Whisper uses input_features
    # See: https://github.com/huggingface/peft/issues/1988
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.config.use_cache = False  # Disable cache for training

# Print trainable parameters
model.print_trainable_parameters()

# %% [markdown]
# ## 6. Data Preprocessing

# %%
def prepare_dataset(batch, max_label_length=448):
    """
    Preprocess audio and text for training.

    Args:
        batch: Single sample with 'audio' and 'sentence' keys
        max_label_length: Maximum token length for labels (Whisper limit is 448)

    Returns:
        dict with 'input_features' and 'labels', or None if sample exceeds token limit
    """
    audio = batch["audio"]

    # Extract features from audio (returns float32 by default)
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # Convert to float16 if using fp16 training
    if CONFIG["fp16"]:
        input_features = input_features.to(torch.float16)

    # Tokenize transcription
    labels = processor.tokenizer(batch["sentence"]).input_ids

    # Check if labels exceed Whisper's max decoder length (448 tokens)
    if len(labels) > max_label_length:
        return None  # Signal to skip this sample

    # Return ONLY the processed features (no audio, sentence, etc.)
    return {
        "input_features": input_features,
        "labels": labels
    }

print("=" * 60)
print("PREPROCESSING TRAINING DATA (converting audio to mel spectrograms)")
print("=" * 60)
print(f"Dataset size: {len(train_dataset)} samples")
print(f"Max label length: {CONFIG['max_label_length']} tokens")

# Keep reference to raw audio dataset for cleanup
train_dataset_raw = train_dataset

# Process manually to avoid multiprocessing crashes
processed_train = {"input_features": [], "labels": []}
skipped_token_limit = 0

for i in range(len(train_dataset_raw)):
    if (i + 1) % 100 == 0:
        print(f"[PREPROCESS] Training progress: {i + 1}/{len(train_dataset_raw)}")
    sample = train_dataset_raw[i]
    processed = prepare_dataset(sample, CONFIG["max_label_length"])

    if processed is None:
        skipped_token_limit += 1
        print(f"  [SKIP] Sample {i}: transcript exceeds {CONFIG['max_label_length']} tokens")
        continue

    processed_train["input_features"].append(processed["input_features"])
    processed_train["labels"].append(processed["labels"])

# Create dataset from dict (this ensures ONLY these columns exist)
import datasets
train_dataset = datasets.Dataset.from_dict(processed_train)
print(f"\nTraining preprocessing complete:")
print(f"  Valid samples: {len(train_dataset)}")
print(f"  Skipped (token limit): {skipped_token_limit}")

# Free memory: delete raw audio dataset and processed dict
del train_dataset_raw, processed_train
gc.collect()
print("[MEMORY] Freed raw training audio data")

print("\n" + "=" * 60)
print("PREPROCESSING VALIDATION DATA")
print("=" * 60)

# Keep reference for cleanup
val_dataset_raw = val_dataset

processed_val = {"input_features": [], "labels": []}
skipped_val_token_limit = 0

for i in range(len(val_dataset_raw)):
    if (i + 1) % 50 == 0:
        print(f"[PREPROCESS] Validation progress: {i + 1}/{len(val_dataset_raw)}")
    sample = val_dataset_raw[i]
    processed = prepare_dataset(sample, CONFIG["max_label_length"])

    if processed is None:
        skipped_val_token_limit += 1
        print(f"  [SKIP] Sample {i}: transcript exceeds {CONFIG['max_label_length']} tokens")
        continue

    processed_val["input_features"].append(processed["input_features"])
    processed_val["labels"].append(processed["labels"])

val_dataset = datasets.Dataset.from_dict(processed_val)
print(f"\nValidation preprocessing complete:")
print(f"  Valid samples: {len(val_dataset)}")
print(f"  Skipped (token limit): {skipped_val_token_limit}")

# Free memory
del val_dataset_raw, processed_val
gc.collect()
print("[MEMORY] Freed raw validation audio data")

print(f"\n{'='*60}")
print(f"PREPROCESSING SUMMARY")
print(f"{'='*60}")
print(f"Train: {len(train_dataset)} samples (skipped {skipped_token_limit} for token limit)")
print(f"Val: {len(val_dataset)} samples (skipped {skipped_val_token_limit} for token limit)")

# %%
# DEBUG: Check dataset columns after preprocessing
print("Train dataset columns:", train_dataset.column_names)
print("Val dataset columns:", val_dataset.column_names)
print("\nTrain dataset features:")
print(train_dataset.features)
print("\nSample from train_dataset[0]:")
print(train_dataset[0].keys())

# %%
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper seq2seq training."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # Extract input_features (mel spectrograms) for padding
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # print(f"[DTYPE DEBUG] Data collator - after pad: {batch['input_features'].dtype}")

        # CRITICAL: Convert to float16 to match model precision
        if batch["input_features"].dtype != torch.float16:
            # print(f"[DTYPE DEBUG] Data collator - converting {batch['input_features'].dtype} -> float16")
            batch["input_features"] = batch["input_features"].to(torch.float16)
        # else:
        #     print(f"[DTYPE DEBUG] Data collator - already float16, no conversion needed")

        # Extract labels (token ids) for padding
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 (ignored by loss)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # print(f"[DTYPE DEBUG] Data collator - final output dtype: {batch['input_features'].dtype}")

        # CRITICAL: Return ONLY input_features and labels
        # Don't return the whole batch which might have extra keys
        return {
            "input_features": batch["input_features"],
            "labels": labels,
        }

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# WER metric (using HuggingFace evaluate library imported at top of notebook)
wer_metric = hf_evaluate.load("wer")

def compute_metrics(pred):
    """Compute WER for evaluation."""
    # For Seq2SeqTrainer with predict_with_generate, predictions are already generated token IDs
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    # Skip special tokens and strip whitespace
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Normalize whitespace
    pred_str = [text.strip() for text in pred_str]
    label_str = [text.strip() for text in label_str]

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("Data collator and metrics ready.")

# %% [markdown]
# ## 8. Training Configuration
# 
# ### Why Seq2SeqTrainer?
# 
# Whisper is a sequence-to-sequence (encoder-decoder) model that takes audio input and generates text output. `Seq2SeqTrainer` from HuggingFace is specifically designed for this architecture and provides:
# 
# 1. **Proper generation during evaluation**: Uses `model.generate()` instead of forward pass
# 2. **Label handling**: Correctly handles the decoder input/output shift
# 3. **Beam search support**: For better generation quality during eval
# 
# **Reference**: [HuggingFace Fine-tune Whisper Guide](https://huggingface.co/blog/fine-tune-whisper)

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    max_steps=CONFIG["max_steps"],
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_steps=CONFIG["save_steps"],
    logging_steps=50,
    save_total_limit=3,
    fp16=CONFIG["fp16"],
    bf16=CONFIG["bf16"],
    weight_decay=0.01,
    dataloader_num_workers=4,
    remove_unused_columns=False,  # Required for PEFT - PeftModel forward signature differs from base model
    label_names=["labels"],
    predict_with_generate=True,  # Use generation for evaluation, not teacher forcing
    generation_max_length=225,   # Max length for generated sequences
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    report_to=["tensorboard"],
)

print(f"Training configuration:")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Batch size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")
print(f"  Max steps: {CONFIG['max_steps']}")
print(f"  Precision: {'fp16' if CONFIG['fp16'] else 'fp32'}")

# %% [markdown]
# ## 9. Train

# %%
# Initialize trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,  # CRITICAL: Use feature_extractor, not tokenizer!
)

# Clear GPU cache before training
torch.cuda.empty_cache()

print("Starting training...")
print("="*60)

# %%
# Train the model
trainer.train()

# %% [markdown]
# ## 10. Fine-tuning Summary
# 
# **Is LoRA "real" fine-tuning?**
# 
# LoRA is a form of **parameter-efficient fine-tuning (PEFT)**, not full fine-tuning. The distinction:
# 
# - **Full fine-tuning**: Updates all 1.5B parameters. Higher capacity but requires more VRAM and risks overfitting.
# - **LoRA**: Updates only ~15M adapter parameters (~1%). More efficient, preserves base knowledge, still achieves strong domain adaptation.
# 
# For domain adaptation (like VHP historical audio), LoRA is often preferred because:
# 1. The base model already has strong ASR capabilities
# 2. We want to adapt to acoustic characteristics, not relearn language
# 3. Limited training data makes full fine-tuning prone to overfitting

# %%
# Save LoRA weights
lora_path = os.path.join(CONFIG["output_dir"], "lora-weights")
model.save_pretrained(lora_path)
processor.save_pretrained(lora_path)

print(f"LoRA weights saved to: {lora_path}")

# Optionally merge and save full model
print("\nMerging LoRA weights with base model...")
merged_model = model.merge_and_unload()
merged_path = os.path.join(CONFIG["output_dir"], "merged-model")
merged_model.save_pretrained(merged_path)
processor.save_pretrained(merged_path)

print(f"Merged model saved to: {merged_path}")

# %%
# Print summary
print("="*60)
print("FINE-TUNING COMPLETE")
print("="*60)
print(f"\nBase Model: {CONFIG['model_name']}")
print(f"LoRA config: r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"\nData:")
print(f"  Train parquet: {CONFIG['train_parquet']}")
print(f"  Val parquet: {CONFIG['val_parquet']}")
print(f"\nOutputs:")
print(f"  LoRA weights: {lora_path}")
print(f"  Merged model: {merged_path}")

# %% [markdown]
# ## 11. Quick Inference Test
#
# Quick sanity check to verify the fine-tuned model can generate transcriptions.
#
# **Note**: This uses a few samples from the validation set (not a real test set). For proper evaluation, use the production pipeline (see Next Steps).
#
# The merged model is in HuggingFace format. Our production `infer_whisper.py` uses faster-whisper (CTranslate2 format) for speed. For this quick test, we use HuggingFace transformers pipeline directly.

# %%
# Quick test using HuggingFace transformers pipeline
from transformers import pipeline as hf_pipeline

# Load the merged fine-tuned model
pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=merged_path,
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1,
)


print(f"Model: {merged_path}")

# %%
# Run quick inference on validation samples
from tempfile import NamedTemporaryFile
from pydub import AudioSegment

inference_results = []

for idx, row in df_val_sample.iterrows():
    # For segmented parquets, use segmented_audio_url directly
    is_segmented = 'segmented_audio_url' in row

    if is_segmented:
        blob_path = row.get('segmented_audio_url', '')
        if not blob_path:
            print(f"[{idx}] SKIP: No segmented_audio_url")
            continue
        blob_paths = [blob_path]
    else:
        # Fallback to original logic for non-segmented parquets
        blob_paths = data_loader.get_blob_path_for_row(row, idx, CONFIG["blob_prefix"])

    for blob_path in blob_paths:
        if azure_utils.blob_exists(blob_path):
            print(f"[{idx}] Processing: {blob_path}")

            try:
                # Download audio
                audio_bytes = azure_utils.download_blob_to_memory(blob_path)

                # Save to temp file
                with NamedTemporaryFile(suffix=Path(blob_path).suffix, delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                # Convert to wav 16kHz mono
                audio_seg = AudioSegment.from_file(tmp_path)
                audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)

                wav_path = tmp_path.replace(Path(blob_path).suffix, '.wav')
                audio_seg.export(wav_path, format='wav')

                # Run inference
                result = pipe(wav_path, return_timestamps=True)
                hypothesis = result["text"]

                # Get ground truth
                if is_segmented:
                    gt = row.get('segmented_audio_transcript', '')
                else:
                    gt = clean_raw_transcript_str(row.get('fulltext_file_str', ''))

                inference_results.append({
                    "file_id": idx,
                    "hypothesis": hypothesis,
                    "ground_truth": gt,
                    "blob_path": blob_path
                })

                # Cleanup
                os.unlink(tmp_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

            except Exception as e:
                print(f"  Error: {e}")

            break  # Only process first available blob path

print(f"\nCompleted: {len(inference_results)} samples")

# %%
# View inference results
print("=" * 70)
print("QUICK INFERENCE TEST RESULTS")
print("=" * 70)

for r in inference_results[:3]:  # Show first 3
    print(f"\nFile ID: {r['file_id']}")
    print(f"Blob: {r['blob_path']}")
    print(f"\nHypothesis (first 300 chars):")
    print(r['hypothesis'][:300] + "..." if len(r['hypothesis']) > 300 else r['hypothesis'])
    print(f"\nGround truth (first 300 chars):")
    print(r['ground_truth'][:300] + "..." if len(r['ground_truth']) > 300 else r['ground_truth'])
    print("-" * 70)

# Save results
quick_test_dir = Path(CONFIG["output_dir"]) / "quick-inference-test"
quick_test_dir.mkdir(parents=True, exist_ok=True)

df_results = pd.DataFrame(inference_results)
df_results.to_parquet(quick_test_dir / "results.parquet", index=False)
print(f"\nResults saved to: {quick_test_dir / 'results.parquet'}")
print(f"\nNote: These are validation samples, not a real test set.")
print(f"For proper evaluation, use the production pipeline (see next cell).")

# %% [markdown]
# ## 12. Model Outputs & Next Steps
# 
# ### Understanding Model Outputs
# 
# After training, you'll have these artifacts in `{output_dir}`:
# 
# ```
# outputs/vhp-pre2010-whisper-large-v3-lora-ft-a6000/
# ├── checkpoint-{step}/              # Training checkpoints (for resuming training)
# │   ├── adapter_config.json         # LoRA configuration
# │   ├── adapter_model.safetensors   # LoRA weights at this checkpoint
# │   └── ...
# ├── lora-weights/                   # Final LoRA adapters (~60MB)
# │   ├── adapter_config.json
# │   ├── adapter_model.safetensors
# │   └── preprocessor_config.json
# └── merged-model/                   # Full model with LoRA merged (HF format, ~3GB)
#     ├── config.json
#     ├── model.safetensors
#     ├── preprocessor_config.json
#     └── ...
# ```
# 
# **Which to use?**
# - **checkpoints/**: Resume training from a specific step
# - **lora-weights/**: Load adapters on top of base model (saves disk space)
# - **merged-model/**: Standalone model ready for inference (what we'll use)
# 
# ---
# 
# ### `outputs/` vs `models/` Directory Convention
# 
# **Important**: The project follows this convention:
# - **`outputs/`**: Job artifacts from training/inference runs (checkpoints, logs, results)
# - **`models/`**: Production-ready models for inference pipeline
# 
# After training, your fine-tuned model lives in `outputs/{experiment_id}/` alongside other job outputs like inference results and evaluation metrics. This can get confusing when you have multiple runs.
# 
# **Best Practice**: Copy the converted model to `models/` for production use:
# 
# ```bash
# # After conversion (see Step 1 below), copy to models directory
# cp -r outputs/vhp-pre2010-whisper-large-v3-lora-ft-a6000/merged-model-ct2 \
#       models/whisper-large-v3-vhp-lora
# 
# # Now you have a clean model directory for inference configs
# models/
# └── whisper-large-v3-vhp-lora/      # Production model (CT2 format)
#     ├── config.json
#     ├── model.bin
#     └── ...
# ```
# 
# This keeps:
# - Training artifacts (checkpoints, HF models) in `outputs/` for reproducibility
# - Production models in `models/` for clean inference pipeline configs
# 
# ---
# 
# ### Production Inference Pipeline
# 
# **Step 1: Convert model to CTranslate2 format (run on RunPod or machine with model)**
# ```bash
# # Install CTranslate2 converter
# uv pip install ctranslate2
# 
# # Convert HuggingFace model to CTranslate2 format
# ct2-transformers-converter \
#   --model ../outputs/vhp-pre2010-whisper-large-v3-lora-ft-a6000/merged-model \
#   --output_dir ../outputs/vhp-pre2010-whisper-large-v3-lora-ft-a6000/merged-model-ct2 \
#   --quantization float16
# 
# # Copy to models directory for production use
# cp -r ../outputs/vhp-pre2010-whisper-large-v3-lora-ft-a6000/merged-model-ct2 \
#       ../models/whisper-large-v3-vhp-lora
# ```
# 
# **Step 2: Create inference config**
# 
# Create `configs/runs/vhp-pre2010-whisper-large-v3-lora-sample100.yaml`:
# ```yaml
# experiment_id: vhp-pre2010-whisper-large-v3-lora-sample100
# 
# model:
#   name: "whisper-large-v3-lora"
#   dir: "./models/whisper-large-v3-vhp-lora"  # Use models/ directory (not outputs/)
#   batch_size: 12
#   device: "cuda"
#   compute_type: "float16"
# 
# input:
#   source: "azure_blob"
#   parquet_path: "data/raw/loc/veterans_history_project_resources_pre2010_test.parquet"
#   blob_prefix: "loc_vhp"
#   sample_size: 100
# 
# output:
#   dir: "outputs/vhp-pre2010-whisper-large-v3-lora-sample100"  # Inference results go to outputs/
# ```
# 
# **Step 3: Run inference with production pipeline**
# ```bash
# uv run python scripts/infer_whisper.py \
#   --config configs/runs/vhp-pre2010-whisper-large-v3-lora-sample100.yaml
# ```
# 
# **Step 4: Evaluate**
# ```bash
# uv run python scripts/evaluate.py \
#   --config configs/runs/vhp-pre2010-whisper-large-v3-lora-sample100.yaml \
#   --inference_results outputs/vhp-pre2010-whisper-large-v3-lora-sample100/inference_results.parquet \
#   --parquet data/raw/loc/veterans_history_project_resources_pre2010_test.parquet
# ```
# 
# ---
# 
# ### Model Format Summary
# 
# | Format | Location | Use Case | Size |
# |--------|----------|----------|------|
# | **LoRA adapters** | `outputs/{exp}/lora-weights/` | Load on base model, save space | ~60MB |
# | **HF merged** | `outputs/{exp}/merged-model/` | Training, HF pipeline, conversion | ~3GB |
# | **CTranslate2** | `models/whisper-large-v3-vhp-lora/` | **Production inference** | ~1.5GB |
# | **Checkpoints** | `outputs/{exp}/checkpoint-N/` | Resume training | ~3GB each |
# 
# **Disk cleanup**: After copying to `models/`, you can delete:
# - `outputs/{exp}/merged-model/` (~3GB) - no longer needed after conversion
# - `outputs/{exp}/checkpoint-*/` (~3GB each) - only needed if resuming training


