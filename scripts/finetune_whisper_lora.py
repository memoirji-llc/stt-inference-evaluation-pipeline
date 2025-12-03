#!/usr/bin/env python
"""
Fine-tuning Whisper with LoRA/PEFT

Supports both Whisper v2 (80 mel bins) and v3 (128 mel bins) models.

OPERATION SEQUENCE:
1. Load configuration (CONFIG dict)
2. Load train/val data from parquet
3. Download and process audio from Azure blob
4. Load Whisper model + apply LoRA adapters
5. Preprocess audio to mel spectrograms
6. Train model with Seq2SeqTrainer
7. Save LoRA weights + merge with base model
8. Optional: Run quick inference test

USAGE:
    # Edit CONFIG section below, then run:
    cd /workspace/amia2025-stt-benchmarking
    screen -S whisper-ft
    python scripts/finetune_whisper_lora.py

REQUIREMENTS:
    - credentials/creds.env with Azure blob credentials
    - Dependencies: peft, accelerate, transformers, torch

OUTPUT:
    CONFIG["output_dir"]/
    ├── checkpoint-*/        # Training checkpoints
    ├── lora-weights/        # Final LoRA adapters (~60MB)
    ├── merged-model/        # Full model with LoRA merged (~3GB)
    └── runs/                # Tensorboard logs
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import os
from pathlib import Path
import gc
import io
import time

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load Azure credentials
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / 'credentials/creds.env')

# Core dependencies
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Audio processing
from pydub import AudioSegment
import librosa

# HuggingFace transformers
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    pipeline
)
from datasets import Dataset, DatasetDict, Audio
import evaluate

# LoRA / PEFT
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Local imports
from scripts.cloud import azure_utils
from scripts.data import data_loader


# ============================================================================
# CONFIGURATION
# ============================================================================
#
# EDIT THIS SECTION to configure fine-tuning
#
# Key parameters:
# - model_name: Which Whisper model to fine-tune
# - model_type: "v2" (80 mel bins) or "v3" (128 mel bins) - AUTO-DETECTED from model
# - lora_r/lora_alpha: LoRA rank and scaling (higher = more parameters)
# - learning_rate: CRITICAL - v3 uses 1e-5, v2 typically uses 1e-3
# - batch_size/gradient_accumulation_steps: Effective batch size = product of both
#
# ============================================================================

CONFIG = {
    # ====================
    # MODEL CONFIGURATION
    # ====================
    # Whisper model to fine-tune from HuggingFace
    # Options:
    #   v3: "openai/whisper-large-v3" (128 mel bins)
    #   v2: "openai/whisper-large-v2" (80 mel bins)
    #   v2: "openai/whisper-medium" (80 mel bins)
    #   v2: "openai/whisper-small" (80 mel bins)
    #   v2: "openai/whisper-base" (80 mel bins)
    #   Local: Use absolute path for pre-downloaded models
    "model_name": str(PROJECT_ROOT / "models/hf-whisper/whisper-base"),

    # ====================
    # LoRA CONFIGURATION
    # ====================
    "lora_r": 32,              # LoRA rank (8, 16, 32, 64)
    "lora_alpha": 64,          # LoRA alpha (typically 2x rank)
    "lora_dropout": 0.05,      # Dropout for LoRA layers
    "target_modules": ["q_proj", "v_proj"],  # Which modules to apply LoRA to

    # ====================
    # TRAINING HYPERPARAMETERS
    # ====================
    "learning_rate": 1e-3,     # CRITICAL: v3 uses 1e-5, v2 uses 1e-3
    "batch_size": 4,           # Per-device batch size
    "gradient_accumulation_steps": 4,  # Effective batch = 4 * 4 = 16
    "max_steps": 5000,         # Total training steps
    "eval_steps": 500,         # Evaluate every N steps
    "save_steps": 500,         # Save checkpoint every N steps
    "warmup_steps": 500,       # Learning rate warmup
    "fp16": True,              # Use fp16 precision (faster, less memory)

    # ====================
    # DATA PATHS
    # ====================
    # Parquet files with transcript + blob_path columns
    # Use NFA-segmented data (<=30s segments) for fine-tuning
    "train_parquet": str(PROJECT_ROOT / "data/raw/loc/veterans_history_project_resources_pre2010_train_nfa_segmented.parquet"),
    "val_parquet": str(PROJECT_ROOT / "data/raw/loc/veterans_history_project_resources_pre2010_val_nfa_segmented.parquet"),
    "is_segmented": True,      # True if using NFA-segmented data (recommended)
    "sample_size": None,       # None = use all data, or set to int for testing

    # ====================
    # OUTPUT CONFIGURATION
    # ====================
    "output_dir": str(PROJECT_ROOT / "outputs/vhp-whisper-base-v2-lora-ft-a6000"),
    "run_quick_test": False,   # Disabled - audio column not preserved after preprocessing
}

# Create output directory
Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("WHISPER LoRA FINE-TUNING")
print("=" * 80)
print(f"Model: {CONFIG['model_name']}")
print(f"LoRA rank: {CONFIG['lora_r']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"Max steps: {CONFIG['max_steps']}")
print(f"Output: {CONFIG['output_dir']}")
print("=" * 80)
print()

# PyTorch/CUDA diagnostics
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print()


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("STEP 1: LOAD DATA")
print("=" * 80)

def load_finetune_data(parquet_path, is_segmented, sample_size=None):
    """Load and filter parquet data for fine-tuning."""
    df = pd.read_parquet(parquet_path)

    if is_segmented:
        # Filter for valid segmented samples
        print(f"Detected segmented parquet with {len(df)} segments")
        df = df[df['segmented_audio_url'].notna() & (df['segmented_audio_url'] != '')]
        print(f"Filtered to {len(df)} segments with segmented_audio_url")
        df = df[df['segmented_audio_transcript'].notna() & (df['segmented_audio_transcript'] != '')]
        print(f"Filtered to {len(df)} segments with segmented_audio_transcript")
    else:
        # Filter for original data
        df = df[df['fulltext_file_str'].notna()]
        df = df[(df['audio_url'].notna()) | (df['video_url'].notna())]
        print(f"Filtered to {len(df)} items with transcripts and media")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} items")

    return df

# Load train/val parquet files
print("Loading training data...")
train_df = load_finetune_data(
    CONFIG["train_parquet"],
    CONFIG["is_segmented"],
    CONFIG["sample_size"]
)

print("\nLoading validation data...")
val_df = load_finetune_data(
    CONFIG["val_parquet"],
    CONFIG["is_segmented"],
    CONFIG["sample_size"] // 10 if CONFIG["sample_size"] else None
)

print(f"\nTrain samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print()


# ============================================================================
# STEP 2: PROCESS AUDIO
# ============================================================================

print("=" * 80)
print("STEP 2: PROCESS AUDIO FROM AZURE BLOB")
print("=" * 80)

def process_single_audio(row, idx, is_segmented):
    """
    Download and process a single audio sample from Azure blob.

    Returns:
        dict with 'audio' (np.array) and 'transcript' (str)
        or None if processing fails
    """
    # Get transcript
    if is_segmented:
        transcript = row.get("segmented_audio_transcript", "").strip()
        if not transcript:
            return None
    else:
        transcript = row.get("transcript", "").strip()
        if not transcript:
            return None

    # Get blob paths to try
    if is_segmented:
        blob_path = row.get("segmented_audio_url")
        if not blob_path:
            return None
        blob_paths = [blob_path]
    else:
        original_idx = row.get("original_index", idx)
        blob_paths = [
            f"loc_vhp/{original_idx}/video.mp4",
            f"loc_vhp/{original_idx}/audio.mp3"
        ]

    # Try to download audio
    for blob_path in blob_paths:
        try:
            audio_bytes = azure_utils.download_blob_to_memory(
                container_name="audio-raw",
                blob_name=blob_path
            )

            if not audio_bytes or len(audio_bytes) < 100:
                continue

            # Convert to 16kHz mono WAV
            audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)

            # For non-segmented data, limit to 30s (Whisper's max context)
            if not is_segmented:
                audio_seg = audio_seg[:30000]  # 30 seconds in milliseconds

            # Convert to numpy float32 array [-1, 1]
            samples = np.array(audio_seg.get_array_of_samples())
            audio_array = samples.astype(np.float32) / 32768.0  # 16-bit normalization

            return {
                "audio": audio_array,
                "transcript": transcript,
                "blob_path": blob_path
            }

        except Exception as e:
            continue

    return None


def prepare_hf_dataset_batched(df, is_segmented, batch_size=100):
    """
    Download and process audio in batches, creating HuggingFace Dataset.

    Memory-efficient: processes 100 samples at a time, clears memory between batches.
    """
    print(f"Processing {len(df)} samples in batches of {batch_size}...")

    all_data = []

    for batch_start in tqdm(range(0, len(df), batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        batch_data = []
        for idx, row in batch_df.iterrows():
            result = process_single_audio(row, idx, is_segmented)
            if result:
                batch_data.append(result)

        all_data.extend(batch_data)

        # Memory cleanup
        del batch_data
        gc.collect()

        print(f"  Batch {batch_start}-{batch_end}: {len(all_data)} samples processed")

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "audio": [d["audio"] for d in all_data],
        "transcript": [d["transcript"] for d in all_data],
        "blob_path": [d["blob_path"] for d in all_data]
    })

    # Cast audio column to Audio type for automatic processing
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Cache to disk for memory-mapped access
    dataset = dataset.with_format("torch")

    print(f"Final dataset: {len(dataset)} samples")
    return dataset


# Process train and val datasets
train_dataset = prepare_hf_dataset_batched(train_df, CONFIG["is_segmented"])
val_dataset = prepare_hf_dataset_batched(val_df, CONFIG["is_segmented"])

# Clear dataframes from memory
del train_df, val_df
gc.collect()

print()


# ============================================================================
# STEP 3: LOAD MODEL + APPLY LoRA
# ============================================================================

print("=" * 80)
print("STEP 3: LOAD MODEL + APPLY LoRA")
print("=" * 80)

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float16 if CONFIG["fp16"] else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

processor = WhisperProcessor.from_pretrained(CONFIG["model_name"])

# Auto-detect model version from mel bins
n_mels = model.config.num_mel_bins
model_type = "v3" if n_mels == 128 else "v2"

print(f"Model loaded: {CONFIG['model_name']}")
print(f"Model type: {model_type} ({n_mels} mel bins)")
print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
print()

# Remove forced decoder IDs for fine-tuning
if hasattr(model.config, "forced_decoder_ids"):
    model.config.forced_decoder_ids = None
if hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None

# Apply LoRA
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none"
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"LoRA configuration:")
print(f"  Rank: {CONFIG['lora_r']}")
print(f"  Alpha: {CONFIG['lora_alpha']}")
print(f"  Target modules: {CONFIG['target_modules']}")
print(f"Trainable parameters: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")
print()


# ============================================================================
# STEP 4: PREPROCESS DATA (Audio → Mel Spectrograms)
# ============================================================================

print("=" * 80)
print("STEP 4: PREPROCESS DATA")
print("=" * 80)

def prepare_dataset(sample, processor, max_label_length=448):
    """
    Preprocess single audio sample.

    Returns dict with input_features and labels, or None if exceeds token limit.
    """
    audio = sample["audio"]

    # Extract features from audio
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    # Convert to fp16 if using fp16 training
    if CONFIG["fp16"]:
        input_features = input_features.to(torch.float16)

    # Tokenize transcript
    labels = processor.tokenizer(sample["transcript"]).input_ids

    # Check if labels exceed Whisper's max decoder length
    if len(labels) > max_label_length:
        return None  # Skip this sample

    return {
        "input_features": input_features,
        "labels": labels
    }


# Preprocess datasets manually (like notebook - avoids multiprocessing issues)
print("Preprocessing train dataset...")
print(f"Dataset size: {len(train_dataset)} samples")
print(f"Max label length: {CONFIG.get('max_label_length', 448)} tokens")

processed_train = {"input_features": [], "labels": []}
skipped_train = 0

for i in range(len(train_dataset)):
    sample = train_dataset[i]
    processed = prepare_dataset(sample, processor, CONFIG.get("max_label_length", 448))

    if processed is None:
        skipped_train += 1
        continue

    processed_train["input_features"].append(processed["input_features"])
    processed_train["labels"].append(processed["labels"])

# Create new dataset from preprocessed data
import datasets
train_dataset = datasets.Dataset.from_dict(processed_train)

print(f"Training preprocessing complete:")
print(f"  Valid samples: {len(train_dataset)}")
print(f"  Skipped (token limit): {skipped_train}")

print("\nPreprocessing val dataset...")
processed_val = {"input_features": [], "labels": []}
skipped_val = 0

for i in range(len(val_dataset)):
    sample = val_dataset[i]
    processed = prepare_dataset(sample, processor, CONFIG.get("max_label_length", 448))

    if processed is None:
        skipped_val += 1
        continue

    processed_val["input_features"].append(processed["input_features"])
    processed_val["labels"].append(processed["labels"])

val_dataset = datasets.Dataset.from_dict(processed_val)

print(f"Validation preprocessing complete:")
print(f"  Valid samples: {len(val_dataset)}")
print(f"  Skipped (token limit): {skipped_val}")

print(f"\n{'='*60}")
print(f"PREPROCESSING SUMMARY")
print(f"{'='*60}")
print(f"Train: {len(train_dataset)} samples (skipped {skipped_train})")
print(f"Val: {len(val_dataset)} samples (skipped {skipped_val})")
print()


# ============================================================================
# STEP 5: SETUP TRAINING
# ============================================================================

print("=" * 80)
print("STEP 5: SETUP TRAINING")
print("=" * 80)

# Data collator
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper fine-tuning."""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split into input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        labels_batch = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Metrics
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    """Compute WER metric."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    max_steps=CONFIG["max_steps"],
    fp16=CONFIG["fp16"],
    eval_strategy="steps",
    per_device_eval_batch_size=CONFIG["batch_size"],
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=CONFIG["save_steps"],
    eval_steps=CONFIG["eval_steps"],
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,  # Required for PEFT
    label_names=["labels"],
)

print(f"Training configuration:")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Batch size: {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']})")
print(f"  Max steps: {CONFIG['max_steps']}")
print(f"  FP16: {CONFIG['fp16']}")
print()


# ============================================================================
# STEP 6: TRAIN
# ============================================================================

print("=" * 80)
print("STEP 6: TRAIN MODEL")
print("=" * 80)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("Starting training...")
print()

trainer.train()

print()
print("Training complete!")
print()


# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================

print("=" * 80)
print("STEP 7: SAVE MODEL ARTIFACTS")
print("=" * 80)

# Save LoRA weights
lora_output_dir = Path(CONFIG["output_dir"]) / "lora-weights"
model.save_pretrained(lora_output_dir)
processor.save_pretrained(lora_output_dir)
print(f"LoRA weights saved: {lora_output_dir}")

# Merge LoRA with base model
merged_output_dir = Path(CONFIG["output_dir"]) / "merged-model"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_output_dir)
processor.save_pretrained(merged_output_dir)
print(f"Merged model saved: {merged_output_dir}")
print()


# ============================================================================
# STEP 8: QUICK INFERENCE TEST (Optional)
# ============================================================================

if CONFIG["run_quick_test"]:
    print("=" * 80)
    print("STEP 8: QUICK INFERENCE TEST")
    print("=" * 80)

    # Load merged model in pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=str(merged_output_dir),
        device=0 if torch.cuda.is_available() else -1
    )

    # Test on validation samples
    test_samples = val_dataset.select(range(min(CONFIG["test_sample_size"], len(val_dataset))))

    results = []
    for i, sample in enumerate(test_samples):
        # Get audio array
        audio = sample["audio"]["array"]
        reference = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True)

        # Run inference
        result = pipe(audio)
        hypothesis = result["text"]

        # Calculate WER
        wer = 100 * wer_metric.compute(predictions=[hypothesis], references=[reference])

        results.append({
            "sample_id": i,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": wer
        })

        print(f"Sample {i}:")
        print(f"  REF: {reference[:100]}...")
        print(f"  HYP: {hypothesis[:100]}...")
        print(f"  WER: {wer:.1f}%")
        print()

    # Save results
    results_df = pd.DataFrame(results)
    results_path = Path(CONFIG["output_dir"]) / "quick_test_results.parquet"
    results_df.to_parquet(results_path, index=False)

    avg_wer = results_df["wer"].mean()
    print(f"Average WER: {avg_wer:.1f}%")
    print(f"Results saved: {results_path}")
    print()


# ============================================================================
# COMPLETE
# ============================================================================

print("=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)
print(f"Model artifacts saved to: {CONFIG['output_dir']}")
print()
print("Next steps:")
print("1. Download merged model from RunPod:")
print(f"   scp -r runpod:/workspace/amia2025-stt-benchmarking/{CONFIG['output_dir']}/merged-model .")
print()
print("2. Upload to Azure blob:")
print("   # Use Azure Storage Explorer GUI or az storage blob upload-batch")
print()
print("3. Run inference on Azure VM:")
print("   python scripts/run_inference.py --config configs/runs/your-config.yaml")
print()
print(f"Model type: {model_type} ({n_mels} mel bins)")
if model_type == "v3":
    print("⚠️  Inference: Use HuggingFace transformers (NOT faster-whisper)")
else:
    print("✅ Inference: Compatible with both HuggingFace and faster-whisper")
print("=" * 80)
