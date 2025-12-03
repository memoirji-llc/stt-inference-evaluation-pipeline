#!/usr/bin/env python3
"""
Fine-tune Whisper models with LoRA on VHP data.
This script is a direct conversion of the working notebook.
"""
import sys
import os
from pathlib import Path

# Load Azure credentials
from dotenv import load_dotenv
load_dotenv(dotenv_path='credentials/creds.env')

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

# Import HuggingFace evaluate
import evaluate as hf_evaluate

# Add scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from data import data_loader
from cloud import azure_utils

# Import clean_raw_transcript_str
import importlib.util
spec = importlib.util.spec_from_file_location("local_evaluate", PROJECT_ROOT / "scripts/eval/evaluate.py")
local_evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_evaluate)
clean_raw_transcript_str = local_evaluate.clean_raw_transcript_str

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "train_parquet": str(PROJECT_ROOT / "data/raw/loc/veterans_history_project_resources_pre2010_train_nfa_segmented.parquet"),
    "val_parquet": str(PROJECT_ROOT / "data/raw/loc/veterans_history_project_resources_pre2010_val_nfa_segmented.parquet"),

    "blob_prefix": "loc_vhp",

    "train_sample_size": 2400,
    "val_sample_size": 600,
    "random_seed": 42,

    "output_dir": str(PROJECT_ROOT / "outputs/vhp-pre2010-whisper-base-lora-ft-a6000"),

    "model_name": str(PROJECT_ROOT / "models/hf-whisper/whisper-base"),

    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],

    "learning_rate": 1e-3,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "warmup_steps": 500,
    "max_steps": 5000,
    "eval_steps": 500,
    "save_steps": 500,

    "fp16": True,
    "bf16": False,

    "max_label_length": 448,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"\nOutput directory: {CONFIG['output_dir']}")
print(f"Model: {CONFIG['model_name']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}\n")

# =============================================================================
# LOAD DATA
# =============================================================================
def load_finetune_dataset(parquet_path: str, sample_size: int = None, random_seed: int = 42):
    """Load dataset for fine-tuning (handles segmented parquets)."""
    df = pd.read_parquet(parquet_path)

    is_segmented = 'segmented_audio_url' in df.columns

    if is_segmented:
        print(f"Detected segmented parquet with {len(df)} segments")
        df = df[df['segmented_audio_url'].notna() & (df['segmented_audio_url'] != '')]
        print(f"Filtered to {len(df)} segments with segmented_audio_url")
        df = df[df['segmented_audio_transcript'].notna() & (df['segmented_audio_transcript'] != '')]
        print(f"Filtered to {len(df)} segments with segmented_audio_transcript")

        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_seed)
            print(f"Sampled {sample_size} segments")
        else:
            print(f"Using all {len(df)} segments")
    else:
        df = data_loader.load_vhp_dataset(
            parquet_path=parquet_path,
            sample_size=sample_size,
            filter_has_transcript=True,
            filter_has_media=True
        )

    print(f"Loaded {len(df)} samples")
    return df

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
print(f"Val: {len(df_val)} samples\n")

# =============================================================================
# PREPARE HF DATASETS (Download audio from Azure)
# =============================================================================
def prepare_hf_dataset(df: pd.DataFrame, blob_prefix: str, max_duration_sec: int = 30):
    """Convert DataFrame to HuggingFace Dataset with audio."""
    from tempfile import NamedTemporaryFile
    from pydub import AudioSegment
    import librosa

    is_segmented = 'segmented_audio_url' in df.columns
    print(f"Preparing dataset: is_segmented={is_segmented}, df size={len(df)}")

    records = []

    for idx, row in df.iterrows():
        # Get transcript
        if is_segmented:
            transcript = row.get('segmented_audio_transcript', '')
        else:
            raw_transcript = row.get('fulltext_file_str', '')
            transcript = clean_raw_transcript_str(raw_transcript)

        if not transcript.strip():
            continue

        # Get blob path
        if is_segmented:
            blob_path = row.get('segmented_audio_url', '')
            if not blob_path:
                continue
            blob_path_candidates = [blob_path]
        else:
            blob_path_candidates = data_loader.get_blob_path_for_row(row, idx, blob_prefix)
            if not blob_path_candidates:
                continue

        # Download audio
        audio_data = None
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

                audio_data, sr = librosa.load(wav_path, sr=16000)

                os.unlink(tmp_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)

                break
            except Exception as e:
                continue

        if audio_data is None:
            continue

        records.append({
            "audio": {"array": audio_data, "sampling_rate": 16000},
            "sentence": transcript
        })

    print(f"Valid samples: {len(records)}")

    if len(records) == 0:
        raise ValueError("No valid samples found")

    dataset = Dataset.from_dict({
        "audio": [r["audio"] for r in records],
        "sentence": [r["sentence"] for r in records]
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset

print("Preparing training dataset (downloading audio from Azure)...")
train_dataset = prepare_hf_dataset(df_train, CONFIG["blob_prefix"])

print("\nPreparing validation dataset...")
val_dataset = prepare_hf_dataset(df_val, CONFIG["blob_prefix"])

print(f"\nFinal dataset sizes:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val: {len(val_dataset)}\n")

# =============================================================================
# LOAD MODEL
# =============================================================================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

processor = WhisperProcessor.from_pretrained(CONFIG["model_name"])

print(f"Loading model: {CONFIG['model_name']}")
model = WhisperForConditionalGeneration.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

n_mels = model.config.num_mel_bins
model_type = "v3" if n_mels == 128 else "v2"

print(f"Model type: {model_type} ({n_mels} mel bins)")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")

# =============================================================================
# APPLY LoRA
# =============================================================================
lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=CONFIG["target_modules"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.config.use_cache = False

model.print_trainable_parameters()

# =============================================================================
# PREPROCESS DATA
# =============================================================================
def prepare_dataset(batch, processor, max_label_length=448):
    """Preprocess audio and text."""
    audio = batch["audio"]

    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]

    if CONFIG["fp16"]:
        input_features = input_features.to(torch.float16)

    labels = processor.tokenizer(batch["sentence"]).input_ids

    if len(labels) > max_label_length:
        return None

    return {
        "input_features": input_features,
        "labels": labels
    }

print("\nPreprocessing training data...")
processed_train = {"input_features": [], "labels": []}
skipped_token_limit = 0

for i in range(len(train_dataset)):
    sample = train_dataset[i]
    processed = prepare_dataset(sample, processor, CONFIG["max_label_length"])

    if processed is None:
        skipped_token_limit += 1
        continue

    processed_train["input_features"].append(processed["input_features"])
    processed_train["labels"].append(processed["labels"])

import datasets
train_dataset = datasets.Dataset.from_dict(processed_train)
print(f"Training preprocessing complete: {len(train_dataset)} samples (skipped {skipped_token_limit})")

print("\nPreprocessing validation data...")
processed_val = {"input_features": [], "labels": []}
skipped_val_token_limit = 0

for i in range(len(val_dataset)):
    sample = val_dataset[i]
    processed = prepare_dataset(sample, processor, CONFIG["max_label_length"])

    if processed is None:
        skipped_val_token_limit += 1
        continue

    processed_val["input_features"].append(processed["input_features"])
    processed_val["labels"].append(processed["labels"])

val_dataset = datasets.Dataset.from_dict(processed_val)
print(f"Validation preprocessing complete: {len(val_dataset)} samples (skipped {skipped_val_token_limit})\n")

# =============================================================================
# DATA COLLATOR & METRICS
# =============================================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if batch["input_features"].dtype != torch.float16:
            batch["input_features"] = batch["input_features"].to(torch.float16)

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        return {
            "input_features": batch["input_features"],
            "labels": labels,
        }

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

wer_metric = hf_evaluate.load("wer")

def compute_metrics(pred):
    """Compute WER."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [text.strip() for text in pred_str]
    label_str = [text.strip() for text in label_str]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# =============================================================================
# TRAINING
# =============================================================================
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
    remove_unused_columns=False,
    label_names=["labels"],
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    report_to=["tensorboard"],
)

print("Training configuration:")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Batch size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")
print(f"  Max steps: {CONFIG['max_steps']}")
print(f"  Precision: {'fp16' if CONFIG['fp16'] else 'fp32'}\n")

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

torch.cuda.empty_cache()

print("Starting training...")
print("="*60)
trainer.train()

# =============================================================================
# SAVE MODEL
# =============================================================================
lora_path = os.path.join(CONFIG["output_dir"], "lora-weights")
model.save_pretrained(lora_path)
processor.save_pretrained(lora_path)
print(f"\nLoRA weights saved to: {lora_path}")

print("\nMerging LoRA weights with base model...")
merged_model = model.merge_and_unload()
merged_path = os.path.join(CONFIG["output_dir"], "merged-model")
merged_model.save_pretrained(merged_path)
processor.save_pretrained(merged_path)
print(f"Merged model saved to: {merged_path}")

print("\n" + "="*60)
print("FINE-TUNING COMPLETE")
print("="*60)
