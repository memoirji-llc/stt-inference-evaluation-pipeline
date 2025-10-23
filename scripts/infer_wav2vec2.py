# config/ io
import glob
import os
from pathlib import Path
import yaml
# audio processing
import soundfile as sf
import librosa
# models
import torch
from transformers import AutoProcessor, AutoModelForCTC
# experiment tracking
import wandb

# configs for local
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.set_num_threads(4) # keep CPU stable

# # load config
# with open("configs/inference.yaml") as f:
#     cfg = yaml.safe_load(f)

def run(cfg):
    out_dir = Path(cfg["output"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(cfg["model"]["dir"])
    device = "cpu"  # safe on local (Mac); can switch to CUDA later
    model = AutoModelForCTC.from_pretrained(cfg["model"]["dir"]).to(device).eval()
    hyp_path = out_dir / "hyp_w2v2.txt"
    
    with open(hyp_path, "w") as hout:
        for p in glob.glob(cfg["input"]["audio_glob"]):
            info = sf.info(p)
            wav, sr = librosa.load(p, sr=cfg["input"]["sample_rate"], mono=True, duration=cfg["input"]["duration_sec"])
            
            # if 2 channels, force convert to mono
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            
            inputs = processor(wav, sampling_rate=sr, return_tensors="pt", padding="longest")
            
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits
                
            ids = torch.argmax(logits, dim=-1)
            hyp = processor.batch_decode(ids)[0].lower()
            print(hyp[:10])
            hout.write(hyp + "\n")

    wandb.save(str(hyp_path))
    wandb.save(str(info))
    print("testing finished")
    return {"hyp_path": str(hyp_path)}



