# config/ io
import glob
from pathlib import Path
import yaml
from tempfile import NamedTemporaryFile
# audio processing
import soundfile as sf
import librosa
# models
from faster_whisper import WhisperModel
# experiment tracking
import wandb

# # load config
# with open("configs/inference.yaml") as f:
#     cfg = yaml.safe_load(f)

def run(cfg):
    out_dir = Path(cfg["output"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(cfg["model"]["dir"])
    model_snap = max((model_root / "snapshots").iterdir(), key=lambda p: p.stat().st_mtime)
    model_file_dir = str(model_snap)  # folder that contains model.bin
    model = WhisperModel(model_file_dir, device="auto")
    print("Using model dir on local:", model_file_dir)
    hyp_path = out_dir / "hyp_whisper.txt"
    
    with open(hyp_path, "w") as hout:
        
        for p in glob.glob(cfg["input"]["audio_glob"]):
            # load exactly first 120s as mono 16k
            wave, _ = librosa.load(p, sr=16000, mono=True, duration=120)
            
            with NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                
                sf.write(tmp.name, wave, 16000)  # write trimmed clip
                
                segments, info = model.transcribe(
                    tmp.name,
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=True,
                    no_speech_threshold=0.6,
                    word_timestamps=False,
                    initial_prompt=None,
                    suppress_tokens=[-1],   # <-- FIX: list of ints, or set to None
                    condition_on_previous_text=False,
                )

                hyp_whisper = " ".join(s.text.strip().lower() for s in segments)
                print(f"[{info.language}] {hyp_whisper[:10]}...")
                wandb.log({"placeholder": 1})
    wandb.save(str(hyp_path))
    wandb.save(str(info))
    print("testing finished")
    return {"hyp_path": str(hyp_path)}