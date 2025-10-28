import argparse, yaml, wandb, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg["wandb"]["project"],
               config=cfg, group=cfg["wandb"].get("group"),
               job_type="inference", tags=cfg["wandb"].get("tags", []))

    # Import the appropriate inference module
    scripts_dir = Path(__file__).parent
    if "whisper" in cfg["model"]["name"]:
        sys.path.insert(0, str(scripts_dir))
        import infer_whisper as mod
    elif "wav2vec2" in cfg["model"]["name"]:
        sys.path.insert(0, str(scripts_dir))
        import infer_wav2vec2 as mod
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    # both infer_* return the path to a hypothesis .txt and maybe per-file outputs
    run_artifacts = mod.run(cfg)
    wandb.log({"status": "inference_done"})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())