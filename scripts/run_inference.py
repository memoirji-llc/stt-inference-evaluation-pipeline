import argparse, yaml, wandb, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", required=False, help="Override model directory from config")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model_dir:
        print(f"[OVERRIDE] Using model directory: {args.model_dir}")
        cfg["model"]["dir"] = args.model_dir
        # Update experiment_id to reflect override
        model_name = Path(args.model_dir).name.replace("models--Systran--faster-", "")
        cfg["experiment_id"] = f"{cfg['experiment_id']}-override-{model_name}"

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
    elif "gcp_chirp" in cfg["model"]["name"]:
        sys.path.insert(0, str(scripts_dir))
        import infer_gcp_chirp as mod
    elif "aws_transcribe" in cfg["model"]["name"]:
        sys.path.insert(0, str(scripts_dir))
        import infer_aws_transcribe as mod
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")

    # both infer_* return the path to a hypothesis .txt and maybe per-file outputs
    run_artifacts = mod.run(cfg)
    wandb.log({"status": "inference_done"})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())