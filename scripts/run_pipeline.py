"""
Simple pipeline orchestrator: runs inference + evaluation for each config.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

def get_hyp_path(config_path):
    """Read config and determine where the hypothesis file will be written."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output"]["dir"]
    model_name = cfg["model"]["name"]

    # Match the naming convention from infer_* scripts
    if model_name == "whisper":
        return Path(output_dir) / "hyp_whisper.txt"
    elif model_name == "wav2vec2":
        return Path(output_dir) / "hyp_w2v2.txt"
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    ap = argparse.ArgumentParser(description="Run inference + evaluation pipeline")
    ap.add_argument("--configs", nargs="+", required=True, help="Config YAML files to process")
    ap.add_argument("--ref", required=True, help="Reference ground truth text file")
    args = ap.parse_args()

    for config_path in args.configs:
        print(f"\n=== PROCESSING: {config_path} ===")

        # Step 1: Run inference
        print(f"=== INFERENCE: {config_path} ===")
        result = subprocess.run(
            ["uv", "run", "python", "scripts/run_inference.py", "--config", config_path],
            check=True
        )

        # Step 2: Get hypothesis path
        hyp_path = get_hyp_path(config_path)
        print(f"Hypothesis file: {hyp_path}")

        # Step 3: Run evaluation
        print(f"=== EVALUATION: {Path(config_path).name} ===")
        result = subprocess.run(
            ["uv", "run", "python", "scripts/evaluate.py",
             "--config", config_path,
             "--hyp", str(hyp_path),
             "--ref", args.ref],
            check=True
        )

    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    sys.exit(main())
