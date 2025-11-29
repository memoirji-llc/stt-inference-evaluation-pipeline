"""
Pipeline orchestrator: runs inference + evaluation for each config.
Supports both old (single .txt reference) and new (parquet) workflows.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def get_output_paths(config_path):
    """Read config and determine where output files will be written."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output"]["dir"])
    model_name = cfg["model"]["name"]

    # Determine hypothesis filename based on model
    if model_name == "whisper":
        hyp_filename = "hyp_whisper.txt"
    elif model_name == "wav2vec2":
        hyp_filename = "hyp_w2v2.txt"
    else:
        hyp_filename = "hyp_whisper.txt"  # Default fallback

    return {
        "output_dir": output_dir,
        "inference_results": output_dir / "inference_results.parquet",
        "hyp_path": output_dir / hyp_filename,
    }


def main():
    ap = argparse.ArgumentParser(description="Run inference + evaluation pipeline")
    ap.add_argument("--configs", nargs="+", required=True, help="Config YAML files to process")
    ap.add_argument("--parquet", required=False, help="Path to parquet with ground truth (new workflow)")
    ap.add_argument("--ref", required=False, help="Reference ground truth text file (legacy workflow)")

    # CLI overrides (optional)
    ap.add_argument("--model-dir", required=False, help="Override model directory from config")
    args = ap.parse_args()

    # Determine workflow type
    use_parquet_workflow = args.parquet is not None

    for config_path in args.configs:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {config_path}")
        if args.model_dir:
            print(f"MODEL OVERRIDE: {args.model_dir}")
        print(f"{'='*60}")

        # Step 1: Run inference
        print(f"\n=== STEP 1: INFERENCE ===")

        # Build inference command with optional overrides
        inference_cmd = ["uv", "run", "python", "scripts/run_inference.py", "--config", config_path]
        if args.model_dir:
            inference_cmd.extend(["--model-dir", args.model_dir])

        result = subprocess.run(inference_cmd, check=True)

        # Step 2: Get output paths
        paths = get_output_paths(config_path)
        print(f"\nInference complete. Results saved to:")
        print(f"  - {paths['inference_results']}")

        # Step 3: Run evaluation
        print(f"\n=== STEP 2: EVALUATION ===")

        if use_parquet_workflow:
            # New workflow: evaluate using parquet
            eval_cmd = [
                "uv", "run", "python", "scripts/eval/evaluate.py",
                "--config", config_path,
                "--inference_results", str(paths["inference_results"]),
                "--parquet", args.parquet,
            ]
        else:
            # Legacy workflow: evaluate using single reference text file
            if not args.ref:
                print("WARNING: No --parquet or --ref specified. Skipping evaluation.")
                continue

            eval_cmd = [
                "uv", "run", "python", "scripts/eval/evaluate.py",
                "--config", config_path,
                "--hyp", str(paths["hyp_path"]),
                "--ref", args.ref,
            ]

        result = subprocess.run(eval_cmd, check=True)

        print(f"\n{'='*60}")
        print(f"COMPLETED: {config_path}")
        print(f"{'='*60}")

    print("\n" + "="*60)
    print("ALL PIPELINES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    sys.exit(main())
