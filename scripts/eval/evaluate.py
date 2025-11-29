import argparse, yaml, jiwer, wandb, re
from pathlib import Path
import sys
from pathlib import Path as P
_scripts_dir = P(__file__).parent
sys.path.insert(0, str(_scripts_dir))
from file_logger import log, init_logger
import pandas as pd
from bs4 import BeautifulSoup


def normalize(s, use_whisper_normalizer=False):
    """
    Normalize text for WER calculation.

    Args:
        s: Input text string
        use_whisper_normalizer: If True, use OpenAI's Whisper normalizer (handles numbers, dates, etc.)
                                If False, use jiwer's standard normalization with contraction expansion

    Returns:
        Normalized text string

    Note:
        Standard normalization (default):
        - Expands contractions (we're → we are, can't → can not)
        - Converts to lowercase
        - Removes punctuation
        - Normalizes whitespace

        This follows best practices from Whisper normalizer and ASR research.
        See: learnings/wer-normalization-guide.md for details.
    """
    if use_whisper_normalizer:
        try:
            from whisper.normalizers import EnglishTextNormalizer
            normalizer = EnglishTextNormalizer()
            return normalizer(s)
        except ImportError:
            log("WARNING: whisper not installed. Falling back to jiwer normalization.")
            log("Install with: pip install openai-whisper")
            use_whisper_normalizer = False

    # Standard jiwer normalization (default)
    tx = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),  # Expand we're → we are, etc.
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])
    return tx(s)


def clean_raw_transcript_str(fulltext_file_str: str) -> str:
    """
    Clean raw XML transcript from VHP dataset.
    Adapted from loc.ipynb notebook.
    """
    if not fulltext_file_str or pd.isna(fulltext_file_str):
        return ""

    l_transcript_lines = []
    # utilize bs4 xml parser
    soup = BeautifulSoup(fulltext_file_str, 'xml')
    # each sp tag in the document represents a "line" in the transcript
    for sp in soup.find_all('sp'):
        try:
            speaker = sp.find('speaker').get_text(strip=True)
        except:
            speaker = "speaker_unknown"
        try:
            spoken_text = sp.find('p').get_text(strip=True)
        except:
            spoken_text = ""

        l_transcript_lines.append(f"<{speaker}>{spoken_text}</{speaker}> ")

    # merge lines into one string
    transcript_lines = ''.join(l_transcript_lines)

    # remove (), [], {} and anything in between
    transcript_lines_stripped = re.sub(r'\([^)]*\)', '', transcript_lines)
    transcript_lines_stripped = re.sub(r'\[[^]]*\]', '', transcript_lines_stripped)
    transcript_lines_stripped = re.sub(r'\{[^}]*\)\}', '', transcript_lines_stripped)

    # remove double dashes and ellipsis
    transcript_lines_stripped = re.sub(r'--+', '', transcript_lines_stripped)
    transcript_lines_stripped = re.sub(r'\.{2,}', '', transcript_lines_stripped)

    # clean whitespace
    transcript_lines_stripped = re.sub(r'\s+', ' ', transcript_lines_stripped).strip()

    # remove speaker tags
    transcript_lines_stripped = re.sub(r'\<[^>]*\>', '', transcript_lines_stripped)

    return transcript_lines_stripped


def main():
    ap = argparse.ArgumentParser(description="Evaluate STT results against ground truth")
    ap.add_argument("--config", required=True, help="Experiment config YAML")

    # New workflow (parquet-based)
    ap.add_argument("--inference_results", required=False, help="Path to inference_results.parquet (new workflow)")
    ap.add_argument("--parquet", required=False, help="Path to VHP parquet with ground truth transcripts (new workflow)")

    # Legacy workflow (single text file)
    ap.add_argument("--hyp", required=False, help="Hypothesis text file (legacy workflow)")
    ap.add_argument("--ref", required=False, help="Reference text file (legacy workflow)")

    args = ap.parse_args()

    # Determine which workflow to use
    if args.inference_results and args.parquet:
        # New parquet-based workflow
        evaluate_parquet_workflow(args)
    elif args.hyp and args.ref:
        # Legacy single-file workflow
        evaluate_legacy_workflow(args)
    else:
        log("ERROR: Must provide either (--inference_results + --parquet) OR (--hyp + --ref)")
        return 1


def evaluate_legacy_workflow(args):
    """Legacy workflow: single hypothesis file vs single reference file"""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize file logger
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logger(out_dir, prefix="evaluation")

    # Get normalization setting from config (default: False = use jiwer)
    use_whisper_normalizer = cfg.get("evaluation", {}).get("use_whisper_normalizer", False)

    # Read reference and hypothesis as single strings
    ref_text = normalize(Path(args.ref).read_text(), use_whisper_normalizer=use_whisper_normalizer)
    hyp_text = normalize(Path(args.hyp).read_text(), use_whisper_normalizer=use_whisper_normalizer)

    # Compute WER
    m = jiwer.process_words(ref_text, hyp_text)
    log(f"\n=== EVALUATION RESULTS ===")
    log(f"WER: {m.wer:.3f}")
    log(f"Substitutions: {m.substitutions}")
    log(f"Deletions: {m.deletions}")
    log(f"Insertions: {m.insertions}")

    # Log to wandb - Add "evaluation" tag to help distinguish from inference runs
    tags = cfg["wandb"].get("tags", []).copy() if cfg["wandb"].get("tags") else []
    if "evaluation" not in tags:
        tags = ["evaluation"] + tags

    wandb.init(project=cfg["wandb"]["project"],
               group=cfg["wandb"].get("group"),
               job_type="evaluation",
               config=cfg,
               tags=tags,
               name=f"{cfg['experiment_id']}-evaluation")  # Explicit name for easy filtering

    wandb.log({
        "wer": m.wer,
        "substitutions": m.substitutions,
        "deletions": m.deletions,
        "insertions": m.insertions,
    })


def evaluate_parquet_workflow(args):
    """New workflow: parquet-based evaluation with per-file metrics"""

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Initialize file logger
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logger(out_dir, prefix="evaluation")

    # Get normalization setting from config (default: False = use jiwer)
    use_whisper_normalizer = cfg.get("evaluation", {}).get("use_whisper_normalizer", False)

    # Print normalization method being used
    if use_whisper_normalizer:
        log("Using Whisper normalizer (numbers, dates, contractions, etc.)")
    else:
        log("Using jiwer normalization with contraction expansion (default)")

    # Load inference results
    df_results = pd.read_parquet(args.inference_results)
    log(f"Loaded {len(df_results)} inference results")

    # Load ground truth parquet
    df_gt = pd.read_parquet(args.parquet)
    log(f"Loaded {len(df_gt)} ground truth entries")

    # Join inference results with ground truth
    # Match by file_id (index in the original dataframe)
    df_eval = df_results.copy()

    # Extract ground truth for each file
    eval_results = []
    for idx, row in df_eval.iterrows():
        file_id = row["file_id"]
        hypothesis = row["hypothesis"]

        # Get ground truth from inference results (already embedded correctly)
        # The ground_truth column was populated during inference with the correct mapping
        reference = row.get('ground_truth', None)

        if reference is None:
            log(f"[{file_id}] No ground truth available in inference results")
            reference = None

        # Compute WER if we have both reference and hypothesis
        if reference and hypothesis and row["status"] == "success":
            ref_norm = normalize(reference, use_whisper_normalizer=use_whisper_normalizer)
            hyp_norm = normalize(hypothesis, use_whisper_normalizer=use_whisper_normalizer)

            try:
                m = jiwer.process_words(ref_norm, hyp_norm)
                wer = m.wer
                subs = m.substitutions
                dels = m.deletions
                ins = m.insertions
            except Exception as e:
                log(f"[{file_id}] WER computation failed: {e}")
                wer = None
                subs = dels = ins = None
        else:
            wer = None
            subs = dels = ins = None

        eval_results.append({
            "file_id": file_id,
            "collection_number": row["collection_number"],
            "wer": wer,
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "duration_sec": row["duration_sec"],
            "processing_time_sec": row["processing_time_sec"],
            "status": row["status"],
        })

        if wer is not None:
            log(f"[{file_id}] WER: {wer:.3f} | S: {subs}, D: {dels}, I: {ins}")

    # Create evaluation DataFrame
    df_eval_results = pd.DataFrame(eval_results)

    # Save evaluation results
    eval_path = Path(cfg["output"]["dir"]) / "evaluation_results.parquet"
    df_eval_results.to_parquet(eval_path, index=False)
    log(f"\nSaved evaluation results to {eval_path}")

    # Also save as CSV for easy inspection
    csv_path = Path(cfg["output"]["dir"]) / "evaluation_results.csv"
    df_eval_results.to_csv(csv_path, index=False)
    log(f"Saved evaluation results to {csv_path}")

    # Compute aggregate metrics (only successful files with WER)
    successful = df_eval_results[df_eval_results["wer"].notna()]

    if len(successful) > 0:
        mean_wer = successful["wer"].mean()
        median_wer = successful["wer"].median()
        total_duration = successful["duration_sec"].sum()
        total_subs = successful["substitutions"].sum()
        total_dels = successful["deletions"].sum()
        total_ins = successful["insertions"].sum()

        log(f"\n=== AGGREGATE METRICS ===")
        log(f"Files evaluated: {len(successful)}")
        log(f"Mean WER: {mean_wer:.3f}")
        log(f"Median WER: {median_wer:.3f}")
        log(f"Total duration: {total_duration:.1f}s")
        log(f"Total errors - S: {total_subs}, D: {total_dels}, I: {total_ins}")

        # Log to wandb - Add "evaluation" tag to help distinguish from inference runs
        tags = cfg["wandb"].get("tags", []).copy() if cfg["wandb"].get("tags") else []
        if "evaluation" not in tags:
            tags = ["evaluation"] + tags

        wandb.init(project=cfg["wandb"]["project"],
                   group=cfg["wandb"].get("group"),
                   job_type="evaluation",
                   config=cfg,
                   tags=tags,
                   name=f"{cfg['experiment_id']}-evaluation")  # Explicit name for easy filtering

        wandb.log({
            "mean_wer": mean_wer,
            "median_wer": median_wer,
            "files_evaluated": len(successful),
            "total_duration_sec": total_duration,
            "total_substitutions": total_subs,
            "total_deletions": total_dels,
            "total_insertions": total_ins,
        })

        # Create wandb table for per-file results
        table = wandb.Table(dataframe=df_eval_results)
        wandb.log({"evaluation_results": table})

        # Upload evaluation artifacts
        wandb.save(str(eval_path))
        wandb.save(str(csv_path))

    else:
        log("\nNo successful evaluations to report")


if __name__ == "__main__":
    main()
