import argparse, yaml, jiwer, wandb
from pathlib import Path

def normalize(s):
    tx = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                        jiwer.Strip(), jiwer.RemoveMultipleSpaces()])
    return tx(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hyp", required=True)     # outputs/.../hyp_*.txt
    ap.add_argument("--ref", required=True)     # your GT file (one line per file)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ref_text = normalize(Path(args.ref).read_text())
    hyp_text = normalize(Path(args.hyp).read_text())

    m = jiwer.process_words(ref_text, hyp_text)
    print("WER:", m.wer, "S:", m.substitutions, "D:", m.deletions, "I:", m.insertions)

    wandb.init(project=cfg["wandb"]["project"], group=cfg["wandb"].get("group"),
               job_type="evaluation", config=cfg, tags=cfg["wandb"].get("tags", []))
    wandb.log({"wer": m.wer, "subs": m.substitutions, "dels": m.deletions, "ins": m.insertions})

if __name__ == "__main__":
    main()