"""Round 6 S0: loss-only evaluation set data/eval_loss_round6.jsonl.

wikitext-103-raw-v1 validation + test; a paragraph = one non-empty, non-heading raw line (as prepare_data.py) with
>= 30 words. Test paragraphs that are also in data/samples.jsonl (rounds 1-5 bank / mean estimation) are kept
(without them the set has only ~309k predicted tokens, below the 500k target) and flagged with in_samples=true;
--drop_samples excludes them. Each row is one sequence; token counts use the Pythia tokenizer
(the only models round 6 evaluates loss on), truncated to --max_len tokens, predicted tokens = T - 1.
Only used for loss; never for fitting means, directions or heads.
"""
import argparse
import json
import os

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

FILES = {"validation": "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
         "test": "wikitext-103-raw-v1/test-00000-of-00001.parquet"}
TOK = "/dataset1/zailong/models/ckpts/pythia-70m/step143000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_words", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--drop_samples", action="store_true")
    ap.add_argument("--out", default="data/eval_loss_round6.jsonl")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(TOK, local_files_only=True)
    used = {json.loads(l)["line"] for l in open("data/samples.jsonl")}
    rows, n_pred, n_skip = [], 0, 0
    for split, f in FILES.items():
        path = hf_hub_download("Salesforce/wikitext", f, repo_type="dataset")
        for i, line in enumerate(load_dataset("parquet", data_files=path, split="train")["text"]):
            t = line.strip()
            if not t or t.startswith("=") or len(t.split()) < args.min_words:
                continue
            in_s = split == "test" and i in used
            if in_s and args.drop_samples:
                n_skip += 1
                continue
            T = min(len(tok(t, add_special_tokens=True)["input_ids"]), args.max_len)
            rows.append({"id": len(rows), "split": split, "line": i, "n_words": len(t.split()), "n_tok_pythia": T,
                         "in_samples": in_s, "text": t})
            n_pred += T - 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"{len(rows)} paragraphs ({sum(r['split'] == 'validation' for r in rows)} validation), "
          f"{n_pred} predicted Pythia tokens (max_len {args.max_len}); skipped {n_skip} round-1 sample paragraphs "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
