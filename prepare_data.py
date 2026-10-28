"""Sample 1000 wikitext-103 test paragraphs (>=100 words, seed 0) into data/samples.jsonl.

A "paragraph" is one non-empty, non-heading line of the raw wikitext file
(headings look like " = Title = "). The raw text is kept verbatim, including
wikitext's " @-@ " / " , " tokenization artifacts.
"""
import argparse
import json
import os
import random

from datasets import load_dataset
from huggingface_hub import hf_hub_download

TEST_FILE = "wikitext-103-raw-v1/test-00000-of-00001.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--min_words", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/samples.jsonl")
    args = ap.parse_args()

    # read the cached parquet directly (works with HF_HUB_OFFLINE=1)
    path = hf_hub_download("Salesforce/wikitext", TEST_FILE, repo_type="dataset", local_files_only=True)
    ds = load_dataset("parquet", data_files=path, split="train")
    paras = []
    for i, line in enumerate(ds["text"]):
        t = line.strip()
        if not t or t.startswith("="):
            continue
        if len(t.split()) >= args.min_words:
            paras.append((i, t))
    print(f"{len(paras)} candidate paragraphs with >= {args.min_words} words")

    rng = random.Random(args.seed)
    picked = sorted(rng.sample(range(len(paras)), args.n))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for k, j in enumerate(picked):
            line_idx, text = paras[j]
            f.write(json.dumps({"id": k, "line": line_idx, "n_words": len(text.split()), "text": text}) + "\n")
    print(f"wrote {args.n} samples to {args.out}")


if __name__ == "__main__":
    main()
