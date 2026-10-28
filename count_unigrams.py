"""Round 3, step 3b (CPU only): unigram counts on a fixed prefix of wikitext-103 train, per distinct tokenizer.

Models whose tokenizer.json has the same model / normalizer / pre-tokenizer are counted once.
Writes cache/unigrams/<group>.npy and results/round3/unigram_groups.json.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

TRAIN = [f"wikitext-103-raw-v1/train-0000{i}-of-00002.parquet" for i in range(2)]


def tok_group(path):
    j = json.load(open(os.path.join(path, "tokenizer.json")))
    key = json.dumps([j.get("model"), j.get("normalizer"), j.get("pre_tokenizer")], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models.txt")
    ap.add_argument("--max_chars", type=int, default=90_000_000, help="same text prefix for every tokenizer")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--out", default="results/round3/unigram_groups.json")
    ap.add_argument("--cache", default="cache/unigrams/")
    args = ap.parse_args()

    files = [hf_hub_download("Salesforce/wikitext", f, repo_type="dataset", local_files_only=True) for f in TRAIN]
    lines, n = [], 0
    for t in load_dataset("parquet", data_files=files, split="train")["text"]:
        if t.strip():
            lines.append(t)
            n += len(t)
            if n >= args.max_chars:
                break
    print(f"{len(lines)} lines, {n} chars", flush=True)

    paths = [l.strip() for l in open(args.models) if l.strip()]
    groups = {}
    for p in paths:
        groups.setdefault(tok_group(p), []).append(p)
    os.makedirs(args.cache, exist_ok=True)
    info = {"n_lines": len(lines), "n_chars": n, "groups": {}, "model_group": {}}
    for g, ps in groups.items():
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(ps[0], local_files_only=True)
        counts = np.zeros(len(tok), dtype=np.int64)
        for i in range(0, len(lines), args.batch):
            ids = tok(lines[i:i + args.batch], add_special_tokens=False)["input_ids"]
            counts += np.bincount(np.concatenate([np.asarray(x, dtype=np.int64) for x in ids]), minlength=len(tok))
        np.save(os.path.join(args.cache, f"{g}.npy"), counts)
        info["groups"][g] = {"models": [os.path.basename(p) for p in ps], "n_tokens": int(counts.sum()),
                             "vocab_len": len(tok), "n_types_ge5": int((counts >= 5).sum())}
        for p in ps:
            info["model_group"][os.path.basename(p)] = g
        print(f"group {g}: {[os.path.basename(p) for p in ps]} -> {counts.sum()} tokens, "
              f"{(counts >= 5).sum()} types with count>=5 ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(info, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
