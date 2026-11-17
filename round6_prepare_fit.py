"""Round 6 phase 2 T3: 24000 sentences (6-50 words, seed 0) cut from wikitext-103 *train* -> data/fit_sentences_round6.jsonl.

Same cutting / detokenization as prepare_fit_sentences.py (round 4), but from train (non-heading lines of the first 60M
characters), so the fit set has >= 4 d sentences for every model (d <= 5120). Used only to fit similarity statistics.
"""
import json

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from prepare_fit_sentences import detok

import re

TRAIN = [f"wikitext-103-raw-v1/train-0000{i}-of-00002.parquet" for i in range(2)]


def main(n=24000, seed=0, min_words=6, max_words=50, max_chars=60_000_000):
    files = [hf_hub_download("Salesforce/wikitext", f, repo_type="dataset", local_files_only=True) for f in TRAIN]
    cands, seen, nch = [], set(), 0
    for text in load_dataset("parquet", data_files=files, split="train")["text"]:
        t = text.strip()
        if not t or t.startswith("="):
            continue
        nch += len(t)
        if nch > max_chars:
            break
        for s in re.split(r"(?<= \.) ", t):
            s = detok(s.strip())
            if s.endswith(" ."):
                s = s[:-2] + "."
            if min_words <= len(s.split()) <= max_words and s not in seen:
                seen.add(s)
                cands.append(s)
    pick = np.random.default_rng(seed).choice(len(cands), size=n, replace=False)
    with open("data/fit_sentences_round6.jsonl", "w") as f:
        for i in sorted(pick):
            f.write(json.dumps({"text": cands[i]}) + "\n")
    print(f"{len(cands)} candidate sentences, wrote {n} to data/fit_sentences_round6.jsonl")


if __name__ == "__main__":
    main()
