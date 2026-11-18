"""Round 6 phase 3 U2: extend the phase-2 fit set (24000 sentences, round6_prepare_fit.py) to 8 d = 40960 for d <= 5120.

Same candidate list and the same generator: rng = default_rng(0); the first draw rng.choice(N, 24000, replace=False) must
reproduce data/fit_sentences_round6.jsonl exactly (asserted); the same rng then draws 16960 more from the remaining
candidates, and finally rng.permutation(40960) fixes a random order. The fit-size subsets n = d, 2d, 4d, 8d are prefixes of
this order. Writes data/fit_sentences_round6_8d.jsonl ({"text", "in_p2"}).
"""
import json
import re

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from prepare_fit_sentences import detok
from round6_prepare_fit import TRAIN


def candidates(min_words=6, max_words=50, max_chars=60_000_000):
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
    return cands


def main(n_total=40960):
    cands = candidates()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(cands), size=24000, replace=False)
    old = [json.loads(l)["text"] for l in open("data/fit_sentences_round6.jsonl")]
    assert [cands[i] for i in sorted(pick)] == old, "phase-2 fit set not reproduced"
    rest = np.setdiff1d(np.arange(len(cands)), pick)
    extra = rng.choice(rest, size=n_total - 24000, replace=False)
    allidx = np.concatenate([np.sort(pick), np.sort(extra)])
    order = rng.permutation(n_total)
    inp2 = set(pick.tolist())
    with open("data/fit_sentences_round6_8d.jsonl", "w") as f:
        for j in order:
            i = int(allidx[j])
            f.write(json.dumps({"text": cands[i], "in_p2": i in inp2}) + "\n")
    print(f"{len(cands)} candidates; reproduced 24000; wrote {n_total} to data/fit_sentences_round6_8d.jsonl")


if __name__ == "__main__":
    main()
