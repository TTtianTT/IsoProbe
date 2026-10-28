"""Round 4 A: 2000 sentences cut from data/samples.jsonl at full stops -> data/fit_sentences.jsonl.

Used only to estimate the mean / std / principal components of pooled sentence vectors (never STS data).
WikiText's tokenization artifacts (" @-@ ", " , ", " 's", ...) are undone so the sentences look like the STS text.
"""
import json
import re

import numpy as np

DETOK = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" ; ", "; "), (" : ", ": "),
         (" 's", "'s"), (" n't", "n't"), (" ( ", " ("), (" ) ", ") "), (" ? ", "? "), (" ! ", "! ")]


def detok(s):
    for a, b in DETOK:
        s = s.replace(a, b)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    return re.sub(r"\s+", " ", s).strip()


def main(n=2000, seed=0, min_words=6, max_words=50):
    cands, seen = [], set()
    for line in open("data/samples.jsonl"):
        text = json.loads(line)["text"]
        for s in re.split(r"(?<= \.) ", text):  # WikiText writes the full stop as " ."
            s = detok(s.strip())
            if s.endswith(" ."):
                s = s[:-2] + "."
            if min_words <= len(s.split()) <= max_words and s not in seen:
                seen.add(s)
                cands.append(s)
    pick = np.random.default_rng(seed).choice(len(cands), size=n, replace=False)
    with open("data/fit_sentences.jsonl", "w") as f:
        for i in sorted(pick):
            f.write(json.dumps({"text": cands[i]}) + "\n")
    print(f"{len(cands)} candidate sentences, wrote {n} to data/fit_sentences.jsonl")


if __name__ == "__main__":
    main()
