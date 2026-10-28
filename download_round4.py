"""Round 4 downloads (login node only): STS test sets into HF_HOME, and the training-dynamics checkpoints in
configs/round4_ckpts.tsv into /dataset1/zailong/models/ckpts/<model>/<revision>/ (safetensors + json/tokenizer only).

Usage: python download_round4.py sts | ckpts [--only pythia-70m]
"""
import argparse
import os
import sys
import time

from huggingface_hub import hf_hub_download, snapshot_download

CKPT_ROOT = "/dataset1/zailong/models/ckpts"
STS = [("sentence-transformers/stsb", "data/test-00000-of-00001.parquet"), ("mteb/sickr-sts", "test.jsonl.gz")]


def ckpt_dir(repo, rev):
    return os.path.join(CKPT_ROOT, repo.split("/")[1], rev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sts", "ckpts"])
    ap.add_argument("--list", default="configs/round4_ckpts.tsv")
    ap.add_argument("--only", default=None, help="substring filter on the model name")
    args = ap.parse_args()
    if args.what == "sts":
        for repo, f in STS:
            print(hf_hub_download(repo, f, repo_type="dataset"), flush=True)
        return
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    for repo, rev in rows:
        if args.only and args.only not in repo:
            continue
        out = ckpt_dir(repo, rev)
        done = os.path.join(out, ".complete")
        if os.path.exists(done):
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                snapshot_download(repo, revision=rev, local_dir=out,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
                break
            except Exception as e:  # transient hub errors: retry
                print(f"{repo}@{rev} attempt {attempt}: {e}", file=sys.stderr, flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"failed {repo}@{rev}")
        open(done, "w").close()
        print(f"{repo}@{rev} -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
