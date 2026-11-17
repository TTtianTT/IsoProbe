"""Round 6 S6 downloads (login node only): PolyPythias final checkpoints listed in configs/round6_polypythia.tsv
into /dataset1/zailong/models/ckpts/<repo name>/<revision>/. These repos ship pytorch_model.bin (no safetensors).
Total estimated before download (HfApi file metadata): 9 x 166 MB (70m) + 9 x 375 MB (160m) ~= 4.9 GB.

Usage: HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 python round6_download.py
"""
import argparse
import os
import sys
import time

from huggingface_hub import snapshot_download

from download_round4 import ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round6_polypythia.tsv")
    args = ap.parse_args()
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    for repo, rev in rows:
        out = ckpt_dir(repo, rev)
        done = os.path.join(out, ".complete")
        if os.path.exists(done):
            continue
        t0 = time.time()
        for attempt in range(3):
            try:
                snapshot_download(repo, revision=rev, local_dir=out,
                                  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"])
                open(done, "w").close()
                print(f"{repo}@{rev} -> {out} ({time.time() - t0:.0f}s)", flush=True)
                break
            except Exception as e:  # transient hub errors: retry, then move on to the next repo
                print(f"{repo}@{rev} attempt {attempt}: {e}", file=sys.stderr, flush=True)
                time.sleep(30)
        else:
            print(f"FAILED {repo}@{rev}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
