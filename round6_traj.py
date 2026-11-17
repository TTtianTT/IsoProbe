"""Round 6 phase 2 T1 (CPU, fp64): is the late collapse visible in the gauge-invariant geometry h*?

For every checkpoint bank listed in configs/round6_p2_traj.tsv (284 Pythia from rounds 4-5, 15 OLMo-2, 18 PolyPythias
endpoints from phase 1): A, B_raw, B_c, RankMe/d, centered top-1 variance share and ||ubar||^2 on V1, V3, h*_unif and h*_pi,
with h* = Mt^(1/2)[h; 1] from the folded head (W diag(gamma), W beta) exactly as in round6_gauge.py. W_U: all head rows
(bf16-rounded as loaded by the GPU jobs); pi = round-3/5 unigram counts of the model's tokenizer (0 beyond the tokenizer).
Writes results/round6/p2/traj/<model>/<rev>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from download_round4 import ckpt_dir
from round5_cpu import load_head as load_head_st
from round6_lib import DT, _normalize64, gauge_root, hstar, metrics, seq_index
from round6_poly import head_from_bin

UNIGRAMS = {"pythia": "cache/unigrams/8fed1dc1b157.npy"}  # same vocabulary for PolyPythias (checked); OLMo-2 from T1 job


def unigram_counts(name):
    if name.startswith("OLMo-2"):
        g = json.load(open("results/round6/p2/unigram_groups_olmo2.json"))["model_group"]["main"]
        return np.load(f"cache/unigrams/{g}.npy")
    return np.load(UNIGRAMS["pythia"])


def ubar_sq(X):
    return torch.nn.functional.normalize(X, dim=-1).mean(0).pow(2).sum().item()


@torch.no_grad()
def run(name, rev, bank_fn, out_fn):
    t0 = time.time()
    b = torch.load(bank_fn, weights_only=False, map_location="cpu")
    x = b["x"].to(DT)
    g = b["gamma"].to(DT)
    be = b["beta"].to(DT) if b["beta"] is not None else None
    V1 = _normalize64(x, b["norm_type"], b["eps"])
    V3 = g * V1 + (be if be is not None else 0)
    sidx = seq_index(np.asarray(b["seq"]))
    path = ckpt_dir(f"EleutherAI/{name}" if name.startswith("pythia") else f"allenai/{name}", rev)
    W = (head_from_bin(path) if os.path.exists(os.path.join(path, "pytorch_model.bin")) else load_head_st(path)).to(DT)
    d = W.shape[1]
    c = unigram_counts(name).astype(np.float64)
    pi = np.zeros(W.shape[0])
    n = min(W.shape[0], len(c))
    pi[:n] = c[:n]
    pi = torch.as_tensor(pi / pi.sum(), dtype=DT)
    Wf, bf = W * g, (W @ be if be is not None else None)
    res = {"model": name, "revision": rev, "bank": bank_fn, "d": d, "vocab_rows": W.shape[0]}
    for tag, H in (("V1", V1), ("V3", V3)):
        res[tag] = {**metrics(H, sidx), "ubar_sq": ubar_sq(H)}
    for pv, w in (("hstar_unif", None), ("hstar_pi", pi)):
        R, ei = gauge_root(Wf, bf, w)
        Hs = hstar(V1, R)
        res[pv] = {**metrics(Hs, sidx, d), "ubar_sq": ubar_sq(Hs), "n_neg_eig": ei["n_neg_eig"]}
    res["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    json.dump(res, open(out_fn, "w"), indent=1)
    print(f"[{name}@{rev}] A V1 {res['V1']['A']:.3f} V3 {res['V3']['A']:.3f} h*u {res['hstar_unif']['A']:.3f} "
          f"h*pi {res['hstar_pi']['A']:.3f}; top1 h*u {res['hstar_unif']['top1']:.3f} ({res['elapsed_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round6_p2_traj.tsv")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="results/round6/p2/traj/")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    k, n = map(int, args.shard.split("/"))
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()][k::n]
    for name, rev, bank_fn in rows:
        out_fn = os.path.join(args.out, name, f"{rev}.json")
        if os.path.exists(out_fn):
            continue
        run(name, rev, bank_fn, out_fn)


if __name__ == "__main__":
    main()
