"""Round 5 items 2-3 (CPU, no GPU). For every Pythia checkpoint with a bank (round 4 and round 5):

  W_U spectrum (first 50277 rows, bf16-rounded head weight as loaded by the model, fp32 SVD):
    sigma1^2 / sum sigma^2, sigma1 / sigma2, singular entropy as KL(p || uniform) with p from sigma and from sigma^2
    (Godey et al. 2024 define it only in words), effective rank exp(H(sigma / sum sigma)) (RankMe, as in measure.py);
  alignment: |cos(ubar_V3, v1(W_U))| and |cos(ubar_V1, v1(W_U diag(gamma)))| plus the squared share of each ubar in
    the top-1/5/10 right-singular subspaces (ubar = mean of the L2-normalized bank tokens, unit-normalized);
  early phase: Spearman(W_U m, log unigram count) over tokens with count >= 5.
Also H_uni: cross-entropy of the add-1 smoothed train unigram model on the evaluation targets (sequences 500-999).
Writes results/round5/spec/<model>/<rev>.json and results/round5/h_uni.json.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open

from download_round4 import ckpt_dir
from measure import load_samples
from round3_lib import spearman, variants

N_VOC = 50277  # Pythia tokenizer vocabulary (the head has 50304 rows; the rest are padding)
HEAD_KEYS = ("embed_out.weight", "lm_head.weight")


def load_head(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = {k: os.path.join(path, f) for k, f in json.load(open(idx))["weight_map"].items()}
    else:
        f = os.path.join(path, "model.safetensors")
        with safe_open(f, "pt") as sf:
            wm = {k: f for k in sf.keys()}
    key = next(k for k in HEAD_KEYS if k in wm)
    with safe_open(wm[key], "pt") as sf:
        return sf.get_tensor(key).to(torch.bfloat16).float()  # the GPU jobs load the model in bf16


def spec_stats(W):
    U, s, Vh = torch.linalg.svd(W, full_matrices=False)
    s = s.double()
    r = len(s)
    out = {"s1sq_over_sum": (s[0] ** 2 / (s ** 2).sum()).item(), "s1_over_s2": (s[0] / s[1]).item(), "rank_n": r}
    for tag, q in (("sigma", s / s.sum()), ("sigma2", s ** 2 / (s ** 2).sum())):
        qq = q[q > 0]
        out[f"kl_uniform_{tag}"] = (math.log(r) + (qq * qq.log()).sum()).item()
    p = s / s.sum()
    p = p[p > 0]
    out["eff_rank"] = math.exp(-(p * p.log()).sum().item())
    return out, Vh[:10].double()


def align(u, Vh):
    u = u.double() / u.double().norm()
    proj = Vh @ u
    return {"abs_cos_v1": proj[0].abs().item(), **{f"sub{k}_sq": proj[:k].pow(2).sum().item() for k in (1, 5, 10)}}


def h_uni(counts, tok_path, args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    q = np.log((counts[:N_VOC] + 1) / (counts[:N_VOC].sum() + N_VOC))
    rows = load_samples(args.data, 0)
    nll, n = 0.0, 0
    for sid in range(args.n_est, len(rows)):
        ids = tok(rows[sid]["text"], add_special_tokens=True, truncation=True, max_length=args.max_len)["input_ids"]
        y = np.asarray(ids[1:])
        nll += -q[y].sum()
        n += len(y)
    return nll / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_est", type=int, default=500)
    ap.add_argument("--unigram_groups", default="results/round5/unigram_groups.json")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ug = json.load(open(args.unigram_groups))
    g = next(iter(ug["groups"]))
    counts = np.load(f"cache/unigrams/{g}.npy")
    keep = counts[:N_VOC] >= 5
    jobs = []
    for rnd in ("round4", "round5"):
        root = f"cache/{rnd}_banks"
        for m in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if not m.startswith("pythia"):
                continue
            for f in sorted(os.listdir(os.path.join(root, m))):
                jobs.append((m, f[:-3], os.path.join(root, m, f)))
    hu, n = h_uni(counts, ckpt_dir("EleutherAI/pythia-70m", "step143000"), args)
    os.makedirs("results/round5", exist_ok=True)
    json.dump({"H_uni": hu, "n_targets": n, "unigram_group": g, "vocab": N_VOC}, open("results/round5/h_uni.json", "w"), indent=1)
    print(f"H_uni = {hu:.4f} nats over {n} targets", flush=True)
    for m, rev, bank_fn in jobs:
        out_fn = f"results/round5/spec/{m}/{rev}.json"
        if os.path.exists(out_fn):
            continue
        b = torch.load(bank_fn, weights_only=False)
        W = load_head(ckpt_dir(f"EleutherAI/{m}", rev))[:N_VOC]
        st, Vh = spec_stats(W)
        stf, Vhf = spec_stats(W * b["gamma"].float())
        V = variants(b["x"], b["gamma"], b["beta"], b["norm_type"], b["eps"])
        u3 = torch.nn.functional.normalize(V["V3"], dim=-1).mean(0)
        u1 = torch.nn.functional.normalize(V["V1"], dim=-1).mean(0)
        dall = b["delta"]["all"][:N_VOC].numpy()
        rho = spearman(dall[keep], np.log(counts[:N_VOC][keep]))[0]
        res = {"model": m, "revision": rev, "bank": bank_fn, "d": W.shape[1], "W_U": st, "W_U_folded": stf,
               "align_V3_WU": align(u3, Vh), "align_V1_WUgamma": align(u1, Vhf),
               "random_abs_cos": math.sqrt(2 / (math.pi * W.shape[1])),
               "spearman_WUm_logfreq": rho, "n_tokens_ge5": int(keep.sum())}
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        json.dump(res, open(out_fn, "w"), indent=1)
        print(f"[{m}@{rev}] s1^2/sum={st['s1sq_over_sum']:.3f} cos3={res['align_V3_WU']['abs_cos_v1']:.3f} "
              f"cos1={res['align_V1_WUgamma']['abs_cos_v1']:.3f} rho_freq={rho:+.3f}", flush=True)


if __name__ == "__main__":
    main()
