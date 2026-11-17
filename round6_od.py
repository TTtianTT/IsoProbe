"""Round 6 S3 (CPU): outlier dimensions (Macocco et al. 2025 rule) under different gauges.

OD(H) = { j : median_i |H_ij| > 0.99-quantile of all |H_ij| } on the bank tokens without position 0
(rule copied from imacocco/llms_outlier_dimensions src/evaluate.py; see related_round6.md).
Representations: V3, V1, and the S1 worst-case diagonal gauges (kappa 100) D_Amax V1, D_Amin V1;
exploratory: V2 = gamma V1 and h*_unif (gauge metric, d+1 coordinates).
Overlaps of OD(V3) with the round-3 top-10 ||ubar||^2 dims and with the top-10 dims by additive functional share
Cov(W[:,k] m_k, delta_all) / Var(delta_all) over tokens with count >= 5 (round-3 convention).
Writes results/round6/od/<model>.json; the summary table is made by round6_summary.py.
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from round6_lib import MODELS12, hstar, load_bank, load_head

MAIN = ["V3", "V1", "D_Amax", "D_Amin"]


def od_set(H):
    A = np.abs(H.numpy())
    med = np.median(A, axis=0)
    thr = np.quantile(A, 0.99)
    idx = np.flatnonzero(med > thr)
    idx = idx[np.argsort(-med[idx])]
    return idx.tolist(), float(thr), float(np.sort(med)[-1])


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


@torch.no_grad()
def run(slug, args):
    fn = os.path.join(args.out, f"{slug}.json")
    if os.path.exists(fn):
        return
    gfn = os.path.join(args.gauge_cache, f"{slug}.pt")
    if not os.path.exists(gfn):
        print(f"[{slug}] no gauge cache yet, skipping", flush=True)
        return
    G = torch.load(gfn, weights_only=False)
    B = load_bank(slug)
    keep = torch.as_tensor(B["pos"] != 0)
    V1, V3 = B["V1"][keep], B["V3"][keep]
    reps = {"V3": V3, "V1": V1, "D_Amax": V1 * G["worst_s"]["k100_A_max"], "D_Amin": V1 * G["worst_s"]["k100_A_min"],
            "V2": V1 * B["gamma"], "hstar_unif": hstar(V1, G["R_unif"])}
    ods = {}
    for k, H in reps.items():
        idx, thr, mx = od_set(H)
        ods[k] = {"dims": idx, "n": len(idx), "threshold_q99": thr, "max_median": mx}
    jac = {f"{a}|{b}": jaccard(ods[a]["dims"], ods[b]["dims"]) for a, b in itertools.combinations(MAIN, 2)}
    jac["V2|V3"] = jaccard(ods["V2"]["dims"], ods["V3"]["dims"])

    # overlaps with round-3 top-10 ||ubar||^2 dims and top-10 additive-share dims
    top10_u = json.load(open("results/round3/analysis.json"))["step2"][slug]["top10_dims"]
    W, _ = load_head(slug)
    m = B["raw"]["m"].double()
    groups = json.load(open("results/round3/unigram_groups.json"))
    counts = np.load(f"cache/unigrams/{groups['model_group'][slug]}.npy")
    kv = np.zeros(W.shape[0], dtype=bool)
    n = min(W.shape[0], len(counts))
    kv[:n] = counts[:n] >= 5
    Wk = W[torch.as_tensor(kv)]
    dall = Wk @ m
    a = dall - dall.mean()
    share = m * ((Wk - Wk.mean(0)).T @ a) / (a.pow(2).sum())  # Cov(W[:,k] m_k, delta_all) / Var(delta_all)
    top10_add = torch.argsort(share, descending=True)[:10].tolist()
    od3 = set(ods["V3"]["dims"])
    res = {"model": slug, "n_tokens": int(keep.sum()), "od": ods, "jaccard": jac,
           "top10_ubar_V3": top10_u, "top10_additive": top10_add,
           "additive_share_top10": share[top10_add].tolist(), "additive_share_sum": share.sum().item(),
           "overlap_OD_V3_top10_ubar": len(od3 & set(top10_u)),
           "overlap_OD_V3_top10_additive": len(od3 & set(top10_add)),
           "gamma_max_over_median": (B["gamma"].abs().max() / B["gamma"].abs().median()).item()}
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(fn, "w"), indent=1)
    print(f"[{slug}] |OD| " + " ".join(f"{k}={ods[k]['n']}" for k in reps) + f"; J(V1,V3)={jac['V3|V1']:.2f}; "
          f"overlap ubar {res['overlap_OD_V3_top10_ubar']}, additive {res['overlap_OD_V3_top10_additive']} "
          f"(sum of shares {res['additive_share_sum']:.3f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS12))
    ap.add_argument("--gauge_cache", default="cache/round6/gauge/")
    ap.add_argument("--out", default="results/round6/od/")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    for slug in args.models.split(","):
        run(slug, args)


if __name__ == "__main__":
    main()
