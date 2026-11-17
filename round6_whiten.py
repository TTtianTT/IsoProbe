"""Round 6 phase 2 T3 (GPU, one model per call): whitening invariance and STS-B under gauge transforms.

Sentence vectors: mean pooling of V1 (position 0 excluded, as round 4), fp64; every representation is a linear (affine for V3)
map of the pooled V1, because such maps commute with mean pooling: V3 = gamma*V1 + beta, the two S1 worst-case diagonal gauges
(kappa 100), and 3 G_lin samples A = Q1 diag(s) Q2 (kappa 100, round6_lib.sample_transform seeds 0-2).
Similarity variants (statistics from the fit set only): raw, zscore (new fit set data/fit_sentences_round6.jsonl, 24000
sentences; and the round-4 fit set, 2000 sentences), ZCA whitening on the new fit set, abtt-10 on the new fit set.
ZCA: z = (e - mu) Q diag(lam^-1/2) Q^T over eigenvalues lam > 1e-12 * lam_max (pseudo-inverse on a structurally null space,
no shrinkage); the number of dropped eigenvalues and the condition number of the kept spectrum are recorded.
Writes results/round6/p2/whiten/<model>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from extract_round3 import norm_params
from measure import final_norm_module, resolve_model
from round3_lib import variants
from round6_lib import sample_transform
from sts_round4 import cosines, load_sts, spearman100

NULL_TOL = 1e-12


def zca(fit):
    mu = fit.mean(0)
    C = torch.cov((fit - mu).T, correction=0)
    lam, Q = torch.linalg.eigh((C + C.T) / 2)
    keep = lam > NULL_TOL * lam.max()
    Wh = (Q[:, keep] * lam[keep].rsqrt()) @ Q[:, keep].T
    info = {"n_dropped": int((~keep).sum()), "cond_kept": (lam[keep].max() / lam[keep].min()).item(),
            "lam_min_dropped_ratio": (lam[~keep].abs().max() / lam.max()).item() if (~keep).any() else None,
            "lam_min_kept_ratio": (lam[keep].min() / lam.max()).item()}
    return mu, Wh, info


def zca_svd(fit, rank_tol=False):
    """Same pseudo-inverse whitening from the SVD of the centred fit matrix (no squared condition number).
    rank_tol=False keeps s^2 > NULL_TOL * s_max^2 (as the eigh route); True keeps s > s_max * max(n, d) * eps (numpy matrix_rank)."""
    mu = fit.mean(0)
    _, S, Vh = torch.linalg.svd(fit - mu, full_matrices=False)
    if rank_tol:
        keep = S > S.max() * max(fit.shape) * torch.finfo(S.dtype).eps
    else:
        keep = S.pow(2) > NULL_TOL * S.max().pow(2)
    Wh = (Vh[keep].T * (fit.shape[0] ** 0.5 / S[keep])) @ Vh[keep]
    return mu, Wh, {"n_dropped": int((~keep).sum()), "cond_kept": (S[keep].max() / S[keep].min()).pow(2).item(),
                    "s2_min_dropped_ratio": (S[~keep].max() / S.max()).pow(2).item() if (~keep).any() else None,
                    "s2_min_kept_ratio": (S[keep].min() / S.max()).pow(2).item()}


def check(reps, i_new, i1, i2, gold, res):
    """Numerical check of P7-3 (configs/round6_prereg.md, T3 amendment): eigh / SVD / SVD with rank tolerance."""
    zc = {}
    routes = (("eigh", zca), ("svd", zca_svd), ("svd_ranktol", lambda f: zca_svd(f, rank_tol=True)))
    for route, fz in routes:
        for r, E in reps.items():
            mu, Wh, info = fz(E[i_new])
            Z = torch.nn.functional.normalize((E - mu) @ Wh, dim=-1)
            zc[(route, r)] = (Z[i1] * Z[i2]).sum(-1)
            res["scores"][f"{r}/{route}"] = spearman100(zc[(route, r)], gold)
            res["zca"][f"{r}/{route}"] = info
        vals = [res["scores"][f"{r}/{route}"] for r in reps]
        res.setdefault("range_all", {})[route] = max(vals) - min(vals)
        res.setdefault("max_abs_cos_diff_vs_V1", {})[route] = {
            r: (zc[(route, r)] - zc[(route, "V1")]).abs().max().item() for r in reps}


def sims(E, fit_new, fit_old, i1, i2):
    """Cosines of every similarity variant for one representation E (rows = all sentences)."""
    out = {"raw": cosines(E, fit_new, "raw", i1, i2), "zscore": cosines(E, fit_new, "zscore", i1, i2),
           "zscore_oldfit": cosines(E, fit_old, "zscore", i1, i2), "abtt10": cosines(E, fit_new, "abtt10", i1, i2)}
    mu, Wh, info = zca(fit_new)
    Z = torch.nn.functional.normalize((E - mu) @ Wh, dim=-1)
    out["whiten"] = (Z[i1] * Z[i2]).sum(-1)
    return out, info


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fit_new", default="data/fit_sentences_round6.jsonl")
    ap.add_argument("--fit_old", default="data/fit_sentences.jsonl")
    ap.add_argument("--gauge_cache", default="cache/round6/gauge/")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_pairs", type=int, default=0)
    ap.add_argument("--max_fit", type=int, default=0, help="smoke test only")
    ap.add_argument("--out", default="results/round6/p2/whiten/")
    ap.add_argument("--check", action="store_true", help="numerical check only (eigh vs SVD whitening); use --out .../whiten_check/")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    path = resolve_model(args.model)
    slug = os.path.basename(path.rstrip("/"))
    fn = os.path.join(args.out, f"{slug}.json")
    if os.path.exists(fn):
        print(f"{fn} exists, skipping", flush=True)
        return
    dev = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev, local_files_only=True).eval()
    norm = final_norm_module(model)
    gamma, beta, eps, ntype = norm_params(norm)
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0]))
    fit_new = [json.loads(l)["text"] for l in open(args.fit_new)]
    fit_old = [json.loads(l)["text"] for l in open(args.fit_old)]
    if args.max_fit:
        fit_new = fit_new[:args.max_fit]
    sts = load_sts()["stsb"]
    if args.max_pairs:
        sts = sts.iloc[:args.max_pairs]
    sents = list(dict.fromkeys(fit_new + fit_old + list(sts["sentence1"]) + list(sts["sentence2"])))
    sid = {s: i for i, s in enumerate(sents)}
    d = gamma.shape[0]
    V1 = torch.zeros(len(sents), d, dtype=torch.float64)
    for i, s in enumerate(sents):
        ids = tok(s, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
        model(input_ids=ids, attention_mask=torch.ones_like(ids))
        v = variants(cap["x"][0], gamma, beta, ntype, eps)["V1"]
        T = ids.shape[1]
        V1[i] = v[1:T if T > 1 else 1].double().mean(0).cpu()
        if (i + 1) % 5000 == 0:
            print(f"[{slug}] {i + 1}/{len(sents)} {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    V1 = V1.to(dev)
    g64, b64 = gamma.double(), (beta.double() if beta is not None else torch.zeros_like(gamma, dtype=torch.float64))
    G = torch.load(os.path.join(args.gauge_cache, f"{slug}.pt"), weights_only=False)
    reps = {"V1": V1, "V3": V1 * g64 + b64,
            "D_Amax": V1 * G["worst_s"]["k100_A_max"].to(dev), "D_Amin": V1 * G["worst_s"]["k100_A_min"].to(dev)}
    for s in range(3):
        fx, _, _ = sample_transform("lin", d, 100, s)
        reps[f"lin{s}"] = fx(V1.cpu()).to(dev)
    i_new = torch.as_tensor([sid[s] for s in fit_new], device=dev)
    i_old = torch.as_tensor([sid[s] for s in fit_old], device=dev)
    i1 = torch.as_tensor([sid[s] for s in sts["sentence1"]], device=dev)
    i2 = torch.as_tensor([sid[s] for s in sts["sentence2"]], device=dev)
    gold = torch.as_tensor(sts["score"].to_numpy().copy(), dtype=torch.float64, device=dev)
    res = {"model": slug, "d": d, "n_fit_new": len(fit_new), "n_fit_old": len(fit_old), "n_pairs": len(sts),
           "null_tol": NULL_TOL, "scores": {}, "zca": {}}
    if args.check:
        check(reps, i_new, i1, i2, gold, res)
        res["elapsed_s"] = time.time() - t0
        os.makedirs(args.out, exist_ok=True)
        json.dump(res, open(fn, "w"), indent=1)
        print(f"[{slug}] check: " + "; ".join(
            f"{k} range {v:.2e} max|cos-cos_V1| {max(res['max_abs_cos_diff_vs_V1'][k].values()):.2e} "
            f"dropped {[res['zca'][f'{r}/{k}']['n_dropped'] for r in reps]}" for k, v in res["range_all"].items()), flush=True)
        return
    cos = {}
    for r, E in reps.items():
        c, info = sims(E, E[i_new], E[i_old], i1, i2)
        res["zca"][r] = info
        for var, cv in c.items():
            cos[(r, var)] = cv
            res["scores"][f"{r}/{var}"] = spearman100(cv, gold)
    for var in ("raw", "zscore", "zscore_oldfit", "whiten", "abtt10"):
        vals = [res["scores"][f"{r}/{var}"] for r in reps]
        res.setdefault("range_all", {})[var] = max(vals) - min(vals)
        lin = [res["scores"][f"lin{s}/{var}"] for s in range(3)]
        res.setdefault("range_lin", {})[var] = max(lin) - min(lin)
    gen = torch.Generator().manual_seed(args.seed)
    boot = torch.randint(0, len(sts), (args.n_boot, len(sts)), generator=gen).to(dev)
    res["delta"] = {}
    for a, b in (("whiten", "zscore"), ("whiten", "raw"), ("zscore", "zscore_oldfit")):
        ca, cb = cos[("V1", a)], cos[("V1", b)]
        dl = np.array([spearman100(ca[bi], gold[bi]) - spearman100(cb[bi], gold[bi]) for bi in boot])
        res["delta"][f"V1/{a}-{b}"] = {"delta": res["scores"][f"V1/{a}"] - res["scores"][f"V1/{b}"],
                                       "ci_lo": float(np.percentile(dl, 2.5)), "ci_hi": float(np.percentile(dl, 97.5))}
    res["elapsed_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(fn, "w"), indent=1)
    print(f"[{slug}] whiten range {res['range_all']['whiten']:.2e}; zscore lin range {res['range_lin']['zscore']:.2f}; "
          f"V1 whiten {res['scores']['V1/whiten']:.1f} zscore {res['scores']['V1/zscore']:.1f} "
          f"raw {res['scores']['V1/raw']:.1f}; zca {res['zca']['V1']} ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
