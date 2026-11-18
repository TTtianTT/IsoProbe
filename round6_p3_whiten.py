"""Round 6 phase 3 U2 (GPU, one model per call): whitening on 7 STS sets x 3 poolings (configs/round6_prereg.md, U2).

Embedding (bf16 forward, as phase-2 T3): V1 at the final-norm input, pooled per sentence -> fp64 -> cached as fp32 in
cache/round6/p3/whiten_emb/<model>.pt (pools: mean without position 0, last token, PromptEOL last token). Sentences: the
phase-2 fit set (24000, main), the first min(8d, 40960) of data/fit_sentences_round6_8d.jsonl (fit-size analysis) and the
29023 sentences of STS12-16 / STS-B / SICK-R test.
Analysis (fp64, per pool): 7 representations (V1, V3, two S1 worst-case diagonal gauges, 3 G_lin, as phase-2 T3) x
{raw, zscore, whiten (SVD + numerical rank tolerance), whiten_eigh (1e-12, record), abtt10, whiten_lw (Ledoit-Wolf)},
whitening at fit sizes d, 2d, 4d, 8d; P8-6 bootstrap; covariance spectrum of the V1 sentence vectors.
Writes results/round6/p3/whiten_wide/<model>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from extract_round3 import norm_params
from measure import final_norm_module, resolve_model
from round3_lib import variants
from round6_lib import sample_transform
from round6_whiten import zca, zca_svd
from sts_round4 import TEMPLATE, load_sts, spearman100, transform

DATASETS = ["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"]
POOLS = ["mean", "last", "eol"]
FIT_MULT = (1, 2, 4, 8)


def load_all_sts():
    from huggingface_hub import hf_hub_download
    out = load_sts()
    for k in (12, 13, 14, 15, 16):
        p = hf_hub_download(f"mteb/sts{k}-sts", "test.jsonl.gz", repo_type="dataset", local_files_only=True)
        out[f"sts{k}"] = pd.read_json(p, lines=True)[["sentence1", "sentence2", "score"]].reset_index(drop=True)
    return {k: out[k] for k in DATASETS}


def ledoit_wolf(X):
    """Ledoit & Wolf (2004) shrinkage towards (tr S / d) I; X centred [n, d]. Returns (Sigma, rho)."""
    n, d = X.shape
    S = X.T @ X / n
    m = torch.trace(S) / d
    d2 = (S - m * torch.eye(d, dtype=X.dtype, device=X.device)).pow(2).sum()
    nrm = X.pow(2).sum(1)
    # sum_i ||x_i x_i^T - S||_F^2 = sum_i ||x_i||^4 - 2 sum_i x_i^T S x_i + n ||S||_F^2
    b2bar = (nrm.pow(2).sum() - 2 * ((X @ S) * X).sum() + n * S.pow(2).sum()) / n ** 2
    b2 = torch.minimum(b2bar, d2)
    rho = (b2 / d2).item()
    return rho * m * torch.eye(d, dtype=X.dtype, device=X.device) + (1 - rho) * S, rho


def cos_pairs(Z, i1, i2):
    Z = torch.nn.functional.normalize(Z, dim=-1)
    return (Z[i1] * Z[i2]).sum(-1)


@torch.no_grad()
def embed(args, slug, path, sents, d_needed=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev, local_files_only=True).eval()
    norm = final_norm_module(model)
    gamma, beta, eps, ntype = norm_params(norm)
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0]))
    d = gamma.shape[0]
    out = {p: torch.zeros(len(sents), d, dtype=torch.float32) for p in POOLS}
    for i, s in enumerate(sents):
        for mode in ("plain", "eol"):
            text = TEMPLATE.format(s) if mode == "eol" else s
            ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids))
            v = variants(cap["x"][0], gamma, beta, ntype, eps)["V1"].double()
            T = ids.shape[1]
            if mode == "eol":
                out["eol"][i] = v[T - 1].float().cpu()
            else:
                out["mean"][i] = v[1:T if T > 1 else 1].mean(0).float().cpu()
                out["last"][i] = v[T - 1].float().cpu()
        if (i + 1) % 10000 == 0:
            print(f"[{slug}] embedded {i + 1}/{len(sents)} {time.time() - t0:.0f}s", flush=True)
    res = {"sents": sents, **out, "gamma": gamma.cpu(), "beta": beta.cpu() if beta is not None else None,
           "embed_s": time.time() - t0}
    del model
    torch.cuda.empty_cache()
    return res


def analyse_pool(V1, g64, b64, G, d, idx, pairs, gold, boot, n_fit8, dev):
    """All scores for one pool. V1 [N, d] fp64 on dev."""
    reps = {"V1": V1, "V3": V1 * g64 + b64, "D_Amax": V1 * G["worst_s"]["k100_A_max"].to(dev),
            "D_Amin": V1 * G["worst_s"]["k100_A_min"].to(dev)}
    for s in range(3):
        fx, _, _ = sample_transform("lin", d, 100, s)
        reps[f"lin{s}"] = fx(V1.cpu()).to(dev)
    i_fit, i_fit8 = idx["fit"], idx["fit8"]
    out = {"scores": {}, "zca": {}, "lw_rho": {}}
    cos_v1 = {}
    for r, E in reps.items():
        fit = E[i_fit]
        Zs = {"raw": E, "zscore": transform(E, fit, "zscore"), "abtt10": transform(E, fit, "abtt10")}
        mu, Wh, info = zca_svd(fit, rank_tol=True)
        Zs["whiten"] = (E - mu) @ Wh
        out["zca"][f"{r}/whiten"] = info
        mu, Wh, info = zca(fit)
        Zs["whiten_eigh"] = (E - mu) @ Wh
        out["zca"][f"{r}/whiten_eigh"] = info
        Xc = fit - fit.mean(0)
        Sig, rho = ledoit_wolf(Xc)
        lam, Q = torch.linalg.eigh((Sig + Sig.T) / 2)
        Zs["whiten_lw"] = (E - fit.mean(0)) @ ((Q * lam.clamp_min(1e-300).rsqrt()) @ Q.T)
        out["lw_rho"][r] = rho
        for k in FIT_MULT:
            n = k * d
            if n > n_fit8:
                continue
            f8 = E[i_fit8[:n]]
            mu, Wh, info = zca_svd(f8, rank_tol=True)
            Zs[f"whiten_n{k}d"] = (E - mu) @ Wh
            out["zca"][f"{r}/whiten_n{k}d"] = info
        for var, Z in Zs.items():
            for ds, (i1, i2) in pairs.items():
                c = cos_pairs(Z, i1, i2)
                out["scores"][f"{r}/{var}/{ds}"] = spearman100(c, gold[ds])
                if r == "V1" and var in ("whiten", "zscore", "raw"):
                    cos_v1[(var, ds)] = c
        del Zs
    variants_ = sorted({k.split("/")[1] for k in out["scores"]})
    out["range"] = {f"{var}/{ds}": max(out["scores"][f"{r}/{var}/{ds}"] for r in reps) - min(out["scores"][f"{r}/{var}/{ds}"] for r in reps)
                    for var in variants_ for ds in pairs if all(f"{r}/{var}/{ds}" in out["scores"] for r in reps)}
    out["range_lin"] = {f"{var}/{ds}": max(out["scores"][f"lin{s}/{var}/{ds}"] for s in range(3)) - min(out["scores"][f"lin{s}/{var}/{ds}"] for s in range(3))
                        for var in ("zscore", "raw", "whiten") for ds in pairs}
    # P8-6 (and the same statistic for the other pools): mean over datasets of whiten - zscore on V1, paired bootstrap
    dl = []
    for b in range(boot["n"]):
        dd = []
        for ds in pairs:
            bi = boot[ds][b]
            dd.append(spearman100(cos_v1[("whiten", ds)][bi], gold[ds][bi]) - spearman100(cos_v1[("zscore", ds)][bi], gold[ds][bi]))
        dl.append(np.mean(dd))
    obs = np.mean([out["scores"][f"V1/whiten/{ds}"] - out["scores"][f"V1/zscore/{ds}"] for ds in pairs])
    out["whiten_minus_zscore_mean7"] = {"delta": float(obs), "ci_lo": float(np.percentile(dl, 2.5)), "ci_hi": float(np.percentile(dl, 97.5))}
    for var in ("raw", "zscore", "whiten", "abtt10", "whiten_lw"):
        out.setdefault("V1_mean7", {})[var] = float(np.mean([out["scores"][f"V1/{var}/{ds}"] for ds in pairs]))
    # covariance spectrum of the V1 sentence vectors on the main fit set
    X = V1[i_fit] - V1[i_fit].mean(0)
    s = torch.linalg.svdvals(X)
    p = s / s.sum()
    p = p[p > 0]
    lam = s.pow(2)
    keep = s > s.max() * max(X.shape) * torch.finfo(s.dtype).eps
    out["spectrum"] = {"rankme": float(torch.exp(-(p * p.log()).sum())), "rankme_over_d": float(torch.exp(-(p * p.log()).sum()) / d),
                       "cond_kept": float(lam[keep].max() / lam[keep].min()), "n_kept": int(keep.sum()),
                       "top1_share": float(lam[0] / lam.sum()), "top10_share": float(lam[:10].sum() / lam.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fit", default="data/fit_sentences_round6.jsonl")
    ap.add_argument("--fit8", default="data/fit_sentences_round6_8d.jsonl")
    ap.add_argument("--gauge_cache", default="cache/round6/gauge/")
    ap.add_argument("--emb_cache", default="cache/round6/p3/whiten_emb/")
    ap.add_argument("--out", default="results/round6/p3/whiten_wide/")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--max_fit", type=int, default=0, help="smoke test only")
    ap.add_argument("--max_pairs", type=int, default=0, help="smoke test only")
    args = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    path = resolve_model(args.model)
    slug = os.path.basename(path.rstrip("/"))
    fn = os.path.join(args.out, f"{slug}.json")
    if os.path.exists(fn):
        print(f"{fn} exists, skipping")
        return
    t0 = time.time()
    dev = "cuda:0"
    G = torch.load(os.path.join(args.gauge_cache, f"{slug}.pt"), weights_only=False)
    d = G["worst_s"]["k100_A_max"].shape[0]
    fit = [json.loads(l)["text"] for l in open(args.fit)]
    fit8 = [json.loads(l)["text"] for l in open(args.fit8)][:min(8 * d, 40960)]
    if args.max_fit:
        fit, fit8 = fit[:args.max_fit], fit8[:args.max_fit]
    sts = load_all_sts()
    if args.max_pairs:
        sts = {k: v.iloc[:args.max_pairs] for k, v in sts.items()}
    sents = list(dict.fromkeys(fit + fit8 + [s for df in sts.values() for s in list(df.sentence1) + list(df.sentence2)]))
    efn = os.path.join(args.emb_cache, f"{slug}.pt")
    if os.path.exists(efn):
        emb = torch.load(efn, weights_only=False)
        assert emb["sents"] == sents
    else:
        emb = embed(args, slug, path, sents)
        os.makedirs(args.emb_cache, exist_ok=True)
        torch.save(emb, efn + ".tmp")
        os.replace(efn + ".tmp", efn)
    sid = {s: i for i, s in enumerate(sents)}
    T = lambda xs: torch.as_tensor([sid[s] for s in xs], device=dev)  # noqa: E731
    idx = {"fit": T(fit), "fit8": T(fit8)}
    pairs = {k: (T(df.sentence1), T(df.sentence2)) for k, df in sts.items()}
    gold = {k: torch.as_tensor(df.score.to_numpy().astype(np.float64), device=dev) for k, df in sts.items()}
    gen = torch.Generator().manual_seed(0)
    boot = {"n": args.n_boot, **{k: torch.randint(0, len(df), (args.n_boot, len(df)), generator=gen).to(dev) for k, df in sts.items()}}
    g64 = emb["gamma"].double().to(dev)
    b64 = emb["beta"].double().to(dev) if emb["beta"] is not None else torch.zeros_like(g64)
    res = {"model": slug, "d": d, "n_fit": len(fit), "n_fit8": len(fit8), "n_sentences": len(sents),
           "n_pairs": {k: len(v) for k, v in sts.items()}, "embed_s": emb.get("embed_s"), "dtype_forward": "bf16",
           "dtype_analysis": "fp64", "pools": {}}
    for pool in POOLS:
        V1 = emb[pool].to(dev).double()
        res["pools"][pool] = analyse_pool(V1, g64, b64, G, d, idx, pairs, gold, boot, len(fit8), dev)
        r = res["pools"][pool]
        print(f"[{slug} {pool}] whiten max range {max(v for k, v in r['range'].items() if k.startswith('whiten/')):.2e} "
              f"raw max range {max(v for k, v in r['range'].items() if k.startswith('raw/')):.1f}; V1 mean7 {r['V1_mean7']}; "
              f"w-z {r['whiten_minus_zscore_mean7']} ({time.time() - t0:.0f}s)", flush=True)
        del V1
        torch.cuda.empty_cache()
    res["elapsed_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(fn + ".tmp", "w"), indent=1)
    os.replace(fn + ".tmp", fn)


if __name__ == "__main__":
    main()
