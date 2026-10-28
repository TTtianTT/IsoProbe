"""Round 4 A (GPU, one model per call): STS-B / SICK-R with V1 vs V3 sentence embeddings.

Pooling: mean over the last layer without position 0, or PromptEOL (last token of a one-word prompt).
V1 / V3 follow round3_lib.variants (computed per token from the final-norm input x, then pooled); V3 is the
reconstruction gamma*V1 + beta (in fp64), which matched hidden_states[-1] to <= 2.5e-3 in round 3 (re-checked here as 0a).
Similarity variants use statistics fitted on data/fit_sentences.jsonl only: raw, center, zscore, abtt-1/3/10.
Also: a layer sweep (raw / center on every hidden_states[l]) and paired bootstrap CIs for V1 - V3.
Writes results/round4/sts/<slug>.json.
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

STS = {"stsb": ("sentence-transformers/stsb", "data/test-00000-of-00001.parquet"),
       "sickr": ("mteb/sickr-sts", "test.jsonl.gz")}
TEMPLATE = 'This sentence : "{}" means in one word:"'
VARIANTS = ["raw", "center", "zscore", "abtt1", "abtt3", "abtt10"]
POOLS = ["mean", "eol"]


def load_sts():
    from huggingface_hub import hf_hub_download
    out = {}
    for name, (repo, f) in STS.items():
        p = hf_hub_download(repo, f, repo_type="dataset", local_files_only=True)
        df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_json(p, lines=True)
        out[name] = df[["sentence1", "sentence2", "score"]].reset_index(drop=True)
    return out


def avg_rank(x):
    """1-based ranks with ties averaged (as in Spearman's rho)."""
    s, idx = torch.sort(x)
    _, inv, counts = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0)
    r = (ends - counts + ends - 1).double() / 2 + 1
    out = torch.empty(x.shape, dtype=torch.float64, device=x.device)
    out[idx] = r[inv]
    return out


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def spearman100(x, y):
    return 100 * pearson(avg_rank(x), avg_rank(y))


def transform(E, fit, variant):
    """Apply a similarity variant with statistics from the fit set (all fp64)."""
    if variant == "raw":
        return E
    mu = fit.mean(0)
    if variant == "center":
        return E - mu
    if variant == "zscore":
        return (E - mu) / fit.std(0, unbiased=False).clamp_min(1e-30)
    k = int(variant[4:])
    U = torch.linalg.svd(fit - mu, full_matrices=False).Vh[:k].T  # [d, k]
    Z = E - mu
    return Z - (Z @ U) @ U.T


def cosines(E, fit, variant, i1, i2):
    Z = torch.nn.functional.normalize(transform(E, fit, variant), dim=-1)
    return (Z[i1] * Z[i2]).sum(-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fit", default="data/fit_sentences.jsonl")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--max_pairs", type=int, default=0, help="smoke test: use only the first N pairs per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/round4/sts/")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="subset of similarity variants (round 5: raw,center,zscore)")
    ap.add_argument("--no_layers", action="store_true", help="skip the layer sweep")
    ap.add_argument("--out_name", default=None, help="output file stem (default: model directory name)")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    path = resolve_model(args.model)
    slug = os.path.basename(path.rstrip("/"))
    variants_run = args.variants.split(",")
    dev = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev,
                                                 local_files_only=True).eval()
    norm = final_norm_module(model)
    gamma, beta, eps, ntype = norm_params(norm)
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0]))
    L = model.config.num_hidden_layers

    fit = [json.loads(l)["text"] for l in open(args.fit)]
    sts = load_sts()
    if args.max_pairs:
        sts = {k: v.iloc[:args.max_pairs] for k, v in sts.items()}
    sents = list(dict.fromkeys(fit + [s for df in sts.values() for c in ("sentence1", "sentence2") for s in df[c]]))
    sid = {s: i for i, s in enumerate(sents)}
    n_fit = len(fit)
    print(f"[{slug}] {len(sents)} unique sentences ({n_fit} fit); "
          + ", ".join(f"{k}: {len(v)} pairs" for k, v in sts.items()), flush=True)

    d = gamma.shape[0]
    emb = {p: {v: torch.zeros(len(sents), d, dtype=torch.float64) for v in ("V1", "V3", "V3hs")} for p in POOLS}
    lay = None if args.no_layers else {p: torch.zeros(L + 1, len(sents), d, dtype=torch.float16)
                                       for p in POOLS}  # layer sweep (raw h_l)
    rec_err, n_tok = 0.0, {p: [] for p in POOLS}
    for i, s in enumerate(sents):
        for p in POOLS:
            text = s if p == "mean" else TEMPLATE.format(s)
            ids = tok(text, add_special_tokens=True, return_tensors="pt")["input_ids"].to(dev)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
            V = variants(cap["x"][0], gamma, beta, ntype, eps)
            # V3 in fp64 from the fp32 V1, so pooled V3 is exactly gamma * pooled V1 + beta: in fp32 the large
            # mean dims (e.g. GPT-2 dim 496, ~200 +- 0.5) lose ~1e-5 relative precision after centering (check A1)
            V["V3"] = gamma.double() * V["V1"].double() + (beta.double() if beta is not None else 0)
            hs = out.hidden_states[-1][0].float()
            rec_err = max(rec_err, ((V["V3"].float() - hs).norm() / hs.norm()).item())
            T = ids.shape[1]
            sel = slice(1, T) if (p == "mean" and T > 1) else (slice(T - 1, T) if p == "eol" else slice(0, T))
            for v, H in (("V1", V["V1"]), ("V3", V["V3"]), ("V3hs", hs)):
                emb[p][v][i] = H[sel].double().mean(0).cpu()  # pool in fp64 (see check A1)
            if lay is not None:
                lay[p][:, i] = torch.stack([h[0, sel].float().mean(0) for h in out.hidden_states]).half().cpu()
            n_tok[p].append(T)
        if (i + 1) % 1000 == 0:
            print(f"[{slug}] {i + 1}/{len(sents)} sentences {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    gen = torch.Generator().manual_seed(args.seed)
    res = {"model": slug, "path": path, "norm_type": ntype, "template": TEMPLATE, "n_fit": n_fit,
           "n_sentences": len(sents), "mean_tokens": {p: float(np.mean(n_tok[p])) for p in POOLS},
           "check0a_max_rel_err": rec_err, "scores": {}, "delta": {}, "layers": {}, "n_boot": args.n_boot}
    for ds, df in sts.items():
        i1 = torch.as_tensor([sid[s] for s in df["sentence1"]], device=dev)
        i2 = torch.as_tensor([sid[s] for s in df["sentence2"]], device=dev)
        gold = torch.as_tensor(df["score"].to_numpy().copy(), dtype=torch.float64, device=dev)
        boot = torch.randint(0, len(df), (args.n_boot, len(df)), generator=gen).to(dev)  # same draws for V1 and V3
        for p in POOLS:
            E = {v: emb[p][v].to(dev) for v in emb[p]}
            cos = {}
            for v in ("V1", "V3", "V3hs"):
                for var in variants_run if v != "V3hs" else ("raw", "center"):
                    c = cosines(E[v], E[v][:n_fit], var, i1, i2)
                    cos[(v, var)] = c
                    res["scores"][f"{ds}/{p}/{v}/{var}"] = spearman100(c, gold)
            for var in variants_run:
                a, b = cos[("V1", var)], cos[("V3", var)]
                dl = np.array([spearman100(a[bi], gold[bi]) - spearman100(b[bi], gold[bi]) for bi in boot])
                pt = res["scores"][f"{ds}/{p}/V1/{var}"] - res["scores"][f"{ds}/{p}/V3/{var}"]
                res["delta"][f"{ds}/{p}/{var}"] = {"delta": pt, "ci_lo": float(np.percentile(dl, 2.5)),
                                                   "ci_hi": float(np.percentile(dl, 97.5))}
            for l in range(L + 1) if lay is not None else []:
                Hl = lay[p][l].to(dev).double()
                for var in ("raw", "center"):
                    res["layers"][f"{ds}/{p}/{l}/{var}"] = spearman100(cosines(Hl, Hl[:n_fit], var, i1, i2), gold)
        print(f"[{slug}] {ds}: " + "  ".join(
            f"{p}/{var} V1={res['scores'][f'{ds}/{p}/V1/{var}']:.1f} V3={res['scores'][f'{ds}/{p}/V3/{var}']:.1f}"
            for p in POOLS for var in ("raw", "center", "zscore")), flush=True)
    a1 = max(abs(res["delta"][f"{ds}/{p}/zscore"]["delta"]) for ds in sts for p in POOLS)
    res["checkA1_zscore_max_abs_delta"] = a1
    res["checkA1_pass"] = a1 < 1e-6
    res["elapsed_s"] = time.time() - t0
    print(f"[{slug}] check 0a max rel err {rec_err:.2e}; A1 zscore |V1-V3| max {a1:.2e} "
          f"({'pass' if a1 < 1e-6 else 'FAIL'}); {res['elapsed_s']:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"{args.out_name or slug}{'_smoke' if args.max_pairs else ''}.json")
    json.dump(res, open(fn, "w"), indent=1)
    print(f"[{slug}] wrote {fn}", flush=True)


if __name__ == "__main__":
    main()
