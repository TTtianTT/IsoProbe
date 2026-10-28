"""Round 3 follow-up (login node, CPU only): uses cache/banks/<slug>.pt and the round-3 / round-1 JSONs.

1. Cancellation check: delta_S = W_U[:,S] m_S (S = top-k dims), delta_R = delta_all - delta_S, on tokens with count >= 5:
   sd(delta_R), corr(delta_S, delta_R), additive share Cov(delta_S, delta_all) / Var(delta_all).
2. Per-dimension detail of the top-20 V3 dims for a few models (W_U read from safetensors, no model load).
3. Architecture covariates from config.json and the checkpoint's weight names.
4. Overlap of the round-1 mid-layer outlier dims with the round-3 top-10 dims.
Writes results/round3/followup.md and results/round3/followup.json.
"""
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from analyze_round3 import KS, load_bank, load_models
from round3_lib import decompose

DETAIL = ["gpt2", "gpt2-medium", "Qwen3-8B-Base", "Qwen3-8B", "Llama-3.1-8B", "Olmo-3.1-32B-Instruct"]
N_DETAIL = 20


def weight_map(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(path, f) for k, f in json.load(open(idx))["weight_map"].items()}
    f = os.path.join(path, "model.safetensors")
    with safe_open(f, "pt") as sf:
        return {k: f for k in sf.keys()}


def load_unembed(path):
    """LM-head weight [V, d] from safetensors: lm_head.weight, else the tied input embedding (GPT-2: wte.weight)."""
    wm = weight_map(path)
    key = next(k for k in ("lm_head.weight", "wte.weight", "transformer.wte.weight", "model.embed_tokens.weight")
               if k in wm)
    with safe_open(wm[key], "pt") as sf:
        return sf.get_tensor(key), key


def keep_mask(slug, n_vocab, groups):
    counts = np.load(f"cache/unigrams/{groups['model_group'][slug]}.npy")
    n = min(n_vocab, len(counts))
    keep = np.zeros(n_vocab, dtype=bool)
    keep[:n] = counts[:n] >= 5
    return torch.as_tensor(keep)


def additive_share(a, total):
    """Cov(a, total) / Var(total): the part of the spread of `total` attributable to `a` (shares of a partition sum to 1)."""
    a, total = a.double(), total.double()
    return (((a - a.mean()) * (total - total.mean())).mean() / total.var(unbiased=False)).item()


def cancellation(b, keep):
    dA = b["delta"]["all"][keep].double()
    out = {"sd_all": dA.std(unbiased=False).item()}
    for k in KS:
        dS = b["delta"][f"top{k}"][keep].double()
        dR = dA - dS
        out[k] = {"sd_S": dS.std(unbiased=False).item(), "sd_R": dR.std(unbiased=False).item(),
                  "corr_S_R": torch.corrcoef(torch.stack([dS, dR]))[0, 1].item(),
                  "additive_share": additive_share(dS, dA)}
    return out


def detail(slug, path, b, V, keep):
    W, key = load_unembed(path)
    W = W.to(torch.bfloat16)  # the GPU job used the bf16-loaded head (GPT-2's checkpoint is fp32)
    order = b["order"]
    top = order[:N_DETAIL]
    Wt = W[:, top].float()  # [V, 20]
    m = b["m"].float()
    # consistency: the GPU job's top-8 offset vector from the same W and m
    rec = Wt[:, :8] @ m[top[:8]]
    rec_err = ((rec - b["delta"]["top8"]).abs().max() / b["delta"]["top8"].abs().max()).item()
    _, contrib = decompose(V["V1"], b["gamma"], b["beta"])
    mu, sd = V["V1"].mean(0), V["V1"].std(0, unbiased=False)
    share = mu.pow(2) / (mu.pow(2) + sd.pow(2))
    pct = lambda x: torch.argsort(torch.argsort(x)).float() / (len(x) - 1)  # noqa: E731
    share_pct, gamma_pct = pct(share), pct(b["gamma"].abs())
    dA = b["delta"]["all"][keep]
    rows = []
    for j, k in enumerate(top.tolist()):
        dk = Wt[keep, j] * m[k]
        rows.append({"dim": k, "ubar_share": contrib[k].item(), "v1_mean_share": share[k].item(),
                     "v1_mean_share_pct": share_pct[k].item(), "gamma": b["gamma"][k].item(),
                     "gamma_abs_pct": gamma_pct[k].item(), "m": m[k].item(),
                     "offset_sd": dk.std(unbiased=False).item(), "additive_share": additive_share(dk, dA)})
    return {"W_key": key, "W_shape": list(W.shape), "top8_reconstruction_max_rel_err": rec_err, "rows": rows}


def covariates(slug, path, r3, r1):
    cfg = json.load(open(os.path.join(path, "config.json")))
    names = list(weight_map(path))
    layer0 = [k for k in names if ".0." in k or k.startswith("h.0.")]
    has = lambda s: any(s in k for k in layer0)  # noqa: E731
    if has("ln_1"):
        pos = "pre-norm (ln_1/ln_2)"
    elif has("input_layernorm"):
        pos = "pre-norm"
    elif has("post_feedforward_layernorm"):
        pos = "post-sublayer (norm on the branch output, before the residual add)"
    else:
        pos = "?"
    if has("q_norm"):
        qk = "yes, per head" if cfg.get("model_type") == "qwen3" else "yes, full width"
    else:
        qk = "no"
    attn_bias = has("q_proj.bias") or has("c_attn.bias")
    return {"norm_type": r3["norm_type"], "norm_beta": r3["has_beta"], "norm_position": pos, "qk_norm": qk,
            "tied": r3["tied"], "lm_head_bias": r3["head_bias"], "attention_bias": attn_bias,
            "config_attention_bias": cfg.get("attention_bias"), "d": r3["d"], "n_layers": r1["n_layers"],
            "vocab_config": cfg.get("vocab_size"), "vocab_lm_head": r3["vocab"], "model_type": cfg.get("model_type")}


def outlier_overlap(r1, top10, b, V, order):
    L = r1["n_layers"]
    mid = [x for x in r1["layers"] if 0 < x["layer"] < L]
    union = sorted({d for x in mid for d in x["all"]["outlier_top5_dims"]})
    last = r1["layers"][-1]["all"]["outlier_top5_dims"]
    # where do the mid-layer outlier dims end up at the final norm? (V1 mean share, |gamma|, V3 rank; percentiles)
    mu, sd = V["V1"].mean(0), V["V1"].std(0, unbiased=False)
    pct = lambda x: torch.argsort(torch.argsort(x)).float() / (len(x) - 1)  # noqa: E731
    share_pct, gamma_pct = pct(mu.pow(2) / (mu.pow(2) + sd.pow(2))), pct(b["gamma"].abs())
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    u = torch.as_tensor(union)
    return {"mid_layers": [x["layer"] for x in mid], "mid_union": union, "top10_V3": top10,
            "overlap": sorted(set(union) & set(top10)), "last_layer_top5": last,
            "last_overlap": len(set(last) & set(top10)),
            "mid_union_v1_share_pct": share_pct[u].tolist(), "mid_union_gamma_pct": gamma_pct[u].tolist(),
            "mid_union_v3_rank": (rank[u] + 1).tolist()}


def main():
    slugs, res = load_models()
    groups = json.load(open("results/round3/unigram_groups.json"))
    a3 = json.load(open("results/round3/analysis.json"))["step2"]
    out = {"cancellation": {}, "detail": {}, "covariates": {}, "outlier_overlap": {}}
    for s in slugs:
        b, V = load_bank(s)
        path = res[s]["path"]
        r1 = json.load(open(f"results/{s}.json"))
        keep = keep_mask(s, b["delta"]["all"].shape[0], groups)
        out["cancellation"][s] = cancellation(b, keep)
        out["covariates"][s] = covariates(s, path, res[s], r1)
        out["outlier_overlap"][s] = outlier_overlap(r1, a3[s]["top10_dims"], b, V, b["order"])
        if s in DETAIL:
            out["detail"][s] = detail(s, path, b, V, keep)
        c = out["cancellation"][s]
        print(f"{s}: " + "  ".join(f"k{k} corr={c[k]['corr_S_R']:+.2f} share={c[k]['additive_share']:+.2f}" for k in KS),
              flush=True)
    json.dump(out, open("results/round3/followup.json", "w"), indent=1)
    write_md(slugs, res, out)
    print("wrote results/round3/followup.{md,json}")


def write_md(slugs, res, out):
    L = ["# Round 3 follow-up (generated by analyze_round3b.py)\n",
         "## 1. Cancellation check\n",
         "δ_S = W_U[:,S] m_S (S = top-k V3 dims), δ_R = δ_all − δ_S; tokens with unigram count ≥ 5. "
         "Additive share = Cov(δ_S, δ_all) / Var(δ_all) (the shares of S and R sum to 1).\n",
         "| model | sd(δ_all) | " + " | ".join(f"k={k} sd(δ_S) / sd(δ_R) / corr / share" for k in KS) + " |",
         "|---" * (2 + len(KS)) + "|"]
    for s in slugs:
        c = out["cancellation"][s]
        L.append(f"| {s} | {c['sd_all']:.2f} | " + " | ".join(
            f"{c[k]['sd_S']:.2f} / {c[k]['sd_R']:.2f} / {c[k]['corr_S_R']:+.2f} / {c[k]['additive_share']:+.2f}"
            for k in KS) + " |")
    L.append("\n### Additive share vs Δloss\n")
    L += ["| model | " + " | ".join(f"k={k} share / Δloss" for k in KS) + " | all Δloss |", "|---" * (2 + len(KS)) + "|"]
    for s in slugs:
        c, d = out["cancellation"][s], res[s]["step3"]["dloss"]
        L.append(f"| {s} | " + " | ".join(f"{c[k]['additive_share']:+.2f} / {d[f'top{k}']:+.3f}" for k in KS)
                 + f" | {d['all']:+.3f} |")
    L.append("\n## 2. Top-20 V3 dims\n")
    for s, dd in out["detail"].items():
        L += [f"\n### {s}\n", f"W_U = `{dd['W_key']}` {dd['W_shape']}; top-8 offset reconstruction vs GPU job: "
              f"max rel err {dd['top8_reconstruction_max_rel_err']:.1e}\n",
              "| rank | dim | ū_k² share | V1 μ²/(μ²+σ²) (pct) | γ_k (pct of \\|γ\\|) | m_k | sd of W_U[:,k] m_k | additive share |",
              "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(dd["rows"], 1):
            L.append(f"| {i} | {r['dim']} | {r['ubar_share']:.4f} | {r['v1_mean_share']:.3f} ({r['v1_mean_share_pct']:.3f}) | "
                     f"{r['gamma']:+.3f} ({r['gamma_abs_pct']:.3f}) | {r['m']:+.2f} | {r['offset_sd']:.3f} | "
                     f"{r['additive_share']:+.3f} |")
    L += ["\n## 3. Architecture covariates\n",
          "| model | norm | norm position | QK-norm | tied emb | lm_head bias | attention bias | d | layers | vocab (config / lm_head) |",
          "|---" * 10 + "|"]
    for s in slugs:
        c = out["covariates"][s]
        norm = c["norm_type"] + (" (with β)" if c["norm_beta"] else "")
        L.append(f"| {s} | {norm} | {c['norm_position']} | {c['qk_norm']} | {c['tied']} | {c['lm_head_bias']} | "
                 f"{c['attention_bias']} | {c['d']} | {c['n_layers']} | {c['vocab_config']} / {c['vocab_lm_head']} |")
    L += ["\n## 4. Round-1 mid-layer outlier dims vs round-3 top-10 V3 dims\n",
          "Mid-layer union = outlier_top5_dims (largest per-dim mean |h|, all positions) of every selected layer "
          "except the embedding and the last layer.\n",
          "Last three columns: medians over the mid-layer union dims of their V1 mean-share percentile, "
          "|γ| percentile and rank by V3 contribution (1 = largest).\n",
          "| model | mid layers | mid-layer union | overlap with V3 top-10 | last-layer top-5 ∩ V3 top-10 | "
          "V1 mean-share pct | \\|γ\\| pct | V3 rank |",
          "|---|---|---|---|---|---|---|---|"]
    for s in slugs:
        o = out["outlier_overlap"][s]
        L.append(f"| {s} | {o['mid_layers']} | {o['mid_union']} | {len(o['overlap'])} {o['overlap']} | "
                 f"{o['last_overlap']}/5 | {np.median(o['mid_union_v1_share_pct']):.3f} | "
                 f"{np.median(o['mid_union_gamma_pct']):.3f} | {int(np.median(o['mid_union_v3_rank']))} |")
    open("results/round3/followup.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
