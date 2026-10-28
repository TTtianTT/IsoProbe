"""Round 3 analysis (login node, CPU): reads results/round3/<slug>.json (GPU part), cache/banks/<slug>.pt and the
unigram counts; does step 2 (exact / approximate per-dimension decomposition, circularity check, base->instruct
counterfactual), aggregates steps 1 and 3, step 3b correlations, figures and tables.

Step-1 metrics (A / B / robust-centered B / RankMe for V0-V3) are computed in the GPU job (SVDs of 10k x d banks),
with the functions in round3_lib.py, and read from its JSON here.
Writes results/round3/summary.md, results/round3/analysis.json, figs/r3_*.png.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from analyze import COLORS, GRID, INK, INK2, KIND  # noqa: E402  (also applies the shared rcParams)
from round3_lib import VNAMES, approx_cos, decompose, spearman, variants  # noqa: E402

ANISO = ["gpt2", "gpt2-medium", "Qwen3-8B-Base"]
PAIRS = [("Qwen3-8B-Base", "Qwen3-8B"), ("Llama-3.1-8B", "Llama-3.1-8B-Instruct")]
CF_KS = [1, 2, 4, 8, 16]
KS = [1, 2, 4, 8, 16, 32]
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue steps 250/400/550/700 for V0..V3


def load_models():
    slugs = [os.path.basename(l.strip()) for l in open("models.txt") if l.strip()]
    res = {s: json.load(open(f"results/round3/{s}.json")) for s in slugs}
    return slugs, res


def load_bank(slug):
    b = torch.load(f"cache/banks/{slug}.pt", weights_only=False)
    V = variants(b["x"], b["gamma"], b["beta"], b["norm_type"], b["eps"])
    return b, V


def step2(slug, b, V, r):
    gamma, beta = b["gamma"], b["beta"]
    ub2, contrib = decompose(V["V1"], gamma, beta)
    order = torch.argsort(contrib, descending=True)
    cum = torch.cumsum(contrib[order], 0)
    # consistency: the GPU job ranked the same dims (allow ties to permute)
    same_top8 = set(order[:8].tolist()) == set(r["top32_dims"][:8])
    ub2_const, _ = decompose(V["V1"])  # gamma -> constant 1, beta -> 0: must equal V1's ||ubar||^2
    ub2_v1 = (torch.nn.functional.normalize(V["V1"], dim=-1).mean(0).pow(2).sum()).item()
    approx = approx_cos(V["V1"], gamma, beta)
    # circularity check on V1 (before gamma): per-dim mean share mu^2 / (mu^2 + sigma^2)
    mu, sd = V["V1"].mean(0), V["V1"].std(0, unbiased=False)
    share = mu.pow(2) / (mu.pow(2) + sd.pow(2))
    pct = (torch.argsort(torch.argsort(share)).float() / (len(share) - 1))  # percentile rank of each dim
    _, contrib_v1 = decompose(V["V1"])
    top10 = order[:10]
    v1_top10 = set(torch.argsort(contrib_v1, descending=True)[:10].tolist())
    absg = gamma.abs()
    gpct = torch.argsort(torch.argsort(absg)).float() / (len(absg) - 1)
    return {
        "ubar_sq_V3": ub2, "ubar_sq_V3_gpu": r["ubar_sq_V3"], "same_top8_as_gpu": same_top8,
        "approx_cos_V3": approx, "approx_minus_exact": approx - ub2,
        "ubar_sq_gamma_const": ub2_const, "ubar_sq_V1": ub2_v1,
        "top10_dims": top10.tolist(), "top10_contrib": contrib[top10].tolist(),
        "cum_share": {k: cum[k - 1].item() for k in (1, 2, 4, 8, 10, 16, 32, 64)},
        "cum_curve": cum[:256].tolist(),
        "v1_share_top10": share[top10].tolist(), "v1_share_pct_top10": pct[top10].tolist(),
        "v1_share_median_all": share.median().item(),
        "v1_contrib_of_v3_top10": contrib_v1[top10].sum().item(),
        "v1_top10_overlap_v3_top10": len(v1_top10 & set(top10.tolist())),
        "gamma_pct_top10": gpct[top10].tolist(), "gamma_top10": gamma[top10].tolist(),
        "gamma_abs_median": absg.median().item(),
    }, order


def counterfactual(base, inst, b_b, V_b, V_i, order_b, ub_base, ub_inst):
    mu_b, mu_i = V_b["V1"].mean(0), V_i["V1"].mean(0)
    out = {"cos_base": ub_base, "cos_inst": ub_inst, "gap": ub_base - ub_inst, "k": {}}
    for k in CF_KS:
        S = order_b[:k]
        V1 = V_b["V1"].clone()
        V1[:, S] += (mu_i[S] - mu_b[S])
        c, _ = decompose(V1, b_b["gamma"], b_b["beta"])
        out["k"][k] = {"cos_cf": c, "closed_frac": (ub_base - c) / (ub_base - ub_inst),
                       "dims": S.tolist(), "mu_base": mu_b[S].tolist(), "mu_inst": mu_i[S].tolist()}
    return out


def unigram_corr(slug, b, groups):
    """Spearman(logit offset, log freq) and the offset's spread over the vocabulary (a constant offset is
    softmax-invariant, so only its spread can change the loss)."""
    g = groups["model_group"][slug]
    counts = np.load(f"cache/unigrams/{g}.npy")
    out = {}
    for name, delta in b["delta"].items():
        dv = delta.numpy()
        n = min(len(dv), len(counts))
        keep = counts[:n] >= 5
        rho, p, nn = spearman(dv[:n][keep], np.log(counts[:n][keep]))
        out[name] = {"rho": rho, "p": p, "n_tokens": nn, "offset_sd": float(dv[:n][keep].std()),
                     "offset_absmean": float(abs(dv[:n][keep].mean()))}
    return out


# ---------------------------------------------------------------- figures

def fig_isotropy(slugs, res, path):
    fig, ax = plt.subplots(figsize=(13, 4.4))
    x = np.arange(len(slugs))
    w = 0.2
    for i, v in enumerate(VNAMES):
        vals = [res[s]["step1"][v]["all"]["isotropy_full"] for s in slugs]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=RAMP[i],
               label={"V0": "V0 = x (norm input)", "V1": "V1 = normalized", "V2": "V2 = γ⊙V1",
                      "V3": "V3 = V2+β (hidden_states[-1])"}[v])
    ax.axhline(0.25, color=INK2, lw=1.2, ls="--")
    ax.set_xticks(x, [f"{s}\n({KIND.get(s, '?')})" for s in slugs], fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("isotropy (A, all positions)")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, fontsize=8.5, loc="upper left")
    fig.suptitle("Isotropy at each step of the final norm (dashed: 0.25)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cum(s2, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, s in enumerate(ANISO):
        c = s2[s]["cum_curve"]
        ax.plot(range(1, len(c) + 1), c, color=COLORS[i], lw=2, label=s)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.axvline(8, color=GRID, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("top-k dimensions (ranked by contribution to ‖ū‖² on V3)")
    ax.set_ylabel("cumulative share of ‖ū‖²")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Per-dimension decomposition of the cross-sequence cosine", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_dloss(slugs, res, path):
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.5), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, slugs):
        d = res[s]["step3"]["dloss"]
        top = [d[f"top{k}"] for k in KS]
        rnd = np.array([[d[f"rand{k}_s{r}"] for r in range(5)] for k in KS])
        ax.fill_between(KS, rnd.min(1), rnd.max(1), color=COLORS[1], alpha=0.25, lw=0)
        ax.plot(KS, rnd.mean(1), color=COLORS[1], lw=2, marker="o", ms=4, label="random k dims (5 seeds)")
        ax.plot(KS, top, color=COLORS[0], lw=2, marker="o", ms=4, label="top-k dims")
        ax.axhline(d["all"], color=INK2, lw=1, ls="--", label="all dims")
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"{s} ({KIND.get(s, '?')})", fontsize=9, loc="left", color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("k")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ loss (nats/token)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Removing the mean of hidden_states[-1] on k dimensions: change in next-token loss (seqs 500–999)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    slugs, res = load_models()
    groups = json.load(open("results/round3/unigram_groups.json")) if os.path.exists(
        "results/round3/unigram_groups.json") else None
    s2, orders, banks = {}, {}, {}
    uni = {}
    for s in slugs:
        b, V = load_bank(s)
        s2[s], orders[s] = step2(s, b, V, res[s])
        if groups:
            uni[s] = unigram_corr(s, b, groups)
        if s in {x for p in PAIRS for x in p}:
            banks[s] = (b, V)
        print(f"{s}: ||ubar||^2={s2[s]['ubar_sq_V3']:.4f} approx-exact={s2[s]['approx_minus_exact']:+.4f} "
              f"top8 share={s2[s]['cum_share'][8]:.3f} same_top8={s2[s]['same_top8_as_gpu']}", flush=True)
    cf = {}
    for bs, ins in PAIRS:
        cf[f"{bs}->{ins}"] = counterfactual(bs, ins, banks[bs][0], banks[bs][1], banks[ins][1], orders[bs],
                                            s2[bs]["ubar_sq_V3"], s2[ins]["ubar_sq_V3"])
    if uni:  # does the loss change track the spread of the logit offset across the vocabulary?
        sd = [uni[s][n]["offset_sd"] for s in slugs for n in uni[s]]
        dl = [res[s]["step3"]["dloss"][n] for s in slugs for n in uni[s]]
        offset_sd_vs_dloss = spearman(sd, dl)
    else:
        offset_sd_vs_dloss = None
    rank = {}
    for key in ("isotropy_full", "cross_cos"):
        a = [res[s]["step1"]["V1"]["all"][key] for s in slugs]
        c = [res[s]["step1"]["V3"]["all"][key] for s in slugs]
        rank[key] = spearman(a, c)[:2]

    os.makedirs("figs", exist_ok=True)
    fig_isotropy(slugs, res, "figs/r3_fig1_isotropy_V0-V3.png")
    fig_cum(s2, "figs/r3_fig2_cum_contrib.png")
    fig_dloss(slugs, res, "figs/r3_fig3_dloss.png")
    json.dump({"step2": s2, "counterfactual": cf, "rank_spearman_V1_vs_V3": rank, "unigram": uni,
               "spearman_offset_sd_vs_dloss": offset_sd_vs_dloss},
              open("results/round3/analysis.json", "w"), indent=1)
    write_summary(slugs, res, s2, cf, rank, uni, groups, offset_sd_vs_dloss)
    print("wrote results/round3/{summary.md,analysis.json}, figs/r3_*.png")


def write_summary(slugs, res, s2, cf, rank, uni, groups, offset_sd_vs_dloss=None):
    L = ["# Round 3 summary (generated by analyze_round3.py)\n", "## Checks\n",
         "| model | 0a max rel err | 0b max dA | 0b max dB | fold max rel err | fold argmax agree | shortcut rel diff | all pass |",
         "|---|---|---|---|---|---|---|---|"]
    for s in slugs:
        r = res[s]
        dA = max(r["check0b"][v]["dA"] for v in VNAMES)
        dB = max(r["check0b"][v]["dB"] for v in VNAMES)
        ok = r["check0a"]["pass"] and all(r["check0b"][v]["pass"] for v in VNAMES) and r["fold"]["pass"]
        L.append(f"| {s} | {r['check0a']['max_rel_err']:.1e} | {dA:.4f} | {dB:.4f} | {r['fold']['max_rel_err']:.1e} | "
                 f"{r['fold']['argmax_agree']:.5f} | {r['step3']['shortcut_vs_direct_max_rel_logit_diff']:.1e} | "
                 f"{'yes' if ok else '**NO**'} |")
    for tag in ("all", "no_pos0"):
        L += [f"\n## Step 1: V0–V3 ({tag})\n",
              "A = isotropy (full sequences); B = raw cross cos; Brc = robust-centered B; R = RankMe/d ÷ Gaussian RankMe/d\n",
              "| model | " + " | ".join(f"{v} A / B / Brc / R" for v in VNAMES) + " |", "|---" * 5 + "|"]
        for s in slugs:
            cells = []
            for v in VNAMES:
                m = res[s]["step1"][v][tag]
                cells.append(f"{m['isotropy_full']:.3f} / {m['cross_cos']:.3f} / {m['cross_cos_robust_centered']:.3f} / "
                             f"{m['rankme_over_d_norm']:.3f}")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append(f"\nSpearman (12 models) V1 vs V3: isotropy rho={rank['isotropy_full'][0]:.3f} (p={rank['isotropy_full'][1]:.3g}); "
             f"raw B rho={rank['cross_cos'][0]:.3f} (p={rank['cross_cos'][1]:.3g})\n")
    L += ["## Step 2: per-dimension decomposition of ||ubar||^2 on V3 (all positions)\n",
          "| model | ‖ū‖² | approx − exact | top-1 | top-8 | top-10 | top-32 | γ→1 vs V1 | top-10 dims | "
          "V1 mean-share pct of top-10 (median) | V1 share of top-10 dims | γ pct of top-10 (median) |",
          "|---" * 12 + "|"]
    for s in slugs:
        d = s2[s]
        L.append(f"| {s} | {d['ubar_sq_V3']:.4f} | {d['approx_minus_exact']:+.4f} | {d['cum_share'][1]:.3f} | "
                 f"{d['cum_share'][8]:.3f} | {d['cum_share'][10]:.3f} | {d['cum_share'][32]:.3f} | "
                 f"{d['ubar_sq_gamma_const'] - d['ubar_sq_V1']:+.1e} | {d['top10_dims']} | "
                 f"{np.median(d['v1_share_pct_top10']):.3f} | {d['v1_contrib_of_v3_top10']:.3f} | "
                 f"{np.median(d['gamma_pct_top10']):.3f} |")
    L += ["\n### base → instruct counterfactual (shift base V1 means on top-k base dims to the instruct values)\n",
          "| pair | cos base | cos instruct | " + " | ".join(f"k={k} cos / closed" for k in CF_KS) + " |",
          "|---" * (3 + len(CF_KS)) + "|"]
    for name, c in cf.items():
        L.append(f"| {name} | {c['cos_base']:.4f} | {c['cos_inst']:.4f} | " + " | ".join(
            f"{c['k'][k]['cos_cf']:.4f} / {c['k'][k]['closed_frac']:.2f}" for k in CF_KS) + " |")
    L += ["\n## Step 3: Δ loss (nats/token) after removing the mean of h on S\n",
          "× = top-k Δloss ÷ the largest |Δloss| among the 5 random draws (the random mean is often ≈ 0)\n",
          "| model | base loss | " + " | ".join(f"k={k} top / rand mean [min,max] / ×" for k in KS) + " | all |",
          "|---" * (3 + len(KS)) + "|"]
    for s in slugs:
        d = res[s]["step3"]["dloss"]
        cells = []
        for k in KS:
            rnd = [d[f"rand{k}_s{r}"] for r in range(5)]
            ratio = d[f"top{k}"] / max(abs(min(rnd)), abs(max(rnd)), 1e-6)
            cells.append(f"{d[f'top{k}']:+.4f} / {np.mean(rnd):+.4f} [{min(rnd):+.4f},{max(rnd):+.4f}] / {ratio:.0f}")
        L.append(f"| {s} | {res[s]['step3']['loss_base']:.4f} | " + " | ".join(cells) + f" | {d['all']:+.4f} |")
    if uni:
        names = ["top1", "top2", "top4", "top8", "top16", "top32", "all"]
        L += ["\n## Step 3 follow-up: spread of the logit offset W_U[:,S] m_S over the vocabulary\n",
              f"Spearman(sd over tokens with count ≥ 5, Δloss) across all {offset_sd_vs_dloss[2]} model × set "
              f"combinations: rho = {offset_sd_vs_dloss[0]:.3f}\n",
              "| model | " + " | ".join(f"{n} sd / \\|mean\\|" for n in names) + " |", "|---" * (1 + len(names)) + "|"]
        for s in slugs:
            L.append(f"| {s} | " + " | ".join(f"{uni[s][n]['offset_sd']:.2f} / {uni[s][n]['offset_absmean']:.1f}"
                                             for n in names) + " |")
        L += ["\n## Step 3b: Spearman(W_U[:,S] m_S, log unigram freq), tokens with count ≥ 5\n",
              f"Text: first {groups['n_chars']:,} characters of wikitext-103 train ({groups['n_lines']:,} lines).\n",
              "| model | n tokens (types) | " + " | ".join(names) + " |", "|---" * (2 + len(names)) + "|"]
        for s in slugs:
            u = uni[s]
            L.append(f"| {s} | {u['top1']['n_tokens']} | " + " | ".join(
                f"{u[n]['rho']:+.3f}" + ("" if u[n]["p"] < 1e-3 else f" (p={u[n]['p']:.2g})") for n in names) + " |")
    open("results/round3/summary.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
