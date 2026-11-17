"""Round 6 summaries and figures (login node): python round6_summary.py s1|s2|s3|s4|s5|s6.

Applies only the preregistered decision rules in configs/round6_prereg.md; everything else is descriptive.
"""
import itertools
import json
import os
import sys

import numpy as np

from round3_lib import spearman
from round6_lib import METRIC_KEYS, MODELS12

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY = "#8a8985"
SHORT = {"Mistral-Small-24B-Instruct-2501": "Mistral-24B-Inst", "Olmo-3.1-32B-Instruct": "Olmo-3.1-32B-Inst",
         "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Inst", "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Inst"}


def load_dir(d, models):
    out = {}
    for m in models:
        f = os.path.join(d, f"{m}.json")
        if os.path.exists(f):
            out[m] = json.load(open(f))
    return out


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})
    return plt


# ---------------------------------------------------------------- S1

def s1():
    R = load_dir("results/round6/gauge", MODELS12)
    ms = [m for m in MODELS12 if m in R]
    L = [f"# Round 6 S1: gauge audit and gauge metric ({len(ms)}/12 models)\n"]
    # P6-1
    p61 = {m: max(R[m]["random"]["orth"]["max_abs_diff_vs_V1"].values()) for m in ms}
    p61_max = max(p61.values())
    # P6-2
    out = {"P6-1": {"per_model_max_abs": p61, "max": p61_max, "pass": p61_max < 1e-5}}
    for kappa in (10, 100):
        iv = {m: (R[m]["worst"][f"k{kappa}_A_min"]["A"], R[m]["worst"][f"k{kappa}_A_max"]["A"]) for m in ms}
        rnd = {m: (R[m]["random"][f"diag_k{kappa}"]["min"]["A"], R[m]["random"][f"diag_k{kappa}"]["max"]["A"])
               for m in ms}
        pairs = list(itertools.combinations(ms, 2))
        inter = [max(iv[a][0], iv[b][0]) <= min(iv[a][1], iv[b][1]) for a, b in pairs]
        out[f"intervals_k{kappa}"] = {m: list(iv[m]) for m in ms}
        out[f"random_range_k{kappa}"] = {m: list(rnd[m]) for m in ms}
        out[f"random_outside_worst_k{kappa}"] = [m for m in ms if rnd[m][0] < iv[m][0] - 1e-9 or rnd[m][1] > iv[m][1] + 1e-9]
        out[f"intersect_frac_k{kappa}"] = float(np.mean(inter))
        out[f"n_pairs_k{kappa}"] = len(pairs)
        out[f"non_intersecting_k{kappa}"] = [f"{a}|{b}" for (a, b), ok in zip(pairs, inter) if not ok]
    out["P6-2"] = {"intersect_frac_k100": out["intersect_frac_k100"], "pass": out["intersect_frac_k100"] >= 0.8}
    # P6-3
    p63 = {m: {pv: {"V3_vs_V1": R[m]["hstar_checks"][pv]["V3_vs_V1_max_rel"],
                    "lin": R[m]["hstar_checks"][pv]["lin_k100_max_rel"]} for pv in ("unif", "pi")} for m in ms}
    p63_max = max(v for m in ms for pv in p63[m].values() for v in pv.values())
    out["P6-3"] = {"per_model": p63, "max_rel": p63_max, "pass": p63_max < 1e-4}
    # rankings
    A1 = [R[m]["base"]["V1"]["A"] for m in ms]
    A3 = [R[m]["base"]["V3"]["A"] for m in ms]
    Au = [R[m]["hstar"]["unif"]["from_V1_folded"]["A"] for m in ms]
    Ap = [R[m]["hstar"]["pi"]["from_V1_folded"]["A"] for m in ms]
    out["spearman"] = {"Astar_unif_vs_V1": spearman(Au, A1)[:2], "Astar_unif_vs_V3": spearman(Au, A3)[:2],
                       "Astar_pi_vs_V1": spearman(Ap, A1)[:2], "Astar_pi_vs_V3": spearman(Ap, A3)[:2],
                       "Astar_unif_vs_pi": spearman(Au, Ap)[:2], "V1_vs_V3": spearman(A1, A3)[:2]}
    json.dump(out, open("results/round6/gauge/summary.json", "w"), indent=1)

    L.append("## Preregistered checks\n")
    L.append(f"- **P6-1** (G_orth invariance): max |Δ| over models × 20 samples × 5 metrics = {p61_max:.2e} -> "
             f"{'PASS' if p61_max < 1e-5 else 'FAIL'} (< 1e-5)")
    L.append(f"- **P6-2** (κ = 100 worst-case intervals intersect): {out['intersect_frac_k100']:.3f} of "
             f"{out['n_pairs_k100']} pairs -> {'PASS' if out['P6-2']['pass'] else 'FAIL'} (≥ 0.8); κ = 10: "
             f"{out['intersect_frac_k10']:.3f}")
    L.append(f"- **P6-3** (h* invariance, V3-path and 20 G_lin κ=100, both P): max rel diff = {p63_max:.2e} -> "
             f"{'PASS' if p63_max < 1e-4 else 'FAIL'} (< 1e-4)")
    L.append(f"- Random G_diag samples outside the worst-case interval: κ=10 {out['random_outside_worst_k10']}, "
             f"κ=100 {out['random_outside_worst_k100']}")
    sp = out["spearman"]
    L.append("- Spearman (12 models, bank A): " + ", ".join(f"{k} ρ = {v[0]:+.2f} (p = {v[1]:.2g})" for k, v in sp.items()))
    L.append("\n## Per model (A = bank isotropy)\n")
    L.append("| model | γ cond | A V1 | A V3 | A random diag κ=100 | A worst κ=10 | A worst κ=100 | A* unif | A* π | "
             "B_raw V1 / V3 / h*unif | top1 V1 / V3 / h*unif | RankMe/d V1 / V3 / h*unif |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in ms:
        r = R[m]
        b, hu, hp = r["base"], r["hstar"]["unif"]["from_V1_folded"], r["hstar"]["pi"]["from_V1_folded"]
        w = r["worst"]
        L.append(f"| {m} | {r['gamma']['cond']:.3g} | {b['V1']['A']:.3f} | {b['V3']['A']:.3f} | "
                 f"{r['random']['diag_k100']['min']['A']:.3f} – {r['random']['diag_k100']['max']['A']:.3f} | "
                 f"{w['k10_A_min']['A']:.3f} – {w['k10_A_max']['A']:.3f} | {w['k100_A_min']['A']:.3f} – "
                 f"{w['k100_A_max']['A']:.3f} | {hu['A']:.3f} | {hp['A']:.3f} | "
                 f"{b['V1']['B_raw']:.3f} / {b['V3']['B_raw']:.3f} / {hu['B_raw']:.3f} | "
                 f"{b['V1']['top1']:.3f} / {b['V3']['top1']:.3f} / {hu['top1']:.3f} | "
                 f"{b['V1']['rankme_over_d']:.3f} / {b['V3']['rankme_over_d']:.3f} / {hu['rankme_over_d']:.3f} |")
    L.append("\n## Other metrics: range over 20 random samples per family\n")
    L.append("| model | family | " + " | ".join(METRIC_KEYS) + " |")
    L.append("|---|---|" + "---|" * len(METRIC_KEYS))
    for m in ms:
        for fam, v in R[m]["random"].items():
            L.append(f"| {m} | {fam} | " + " | ".join(f"{v['min'][k]:.3f} – {v['max'][k]:.3f}" for k in METRIC_KEYS) + " |")
    open("results/round6/gauge/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:8]))

    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, m in enumerate(ms):
        y = len(ms) - 1 - i
        k100, k10 = out["intervals_k100"][m], out["intervals_k10"][m]
        ax.plot(k100, [y, y], color=GRAY, lw=2, solid_capstyle="round", zorder=1)
        ax.plot(k10, [y, y], color="#3d3c39", lw=4.5, solid_capstyle="round", zorder=2)
        r = R[m]
        ax.scatter(r["base"]["V1"]["A"], y, marker="o", s=42, color=C1, edgecolor="white", lw=1.2, zorder=3,
                   label="V1" if i == 0 else None)
        ax.scatter(r["base"]["V3"]["A"], y, marker="s", s=38, color=C2, edgecolor="white", lw=1.2, zorder=3,
                   label="V3 (= hidden_states[-1])" if i == 0 else None)
        ax.scatter(r["hstar"]["unif"]["from_V1_folded"]["A"], y, marker="D", s=36, color=C3, edgecolor="white",
                   lw=1.2, zorder=4, label="A* (uniform P)" if i == 0 else None)
        ax.scatter(r["hstar"]["pi"]["from_V1_folded"]["A"], y, marker="^", s=40, color=C4, edgecolor="white",
                   lw=1.2, zorder=4, label="A* (unigram P_π)" if i == 0 else None)
    ax.plot([], [], color="#3d3c39", lw=4.5, label="reachable, cond ≤ 10")
    ax.plot([], [], color=GRAY, lw=2, label="reachable, cond ≤ 100")
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([SHORT.get(m, m) for m in ms][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("isotropy A (bank estimate)")
    ax.set_title("Diagonal gauge G_diag: reachable isotropy per model (output-preserving)", fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/r6_fig1_gauge_intervals.png", dpi=160)
    print("wrote figs/r6_fig1_gauge_intervals.png")


# ---------------------------------------------------------------- S2

def s2():
    R = load_dir("results/round6/sts", MODELS12)
    ms = [m for m in MODELS12 if m in R]
    p = "stsb/mean"
    L = [f"# Round 6 S2: STS-B test, mean pooling, under gauge transforms ({len(ms)}/12 models)\n"]
    chk, p64 = {}, {}
    for m in ms:
        r4 = json.load(open(f"results/round4/sts/{m}.json"))["scores"]
        chk[m] = {f"{v}/{var}": R[m]["scores"][f"{p}/{v}/{var}"] - r4[f"{p}/{v}/{var}"]
                  for v in ("V1", "V3") for var in ("raw", "zscore")}
        p64[m] = {k.split("/")[-1]: v for k, v in R[m]["gauge"].items() if "/P6-4/" in k}
    chk_max = max(abs(v) for d in chk.values() for v in d.values())
    p64_max = max(abs(v) for d in p64.values() for v in d.values())
    out = {"round4_check": chk, "round4_check_max_abs": chk_max, "P6-4": p64, "P6-4_max_abs": p64_max,
           "P6-4_pass": p64_max == 0.0}
    L.append(f"- Round-4 check (V1/V3 raw and zscore minus round 4): max |Δ| = {chk_max:.3g}")
    L.append(f"- **P6-4** (zscore invariant under V3 and 5 diagonal gauges): max |Δ| = {p64_max:.3g} -> "
             f"{'PASS' if p64_max == 0 else 'FAIL'} (must be exactly 0)\n")
    L.append("| model | V1 raw | V3 raw | D_Amax raw | D_Amin raw | 3 random diag raw | h* unif raw | h* π raw | "
             "V1 zscore | h* unif zscore | h* unif raw − V3 raw [95% CI] | h* unif raw − V3 zscore [95% CI] | "
             "h* π raw − V3 raw [95% CI] |")
    L.append("|---|" + "---|" * 12)
    rows = {}
    for m in ms:
        s, g = R[m]["scores"], R[m]["gauge"]
        rnd = [g[f"{p}/D_rand{k}/raw"] for k in range(3)]
        ci = lambda key: g[f"{p}/{key}"]  # noqa: E731
        f = lambda d: f"{d['delta']:+.1f} [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}]"  # noqa: E731
        rows[m] = {"V1_raw": s[f"{p}/V1/raw"], "V3_raw": s[f"{p}/V3/raw"], "Amax_raw": g[f"{p}/D_Amax/raw"],
                   "Amin_raw": g[f"{p}/D_Amin/raw"], "rand_raw": rnd, "hu_raw": g[f"{p}/hstar_unif/raw"],
                   "hp_raw": g[f"{p}/hstar_pi/raw"], "V1_z": s[f"{p}/V1/zscore"], "hu_z": g[f"{p}/hstar_unif/zscore"],
                   "hp_z": g[f"{p}/hstar_pi/zscore"]}
        L.append(f"| {m} | {rows[m]['V1_raw']:.1f} | {rows[m]['V3_raw']:.1f} | {rows[m]['Amax_raw']:.1f} | "
                 f"{rows[m]['Amin_raw']:.1f} | {min(rnd):.1f} – {max(rnd):.1f} | {rows[m]['hu_raw']:.1f} | "
                 f"{rows[m]['hp_raw']:.1f} | {rows[m]['V1_z']:.1f} | {rows[m]['hu_z']:.1f} | "
                 f"{f(ci('delta_hstar_unif_raw_minus_V3_raw'))} | {f(ci('delta_hstar_unif_raw_minus_V3_zscore'))} | "
                 f"{f(ci('delta_hstar_pi_raw_minus_V3_raw'))} |")
    out["rows"] = rows
    json.dump(out, open("results/round6/sts/summary.json", "w"), indent=1)
    open("results/round6/sts/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:3]))
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for i, m in enumerate(ms):
        y = len(ms) - 1 - i
        r = rows[m]
        lo, hi = min(r["Amin_raw"], r["Amax_raw"], *r["rand_raw"]), max(r["Amin_raw"], r["Amax_raw"], *r["rand_raw"])
        ax.plot([lo, hi], [y, y], color=GRAY, lw=2, zorder=1, label="diagonal gauges (raw)" if i == 0 else None)
        for x, mk, c, lab in ((r["V1_raw"], "o", C1, "V1 raw"), (r["V3_raw"], "s", C2, "V3 raw"),
                              (r["hu_raw"], "D", C3, "h* uniform raw"), (r["hp_raw"], "^", C4, "h* unigram raw")):
            ax.scatter(x, y, marker=mk, s=38, color=c, edgecolor="white", lw=1.2, zorder=3, label=lab if i == 0 else None)
        ax.scatter(r["V1_z"], y, marker="|", s=160, color="#0b0b0b", lw=1.6, zorder=4,
                   label="zscore (identical for every diagonal gauge)" if i == 0 else None)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([SHORT.get(m, m) for m in ms][::-1])
    ax.set_xlabel("STS-B test Spearman × 100 (mean pooling)")
    ax.set_title("Sentence similarity under output-preserving gauge transforms", fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/r6_fig2_sts_gauge.png", dpi=160)
    print("wrote figs/r6_fig2_sts_gauge.png")


# ---------------------------------------------------------------- S3

def s3():
    R = load_dir("results/round6/od", MODELS12)
    ms = [m for m in MODELS12 if m in R]
    four = ["gpt2", "gpt2-medium", "Qwen3-8B-Base", "Qwen3-8B"]
    L = [f"# Round 6 S3: outlier dimensions (Macocco et al. rule) under gauges ({len(ms)}/12 models)\n",
         "OD = dims whose median |h| over ~9940 bank tokens (position 0 removed) exceeds the global 0.99 quantile of |h|.\n",
         "| model | max/med \\|γ\\| | \\|OD\\| V3 / V1 / D_Amax / D_Amin | J(V1,V3) | J(V3,D_Amax) | J(V3,D_Amin) | "
         "J(V1,D_Amax) | J(V1,D_Amin) | J(D_Amax,D_Amin) | OD(V3) | OD(V3) ∩ top10 ‖ū‖² | OD(V3) ∩ top10 additive | "
         "(expl.) \\|OD\\| V2 / h*unif |", "|---|" + "---|" * 13]
    for m in ms:
        r = R[m]
        o, j = r["od"], r["jaccard"]
        L.append(f"| {m} | {r['gamma_max_over_median']:.1f} | {o['V3']['n']} / {o['V1']['n']} / {o['D_Amax']['n']} / "
                 f"{o['D_Amin']['n']} | {j['V3|V1']:.2f} | {j['V3|D_Amax']:.2f} | {j['V3|D_Amin']:.2f} | "
                 f"{j['V1|D_Amax']:.2f} | {j['V1|D_Amin']:.2f} | {j['D_Amax|D_Amin']:.2f} | {o['V3']['dims'][:12]} | "
                 f"{r['overlap_OD_V3_top10_ubar']} | {r['overlap_OD_V3_top10_additive']} | {o['V2']['n']} / "
                 f"{o['hstar_unif']['n']} |")
    p65 = {m: R[m]["jaccard"]["V3|V1"] for m in four if m in R}
    n_ok = sum(v < 0.5 for v in p65.values())
    L.append(f"\n- **P6-5**: J(OD(V1), OD(V3)) on the four |γ|-spiky models = "
             + ", ".join(f"{m} {v:.2f}" for m, v in p65.items())
             + f" -> {n_ok}/{len(p65)} below 0.5 -> {'PASS' if n_ok == 4 else 'FAIL'}")
    json.dump({"P6-5": p65, "P6-5_n_below": n_ok, "P6-5_pass": n_ok == 4}, open("results/round6/od/summary.json", "w"),
              indent=1)
    open("results/round6/od/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- S4 / S5

CAUSAL = [l.rstrip("\n").split("\t") for l in open("configs/round6_causal.tsv") if l.strip()]


def s4():
    L = ["# Round 6 S4: inference-time interventions at the final norm (Pythia, eval_loss_round6)\n",
         "Δloss in nats/token relative to the unmodified model (same fp32 path). rand = max |Δ| over 5 random unit "
         "directions. ctrl = bias fitted on the untruncated head (domain-adaptation control).\n",
         "| model | step | n tok | base | V1 ‖ū‖² | ū mean (1) | ū fix (2) | ū proj (3) | (2)/(1) | rand max \\|Δ\\| (1)/(2)/(3) | "
         "σ₁²/Σσ² | \\|cos(v₁, ū)\\| | trunc | trunc+b₀ | trunc+bias | ctrl | trunc+bias − ctrl |",
         "|---|" + "---|" * 16]
    R = {}
    for m, rev in CAUSAL:
        f = f"results/round6/causal/ablate/{m}/{rev}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        R[(m, rev)] = r
        d = r["dloss"]
        rnd = [max(abs(d[f"rand{k}/{iv}"]) for k in range(5)) for iv in ("mean", "fix", "proj")]
        L.append(f"| {m} | {rev[4:]} | {r['n_eval_tokens']} | {r['loss']['base']:.4f} | {r['ubar_sq_V1_bank']:.3f} | "
                 f"{d['ubar/mean']:+.4f} | {d['ubar/fix']:+.4f} | {d['ubar/proj']:+.4f} | "
                 f"{d['ubar/fix'] / d['ubar/mean'] if d['ubar/mean'] > 0 else float('nan'):.3f} | "
                 + " / ".join(f"{x:.4f}" for x in rnd) + f" | {r['svd']['s1sq_over_sum']:.3f} | "
                 f"{r['svd']['abs_cos_v1_ubar']:.3f} | {d['trunc']:+.4f} | {d['trunc+b0']:+.4f} | {d['trunc+bias']:+.4f} | "
                 f"{d['control_bias']:+.4f} | {r['trunc_bias_minus_control']:+.4f} |")
    out = {}
    for m in ("pythia-70m", "pythia-160m"):
        r = R.get((m, "step143000"))
        if r:
            d = r["dloss"]
            out[m] = {"dl1": d["ubar/mean"], "dl2": d["ubar/fix"], "P6-8": d["ubar/fix"] < 0.2 * d["ubar/mean"],
                      "dl_trunc_bias": d["trunc+bias"], "P6-9": d["trunc+bias"] < 0.02,
                      "trunc_bias_minus_control": r["trunc_bias_minus_control"]}
    if len(out) == 2:
        out["P6-8_pass"] = all(out[m]["P6-8"] for m in ("pythia-70m", "pythia-160m"))
        out["P6-9_pass"] = all(out[m]["P6-9"] for m in ("pythia-70m", "pythia-160m"))
    L.append("")
    for m in ("pythia-70m", "pythia-160m"):
        if m in out:
            o = out[m]
            L.append(f"- {m} step 143000: Δ(2) = {o['dl2']:+.4f} vs 0.2·Δ(1) = {0.2 * o['dl1']:+.4f} -> P6-8 "
                     f"{'holds' if o['P6-8'] else 'fails'}; trunc+bias Δ = {o['dl_trunc_bias']:+.4f} -> P6-9 "
                     f"{'holds' if o['P6-9'] else 'fails'} (vs control {o['trunc_bias_minus_control']:+.4f})")
    if "P6-8_pass" in out:
        L.append(f"- **P6-8**: {'PASS' if out['P6-8_pass'] else 'FAIL'}; **P6-9**: {'PASS' if out['P6-9_pass'] else 'FAIL'}")
    os.makedirs("results/round6/causal", exist_ok=True)
    json.dump(out, open("results/round6/causal/ablate_summary.json", "w"), indent=1)
    open("results/round6/causal/ablate_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def s5():
    L = ["# Round 6 S5 (preliminary): refit final norm + W_U with the body frozen\n",
         "Single seed, single lr (AdamW 1e-4, wd 0), 1000 steps × 16 × 512 wikitext-103 train tokens. Eval: eval_loss_round6.\n",
         "| model | step | base | refit 250 | 500 | 750 | 1000 | base (no in_samples) | refit 1000 (no in_samples) |",
         "|---|---|---|---|---|---|---|---|---|"]
    R = {}
    for m, rev in CAUSAL:
        f = f"results/round6/causal/refit/{m}/{rev}.json"
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        R[(m, rev)] = r
        e = r["evals"]
        g = lambda k, f_="loss": f"{e[k][f_]:.4f}" if k in e else "–"  # noqa: E731
        L.append(f"| {m} | {rev[4:]} | {g('0')} | {g('250')} | {g('500')} | {g('750')} | {g('1000')} | "
                 f"{g('0', 'loss_out')} | {g('1000', 'loss_out')} |")
    out = {}
    L.append("")
    for m in ("pythia-70m", "pythia-160m", "pythia-410m"):
        revs = [rev for mm, rev in CAUSAL if mm == m]
        if not all((m, rv) in R and "1000" in R[(m, rv)]["evals"] for rv in revs):
            L.append(f"- {m}: incomplete ({sum((m, rv) in R and '1000' in R[(m, rv)]['evals'] for rv in revs)}/{len(revs)} "
                     "checkpoints finished), no reading")
            continue
        base = {rv: R[(m, rv)]["evals"]["0"]["loss"] for rv in revs}
        ref = {rv: R[(m, rv)]["evals"]["1000"]["loss"] for rv in revs}
        d_raw = base["step143000"] - min(base.values())
        d_ref = ref["step143000"] - min(ref.values())
        if d_raw <= 0:
            cls = "no reading (D_raw <= 0)"
        else:
            q = d_ref / d_raw
            cls = "readout side" if q <= 0.25 else ("body side" if q >= 0.75 else "mixed")
        out[m] = {"D_raw": d_raw, "D_refit": d_ref, "ratio": d_ref / d_raw if d_raw > 0 else None, "class": cls,
                  "argmin_base": min(base, key=base.get), "argmin_refit": min(ref, key=ref.get)}
        L.append(f"- {m}: D_raw = {d_raw:+.4f} (min at {out[m]['argmin_base']}), D_refit = {d_ref:+.4f} "
                 f"(min at {out[m]['argmin_refit']}), ratio {out[m]['ratio'] if out[m]['ratio'] is None else round(out[m]['ratio'], 3)} -> {cls}")
    json.dump(out, open("results/round6/causal/refit_summary.json", "w"), indent=1)
    open("results/round6/causal/refit_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- S6

def s6():
    rows = [l.rstrip("\n").split("\t") for l in open("configs/round6_polypythia.tsv") if l.strip()]
    L = ["# Round 6 S6: PolyPythias step 143000 (descriptive; collapse = V1 ||ubar||^2 >= 0.9 and W_U s1^2/sum >= 0.5)\n",
         "| model | V1 ‖ū‖² | A V1 | A V3 | W_U σ₁²/Σσ² | \\|cos(ū_V1, v₁(W_U·γ))\\| | max/med \\|γ\\| | ‖β‖/‖γ‖ | loss (round-3 eval) | 0a | fold | collapsed |",
         "|---|" + "---|" * 11]
    out = {}
    ref = {}
    for m in ("pythia-70m", "pythia-160m"):
        r = json.load(open(f"results/round4/dyn/{m}/step143000.json"))
        sp = json.load(open(f"results/round5/spec/{m}/step143000.json"))
        ref[m] = (r, sp["W_U"]["s1sq_over_sum"], sp["align_V1_WUgamma"]["abs_cos_v1"])
    for repo, rev in [(f"EleutherAI/{m}", "step143000") for m in ("pythia-70m", "pythia-160m")] + rows:
        name = repo.split("/")[1]
        if name in ref:
            r, s1, cs = ref[name]
            tag = f"{name} (seed 0, round 4/5)"
            col = r["step1"]["V1"]["all"]["ubar_sq"] >= 0.9 and s1 >= 0.5
        else:
            f = f"results/round6/poly/{name}/{rev}.json"
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            s1, cs, col, tag = r["spec"]["W_U"]["s1sq_over_sum"], r["spec"]["align_V1_WUgamma"]["abs_cos_v1"], r["collapsed"], name
            out[name] = {"ubar_sq_V1": r["step1"]["V1"]["all"]["ubar_sq"], "s1sq_over_sum": s1, "collapsed": col,
                         "loss": r["function"]["loss_base"]}
        g = r["gamma_beta"]
        L.append(f"| {tag} | {r['step1']['V1']['all']['ubar_sq']:.3f} | {r['step1']['V1']['all']['isotropy_full']:.3f} | "
                 f"{r['step1']['V3']['all']['isotropy_full']:.3f} | {s1:.3f} | {cs:.3f} | {g['gamma_max_over_median']:.1f} | "
                 f"{g['beta_over_gamma']:.2f} | {r['function']['loss_base']:.3f} | {r['check0a']['max_rel_err']:.1e} | "
                 f"{r['fold']['max_rel_err']:.1e} | {'yes' if col else 'no'} |")
    for m in ("pythia-70m", "pythia-160m"):
        ks = [k for k in out if k.startswith(m + "-seed")]
        n = sum(out[k]["collapsed"] for k in ks)
        out[f"{m}_n_collapsed"] = n
        L.append(f"\n- {m}: {n}/{len(ks)} new seeds collapsed (seed 0 from rounds 4/5: collapsed)")
    json.dump(out, open("results/round6/poly/summary.json", "w"), indent=1)
    open("results/round6/poly/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6}[sys.argv[1]]()
