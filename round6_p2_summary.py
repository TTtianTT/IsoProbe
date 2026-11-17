"""Round 6 phase 2 summaries and figures (login node): python round6_p2_summary.py t1|t2|t3|t4|t5|t6.

Applies only the decision rules preregistered in configs/round6_prereg.md (phase 2); everything else is descriptive.
Commit messages must quote numbers printed by this script.
"""
import glob
import json
import os
import sys

import numpy as np

from round6_summary import C1, C2, C3, C4, GRAY as GRAY_, plt_setup

SIZES = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1.4b"]
REPS = [("V1", "V1", C1, "o"), ("V3", "V3", C2, "s"), ("hstar_unif", "h* (uniform)", C3, "D"), ("hstar_pi", "h* (unigram)", C4, "^")]


def load_traj(name):
    out = {}
    for f in glob.glob(f"results/round6/p2/traj/{name}/*.json"):
        r = json.load(open(f))
        out[r["revision"]] = r
    return out


def step_of(rev):
    return int(rev[4:]) if rev.startswith("step") else None


# ---------------------------------------------------------------- T1

def t1():
    T = {m: load_traj(m) for m in SIZES + ["OLMo-2-0425-1B"]}
    L = ["# Round 6 phase 2 T1: collapse in the gauge-invariant geometry h*\n"]
    pairs = {"pythia-70m": ("step50000", "step143000"), "pythia-160m": ("step80000", "step143000"),
             "pythia-410m": ("step80000", "step143000"), "pythia-1.4b": ("step80000", "step143000")}
    out = {"P7-1": {}}
    L.append("| model | from → to | " + " | ".join(f"top1 {lab}" for _, lab, _, _ in REPS) + " | " +
             " | ".join(f"A {lab}" for _, lab, _, _ in REPS) + " | ‖ū‖² h*u |")
    L.append("|---|---|" + "---|" * 9)
    for m, (a, b) in pairs.items():
        ra, rb = T[m][a], T[m][b]
        dtop = rb["hstar_unif"]["top1"] - ra["hstar_unif"]["top1"]
        ok = dtop >= 0.2 if m in ("pythia-70m", "pythia-160m") else abs(dtop) < 0.05
        out["P7-1"][m] = {"from": a, "to": b, "delta_top1_hstar_unif": dtop, "ok": bool(ok),
                          "delta_top1_hstar_pi": rb["hstar_pi"]["top1"] - ra["hstar_pi"]["top1"]}
        L.append(f"| {m} | {a[4:]} → {b[4:]} | " + " | ".join(f"{ra[k]['top1']:.3f} → {rb[k]['top1']:.3f}" for k, _, _, _ in REPS)
                 + " | " + " | ".join(f"{ra[k]['A']:.3f} → {rb[k]['A']:.3f}" for k, _, _, _ in REPS)
                 + f" | {ra['hstar_unif']['ubar_sq']:.3f} → {rb['hstar_unif']['ubar_sq']:.3f} |")
    out["P7-1"]["pass"] = all(v["ok"] for v in out["P7-1"].values())
    L.append("\n- **P7-1** (uniform h*): " + "; ".join(f"{m} Δtop1 = {v['delta_top1_hstar_unif']:+.3f} ({'ok' if v['ok'] else 'not ok'})"
                                                  for m, v in out["P7-1"].items() if m != "pass")
             + f" -> {'PASS' if out['P7-1']['pass'] else 'FAIL'}")
    L.append("- unigram h*_π Δtop1 (report only): " + "; ".join(f"{m} {v['delta_top1_hstar_pi']:+.3f}"
                                                          for m, v in out["P7-1"].items() if m != "pass"))
    # endpoints across seeds
    L.append("\n## Seed variation at step 143000 (original seed + 9 PolyPythias)\n")
    L.append("| size | n | " + " | ".join(f"A {lab} mean ± sd (CV)" for _, lab, _, _ in REPS) + " | top1 h*u mean ± sd |")
    L.append("|---|---|" + "---|" * 5)
    out["seeds"] = {}
    for size in ("pythia-70m", "pythia-160m"):
        rs = [T[size]["step143000"]] + [json.load(open(f"results/round6/p2/traj/{size}-seed{k}/step143000.json")) for k in range(1, 10)]
        row = {}
        for k, _, _, _ in REPS:
            v = np.array([r[k]["A"] for r in rs])
            row[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "cv": float(v.std(ddof=1) / v.mean()),
                      "values": v.tolist()}
        t1v = np.array([r["hstar_unif"]["top1"] for r in rs])
        out["seeds"][size] = row
        L.append(f"| {size} | {len(rs)} | " + " | ".join(f"{row[k]['mean']:.3f} ± {row[k]['sd']:.3f} ({row[k]['cv']:.2f})" for k, _, _, _ in REPS)
                 + f" | {t1v.mean():.3f} ± {t1v.std(ddof=1):.3f} |")
    # OLMo-2 and trajectory endpoints
    L.append("\n## Trajectories (min over training / final), A\n")
    L.append("| model | " + " | ".join(f"{lab} min (step) / final" for _, lab, _, _ in REPS) + " |")
    L.append("|---|" + "---|" * 4)
    out["traj"] = {}
    for m in SIZES + ["OLMo-2-0425-1B"]:
        revs = sorted(T[m], key=lambda r: (step_of(r) if step_of(r) is not None else 10 ** 9) if m.startswith("pythia")
                      else (int(r.split("-step")[1].split("-")[0]) if "-step" in r else 10 ** 9))
        fin = "step143000" if m.startswith("pythia") else "main"
        cells = []
        out["traj"][m] = {}
        for k, _, _, _ in REPS:
            v = [(T[m][r][k]["A"], r) for r in revs if not (m.startswith("pythia") and step_of(r) < 1000)]
            mn = min(v)
            out["traj"][m][k] = {"min": mn[0], "min_rev": mn[1], "final": T[m][fin][k]["A"]}
            cells.append(f"{mn[0]:.3f} ({mn[1]}) / {T[m][fin][k]['A']:.3f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("\n(min taken over checkpoints with step ≥ 1000 for Pythia, i.e. excluding the early transient)")
    os.makedirs("results/round6/p2/traj", exist_ok=True)
    json.dump(out, open("results/round6/p2/traj/summary.json", "w"), indent=1)
    open("results/round6/p2/traj/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    plt = plt_setup()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True)
    for j, m in enumerate(SIZES):
        revs = sorted((r for r in T[m] if step_of(r) is not None and step_of(r) >= 1000), key=step_of)
        x = [step_of(r) for r in revs]
        for i, met in enumerate(("A", "top1")):
            ax = axes[i, j]
            for k, lab, col, mk in REPS:
                ax.plot(x, [T[m][r][k][met] for r in revs], color=col, lw=1.6, label=lab, marker=mk, ms=2.5, markevery=6)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(m.replace("pythia-", "Pythia-"), fontsize=10, loc="left")
            if j == 0:
                ax.set_ylabel("isotropy A" if met == "A" else "centered top-1 variance share")
            ax.set_ylim(0, 1)
            if i == 1:
                ax.set_xlabel("training step (log)")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=7.5)
    fig.suptitle("Late collapse seen in V1 / V3 and in the gauge-invariant h* (step ≥ 1000)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig("figs/r6p2_fig1_collapse_invariant.png", dpi=150)
    print("wrote figs/r6p2_fig1_collapse_invariant.png")


# ---------------------------------------------------------------- T2

def t2():
    r = json.load(open("results/round6/p2/tokens/summary.json"))
    dt = json.load(open("results/round6/p2/tokens/deviation_traj.json"))
    L = ["# Round 6 phase 2 T2: saturation split and the high-deviation tokens\n",
         "| model | Δloss (mean) | q | k | share of Σd in H | share in complement [95% CI] | mean d in H / complement | class |",
         "|---|---|---|---|---|---|---|---|"]
    for m, v in r.items():
        for q, s_ in v["split"].items():
            L.append(f"| {m} | {v['mean_d']:+.4f} | {q} | {s_['k']} | {s_['share_H']:.4f} | {s_['share_complement']:.4f} "
                     f"[{s_['share_complement_ci'][0]:.4f}, {s_['share_complement_ci'][1]:.4f}] | {s_['mean_d_H']:+.3f} / "
                     f"{s_['mean_d_complement']:+.3f} | {s_['class']} |")
    L.append("\n| model | ρ(dev, −log f input) | ρ(dev, −log f target) | ρ(dev, entropy) | loss H / compl. | entropy H / compl. | "
             "incomplete-UTF-8 input: H / all | share of all such inputs in H |")
    L.append("|---|---|---|---|---|---|---|---|")
    p72 = {}
    for m, v in r.items():
        b = v["2b"]
        e = b["explore_incomplete_utf8"]
        p72[m] = b["spearman_dev_vs_neglogfreq_input"]
        L.append(f"| {m} | {b['spearman_dev_vs_neglogfreq_input']:+.3f} | {b['spearman_dev_vs_neglogfreq_target']:+.3f} | "
                 f"{b['spearman_dev_vs_entropy']:+.3f} | {b['loss_H']:.3f} / {b['loss_complement']:.3f} | "
                 f"{b['entropy_H']:.3f} / {b['entropy_complement']:.3f} | {e['input_H']:.3f} / {e['input_all']:.4f} | "
                 f"{e['share_of_all_incomplete_inputs_in_H']:.3f} |")
    ok = all(p72[m] > 0.2 for m in ("pythia-70m", "pythia-160m"))
    L.append(f"\n- **P7-2**: ρ(dev, −log f input) = 70m {p72['pythia-70m']:+.3f}, 160m {p72['pythia-160m']:+.3f} -> "
             f"{'PASS' if ok else 'FAIL'} (both > 0.2 required)")
    L.append("\n## 2c t50 (report only)\n")
    for m, v in dt.items():
        L.append(f"- {m} (window from {v['window_start']}): " + ", ".join(f"{k} {t}" for k, t in v["t50"].items()))
    json.dump({"P7-2": p72, "P7-2_pass": ok}, open("results/round6/p2/tokens/decisions.json", "w"), indent=1)
    open("results/round6/p2/tokens/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cols = {"pythia-70m": C1, "pythia-160m": C2, "pythia-410m": C3}
    mk = {"pythia-70m": "o", "pythia-160m": "s", "pythia-410m": "D"}
    for m, v in r.items():
        qs = [float(q[:-1]) for q in v["split"]]
        axes[0].plot(qs, [100 * s_["share_H"] for s_ in v["split"].values()], color=cols[m], marker=mk[m], lw=1.6,
                     label=m.replace("pythia-", "Pythia-") + (" (loss fell)" if v["mean_d"] < 0 else ""))
        axes[1].plot(qs, [s_["mean_d_H"] for s_ in v["split"].values()], color=cols[m], marker=mk[m], lw=1.6)
        axes[1].axhline(v["mean_d"], color=cols[m], lw=1, ls=":")
    axes[0].plot([0.1, 1], [0.1, 1], color=GRAY_, lw=1, ls="--", label="share = token fraction")
    axes[0].set_xlabel("H = top q% of tokens by |p − c| at step 143000 (q, %)")
    axes[0].set_ylabel("share of Σ Δloss in H (%)")
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[0].set_title("Saturation is not carried by H", fontsize=10, loc="left")
    axes[1].set_xlabel("q (%)")
    axes[1].set_ylabel("mean Δloss per token in H (nats)")
    axes[1].set_title("Δloss in H vs all tokens (dotted)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig("figs/r6p2_fig2_saturation_split.png", dpi=150)
    print("wrote figs/r6p2_fig2_saturation_split.png")


# ---------------------------------------------------------------- T3

def t3():
    from round6_lib import MODELS12
    R = {m: json.load(open(f"results/round6/p2/whiten/{m}.json")) for m in MODELS12
         if os.path.exists(f"results/round6/p2/whiten/{m}.json")}
    reps = ["V1", "V3", "D_Amax", "D_Amin", "lin0", "lin1", "lin2"]
    L = [f"# Round 6 phase 2 T3: whitening invariance and STS-B ({len(R)}/12 models)\n",
         "| model | whiten range (7 reps) | zscore range (3 G_lin) | zscore range (V1,V3,D) | raw | zscore | zscore (old fit) | whiten | abtt-10 | "
         "whiten − zscore [95% CI] | dropped eig / cond kept (V1) |", "|---|" + "---|" * 10]
    p73 = max(r["range_all"]["whiten"] for r in R.values())
    n74 = sum(r["range_lin"]["zscore"] > 0.5 for r in R.values())
    out = {"P7-3_max_range": p73, "P7-3_pass": p73 < 1e-6, "P7-4_n_models": n74, "P7-4_pass": n74 >= 10, "rows": {}}
    for m, r in R.items():
        sc = r["scores"]
        zd = max(sc[f"{x}/zscore"] for x in ("V1", "V3", "D_Amax", "D_Amin")) - min(sc[f"{x}/zscore"] for x in ("V1", "V3", "D_Amax", "D_Amin"))
        zo = max(sc[f"{x}/zscore_oldfit"] for x in ("V1", "V3", "D_Amax", "D_Amin")) - min(sc[f"{x}/zscore_oldfit"] for x in ("V1", "V3", "D_Amax", "D_Amin"))
        dw = r["delta"]["V1/whiten-zscore"]
        out["rows"][m] = {"zscore_range_diag": zd, "zscore_old_range_diag": zo, "lin_zscore": [sc[f"lin{k}/zscore"] for k in range(3)],
                          "lin_raw": [sc[f"lin{k}/raw"] for k in range(3)], "zca_V1": r["zca"]["V1"],
                          "cond_kept_max": max(z["cond_kept"] for z in r["zca"].values()),
                          "dropped": {k: z["n_dropped"] for k, z in r["zca"].items()}}
        L.append(f"| {m} | {r['range_all']['whiten']:.2e} | {r['range_lin']['zscore']:.2f} | {zd:.2e} | {sc['V1/raw']:.1f} | "
                 f"{sc['V1/zscore']:.1f} | {sc['V1/zscore_oldfit']:.1f} | {sc['V1/whiten']:.1f} | {sc['V1/abtt10']:.1f} | "
                 f"{dw['delta']:+.1f} [{dw['ci_lo']:+.1f}, {dw['ci_hi']:+.1f}] | {r['zca']['V1']['n_dropped']} / {r['zca']['V1']['cond_kept']:.2e} |")
    L.append(f"\n- **P7-3**: max whitened-STS range over 7 representations = {p73:.3g} -> {'PASS' if p73 < 1e-6 else 'FAIL'} (< 1e-6)")
    L.append(f"- **P7-4**: models with zscore range over 3 G_lin samples > 0.5: {n74}/{len(R)} -> {'PASS' if n74 >= 10 else 'FAIL'} (≥ 10)")
    L.append("- zscore range over V1/V3/D_Amax/D_Amin (new fit / old fit): max " +
             f"{max(v['zscore_range_diag'] for v in out['rows'].values()):.2e} / {max(v['zscore_old_range_diag'] for v in out['rows'].values()):.2e}")
    dws = {m: r["delta"]["V1/whiten-zscore"] for m, r in R.items()}
    L.append(f"- V1 whiten − zscore: 95% CI above 0 in {sum(d['ci_lo'] > 0 for d in dws.values())}/{len(dws)} models, "
             f"includes 0 in {sum(d['ci_lo'] <= 0 <= d['ci_hi'] for d in dws.values())}, below 0 in {sum(d['ci_hi'] < 0 for d in dws.values())}; "
             f"delta range {min(d['delta'] for d in dws.values()):+.2f} – {max(d['delta'] for d in dws.values()):+.2f}")
    zz = {m: r["delta"]["V1/zscore-zscore_oldfit"] for m, r in R.items()}
    L.append(f"- V1 zscore (new fit) − zscore (old fit): range {min(d['delta'] for d in zz.values()):+.2f} – {max(d['delta'] for d in zz.values()):+.2f}; "
             f"CI excludes 0 in {sum(d['ci_lo'] > 0 or d['ci_hi'] < 0 for d in zz.values())}/{len(zz)}")
    L.append(f"- models with zscore G_lin range ≤ 0.5: " + ", ".join(f"{m} {r['range_lin']['zscore']:.2f}" for m, r in R.items() if r["range_lin"]["zscore"] <= 0.5))
    L.append("- dropped eigenvalues per representation (V1, V3, D_Amax, D_Amin, lin0-2): "
             + "; ".join(f"{m} {list(v['dropped'].values())}" for m, v in out["rows"].items()))
    L.append("- max condition number of the kept spectrum over the 7 representations: " +
             "; ".join(f"{m} {v['cond_kept_max']:.1e}" for m, v in out["rows"].items()))
    json.dump(out, open("results/round6/p2/whiten/summary.json", "w"), indent=1)
    open("results/round6/p2/whiten/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plt = plt_setup()
    from round6_summary import SHORT
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ms = list(R)
    for i, m in enumerate(ms):
        y = len(ms) - 1 - i
        sc = R[m]["scores"]
        for var, col, mk, lab in (("raw", GRAY_, "o", "raw"), ("zscore", C1, "s", "zscore"), ("whiten", C3, "D", "ZCA whitening"),
                                  ("abtt10", C2, "^", "abtt-10")):
            xs = [sc[f"{r}/{var}"] for r in reps]
            ax.plot([min(xs), max(xs)], [y, y], color=col, lw=2, alpha=0.6)
            ax.scatter(xs, [y] * len(xs), color=col, marker=mk, s=26, edgecolor="white", lw=0.8, zorder=3,
                       label=lab if i == 0 else None)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([SHORT.get(m, m) for m in ms][::-1])
    ax.set_xlabel("STS-B test Spearman × 100 (mean pooling) across V1, V3, 2 diagonal and 3 linear gauges")
    ax.set_title("Only whitening gives one score for every output-preserving linear gauge", fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/r6p2_fig3_whiten.png", dpi=160)
    print("wrote figs/r6p2_fig3_whiten.png")



def fb3():
    """Fallback 3: T3 quantities on SICK-R (mean pooling) and STS-B (PromptEOL). Descriptive, no verdicts."""
    from round6_lib import MODELS12
    base = "results/round6/p2/whiten_fb3"
    L = ["# Round 6 phase 2 fallback 3: T3 on SICK-R (mean) and STS-B (PromptEOL), descriptive\n"]
    out = {}
    for cfg in ("sickr_mean", "stsb_eol"):
        R = {m: json.load(open(f"{base}/{cfg}/{m}.json")) for m in MODELS12 if os.path.exists(f"{base}/{cfg}/{m}.json")}
        if not R:
            continue
        L += [f"\n## {cfg} ({len(R)}/12 models)\n",
              "| model | whiten range (7 reps, eigh 1e-12) | whiten range (svd rank-tol) | zscore range (3 G_lin) | raw | zscore | whiten | whiten − zscore [95% CI] |",
              "|---|" + "---|" * 7]
        for m, r in R.items():
            sc, dw = r["scores"], r["delta"]["V1/whiten-zscore"]
            L.append(f"| {m} | {r['range_all']['whiten']:.2e} | {r['range_all']['whiten_rt']:.2e} | {r['range_lin']['zscore']:.2f} | "
                     f"{sc['V1/raw']:.1f} | {sc['V1/zscore']:.1f} | {sc['V1/whiten']:.1f} | {dw['delta']:+.1f} [{dw['ci_lo']:+.1f}, {dw['ci_hi']:+.1f}] |")
        mx = max(r["range_all"]["whiten"] for r in R.values())
        mxr = max(r["range_all"]["whiten_rt"] for r in R.values())
        nz = sum(r["range_all"]["whiten"] > 0 for r in R.values())
        nzr = sum(r["range_all"]["whiten_rt"] > 0 for r in R.values())
        lz = [r["range_lin"]["zscore"] for r in R.values()]
        n05 = sum(x > 0.5 for x in lz)
        dws = [r["delta"]["V1/whiten-zscore"] for r in R.values()]
        up = sum(d["ci_lo"] > 0 for d in dws); dn = sum(d["ci_hi"] < 0 for d in dws)
        L.append(f"\n- whitened-score range over 7 representations: eigh path max {mx:.3g} (> 0 in {nz}/{len(R)}); svd rank-tol path max {mxr:.3g} (> 0 in {nzr}/{len(R)})")
        L.append(f"- zscore range over 3 G_lin samples > 0.5 in {n05}/{len(R)} (range {min(lz):.2f} – {max(lz):.2f})")
        L.append(f"- V1 whiten − zscore: CI above 0 in {up}/{len(R)}, below 0 in {dn}, includes 0 in {len(R) - up - dn}; "
                 f"delta range {min(d['delta'] for d in dws):+.2f} – {max(d['delta'] for d in dws):+.2f}")
        L.append("- dropped eigenvalues (eigh path) per representation: " +
                 "; ".join(f"{m} {[z['n_dropped'] for z in r['zca'].values()]}" for m, r in R.items() if any(z['n_dropped'] for z in r['zca'].values())))
        out[cfg] = {"n": len(R), "whiten_range_max": mx, "whiten_range_nonzero": nz, "whiten_rt_range_max": mxr, "whiten_rt_range_nonzero": nzr,
                    "zscore_lin_gt05": n05, "whiten_minus_zscore_ci_above0": up, "ci_below0": dn,
                    "rows": {m: {"range_all": r["range_all"], "range_lin": r["range_lin"], "delta": r["delta"]} for m, r in R.items()}}
    json.dump(out, open(f"{base}/summary.json", "w"), indent=1)
    open(f"{base}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------- T4

def t50(steps, vals, start, kind="level"):
    s, v = np.asarray(steps), np.asarray(vals, dtype=float)
    k = s >= start
    s, v = s[k], v[k]
    o = np.argsort(s)
    s, v = s[o], v[o]
    if kind == "loss":
        i0 = int(np.argmin(v))
        tot = v[-1] - v[i0]
        if tot <= 0:
            return None
        hit = np.flatnonzero(v[i0:] >= v[i0] + 0.5 * tot)
        return int(s[i0 + hit[0]])
    tot = v[-1] - v[0]
    if tot == 0:
        return None
    hit = np.flatnonzero((v - v[0]) / tot >= 0.5)
    return int(s[hit[0]])


def seed_series(name):
    """Per checkpoint: V1 ||ubar||^2, W_U s1^2/sum, eval loss (round6_eval) for a seed (original seed: rounds 4/5)."""
    rows = {}
    if "-seed" in name:
        for f in glob.glob(f"results/round6/poly/{name}/step*.json"):
            r = json.load(open(f))
            rows[int(r["revision"][4:])] = {"ub": r["step1"]["V1"]["all"]["ubar_sq"], "s1": r["spec"]["W_U"]["s1sq_over_sum"]}
    else:
        for rnd in ("round4", "round5"):
            for f in glob.glob(f"results/{rnd}/dyn/{name}/step*.json"):
                st = int(os.path.basename(f)[4:-5])
                sp = f"results/round5/spec/{name}/step{st}.json"
                if os.path.exists(sp):
                    rows[st] = {"ub": json.load(open(f))["step1"]["V1"]["all"]["ubar_sq"],
                                "s1": json.load(open(sp))["W_U"]["s1sq_over_sum"]}
    for st in list(rows):
        e = f"results/round6/p2/poly_eval_fp32/{name}/step{st}.json"  # fp32 path (T4 amendment); poly_eval/ (bf16) is superseded
        rows[st]["loss"] = json.load(open(e))["loss"] if os.path.exists(e) else None
    return rows


def t4():
    L = ["# Round 6 phase 2 T4: ordering of residual-stream collapse and head spike across seeds\n"]
    out = {"seeds": {}}
    grid70 = list(range(50000, 143001, 1000))
    L.append("| model | n ckpts | V1 ‖ū‖² start → end | σ₁²/Σσ² start → end | t50 ‖ū‖² | t50 σ₁² | t50 loss | ‖ū‖² first? | t50 σ₁² − t50 ‖ū‖² | t50 loss − t50 ‖ū‖² |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for size, start, seeds in (("pythia-70m", 50000, [None] + list(range(1, 10))), ("pythia-160m", 80000, [None] + list(range(1, 10)))):
        for k in seeds:
            name = size if k is None else f"{size}-seed{k}"
            rows = seed_series(name)
            st = sorted(s for s in rows if s >= start)
            if not st:
                continue
            ub = [rows[s]["ub"] for s in st]
            s1 = [rows[s]["s1"] for s in st]
            lo = [(s, rows[s]["loss"]) for s in st if rows[s]["loss"] is not None]
            full = (set(grid70) <= set(st)) if size == "pythia-70m" else None
            tu, ts = t50(st, ub, start), t50(st, s1, start)
            tl = t50([a for a, _ in lo], [b for _, b in lo], start, "loss") if len(lo) >= 3 else None
            out["seeds"][name] = {"n": len(st), "complete_grid": full, "t50_ubar": tu, "t50_s1": ts, "t50_loss": tl,
                                  "ubar_first": (tu is not None and ts is not None and tu < ts),
                                  "ub_start_end": [ub[0], ub[-1]], "s1_start_end": [s1[0], s1[-1]], "steps": st,
                                  "ub": ub, "s1": s1, "loss": [b for _, b in lo], "loss_steps": [a for a, _ in lo]}
            L.append(f"| {name} | {len(st)} | {ub[0]:.3f} → {ub[-1]:.3f} | {s1[0]:.3f} → {s1[-1]:.3f} | {tu} | {ts} | {tl} | "
                     f"{'yes' if out['seeds'][name]['ubar_first'] else 'no'} | {(ts - tu) if (ts and tu) else '–'} | {(tl - tu) if (tl and tu) else '–'} |")
    col = [f"pythia-70m-seed{k}" for k in range(2, 10)]
    use = [n for n in col if n in out["seeds"] and out["seeds"][n]["complete_grid"]]
    n_yes = sum(out["seeds"][n]["ubar_first"] for n in use)
    need = 7 if len(use) == 8 else len(use) - 1
    out["P7-5"] = {"seeds_used": use, "n_yes": n_yes, "n": len(use), "need": need, "pass": len(use) > 0 and n_yes >= need}
    L.append(f"\n- **P7-5**: t50(V1 ‖ū‖²) < t50(σ₁²/Σσ²) in {n_yes}/{len(use)} collapsed 70m seeds with a complete grid "
             f"(need ≥ {need}) -> {('PASS' if out['P7-5']['pass'] else 'FAIL') if use else 'NOT EVALUABLE (no complete grid yet)'}")
    for grp, names in (("70m seeds 2-9", col), ("160m all seeds", ["pythia-160m"] + [f"pythia-160m-seed{k}" for k in range(1, 10)])):
        v = [out["seeds"][n] for n in names if n in out["seeds"]]
        g1 = [x["t50_s1"] - x["t50_ubar"] for x in v]
        g2 = [x["t50_loss"] - x["t50_ubar"] for x in v if x["t50_loss"] is not None]
        nf = sum(x["ubar_first"] for x in v)
        L.append(f"- {grp}: ‖ū‖² strictly first in {nf}/{len(v)}; t50 σ₁² − t50 ‖ū‖² median {int(np.median(g1))} (range {min(g1)} – {max(g1)}); "
                 f"t50 loss − t50 ‖ū‖² median {int(np.median(g2))} (range {min(g2)} – {max(g2)}, {sum(x > 0 for x in g2)}/{len(g2)} > 0)")
        rise = [x["loss"][-1] - min(x["loss"]) for x in v if len(x["loss"]) >= 3]
        L.append(f"  - {grp}: fp32 eval-loss rise (end − in-window min) median {np.median(rise):.4f} (range {min(rise):.4f} – {max(rise):.4f}, n={len(rise)})")
    # 160m cross-correlation of first differences
    xc = {}
    for k in (1, 3, 6, 9):
        n = f"pythia-160m-seed{k}"
        if n not in out["seeds"]:
            continue
        a = np.diff(out["seeds"][n]["ub"])
        b = np.diff(out["seeds"][n]["s1"])
        if len(a) < 20:
            continue
        cc = {}
        for lag in range(-5, 6):  # positive lag: ubar change at t correlates with s1 change at t + lag (ubar leads)
            x, y = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
            cc[lag] = float(np.corrcoef(x, y)[0, 1])
        xc[n] = cc
    if xc:
        avg = {lag: float(np.mean([c[lag] for c in xc.values()])) for lag in range(-5, 6)}
        out["xcorr_160m"] = {"per_seed": xc, "mean": avg, "argmax_lag": max(avg, key=avg.get)}
        L.append("- 160m cross-correlation of first differences (positive lag = ‖ū‖² leads): mean " +
                 ", ".join(f"{lag:+d}: {v:+.2f}" for lag, v in avg.items()) + f"; argmax lag {out['xcorr_160m']['argmax_lag']:+d}")
    json.dump(out, open("results/round6/p2/poly_summary.json", "w"), indent=1)
    open("results/round6/p2/poly_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = [n for n in out["seeds"] if n.startswith("pythia-70m")]
    for i, n in enumerate(names):
        v = out["seeds"][n]
        y = len(names) - 1 - i
        for key, col, mk, lab in (("t50_ubar", C1, "o", "t50 V1 ‖ū‖²"), ("t50_s1", C2, "s", "t50 W_U σ₁²/Σσ²"),
                                  ("t50_loss", C3, "D", "t50 eval loss")):
            if v[key] is not None:
                ax.scatter(v[key], y, color=col, marker=mk, s=40, edgecolor="white", lw=1, zorder=3, label=lab if i == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("pythia-70m", "70m").replace("-seed", " seed ") if "seed" in n else "70m seed 0 (orig.)"
                        for n in names][::-1])
    ax.set_xlabel("training step")
    ax.set_title("Pythia-70m: when each quantity reaches 50% of its late change", fontsize=10, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figs/r6p2_fig4_t50_by_seed.png", dpi=160)
    print("wrote figs/r6p2_fig4_t50_by_seed.png")


def t3check():
    L = ["# Round 6 phase 2 T3 numerical check (prereg T3 amendment): eigh vs SVD vs SVD with numerical-rank tolerance\n",
         "| model | route | 7-rep range | dropped (V1, V3, D_Amax, D_Amin, lin0-2) | max |cos − cos_V1| |", "|---|---|---|---|---|"]
    out = {}
    for f in sorted(glob.glob("results/round6/p2/whiten_check/*.json")):
        r = json.load(open(f))
        reps = ["V1", "V3", "D_Amax", "D_Amin", "lin0", "lin1", "lin2"]
        out[r["model"]] = {k: {"range": v, "dropped": [r["zca"][f"{x}/{k}"]["n_dropped"] for x in reps],
                               "max_cos_diff": max(r["max_abs_cos_diff_vs_V1"][k].values())} for k, v in r["range_all"].items()}
        for k, v in out[r["model"]].items():
            L.append(f"| {r['model']} | {k} | {v['range']:.2e} | {v['dropped']} | {v['max_cos_diff']:.2e} |")
    ok = [m for m, v in out.items() if v["svd_ranktol"]["range"] < 1e-6]
    L.append(f"\n- route (c) svd_ranktol range < 1e-6 in {len(ok)}/{len(out)} checked models: {ok}")
    L.append("- reading (prereg amendment): (c) < 1e-6 for Qwen3-8B and Qwen3-8B-Base -> numerical/threshold issue, stop condition not triggered; "
             "otherwise stop condition")
    json.dump(out, open("results/round6/p2/whiten_check/summary.json", "w"), indent=1)
    open("results/round6/p2/whiten_check/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def last_eval(r):
    k = max(r["evals"], key=int)
    return int(k), r["evals"][k]


def t5(base="results/round6/p2/refit"):
    """base = results/round6/p2/refit (bf16 body, preregistered) or results/round6/p2/refit_fp32 (precision check)."""
    fp32 = base.rstrip("/").endswith("_fp32")
    runs = {}
    for f in glob.glob(f"{base}/*/*/*.json"):
        if f.endswith("summary.json"):
            continue
        r = json.load(open(f))
        runs[(r["model"], r["revision"], r["arm"], r["seed"])] = r
    L = [f"# Round 6 phase 2 T5: reduced-head refit (gamma, beta, W_U) on frozen bodies ({'fp32 body, precision check' if fp32 else 'bf16 body'})\n",
         "| model | rev | arm | seed | complete | extended | loss orig. head | loss refit (step) | σ₁²/Σσ² start → end | σ₁/σ₂ end | |cos(ū, v₁)| end |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(runs, key=lambda k: (SIZES.index(k[0]), step_of(k[1]), k[2], k[3])):
        r = runs[key]
        st, le = last_eval(r)
        se = r.get("spec_end") or {}
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {r['complete']} | {r['extended']} | {r.get('loss_orig_head', float('nan')):.4f} | "
                 f"{le:.4f} ({st}) | {r['spec_start']['s1sq_over_sum']:.3f} → {se.get('s1sq_over_sum', float('nan')):.3f} | "
                 f"{se.get('s1_over_s2', float('nan')):.2f} | {se.get('abs_cos_ubar_v1', float('nan')):.3f} |")
    out = {"runs": {"/".join(map(str, k)): {"complete": v["complete"], "extended": v["extended"], "loss_orig_head": v.get("loss_orig_head"),
                                            "loss_refit": last_eval(v)[1], "last_step": last_eval(v)[0], "spec_start": v["spec_start"],
                                            "spec_end": v.get("spec_end")} for k, v in runs.items()}, "D": {}}
    L.append("\n| model | pre → post | D_raw | D_refit (random head) | per-seed D_refit | D_refit / D_raw | reading |")
    L.append("|---|---|---|---|---|---|---|")
    for m, pre in (("pythia-70m", "step50000"), ("pythia-160m", "step80000"), ("pythia-410m", "step80000")):
        post = "step143000"
        seeds = sorted({k[3] for k in runs if k[0] == m and k[1] == post and k[2] == "random"}
                       & {k[3] for k in runs if k[0] == m and k[1] == pre and k[2] == "random"})
        seeds = [s for s in seeds if runs[(m, post, "random", s)]["complete"] and runs[(m, pre, "random", s)]["complete"]]
        if not seeds:
            continue
        a, b = runs[(m, post, "random", seeds[0])], runs[(m, pre, "random", seeds[0])]
        draw = a["loss_orig_head"] - b["loss_orig_head"]
        per = [last_eval(runs[(m, post, "random", s)])[1] - last_eval(runs[(m, pre, "random", s)])[1] for s in seeds]
        dref = float(np.mean(per))
        ratio = dref / draw if draw > 0 else None
        read = ("not interpreted (D_raw ≤ 0)" if draw <= 0 else "readout side" if ratio <= 0.25 else "body side" if ratio >= 0.75 else "mixed")
        out["D"][m] = {"pre": pre, "post": post, "D_raw": draw, "D_refit": dref, "per_seed": dict(zip(seeds, per)), "ratio": ratio, "reading": read}
        L.append(f"| {m} | {pre} → {post} | {draw:+.4f} | {dref:+.4f} | {', '.join(f'{x:+.4f}' for x in per)} | "
                 f"{'–' if ratio is None else f'{ratio:.2f}'} | {read} |")
    L.append("\n- last-interval eval drop (10000 steps: 9000 → 10000; extended: 19000 → 20000): " + "; ".join(
        f"{k[0][7:]} {k[1][4:]} {k[2]} s{k[3]} {runs[k].get('drop_last_1000_final', float('nan')):+.4f}" for k in sorted(runs) if runs[k]["complete"]))
    p6 = [runs.get(("pythia-70m", "step143000", "random", s)) for s in (0, 1, 2)]
    if all(r is not None and r["complete"] for r in p6):
        v = [r["spec_end"]["s1sq_over_sum"] for r in p6]
        out["P7-6"] = {"s1sq_over_sum": v, "pass": all(x < 0.5 for x in v)}
        L.append(f"\n- **P7-6**: 70m step143000 random-init refit σ₁²/Σσ² = {', '.join(f'{x:.3f}' for x in v)} -> "
                 f"{'PASS' if out['P7-6']['pass'] else 'FAIL'} (all < 0.5)")
    else:
        L.append("\n- **P7-6**: not evaluable yet (runs incomplete)")
    json.dump(out, open(f"{base}/summary.json", "w"), indent=1)
    open(f"{base}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    if fp32:
        return
    plt = plt_setup()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = {"step50000": C1, "step80000": C1, "step110000": C3, "step143000": C2}
    for (m, rev, arm, sd), r in sorted(runs.items()):
        if m != "pythia-70m":
            continue
        ks = sorted(r["evals"], key=int)
        ls = {"random": "-", "own": "--", "trunc": ":"}[arm]
        axs[0].plot([int(k) for k in ks[1:]], [r["evals"][k] for k in ks[1:]], ls, color=cols[rev], lw=1.5,
                    label=f"{rev} {arm}" if sd == 0 else None)
        if arm == "random":
            axs[0].axhline(r["loss_orig_head"], color=cols[rev], lw=0.8, alpha=0.6)
    axs[0].set_ylim(3.5, 6.0)
    axs[0].set_xlabel("refit step"); axs[0].set_ylabel("eval loss (nats/token)")
    axs[0].set_title("70m refit eval loss (thin lines: original head;\nown/trunc arms peak above 30 at step 1000, clipped)", fontsize=9, loc="left")
    axs[0].legend(frameon=False, fontsize=7)
    names = [k for k in sorted(runs, key=lambda k: (SIZES.index(k[0]), step_of(k[1]), k[2], k[3])) if runs[k].get("spec_end")]
    x = np.arange(len(names))
    axs[1].bar(x - 0.2, [runs[k]["spec_start"]["s1sq_over_sum"] if k[2] != "random" else np.nan for k in names], 0.4, color=GRAY_, label="start (own/trunc)")
    axs[1].bar(x + 0.2, [runs[k]["spec_end"]["s1sq_over_sum"] for k in names], 0.4, color=C1, label="after refit")
    axs[1].axhline(0.5, color=C2, lw=1, ls="--")
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"{k[0][7:]} {k[1][4:]} {k[2]} s{k[3]}" for k in names], rotation=70, fontsize=6)
    axs[1].set_ylabel("σ₁²/Σσ² of W·diag(γ)"); axs[1].legend(frameon=False, fontsize=7)
    axs[1].set_title("head spectrum after refit", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig("figs/r6p2_fig5_refit.png", dpi=160)
    print("wrote figs/r6p2_fig5_refit.png")


def t6():
    L = ["# Round 6 phase 2 T6 (exploratory): full fine-tuning of pythia-70m\n",
         "| start | arm | λ | V1 ‖ū‖² 0 → end | σ₁²/Σσ² 0 → end | eval loss 0 → end | steps |", "|---|---|---|---|---|---|---|"]
    out = {}
    runs = {}
    for f in sorted(glob.glob("results/round6/p2/ft/*.json")):
        if f.endswith("summary.json"):
            continue
        r = json.load(open(f))
        runs[(r["revision"], r["arm"], r["lambda"])] = r
    for k, r in sorted(runs.items()):
        ks = sorted(r["metrics"], key=int)
        a, b = r["metrics"][ks[0]], r["metrics"][ks[-1]]
        out["/".join(map(str, k))] = {"metrics": r["metrics"]}
        L.append(f"| {k[0]} | {k[1]} | {k[2]:g} | {a['ubar_sq_V1']:.3f} → {b['ubar_sq_V1']:.3f} | {a['s1sq_over_sum']:.3f} → {b['s1sq_over_sum']:.3f} | "
                 f"{a['eval_loss']:.4f} → {b['eval_loss']:.4f} | {ks[-1]} |")
    L.append("\n- eval-loss trajectory (step: loss) of the R-pen runs:")
    for k, r in sorted(runs.items()):
        if k[1] == "rpen":
            ks = sorted(r["metrics"], key=int)
            L.append(f"  - {k[0]} λ={k[2]:g}: " + ", ".join(f"{s}: {r['metrics'][s]['eval_loss']:.3f}" for s in ks)
                     + f"; V1 ‖ū‖² at step {ks[1]}: {r['metrics'][ks[1]]['ubar_sq_V1']:.3f}")
    os.makedirs("results/round6/p2/ft", exist_ok=True)
    json.dump(out, open("results/round6/p2/ft/summary.json", "w"), indent=1)
    open("results/round6/p2/ft/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    plt = plt_setup()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    cols = {"plain": C1, "ctrlpen": C3}
    rcol = {1.0: C2, 0.1: C4, 10.0: GRAY_}
    for (rev, arm, lam), r in sorted(runs.items()):
        ks = sorted(r["metrics"], key=int)
        for ax, key in zip(axs, ("ubar_sq_V1", "s1sq_over_sum", "eval_loss")):
            ax.plot([int(k) for k in ks], [r["metrics"][k][key] for k in ks], "-o" if rev == "step143000" else "--s",
                    color=rcol[float(lam)] if arm == "rpen" else cols[arm], ms=4, lw=1.5, label=f"{rev[4:]} {arm}" + (f" λ={lam:g}" if arm == "rpen" else ""))
    for ax, t in zip(axs, ("V1 ‖ū‖²", "W_U σ₁²/Σσ²", "eval loss")):
        ax.set_title(t, fontsize=10, loc="left"); ax.set_xlabel("fine-tuning step")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig("figs/r6p2_fig6_finetune.png", dpi=160)
    print("wrote figs/r6p2_fig6_finetune.png")


if __name__ == "__main__":
    {"t1": t1, "t2": t2, "t3": t3, "t3check": t3check, "t4": t4, "t5": t5, "t6": t6, "fb3": fb3}[sys.argv[1]](*sys.argv[2:])
