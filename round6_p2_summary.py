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


if __name__ == "__main__":
    {"t1": t1, "t2": t2}[sys.argv[1]]()
