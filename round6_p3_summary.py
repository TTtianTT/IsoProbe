"""Round 6 phase 3 summaries. Usage: $PY round6_p3_summary.py u0|u1

u0: P8-1 (fp32 Spearman of the second-order predictor vs dloss on the U0 checkpoints) and the original / fp32 table of
    dloss, additive share and corr(delta_S, delta_R); saturation magnitude from the phase-2 T4 fp32 eval loss.
    -> results/round6/p3/fp32_audit/summary.{md,json}
u1: P8-3 / P8-4 from results/round6/p3/gauge_resid/*.json -> results/round6/p3/gauge_resid/summary.{md,json}
"""
import glob
import json
import os
import sys

import numpy as np
from round3_lib import spearman as _sp


def spearmanr(a, b):
    class R:
        statistic = _sp(a, b)[0]
    return R

SETS = ["all", "top1", "top8", "top32"]
A = "results/round6/p3/fp32_audit"


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def orig(name, rev):
    for d in ("results/round4/dyn", "results/round5/dyn"):
        r = jl(f"{d}/{name}/{rev}.json")
        if r is not None:
            return r
    return None


def u0():
    rows = [l.split("\t")[:2] for l in open("configs/round6_p3_audit_ckpts.tsv").read().splitlines() if l.strip()]
    pts, tab, missing = [], [], []
    for repo, rev in rows:
        name = repo.split("/")[1]
        dn, lg, do, lo = jl(f"{A}/dyn/{name}/{rev}.json"), jl(f"{A}/logit/{name}/{rev}.json"), orig(name, rev), \
            jl(f"results/round5/logit/{name}/{rev}.json")
        if dn is None or lg is None:
            missing.append(f"{name}@{rev}")
            continue
        for n in SETS:
            pts.append({"model": name, "rev": rev, "set": n, "pred": lg["sets"][n]["half_E_var"],
                        "dloss": dn["function"]["sets"][n]["dloss"],
                        "pred_orig": lo["sets"][n]["half_E_var"] if lo else None,
                        "dloss_orig": do["function"]["sets"][n]["dloss"] if do else None})
        f, fo = dn["function"], do["function"]
        e = {"model": name, "rev": rev, "loss_fp32": f["loss_base"], "loss_orig": fo["loss_base"]}
        for n in ("all", "top8", "top16", "top32"):
            e[f"dl_{n}"] = (fo["sets"][n]["dloss"], f["sets"][n]["dloss"])
        for n in ("top8", "top16", "top32"):
            e[f"add_{n}"] = (fo["sets"][n]["additive_share"], f["sets"][n]["additive_share"])
            e[f"corr_{n}"] = (fo["sets"][n]["corr_S_R"], f["sets"][n]["corr_S_R"])
        tab.append(e)
    x, y = [p["pred"] for p in pts], [p["dloss"] for p in pts]
    rho = spearmanr(x, y).statistic if pts else float("nan")
    po = [p for p in pts if p["pred_orig"] is not None]
    rho_o = spearmanr([p["pred_orig"] for p in po], [p["dloss_orig"] for p in po]).statistic if po else float("nan")
    per = {}
    for m in sorted({p["model"] for p in pts}):
        q = [p for p in pts if p["model"] == m]
        per[m] = {"n": len(q), "rho_fp32": spearmanr([p["pred"] for p in q], [p["dloss"] for p in q]).statistic,
                  "rho_orig": spearmanr([p["pred_orig"] for p in q], [p["dloss_orig"] for p in q]).statistic}
    flips = {k: int(sum(np.sign(e[k][0]) != np.sign(e[k][1]) for e in tab)) for k in tab[0] if k.startswith(("dl_", "add_", "corr_"))} if tab else {}
    sub_gt_all = [(e["model"], e["rev"], k) for e in tab for k in ("dl_top8", "dl_top16", "dl_top32")
                  if (e[k][0] > e["dl_all"][0]) != (e[k][1] > e["dl_all"][1])]
    # saturation magnitude, phase-2 T4 fp32 eval loss, original seed dense window
    sat = {}
    for m, w in (("pythia-70m", 50000), ("pythia-160m", 80000)):
        ls = []
        for fn in glob.glob(f"results/round6/p2/poly_eval_fp32/{m}/step*.json"):
            s = int(os.path.basename(fn)[4:-5])
            if s >= w:
                r = json.load(open(fn))
                ls.append((s, r.get("eval_loss", r.get("loss"))))
        ls.sort()
        if ls and ls[0][1] is not None:
            mn = min(ls, key=lambda t: t[1])
            sat[m] = {"n": len(ls), "min_step": mn[0], "min": mn[1], "end": ls[-1][1], "end_minus_min": ls[-1][1] - mn[1],
                      "max_minus_min": max(t[1] for t in ls) - mn[1]}
    res = {"n_points": len(pts), "n_checkpoints": len(tab), "missing": missing, "P8_1": {"rho_fp32": rho, "rho_orig_same_points": rho_o,
           "pass": bool(rho > 0.95) if not missing else None}, "per_model": per, "sign_flips": flips,
           "subset_gt_all_changes": sub_gt_all, "saturation_fp32_T4": sat,
           "saturation_orig_round5": {"pythia-70m": {"end_minus_min": 0.1934, "max_minus_min": 0.2153},
                                      "pythia-160m": {"end_minus_min": 0.1263, "max_minus_min": 0.1702}}, "table": tab}
    json.dump(res, open(f"{A}/summary.json", "w"), indent=1)
    L = ["# U0 fp32 audit (dyn / logit part)\n", f"checkpoints {len(tab)} / {len(rows)}; points {len(pts)}; missing {missing}\n",
         f"P8-1: Spearman(pred, dloss) fp32 = {rho:.4f} (same points, original bf16 = {rho_o:.4f}) -> "
         f"{'pass' if res['P8_1']['pass'] else 'FAIL' if res['P8_1']['pass'] is False else 'incomplete'}\n",
         "| model | n | rho fp32 | rho orig |", "|---|---|---|---|"]
    L += [f"| {m} | {v['n']} | {v['rho_fp32']:.4f} | {v['rho_orig']:.4f} |" for m, v in per.items()]
    L += ["", f"sign flips (orig vs fp32) over {len(tab)} checkpoints: {flips}",
          f"'subset dloss > all' changes: {sub_gt_all}", "", "| model | step | loss orig / fp32 | dl all | dl top8 | dl top32 | add top8 | corr top32 |",
          "|---|---|---|---|---|---|---|---|"]
    for e in tab:
        L.append(f"| {e['model']} | {e['rev'][4:]} | {e['loss_orig']:.4f} / {e['loss_fp32']:.4f} | "
                 + " | ".join(f"{e[k][0]:+.3f} / {e[k][1]:+.3f}" for k in ("dl_all", "dl_top8", "dl_top32", "add_top8", "corr_top32")) + " |")
    L += ["", f"saturation (T4 fp32, round-6 eval set): {json.dumps(sat)}",
          "saturation (round 5, bf16, round-5 eval set): 70m end-min 0.1934 / max-min 0.2153; 160m 0.1263 / 0.1702"]
    open(f"{A}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def u0s4():
    """S4 part of U0: ablation and per-token shares, original (bf16 body) vs fp32; P8-2."""
    rows = [l.split("\t")[:2] for l in open("configs/round6_causal.tsv").read().splitlines() if l.strip()]
    ex = jl("results/round6/causal/explore_tokens.json") or {}
    tab = []
    for m, rev in rows:
        o, f = jl(f"results/round6/causal/ablate/{m}/{rev}.json"), jl(f"{A}/s4/{m}/{rev}.json")
        t = jl(f"{A}/s4_tokens/{m}/{rev}.json")
        e = {"model": m, "rev": rev}
        for tag, r in (("orig", o), ("fp32", f)):
            dl = r["dloss"]
            e[tag] = {"base": r["loss"]["base"], "mean": dl["ubar/mean"], "fix": dl["ubar/fix"], "ratio": dl["ubar/fix"] / dl["ubar/mean"],
                      "trunc_bias": dl["trunc+bias"], "control": dl["control_bias"], "trunc_bias_minus_control": r["trunc_bias_minus_control"],
                      "trunc_b0": dl["trunc+b0"], "s1sq": r["svd"]["s1sq_over_sum"]}
        e["fp32"]["share_top0.1pct_fix"] = t["fix"]["share_top0.1pct"]
        e["fp32"]["share_top0.1pct_trunc"] = t["trunc_meancomp"]["share_top0.1pct"]
        k = f"{m}/{rev}"
        if k in ex:
            e["orig"]["share_top0.1pct_fix"] = ex[k]["fix"]["share_top0.1pct"]
            e["orig"]["share_top0.1pct_trunc"] = ex[k]["trunc_meancomp"]["share_top0.1pct"]
        tab.append(e)
    get = lambda m, r: next(e for e in tab if e["model"] == m and e["rev"] == r)  # noqa: E731
    p82 = {m: get(m, "step143000")["fp32"]["share_top0.1pct_fix"] for m in ("pythia-70m", "pythia-160m")}
    res = {"P8_2": {"shares": p82, "pass": bool(all(v >= 0.8 for v in p82.values()))}, "table": tab}
    json.dump(res, open(f"{A}/summary_s4.json", "w"), indent=1)
    L = ["# U0 fp32 audit: phase-1 S4 (orig = bf16 body + fp32 head; fp32 = whole model fp32)\n",
         f"P8-2: top-0.1% share of the fix-ubar cost, fp32: 70m {p82['pythia-70m']:.4f}, 160m {p82['pythia-160m']:.4f} -> "
         f"{'pass' if res['P8_2']['pass'] else 'FAIL'}\n",
         "| model | step | base loss | (1) mean | (2) fix | (2)/(1) | trunc+b0 | trunc+bias | control | trunc+bias - control | top-0.1% share (2) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in tab:
        o, f = e["orig"], e["fp32"]
        sh = f"{o['share_top0.1pct_fix']:.3f} / {f['share_top0.1pct_fix']:.3f}" if "share_top0.1pct_fix" in o else f"– / {f['share_top0.1pct_fix']:.3f}"
        L.append(f"| {e['model'][7:]} | {e['rev'][4:]} | {o['base']:.4f} / {f['base']:.4f} | "
                 + " | ".join(f"{o[k]:+.3f} / {f[k]:+.3f}" for k in ("mean", "fix", "ratio", "trunc_b0", "trunc_bias", "control", "trunc_bias_minus_control"))
                 + f" | {sh} |")
    open(f"{A}/summary_s4.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def u2():
    """U2: P8-5, P8-6, last / eol comparisons, raw ranges, fit size, Ledoit-Wolf, spectrum; figure 1."""
    D = "results/round6/p3/whiten_wide"
    models = [os.path.basename(l.strip().rstrip("/")) for l in open("models.txt") if l.strip()]
    R = {m: jl(f"{D}/{m}.json") for m in models}
    missing = [m for m, r in R.items() if r is None]
    R = {m: r for m, r in R.items() if r is not None}
    DS = ["sts12", "sts13", "sts14", "sts15", "sts16", "stsb", "sickr"]
    REPS = ["V1", "V3", "D_Amax", "D_Amin", "lin0", "lin1", "lin2"]
    out = {"missing": missing, "per_model": {}}
    n_cases = n_ok = 0
    worst = []
    for m, r in R.items():
        pm = {}
        for pool, p in r["pools"].items():
            wr = {ds: p["range"][f"whiten/{ds}"] for ds in DS}
            n_cases += len(DS)
            n_ok += sum(v < 1e-6 for v in wr.values())
            worst += [(v, m, pool, ds) for ds, v in wr.items() if v >= 1e-6]
            mean7 = lambda rep, var: float(np.mean([p["scores"][f"{rep}/{var}/{ds}"] for ds in DS]))  # noqa: E731
            pm[pool] = {"whiten_range_max": max(wr.values()), "w_minus_z": p["whiten_minus_zscore_mean7"],
                        "V1_mean7": p["V1_mean7"],
                        "raw_mean7_by_rep": {rep: mean7(rep, "raw") for rep in REPS},
                        "zscore_mean7_by_rep": {rep: mean7(rep, "zscore") for rep in REPS},
                        "whiten_mean7_by_rep": {rep: mean7(rep, "whiten") for rep in REPS},
                        "raw_range_by_ds": {ds: p["range"][f"raw/{ds}"] for ds in DS},
                        "zscore_range_lin_by_ds": {ds: p["range_lin"][f"zscore/{ds}"] for ds in DS},
                        "lw_range_max": max(p["range"][f"whiten_lw/{ds}"] for ds in DS), "lw_rho_V1": p["lw_rho"]["V1"],
                        "fit_size": {k: {"V1_mean7": mean7("V1", f"whiten_n{k}d"),
                                         "range_max": max(p["range"][f"whiten_n{k}d/{ds}"] for ds in DS),
                                         "n_dropped_V1": p["zca"][f"V1/whiten_n{k}d"]["n_dropped"]}
                                     for k in (1, 2, 4, 8) if f"V1/whiten_n{k}d/sts12" in p["scores"]},
                        "n_dropped_main": {rep: p["zca"][f"{rep}/whiten"]["n_dropped"] for rep in REPS},
                        "cond_kept_main_max": max(p["zca"][f"{rep}/whiten"]["cond_kept"] for rep in REPS),
                        "spectrum": p["spectrum"]}
        out["per_model"][m] = pm
    ci = {pool: sum(out["per_model"][m][pool]["w_minus_z"]["ci_lo"] > 0 for m in R) for pool in ("mean", "last", "eol")}
    ci_neg = {pool: sum(out["per_model"][m][pool]["w_minus_z"]["ci_hi"] < 0 for m in R) for pool in ("mean", "last", "eol")}
    out["P8_5"] = {"n_cases": n_cases, "n_below_1e-6": n_ok, "pass": bool(n_ok == n_cases and not missing), "failures": sorted(worst, reverse=True)[:20]}
    out["P8_6"] = {"n_models_ci_above_0": ci["mean"], "pass": bool(ci["mean"] >= 9 and not missing)}
    out["ci_above_0"], out["ci_below_0"] = ci, ci_neg
    json.dump(out, open(f"{D}/summary.json", "w"), indent=1)
    L = ["# U2 whitening on 7 STS sets x 3 poolings\n", f"models {len(R)} / 12; missing {missing}\n",
         f"P8-5: whitening 7-rep range < 1e-6 in {n_ok} / {n_cases} (model x dataset x pool) -> {'pass' if out['P8_5']['pass'] else 'FAIL'}",
         f"P8-6: mean pooling, mean over 7 sets, whiten - zscore CI above 0 in {ci['mean']} / {len(R)} models -> {'pass' if out['P8_6']['pass'] else 'FAIL'}",
         f"CI above 0 / below 0: mean {ci['mean']} / {ci_neg['mean']}, last {ci['last']} / {ci_neg['last']}, eol {ci['eol']} / {ci_neg['eol']}\n",
         "| model | pool | V1 raw | zscore | whiten | abtt10 | whiten LW | whiten - zscore [CI] | raw range (mean7 over reps) | zscore G_lin range max | LW range max | rankme/d | cond kept | top1 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in R:
        for pool in ("mean", "last", "eol"):
            q = out["per_model"][m][pool]
            v = q["V1_mean7"]
            w = q["w_minus_z"]
            rr = q["raw_mean7_by_rep"]
            L.append(f"| {m} | {pool} | {v['raw']:.1f} | {v['zscore']:.1f} | {v['whiten']:.1f} | {v['abtt10']:.1f} | {v['whiten_lw']:.1f} | "
                     f"{w['delta']:+.2f} [{w['ci_lo']:+.2f}, {w['ci_hi']:+.2f}] | {min(rr.values()):.1f} – {max(rr.values()):.1f} | "
                     f"{max(q['zscore_range_lin_by_ds'].values()):.2f} | {q['lw_range_max']:.2f} | {q['spectrum']['rankme_over_d']:.3f} | "
                     f"{q['spectrum']['cond_kept']:.1e} | {q['spectrum']['top1_share']:.3f} |")
    L += ["", "## fit size (mean pooling; V1 whiten mean over 7 sets / max 7-rep range / V1 dropped)", "| model | n = d | 2d | 4d | 8d | main (24000) |", "|---|---|---|---|---|---|"]
    for m in R:
        fsz = out["per_model"][m]["mean"]["fit_size"]
        L.append(f"| {m} | " + " | ".join(f"{fsz[k]['V1_mean7']:.2f} / {fsz[k]['range_max']:.1e} / {fsz[k]['n_dropped_V1']}" if k in fsz else "–"
                                          for k in (1, 2, 4, 8)) + f" | {out['per_model'][m]['mean']['V1_mean7']['whiten']:.2f} |")
    open(f"{D}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))
    # figure 1: mean pooling, mean over 7 sets
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = list(R)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, m in enumerate(ms):
        q = out["per_model"][m]["mean"]
        raw = list(q["raw_mean7_by_rep"].values())
        zs = list(q["zscore_mean7_by_rep"].values())
        wh = list(q["whiten_mean7_by_rep"].values())
        ax.plot([i - 0.2, i - 0.2], [min(raw), max(raw)], color="#2a78d6", lw=6, solid_capstyle="round", label="raw (range over 7 gauges)" if i == 0 else None)
        ax.plot([i, i], [min(zs), max(zs)], color="#eb6834", lw=6, solid_capstyle="round", label="zscore (range over 7 gauges)" if i == 0 else None)
        ax.scatter([i + 0.2], [np.mean(wh)], color="#1baf7a", s=40, zorder=3, label="whitening (cosines equal across 7 gauges to < 1e-8)" if i == 0 else None)
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([m.replace("-Instruct", "-I").replace("Mistral-Small-24B-I-2501", "Mistral-24B-I") for m in ms], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spearman x100, mean over 7 STS sets")
    ax.set_title("Mean pooling of V1: score under 7 output-preserving gauges")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("figs/r6p3_fig1_whiten_wide.png", dpi=150)


def u4():
    """U4: P8-8 (scatter), mechanism grid, P8-9 (12 models; 9 <= 8.2B); figure 3."""
    D = "results/round6/p3/precision"
    sc, mech = jl(f"{D}/scatter.json"), jl(f"{D}/mech.json") or {}
    models = [os.path.basename(l.strip().rstrip("/")) for l in open("models.txt") if l.strip()]
    small = models[:9]
    big = {m: jl(f"{D}/big/{m}.json") for m in models}
    rows = []
    for m in models:
        r = big[m]
        if r is None or "fp32" not in r or "bf16" not in r:
            rows.append((m, None))
            continue
        rows.append((m, {"loss_fp32": r["fp32"]["loss"], "loss_bf16": r["bf16"]["loss"], "diff": r["bf16"]["loss"] - r["fp32"]["loss"],
                         "dl_fp32": r["fp32"]["dloss_demeaned"], "dl_bf16": r["bf16"]["dloss_demeaned"],
                         "dl_diff": r["bf16"]["dloss_demeaned"] - r["fp32"]["dloss_demeaned"], "n": r["fp32"]["n_tokens"]}))
    R = dict(rows)
    p89 = [abs(R[m]["diff"]) < 0.01 for m in small if R[m] is not None]
    res = {"P8_8": sc["P8_8"], "rho_s1": sc["rho_s1"], "P8_9": {"n": len(p89), "n_pass": sum(p89),
           "pass": bool(len(p89) == 9 and all(p89))}, "big": R, "mech": mech}
    json.dump(res, open(f"{D}/summary.json", "w"), indent=1)
    L = ["# U4 precision\n", f"P8-8: Spearman(bf16 - fp32 loss, V1 ubar^2) = {sc['P8_8']['rho_ubar']:.4f} over {sc['n']} checkpoints -> "
         f"{'pass' if sc['P8_8']['pass'] else 'FAIL'}; vs s1^2/sum {sc['rho_s1']:.4f}",
         f"P8-9: |bf16 - fp32| < 0.01 in {sum(p89)} / {len(p89)} of the 9 models <= 8.2B -> {'pass' if res['P8_9']['pass'] else 'FAIL'}\n",
         "| model | loss fp32 | loss bf16 | bf16 - fp32 | demeaned dloss fp32 | bf16 | bf16 - fp32 | tokens |", "|---|---|---|---|---|---|---|---|"]
    for m, v in rows:
        L.append(f"| {m} | " + (" | ".join([f"{v['loss_fp32']:.4f}", f"{v['loss_bf16']:.4f}", f"{v['diff']:+.4f}", f"{v['dl_fp32']:+.4f}",
                                            f"{v['dl_bf16']:+.4f}", f"{v['dl_diff']:+.4f}", str(v['n'])]) if v else "missing | | | | | |") + " |")
    L += ["", "## mechanism (pythia-70m; body / final norm / head)", "| setting | step 143000 | step 50000 |", "|---|---|---|"]
    keys = sorted({k.split("/", 1)[1] for k in mech}, key=lambda k: list(dict.fromkeys(
        ["fp32-fp32-fp32", "bf16-bf16-bf16", "fp32-bf16-bf16", "bf16-fp32-bf16", "bf16-bf16-fp32", "bf16-fp32-fp32", "fp16-fp16-fp16", "fp16-fp32-fp32"])).index(k))
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{mech[f'{rv}/{k}']['loss']:.4f}" if f"{rv}/{k}" in mech else "–" for rv in ("step143000", "step50000")) + " |")
    open(f"{D}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    pts = sc["points"]
    for sz, c in (("70m", "#2a78d6"), ("160m", "#eb6834")):
        q = [p for p in pts if p["model"].startswith(f"pythia-{sz}")]
        for ax, key in zip(axes[:2], ("ubar", "s1")):
            ax.scatter([p[key] for p in q], [p["diff"] for p in q], s=8, color=c, alpha=0.6, linewidths=0, label=f"pythia-{sz} (10 seeds)")
    axes[0].set_xlabel("V1 ||ubar||^2")
    axes[1].set_xlabel("W_U sigma_1^2 / sum sigma^2")
    axes[0].set_ylabel("eval loss: bf16 - fp32 (nat)")
    axes[0].set_title(f"Spearman {sc['P8_8']['rho_ubar']:.3f} (n = {sc['n']})", fontsize=10)
    axes[1].set_title(f"Spearman {sc['rho_s1']:.3f}", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[2]
    lab = [k.replace("-", "/") for k in keys]
    for j, (rv, c) in enumerate((("step143000", "#2a78d6"), ("step50000", "#1baf7a"))):
        ax.barh([i + (j - 0.5) * 0.4 for i in range(len(keys))], [mech.get(f"{rv}/{k}", {}).get("loss", np.nan) - mech.get(f"{rv}/fp32-fp32-fp32", {}).get("loss", np.nan) for k in keys],
                height=0.36, color=c, label=f"70m {rv}")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("loss - all-fp32 loss (nat)")
    ax.set_title("pythia-70m: body / final norm / head dtype", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for a in axes:
        a.grid(color="#e6e6e3", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("figs/r6p3_fig3_precision.png", dpi=150)


def u1():
    D = "results/round6/p3/gauge_resid"
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/*.json")) if not f.endswith("summary.json")]
    pos = [r for r in rs if r["expected_invariant"]]
    key = lambda r: "max_abs_diff_n" if r["arch"] == "ln" else "max_abs_diff_raw"  # noqa: E731
    p83 = all(r["logits_max_rel_err"] < 1e-6 for r in pos)
    p84 = all(max(r[key(r)].values()) < 1e-6 for r in pos)
    L = ["# U1 residual-stream gauge check (fp64)\n", "| case | arch | transform | expected invariant | logits max rel err | argmax agree | "
         "max abs diff A / Bc / RankMe (judged) | raw-x diff A / RankMe (LN only) |", "|---|---|---|---|---|---|---|---|"]
    for r in rs:
        m = r[key(r)]
        raw = r["max_abs_diff_raw"] if r["arch"] == "ln" else None
        L.append(f"| {r['case']} | {r['arch']} | {r['transform']} | {r['expected_invariant']} | {r['logits_max_rel_err']:.2e} | "
                 f"{r['argmax_agree']:.5f} | {m['A']:.1e} / {m['Bc']:.1e} / {m['RankMe']:.1e} | "
                 + (f"{raw['A']:.3g} / {raw['RankMe']:.3g}" if raw else "–") + " |")
    L += ["", f"P8-3 (logits rel err < 1e-6 on all {len(pos)} in-group cases): {'pass' if p83 else 'FAIL'}",
          f"P8-4 (per-layer metric diffs < 1e-6): {'pass' if p84 else 'FAIL'}"]
    json.dump({"P8_3": p83, "P8_4": p84, "n_cases": len(rs), "n_in_group": len(pos),
               "cases": {r["case"]: {k: r[k] for k in ("arch", "transform", "expected_invariant", "logits_max_rel_err", "argmax_agree",
                                                     "max_abs_diff_raw")} | ({"max_abs_diff_n": r["max_abs_diff_n"]} if r["arch"] == "ln" else {})
                         for r in rs}}, open(f"{D}/summary.json", "w"), indent=1)
    open(f"{D}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"u0": u0, "u0s4": u0s4, "u1": u1, "u2": u2, "u4": u4}[sys.argv[1]]()
