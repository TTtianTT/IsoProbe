"""Round 5 analysis (login node, CPU). Implements only the tests defined in configs/round5_prereg.md;
everything else is written under "exploratory".
Writes results/round5/summary.md, results/round5/analysis.json, figs/r5_figC1-C4_*.png.
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze import COLORS, GRID, INK, INK2  # noqa: E402
from round3_lib import spearman  # noqa: E402

DYN5 = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1.4b", "OLMo-2-0425-1B"]
PYTHIA = DYN5[:4]
WINDOW = {"pythia-70m": 50000, "pythia-160m": 80000, "pythia-410m": 80000, "pythia-1.4b": 80000}
EARLY = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
PRE_SETS = ["all", "top1", "top8", "top32"]


def jl(fn):
    return json.load(open(fn)) if os.path.exists(fn) else None


def dyn(m, rev):
    return jl(f"results/round4/dyn/{m}/{rev}.json") or jl(f"results/round5/dyn/{m}/{rev}.json")


def step(rev):
    return int(re.search(r"step(\d+)", rev).group(1)) if "step" in rev else None


# ---------------------------------------------------------------- item 1

def item1(L, out):
    rows = [l.rstrip("\n").split("\t") for l in open("configs/round5_sts.tsv") if l.strip()]
    pts = {m: [] for m in DYN5}
    a1 = 0.0
    for repo, rev in rows:
        m = repo.split("/")[1]
        s = jl(f"results/round5/sts/{m}__{rev}.json")
        d = dyn(m, rev)
        if s is None:
            continue
        a1 = max(a1, s["checkA1_zscore_max_abs_delta"])
        sc = s["scores"]
        e = {"rev": rev, "dtop1": d["step1"]["V3"]["all"]["top1_var"] - d["step1"]["V1"]["all"]["top1_var"],
             "dA": d["step1"]["V1"]["all"]["isotropy_full"] - d["step1"]["V3"]["all"]["isotropy_full"]}
        for ds in ("stsb", "sickr"):
            for p in ("mean", "eol"):
                e[f"{ds}/{p}"] = sc[f"{ds}/{p}/V1/raw"] - sc[f"{ds}/{p}/V3/raw"]
                e[f"{ds}/{p}/ci"] = (s["delta"][f"{ds}/{p}/raw"]["ci_lo"], s["delta"][f"{ds}/{p}/raw"]["ci_hi"])
        pts[m].append(e)
    n = sum(len(v) for v in pts.values())
    key = "stsb/mean"
    rho_t = {m: spearman([e[key] for e in v], [e["dtop1"] for e in v])[0] for m, v in pts.items()}
    rho_a = {m: spearman([e[key] for e in v], [e["dA"] for e in v])[0] for m, v in pts.items()}
    obs = float(np.mean(list(rho_t.values())))
    rng = np.random.default_rng(0)
    perm = np.empty(10000)
    xs = {m: np.array([e["dtop1"] for e in v]) for m, v in pts.items()}
    ys = {m: np.array([e[key] for e in v]) for m, v in pts.items()}
    for i in range(10000):
        perm[i] = np.mean([spearman(rng.permutation(ys[m]), xs[m])[0] for m in pts])
    p1 = (1 + (perm >= obs).sum()) / 10001
    p2 = (1 + (np.abs(perm) >= abs(obs)).sum()) / 10001
    n_better = sum(abs(rho_t[m]) > abs(rho_a[m]) for m in pts)
    out["item1"] = {"n": n, "A1_max": a1, "rho_top1": rho_t, "rho_dA": rho_a, "mean_rho": obs, "p_one_sided": p1,
                    "p_two_sided": p2, "P1_1": bool(obs > 0 and p1 < 0.05), "P1_2_count": n_better,
                    "P1_2": bool(n_better >= 4), "points": pts}
    sec = {}
    for k2 in ("stsb/eol", "sickr/mean", "sickr/eol"):
        sec[k2] = {m: (spearman([e[k2] for e in v], [e["dtop1"] for e in v])[0],
                       spearman([e[k2] for e in v], [e["dA"] for e in v])[0]) for m, v in pts.items()}
    out["item1"]["secondary"] = sec
    L += ["## 第 1 项：A2' 的样本外检验\n",
          f"样本 {n} 个 checkpoint；zscore 下 \\|V1 − V3\\| 最大值 = {a1:.1e}（A1）。主要终点 = STS-B、mean、raw 的 ΔSTS(V1 − V3)。\n",
          "| model | n | ρ(ΔSTS, Δtop1) | ρ(ΔSTS, A_V1 − A_V3) | \\|ρ_top1\\| > \\|ρ_A\\| |", "|---|---|---|---|---|"]
    for m in pts:
        L.append(f"| {m} | {len(pts[m])} | {rho_t[m]:+.3f} | {rho_a[m]:+.3f} | {'yes' if abs(rho_t[m]) > abs(rho_a[m]) else 'no'} |")
    L.append(f"\n平均 ρ = {obs:+.3f}；分层置换检验（10000 次，seed 0）单侧 p = {p1:.4f}（双侧 p = {p2:.4f}，只报告）。"
             f"P1-1：{'成立' if out['item1']['P1_1'] else '不成立'}。P1-2：{n_better}/5 个模型 \\|ρ_top1\\| > \\|ρ_A\\|，"
             f"{'成立' if out['item1']['P1_2'] else '不成立'}。\n")
    L += ["### 次要终点（只报告）\n", "| setting | " + " | ".join(f"{m} ρ_top1 / ρ_A" for m in pts) + " |",
          "|---" * (1 + len(pts)) + "|"]
    for k2, v in sec.items():
        L.append(f"| {k2} | " + " | ".join(f"{v[m][0]:+.2f} / {v[m][1]:+.2f}" for m in pts) + " |")
    fig_c1(pts)


def fig_c1(pts):
    fig, ax = plt.subplots(figsize=(7.4, 5))
    S = [os.path.basename(l.strip()) for l in open("models.txt") if l.strip()]
    rx, ry = [], []
    for s in S:
        r3 = jl(f"results/round3/{s}.json")["step1"]
        r4 = jl(f"results/round4/sts/{s}.json")
        rx.append(r3["V3"]["all"]["top1_var"] - r3["V1"]["all"]["top1_var"])
        ry.append(r4["scores"]["stsb/mean/V1/raw"] - r4["scores"]["stsb/mean/V3/raw"])
    ax.scatter(rx, ry, s=40, facecolors="none", edgecolors=INK2, linewidths=1.2, label="round 4: 12 models (reference)")
    for i, m in enumerate(DYN5):
        v = pts[m]
        ax.plot([e["dtop1"] for e in v], [e["stsb/mean"] for e in v], color=COLORS[i], lw=1, alpha=0.5)
        ax.scatter([e["dtop1"] for e in v], [e["stsb/mean"] for e in v], s=22, color=COLORS[i], label=m, zorder=3)
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_xlabel("Δtop1 = top-1 variance share (V3) − (V1)")
    ax.set_ylabel("ΔSTS(V1 − V3), STS-B, mean pooling, raw (×100)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Out-of-sample test of A2' on training checkpoints (lines connect checkpoints in step order)",
                 x=0.01, ha="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig("figs/r5_figC1_a2prime.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- item 2

def series(m):
    """All Pythia checkpoints of model m that have the round-5 spec + logit outputs, sorted by step."""
    out = []
    for fn in os.listdir(f"results/round5/spec/{m}"):
        rev = fn[:-5]
        d, sp, lg = dyn(m, rev), jl(f"results/round5/spec/{m}/{fn}"), jl(f"results/round5/logit/{m}/{rev}.json")
        if d is None or sp is None or lg is None:
            continue
        out.append({"rev": rev, "step": step(rev), "loss": d["function"]["loss_base"], "loss_r5": lg["loss_base"],
                    "ub_V1": d["decomp"]["ubar_sq_V1"], "A_V1": d["step1"]["V1"]["all"]["isotropy_full"],
                    "A_V3": d["step1"]["V3"]["all"]["isotropy_full"], "ub_V3": d["decomp"]["ubar_sq_V3"],
                    "s1sq": sp["W_U"]["s1sq_over_sum"], "s1s2": sp["W_U"]["s1_over_s2"],
                    "kl_sigma": sp["W_U"]["kl_uniform_sigma"], "kl_sigma2": sp["W_U"]["kl_uniform_sigma2"],
                    "eff_rank": sp["W_U"]["eff_rank"], "b_over_g": d["gamma_beta"]["beta_over_gamma"],
                    "cos3": sp["align_V3_WU"]["abs_cos_v1"], "cos1": sp["align_V1_WUgamma"]["abs_cos_v1"],
                    "sub3": sp["align_V3_WU"], "sub1": sp["align_V1_WUgamma"], "rand_cos": sp["random_abs_cos"],
                    "rho_freq": sp["spearman_WUm_logfreq"], "S_const": lg["S_const"],
                    "pred": {n: lg["sets"][n]["half_E_var"] for n in lg["sets"]},
                    "sdv": {n: lg["sets"][n]["sd_vocab"] for n in lg["sets"]},
                    "first": {n: lg["sets"][n]["first_order"] for n in lg["sets"]},
                    "dloss": {n: d["function"]["sets"][n]["dloss"] for n in d["function"]["sets"]}})
    return sorted(out, key=lambda r: r["step"])


def t50(rs, key, from_min=False):
    v = [r[key] for r in rs]
    if from_min:
        i0 = int(np.argmin(v))
        base, tot = v[i0], v[-1] - v[i0]
        if tot <= 0:
            return None, base, tot
        thr = base + 0.5 * tot
        return next(rs[i]["step"] for i in range(i0, len(v)) if v[i] >= thr), base, tot
    base, tot = v[0], v[-1] - v[0]
    thr = base + 0.5 * tot
    return next(rs[i]["step"] for i in range(len(v)) if (v[i] - thr) * np.sign(tot) >= 0), base, tot


def item2(L, out, S):
    W = {m: [r for r in S[m] if r["step"] >= WINDOW[m]] for m in PYTHIA}
    res = {"t50": {}, "order": {}}
    L += ["\n## 第 2 项：与 Godey 等人（2024）饱和现象的联系\n", "### Q2-1：t50（step）\n",
          "| model | 窗口 | n | loss（自最小值） | W_U σ₁²/Σσ² | V1 ‖ū‖² | ‖β‖/‖γ‖ | 判定（头部谱 vs 残差流） |", "|---|---|---|---|---|---|---|---|"]
    for m in ("pythia-70m", "pythia-160m"):
        rs = W[m]
        t = {"loss": t50(rs, "loss", from_min=True), "s1sq": t50(rs, "s1sq"), "ub_V1": t50(rs, "ub_V1"),
             "b_over_g": t50(rs, "b_over_g")}
        res["t50"][m] = {k: {"t50": v[0], "base": v[1], "total_change": v[2]} for k, v in t.items()}
        th, tr = t["s1sq"][0], t["ub_V1"][0]
        verdict = "H-sim" if abs(th - tr) <= 1000 else ("H-head" if th < tr else "H-rep")
        res["order"][m] = verdict
        L.append(f"| {m} | {rs[0]['step']}–{rs[-1]['step']} | {len(rs)} | {t['loss'][0]} | {th} | {tr} | {t['b_over_g'][0]} | {verdict} |")
    # Q2-2
    t_min = min(res["t50"][m]["ub_V1"]["t50"] for m in ("pythia-70m", "pythia-160m"))
    q22 = {}
    for m in PYTHIA:
        if m in ("pythia-70m", "pythia-160m"):
            rs = [r for r in W[m] if r["step"] >= res["t50"][m]["ub_V1"]["t50"]]
            ok = all(r["cos3"] > 0.5 and r["cos1"] > 0.5 for r in rs)
        else:
            rs = [r for r in S[m] if r["step"] >= t_min]
            ok = all(r["cos3"] < 0.2 and r["cos1"] < 0.2 for r in rs)
        fin = S[m][-1]
        q22[m] = {"n": len(rs), "pass": bool(ok), "cos3_range": [min(r["cos3"] for r in rs), max(r["cos3"] for r in rs)],
                  "cos1_range": [min(r["cos1"] for r in rs), max(r["cos1"] for r in rs)],
                  "final_cos3": fin["cos3"], "final_cos1": fin["cos1"], "rand": fin["rand_cos"]}
    res["Q2_2"] = q22
    res["Q2_2_pass"] = all(v["pass"] for v in q22.values())
    L += ["\n### Q2-2：方向对齐\n", f"70m/160m：各自 V1 ‖ū‖² 的 t50 起的全部 checkpoint，两种 \\|cos\\| 都 > 0.5；"
          f"410m/1.4b：step ≥ {t_min} 的全部 checkpoint，两种 \\|cos\\| 都 < 0.2。\n",
          "| model | n | \\|cos(ū_V3, v₁(W_U))\\| 范围 | \\|cos(ū_V1, v₁(W_U·diag γ))\\| 范围 | step 143000 | 随机基线 | 满足 |",
          "|---|---|---|---|---|---|---|"]
    for m, v in q22.items():
        L.append(f"| {m} | {v['n']} | {v['cos3_range'][0]:.3f} – {v['cos3_range'][1]:.3f} | {v['cos1_range'][0]:.3f} – "
                 f"{v['cos1_range'][1]:.3f} | {v['final_cos3']:.3f} / {v['final_cos1']:.3f} | {v['rand']:.3f} | "
                 f"{'yes' if v['pass'] else 'no'} |")
    L.append(f"\nQ2-2：{'成立' if res['Q2_2_pass'] else '不成立'}。\n")
    # Q2-3
    rs = W["pythia-410m"]
    lmin = min(r["loss"] for r in rs)
    rise = rs[-1]["loss"] - lmin
    v1min = min(r["A_V1"] for r in rs)
    res["Q2_3"] = {"loss_min": lmin, "loss_final": rs[-1]["loss"], "rise": rise, "V1_iso_min": v1min,
                   "pass": bool(rise > 0 and v1min >= 0.4), "n": len(rs)}
    L += ["### Q2-3：410m\n", f"窗口 step {rs[0]['step']}–{rs[-1]['step']}（{len(rs)} 个 checkpoint）：评估 loss 最小值 {lmin:.4f}，"
          f"终点 {rs[-1]['loss']:.4f}，上升 {rise:+.4f}；V1 isotropy 最小值 {v1min:.3f}。Q2-3："
          f"{'成立' if res['Q2_3']['pass'] else '不成立'}。\n"]
    # Q2-4
    xs, ys, sds, tags = [], [], [], []
    for m in PYTHIA:
        for r in S[m]:
            for n in PRE_SETS:
                xs.append(r["pred"][n]); ys.append(r["dloss"][n]); sds.append(r["sdv"][n]); tags.append(f"{m}@{r['rev']}/{n}")
    n_ck = len(xs) // len(PRE_SETS)
    r3 = {}
    for s in [os.path.basename(l.strip()) for l in open("models.txt") if l.strip()]:
        lg, d3 = jl(f"results/round5/logit/round3/{s}.json"), jl(f"results/round3/{s}.json")
        if lg is None:
            continue
        r3[s] = lg
        for n in PRE_SETS:
            xs.append(lg["sets"][n]["half_E_var"]); ys.append(d3["step3"]["dloss"][n]); sds.append(lg["sets"][n]["sd_vocab"])
            tags.append(f"{s}/{n}")
    rho_p = spearman(xs, ys)
    rho_sd = spearman(sds, ys)
    c70 = next(r for r in S["pythia-70m"] if r["step"] == 143000)
    res["Q2_4"] = {"n": len(xs), "n_checkpoints": n_ck, "n_round3": len(r3), "rho_pred": rho_p[0], "rho_sd_vocab": rho_sd[0],
                   "pass_rho": bool(rho_p[0] > 0.95),
                   "contradiction": {"pred_top32": c70["pred"]["top32"], "pred_all": c70["pred"]["all"],
                                     "dloss_top32": c70["dloss"]["top32"], "dloss_all": c70["dloss"]["all"],
                                     "resolved": bool(c70["pred"]["top32"] > c70["pred"]["all"])}}
    res["Q2_4"]["pass"] = res["Q2_4"]["pass_rho"] and res["Q2_4"]["contradiction"]["resolved"]
    c = res["Q2_4"]["contradiction"]
    L += ["### Q2-4：Δloss 预测量\n",
          f"样本：{n_ck} 个 Pythia checkpoint + {len(r3)} 个第三轮模型，× δ ∈ {{all, top1, top8, top32}}，共 {len(xs)} 个点。"
          f"Spearman(½·E_t Var_p(δ), Δloss) = {rho_p[0]:.4f}（判定阈值 > 0.95）；同一样本上 Spearman(词表 sd, Δloss) = "
          f"{rho_sd[0]:.4f}（只报告）。70m step 143000：预测量 top32 = {c['pred_top32']:.3f}、all = {c['pred_all']:.3f}"
          f"（实际 Δloss {c['dloss_top32']:.3f} / {c['dloss_all']:.3f}）；矛盾{'消失' if c['resolved'] else '仍在'}。"
          f"Q2-4：{'成立' if res['Q2_4']['pass'] else '不成立'}。\n"]
    out["item2"] = res
    fig_c2(S, res)
    fig_c4(xs, ys, tags)
    return W


def fig_c2(S, res):
    rows = [("loss", "eval loss"), ("ub_V1", "V1 ‖ū‖²"), ("s1sq", "W_U σ₁²/Σσ²"), ("cos", "|cos| (V3: solid, V1/folded: dashed)"),
            ("b_over_g", "‖β‖/‖γ‖")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 11), sharex="col")
    for j, m in enumerate(PYTHIA):
        rs = [r for r in S[m] if r["step"] >= WINDOW[m]]
        x = [r["step"] for r in rs]
        for i, (k, lab) in enumerate(rows):
            ax = axes[i][j]
            if k == "cos":
                ax.plot(x, [r["cos3"] for r in rs], color=COLORS[0], lw=1.8, marker="o", ms=2)
                ax.plot(x, [r["cos1"] for r in rs], color=COLORS[1], lw=1.8, ls="--", marker="o", ms=2)
                ax.axhline(rs[0]["rand_cos"], color=INK2, lw=1, ls=":")
                ax.set_ylim(0, 1)
            else:
                ax.plot(x, [r[k] for r in rs], color=COLORS[0], lw=1.8, marker="o", ms=2)
            tk = {"loss": "loss", "ub_V1": "ub_V1", "s1sq": "s1sq", "b_over_g": "b_over_g"}.get(k)
            if m in res["t50"] and tk:
                ax.axvline(res["t50"][m][tk]["t50"], color=COLORS[2], lw=1.2)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8.5)
            if i == 0:
                ax.set_title(m, fontsize=9.5, loc="left", color=INK)
        axes[-1][j].set_xlabel("step")
    fig.suptitle("Dense late-training window (green line: t50 of that quantity; dotted: random |cos| baseline)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig("figs/r5_figC2_saturation.png", bbox_inches="tight")
    plt.close(fig)


def fig_c4(xs, ys, tags):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    grp = ["round3" if "@" not in t else t.split("@")[0] for t in tags]
    for i, g in enumerate(PYTHIA + ["round3"]):
        idx = [k for k, gg in enumerate(grp) if gg == g]
        ax.scatter([xs[k] for k in idx], [ys[k] for k in idx], s=10 if g != "round3" else 30,
                   color=COLORS[i] if g != "round3" else INK, alpha=0.6 if g != "round3" else 1,
                   label=g if g != "round3" else "round-3 models")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.plot([0, 10], [0, 10], color=GRID, lw=1)
    ax.set_xlabel("½·E_t Var_p(δ)")
    ax.set_ylabel("actual Δloss (nats/token)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Second-order predictor of the mean-removal Δloss", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig("figs/r5_figC4_predictor.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- item 3

def item3(L, out, S):
    hu = jl("results/round5/h_uni.json")["H_uni"]
    res = {"H_uni": hu, "models": {}}
    L += ["\n## 第 3 项：早期瞬态与 unigram 阶段\n", f"H_uni = {hu:.4f} nats（add-1 平滑的 train 词频，评估目标与 loss_base 相同）。\n",
          "| model | A 最小 | S_const 最大 | ρ(W_U m, log 词频) 最大 | \\|loss − H_uni\\| 最小 | P3-1 | P3-2 | S(1000)/max S | A(1000) − min A | P3-3 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in PYTHIA:
        rs = {r["step"]: r for r in S[m]}
        seq = [rs[s] for s in EARLY]
        A = [r["A_V3"] for r in seq]
        Sc = [r["S_const"] for r in seq]
        rho = [r["rho_freq"] for r in seq]
        gap = [abs(r["loss"] - hu) for r in seq]
        iA, iS, iR, iG = int(np.argmin(A)), int(np.argmax(Sc)), int(np.argmax(rho)), int(np.argmin(gap))
        p31 = abs(iA - iS) <= 1 and abs(iA - iR) <= 1
        p32 = iG == iA
        sr, ar = Sc[-1] / max(Sc), A[-1] - min(A)
        p33 = sr <= 0.5 and ar >= 0.2
        res["models"][m] = {"steps": EARLY, "A": A, "S_const": Sc, "rho_freq": rho,
                            "loss_minus_Huni": [r["loss"] - hu for r in seq], "ub": [r["ub_V3"] for r in seq],
                            "argmin_A": EARLY[iA], "argmax_S": EARLY[iS], "argmax_rho": EARLY[iR], "argmin_gap": EARLY[iG],
                            "P3_1": p31, "P3_2": p32, "P3_3": p33, "S_ratio": sr, "A_rise": ar}
        L.append(f"| {m} | step {EARLY[iA]} ({min(A):.3f}) | step {EARLY[iS]} ({max(Sc):.3f}) | step {EARLY[iR]} ({max(rho):+.3f}) | "
                 f"step {EARLY[iG]} ({min(gap):.3f}) | {'yes' if p31 else 'no'} | {'yes' if p32 else 'no'} | {sr:.3f} | {ar:.3f} | "
                 f"{'yes' if p33 else 'no'} |")
    for k in ("P3_1", "P3_2", "P3_3"):
        cnt = sum(res["models"][m][k] for m in PYTHIA)
        res[k] = {"count": cnt, "pass": cnt == 4}
    L.append("\n" + "；".join(f"{k.replace('_', '-')}：{res[k]['count']}/4 → {'成立' if res[k]['pass'] else '不成立'}"
                              for k in ("P3_1", "P3_2", "P3_3")) + "。\n")
    out["item3"] = res
    fig_c3(S, hu)


def fig_c3(S, hu):
    keys = [("A_V3", "isotropy A"), ("S_const", "S_const"), ("rho_freq", "ρ(W_U m, log freq)"), ("gap", "loss − H_uni")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for i, m in enumerate(PYTHIA):
        rs = [r for r in S[m] if r["step"] <= 2000]
        x = [max(r["step"], 0.5) for r in rs]
        for j, (k, lab) in enumerate(keys):
            y = [r["loss"] - hu for r in rs] if k == "gap" else [r[k] for r in rs]
            axes[j].plot(x, y, color=COLORS[i], lw=1.8, marker="o", ms=3, label=m)
            axes[j].set_title(lab, fontsize=9.5, loc="left", color=INK)
            axes[j].set_xscale("log")
            axes[j].set_xlabel("step (step 0 drawn at 0.5)")
    axes[3].axhline(0, color=GRID, lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Early training (step 0–2000)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig("figs/r5_figC3_early.png", bbox_inches="tight")
    plt.close(fig)


def main():
    out, L = {}, ["# Round 5 summary (generated by analyze_round5.py; tests as preregistered in configs/round5_prereg.md)\n"]
    item1(L, out)
    S = {m: series(m) for m in PYTHIA}
    out["n_series"] = {m: len(v) for m, v in S.items()}
    item2(L, out, S)
    item3(L, out, S)
    json.dump(out, open("results/round5/analysis.json", "w"), indent=1, default=float)
    open("results/round5/summary.md", "w").write("\n".join(L) + "\n")
    print("wrote results/round5/summary.md, analysis.json, figs/r5_figC*.png")


if __name__ == "__main__":
    main()
