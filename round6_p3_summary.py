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
    {"u0": u0, "u1": u1}[sys.argv[1]]()
