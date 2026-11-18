"""Round 6 phase 3 U3: re-check published anisotropy conclusions with gauge-invariant quantities (configs/round6_prereg.md, U3).

  measure : pythia-14m / pythia-2.8b step143000 -> dyn_round4.measure (whole-model fp32, CPU; ISO_DEV=cpu ISO_MODEL_DTYPE=fp32)
            -> bank cache/round6/p3/reanalysis_banks/, then round6_traj.run -> results/round6/p3/reanalysis/traj/<model>/<rev>.json;
            also compares the 2.8b main and step143000 weights.
  table   : the six-row table (results/round6/p3/reanalysis/table.{md,json}) and figs/r6p3_fig2_reanalysis.png.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

OUT = "results/round6/p3/reanalysis"
BANKS = "cache/round6/p3/reanalysis_banks/"
T1 = "results/round6/p2/traj"


def measure():
    import torch
    from download_round4 import ckpt_dir
    from dyn_round4 import measure as dyn_measure
    from round6_traj import run as traj_run
    assert os.environ.get("ISO_DEV") == "cpu" and os.environ.get("ISO_MODEL_DTYPE") == "fp32"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    for name in ("pythia-14m", "pythia-2.8b"):
        rev = "step143000"
        fn = f"{OUT}/dyn/{name}/{rev}.json"
        if not os.path.exists(fn):
            torch.manual_seed(0)
            a = SimpleNamespace(data="data/samples.jsonl", max_len=256, bank_per_seq=10, n_pairs=20000, n_fold=20, n_est=500,
                                seed=0, cache=BANKS)
            res = dyn_measure(f"EleutherAI/{name}", rev, a)
            res["dtype"] = "fp32"
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            json.dump(res, open(fn, "w"), indent=1)
            print(f"[{name}] dyn: A V1 {res['step1']['V1']['all']['isotropy_full']:.3f} V3 {res['step1']['V3']['all']['isotropy_full']:.3f} "
                  f"ubar V1 {res['decomp']['ubar_sq_V1']:.3f} loss {res['function']['loss_base']:.4f} ({res['elapsed_s']:.0f}s)", flush=True)
        tfn = f"{OUT}/traj/{name}/{rev}.json"
        if not os.path.exists(tfn):
            traj_run(name, rev, f"{BANKS}{name}/{rev}.pt", tfn)
    # 2.8b: main vs step143000
    from safetensors import safe_open
    a, b = ckpt_dir("EleutherAI/pythia-2.8b", "step143000"), ckpt_dir("EleutherAI/pythia-2.8b", "main")
    diff, keys_a, keys_b = 0.0, set(), set()
    with safe_open(os.path.join(a, "model.safetensors"), "pt") as fa, safe_open(os.path.join(b, "model.safetensors"), "pt") as fb:
        keys_a, keys_b = set(fa.keys()), set(fb.keys())
        for k in sorted(keys_a & keys_b):
            ta, tb = fa.get_tensor(k), fb.get_tensor(k)
            if ta.shape != tb.shape:
                diff = float("inf")
                continue
            diff = max(diff, (ta.float() - tb.float()).abs().max().item())
    chk = {"max_abs_diff_shared": diff, "only_step143000": sorted(keys_a - keys_b), "only_main": sorted(keys_b - keys_a),
           "dtypes": None}
    json.dump(chk, open(f"{OUT}/pythia-2.8b_main_vs_step143000.json", "w"), indent=1)
    print("2.8b main vs step143000:", chk)


def A4(name, rev, base=T1):
    fn = f"{base}/{name}/{rev}.json"
    if not os.path.exists(fn):
        fn = f"{OUT}/traj/{name}/{rev}.json"
    r = json.load(open(fn))
    return {"V3": r["V3"]["A"], "V1": r["V1"]["A"], "Astar": r["hstar_unif"]["A"], "Astar_pi": r["hstar_pi"]["A"]}


def verdict(d_orig, d_inv):
    if np.sign(d_inv) != np.sign(d_orig) or abs(d_inv) < 0.02:
        return "不成立"
    return "成立" if abs(d_inv) >= 0.5 * abs(d_orig) else "减弱"


def table():
    gauge = {}
    for f in os.listdir("results/round6/gauge"):
        if f.endswith(".json") and f != "summary.json":
            gauge[f[:-5]] = json.load(open(f"results/round6/gauge/{f}"))

    def g4(m):  # phase-1 S1 bank metrics
        r = gauge[m]
        return {"V3": r["base"]["V3"]["A"], "V1": r["base"]["V1"]["A"], "Astar": r["hstar"]["unif"]["from_V1_folded"]["A"],
                "Astar_pi": r["hstar"]["pi"]["from_V1_folded"]["A"]}
    rows = []

    def group_row(i, claim, src, orig_key, a_groups, b_groups, note=""):
        mean = lambda grp, k: float(np.mean([x[k] for x in grp]))  # noqa: E731
        d = {k: mean(a_groups, k) - mean(b_groups, k) for k in ("V3", "V1", "Astar", "Astar_pi")}
        rows.append({"row": i, "claim": claim, "source": src, "orig_quantity": orig_key, "delta": d,
                     "verdict_Astar": verdict(d[orig_key], d["Astar"]), "verdict_Astar_pi": verdict(d[orig_key], d["Astar_pi"]), "note": note})

    gp = {m: g4(m) for m in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")}
    group_row(1, "GPT-2 small / medium 强各向异性，large / xl 不是", "Su & Collier 2022 §3.1", "V3",
              [gp["gpt2-large"], gp["gpt2-xl"]], [gp["gpt2"], gp["gpt2-medium"]], "已知，非预登记")
    py = {m: A4(m, "step143000") for m in ("pythia-14m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-2.8b")}
    group_row(2, "大号 Pythia 各向同性、小号各向异性", "Machina & Mercer 2024（按摘要）", "V3",
              [py["pythia-410m"], py["pythia-1.4b"], py["pythia-2.8b"]], [py["pythia-14m"], py["pythia-70m"], py["pythia-160m"]])
    sub = []
    for m, s0 in (("pythia-70m", 50000), ("pythia-160m", 80000)):
        a, b = A4(m, "step143000"), A4(m, f"step{s0}")
        d = {k: a[k] - b[k] for k in a}
        sub.append({"model": m, "delta": d, "verdict_Astar": verdict(d["V3"], d["Astar"]), "verdict_Astar_pi": verdict(d["V3"], d["Astar_pi"])})
    order = ["不成立", "减弱", "成立"]
    rows.append({"row": 3, "claim": "小模型训练后期最后一层各向异性陡增", "source": "Godey et al. 2024 §4.1", "orig_quantity": "V3", "sub": sub,
                 "verdict_Astar": min((s["verdict_Astar"] for s in sub), key=order.index),
                 "verdict_Astar_pi": min((s["verdict_Astar_pi"] for s in sub), key=order.index), "note": ""})
    sub = []
    for m in ("pythia-410m", "pythia-1.4b"):
        st = [s for s in (8000, 13000, 20000, 32000, 50000, 80000)]
        vals = {s: A4(m, f"step{s}") for s in st}
        d = {k: vals[8000][k] - min(v[k] for v in vals.values()) for k in vals[8000]}
        sub.append({"model": m, "dip": d, "verdict_Astar": verdict(d["V1"], d["Astar"]), "verdict_Astar_pi": verdict(d["V1"], d["Astar_pi"])})
    rows.append({"row": 4, "claim": "大模型训练中段残差流变得各向异性", "source": "本项目第四轮 B1", "orig_quantity": "V1", "sub": sub,
                 "verdict_Astar": min((s["verdict_Astar"] for s in sub), key=order.index),
                 "verdict_Astar_pi": min((s["verdict_Astar_pi"] for s in sub), key=order.index), "note": "dip = A(8000) − 窗口内最小值"})
    sub = []
    for base, post in (("Llama-3.1-8B", "Llama-3.1-8B-Instruct"), ("Qwen3-8B-Base", "Qwen3-8B")):
        a, b = g4(post), g4(base)
        d = {k: a[k] - b[k] for k in a}
        sub.append({"pair": f"{base} → {post}", "delta": d, "verdict_Astar": verdict(d["V3"], d["Astar"]),
                    "verdict_Astar_pi": verdict(d["V3"], d["Astar_pi"])})
    signs = {k: [np.sign(s["delta"][k]) * (abs(s["delta"][k]) >= 0.02) for s in sub] for k in ("V3", "Astar", "Astar_pi")}
    rows.append({"row": 5, "claim": "后训练改变 isotropy，方向在两对之间不一致", "source": "本项目第二轮 3.5", "orig_quantity": "V3", "sub": sub,
                 "directions": {k: [int(x) for x in v] for k, v in signs.items()},
                 "inconsistent_in": {k: bool(len(set(v)) > 1) for k, v in signs.items()}, "note": ""})
    od = json.load(open("results/round6/od/summary.json")) if os.path.exists("results/round6/od/summary.json") else None
    rows.append({"row": 6, "claim": "存在一组特定的 rogue / outlier 维度", "source": "Timkey & van Schijndel 2021 §3.1.3；Macocco et al. 2025",
                 "orig_quantity": "OD 集合（V3）", "verdict_Astar": "不成立（作为模型的函数性质；Olmo-3 除外）", "verdict_Astar_pi": "无对应量",
                 "od": od, "note": "pre-norm：残差流坐标只确定到 O(d)（U1 命题 2–3）"})
    p87 = py["pythia-2.8b"]
    res = {"P8_7": {"Astar": p87["Astar"], "Astar_pi": p87["Astar_pi"],
                    "pass": bool(0.40 <= p87["Astar"] <= 0.55 and 0.25 <= p87["Astar_pi"] <= 0.35)},
           "pythia_step143000": py, "gpt2": gp, "rows": rows}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/table.json", "w"), indent=1, ensure_ascii=False)
    return res


if __name__ == "__main__":
    {"measure": measure, "table": table}[sys.argv[1]]()
