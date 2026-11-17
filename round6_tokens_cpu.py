"""Round 6 phase 2 T2c (CPU, report only): when do the large deviations along ubar appear?

For every Pythia bank (rounds 4-5): p = ubar . V1 over the 10000 bank tokens (ubar, c from the same bank);
kurtosis m4 / m2^2 of p - c and the ratio q99.9(|p - c|) / median(|p - c|). NB: q99.9 of 10000 tokens rests on ~10 points.
t50 as in configs/round5_prereg.md (first step reaching 50% of the change from the window start to the final checkpoint;
windows 70m 50000-143000, 160m 80000-143000), next to t50 of V1 ||ubar||^2 (round 4/5 dyn) and W_U s1^2/sum (round 5 spec).
Writes results/round6/p2/tokens/deviation_traj.json.
"""
import json
import os

import numpy as np
import torch

from round3_lib import normalize

WIN = {"pythia-70m": 50000, "pythia-160m": 80000, "pythia-410m": 80000, "pythia-1.4b": 80000}


def t50(steps, vals, start):
    s = np.asarray(steps)
    v = np.asarray(vals)
    keep = s >= start
    s, v = s[keep], v[keep]
    o = np.argsort(s)
    s, v = s[o], v[o]
    tot = v[-1] - v[0]
    if tot == 0:
        return None
    frac = (v - v[0]) / tot
    hit = np.flatnonzero(frac >= 0.5)
    return int(s[hit[0]]) if len(hit) else None


def main():
    out = {}
    for m in WIN:
        rows = []
        for rnd in ("round4", "round5"):
            root = f"cache/{rnd}_banks/{m}"
            for f in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                rev = f[:-3]
                b = torch.load(os.path.join(root, f), weights_only=False, map_location="cpu")
                V1 = normalize(b["x"].double(), b["norm_type"], b["eps"]).double()
                u = torch.nn.functional.normalize(V1, dim=-1).mean(0)
                u = u / u.norm()
                p = (V1 @ u).numpy()
                dv = p - p.mean()
                a = np.abs(dv)
                dyn = next((json.load(open(f"results/{r}/dyn/{m}/{rev}.json")) for r in ("round4", "round5")
                            if os.path.exists(f"results/{r}/dyn/{m}/{rev}.json")), None)
                spec = f"results/round5/spec/{m}/{rev}.json"
                rows.append({"rev": rev, "step": int(rev[4:]), "kurtosis": float((dv ** 4).mean() / (dv ** 2).mean() ** 2),
                             "q999_over_median": float(np.quantile(a, 0.999) / np.median(a)),
                             "ubar_sq_V1": dyn["step1"]["V1"]["all"]["ubar_sq"] if dyn else None,
                             "s1sq_over_sum": json.load(open(spec))["W_U"]["s1sq_over_sum"] if os.path.exists(spec) else None})
        rows.sort(key=lambda r: r["step"])
        st = WIN[m]
        t = {}
        for key in ("kurtosis", "q999_over_median", "ubar_sq_V1", "s1sq_over_sum"):
            pts = [(r["step"], r[key]) for r in rows if r[key] is not None]
            t[key] = t50([a for a, _ in pts], [b for _, b in pts], st) if pts else None
        out[m] = {"window_start": st, "t50": t, "rows": rows}
        print(m, t, flush=True)
    os.makedirs("results/round6/p2/tokens", exist_ok=True)
    json.dump(out, open("results/round6/p2/tokens/deviation_traj.json", "w"), indent=1)


if __name__ == "__main__":
    main()
