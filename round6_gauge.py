"""Round 6 S1 (CPU, fp64; one model per call): gauge audit (1a) and gauge-invariant metric h* (1b).

1a  metrics of V1 under 20 random G_diag (kappa 10, 100), G_orth, G_lin (kappa 10, 100) transforms; worst-case
    [A_min, A_max] over G_diag with condition number <= kappa (projected Adam on t = ln s in [0, ln kappa]^d,
    equivalent because A is scale invariant); condition number of gamma.
1b  h* = Mt^(1/2)[h; 1] for P uniform and P_pi (unigram); invariance checks (V1, W diag g, W beta) vs (V3, W, 0)
    and 20 G_lin (kappa 100) samples with the transformed head computed explicitly.
Writes results/round6/gauge/<model>.json and cache/round6/gauge/<model>.pt (worst-case s, random diag s, R matrices).
See configs/round6_prereg.md.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from round6_lib import (DT, METRIC_KEYS, gauge_root, hstar, isotropy_bank, load_bank, load_head, metrics,
                        rel_diff, sample_transform, seq_index, unigram_pi)

KAPPAS = [10, 100]
N_SAMPLES = 20


def worst_case(V1, sidx, kappa, gamma, sign, steps=400, lr=0.05, n_rand=3):
    """Projected Adam on t in [0, ln kappa]^d maximizing sign * A(V1 * exp(t)); returns best (A, t, per-init)."""
    d = V1.shape[1]
    hi = math.log(kappa)
    lg = gamma.abs().clamp_min(1e-12).log()
    inits = {"const": torch.full((d,), hi / 2, dtype=DT),
             "gamma": ((lg - lg.min()) / (lg.max() - lg.min()).clamp_min(1e-12) * hi)}
    g = torch.Generator().manual_seed(int(kappa) * 10 + (sign > 0))
    for r in range(n_rand):
        inits[f"rand{r}"] = torch.rand(d, generator=g, dtype=DT) * hi
    best, per = None, {}
    for name, t0 in inits.items():
        t = t0.clone().requires_grad_(True)
        opt = torch.optim.Adam([t], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = -sign * isotropy_bank(V1 * t.exp(), sidx)
            loss.backward()
            opt.step()
            with torch.no_grad():
                t.clamp_(0, hi)
        with torch.no_grad():
            a = isotropy_bank(V1 * t.exp(), sidx).item()
        per[name] = a
        if best is None or sign * a > sign * best[0]:
            best = (a, t.detach().clone(), name)
    return best, per


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/round6/gauge/")
    ap.add_argument("--cache", default="cache/round6/gauge/")
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    slug = args.model
    fn = os.path.join(args.out, f"{slug}.json")
    if os.path.exists(fn):
        print(f"{fn} exists, skipping", flush=True)
        return
    t0 = time.time()
    B = load_bank(slug)
    V1, V3, gamma, beta = B["V1"], B["V3"], B["gamma"], B["beta"]
    sidx = seq_index(B["seq"])
    d = V1.shape[1]
    res = {"model": slug, "d": d, "n_tokens": V1.shape[0], "has_beta": beta is not None}
    ag = gamma.abs()
    res["gamma"] = {"cond": (ag.max() / ag.min()).item(), "max_over_median": (ag.max() / ag.median()).item(),
                    "min_abs": ag.min().item(), "n_nonpositive": int((gamma <= 0).sum())}
    res["base"] = {"V1": metrics(V1, sidx), "V2": metrics(gamma * V1, sidx), "V3": metrics(V3, sidx)}
    print(f"[{slug}] d={d} gamma cond={res['gamma']['cond']:.3g}; V1 {res['base']['V1']}; V3 {res['base']['V3']} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 1a random families
    fam = {}
    keep_diag100 = []
    for family, kappas in (("orth", [0]), ("diag", KAPPAS), ("lin", KAPPAS)):
        for kappa in kappas:
            key = f"{family}" + (f"_k{kappa}" if family != "orth" else "")
            rows = []
            for s in range(args.n_samples):
                fx, _, info = sample_transform(family, d, kappa, s)
                rows.append(metrics(fx(V1), sidx))
                if family == "diag" and kappa == 100 and s < 3:
                    keep_diag100.append(info["s"])
            fam[key] = {"samples": rows,
                        "min": {k: min(r[k] for r in rows) for k in METRIC_KEYS},
                        "max": {k: max(r[k] for r in rows) for k in METRIC_KEYS}}
            if family == "orth":
                fam[key]["max_abs_diff_vs_V1"] = {k: max(abs(r[k] - res["base"]["V1"][k]) for r in rows)
                                                  for k in METRIC_KEYS}
            print(f"[{slug}] {key}: A in [{fam[key]['min']['A']:.4f}, {fam[key]['max']['A']:.4f}] "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res["random"] = fam

    # ---- 1a worst case over G_diag
    worst, worst_s = {}, {}
    with torch.enable_grad():
        for kappa in KAPPAS:
            for tag, sign in (("A_max", 1), ("A_min", -1)):
                (a, t, name), per = worst_case(V1, sidx, kappa, gamma, sign, steps=args.steps)
                s = t.exp()
                worst[f"k{kappa}_{tag}"] = {"A": a, "best_init": name, "per_init": per,
                                            "metrics": metrics(V1 * s, sidx),
                                            "frac_at_bounds": float(((t < 1e-6) | (t > math.log(kappa) - 1e-6))
                                                                    .double().mean())}
                worst_s[f"k{kappa}_{tag}"] = s
                print(f"[{slug}] worst k={kappa} {tag}: {a:.4f} (init {name}; {per}) ({time.time() - t0:.0f}s)",
                      flush=True)
    res["worst"] = worst

    # ---- 1b gauge metric h*
    W, wkey = load_head(slug)
    pi, group = unigram_pi(slug, W.shape[0])
    Wf = W * gamma
    bf = W @ beta if beta is not None else None
    hs, Rs, checks = {}, {}, {}
    P = {"unif": None, "pi": pi}
    for pv, wts in P.items():
        R, ei = gauge_root(Wf, bf, wts)
        R3, _ = gauge_root(W, None, wts)
        m_base = metrics(hstar(V1, R), sidx, d)
        m_v3 = metrics(hstar(V3, R3), sidx, d)
        hs[pv] = {"from_V1_folded": m_base, "from_V3_orig": m_v3}
        checks[pv] = {"V3_vs_V1_max_rel": max(rel_diff(m_v3[k], m_base[k]) for k in METRIC_KEYS),
                      "V3_vs_V1_per_metric": {k: rel_diff(m_v3[k], m_base[k]) for k in METRIC_KEYS},
                      "lin_k100_per_sample": [], "eig": ei}
        Rs[pv] = R
        del R3
    for s in range(args.n_samples):  # explicit transformed head W diag(g) A^-1, shared by both P versions
        fx, fw, _ = sample_transform("lin", d, 100, s)
        Wl, Xl = fw(Wf), fx(V1)
        for pv, wts in P.items():
            Rl, _ = gauge_root(Wl, bf, wts)
            ml = metrics(hstar(Xl, Rl), sidx, d)
            checks[pv]["lin_k100_per_sample"].append(
                max(rel_diff(ml[k], hs[pv]["from_V1_folded"][k]) for k in METRIC_KEYS))
        del Wl, Xl
    for pv in P:
        checks[pv]["lin_k100_max_rel"] = max(checks[pv]["lin_k100_per_sample"])
        print(f"[{slug}] h*[{pv}]: {hs[pv]['from_V1_folded']}; check V3 {checks[pv]['V3_vs_V1_max_rel']:.2e} "
              f"lin {checks[pv]['lin_k100_max_rel']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    res["hstar"] = hs
    res["hstar_checks"] = checks
    res["head_key"] = wkey
    res["vocab_rows"] = W.shape[0]
    res["unigram_group"] = group
    res["elapsed_s"] = time.time() - t0
    os.makedirs(args.cache, exist_ok=True)
    torch.save({"worst_s": worst_s, "diag_k100_s": keep_diag100, "R_unif": Rs["unif"], "R_pi": Rs["pi"],
                "gamma": gamma, "beta": beta}, os.path.join(args.cache, f"{slug}.pt"))
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(fn, "w"), indent=1)
    print(f"[{slug}] wrote {fn} ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
