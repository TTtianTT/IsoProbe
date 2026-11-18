"""Round 6 phase 3 U0: fp32 audit of the loss-based conclusions of rounds 3-5 and phase-1 S4 (configs/round6_prereg.md, U0).

  dyn --rows a-b : rows of configs/round6_p3_audit_ckpts.tsv. Whole-model fp32 rerun of dyn_round4.measure (bank to
                   cache/round6/p3/fp32_banks/) and of round5_gpu.logit_metrics with the fp32 delta sets.
                   -> results/round6/p3/fp32_audit/{dyn,logit}/<model>/<rev>.json
  s4  --rows a-b : rows of configs/round6_causal.tsv. round6_causal.ablate with an fp32 body and the fp32 bank, plus the
                   per-token shares of round6_explore_s4 (intervention 2 = fix the ubar component; mean-compensated
                   sigma_1 truncation). -> results/round6/p3/fp32_audit/{s4,s4_tokens}/<model>/<rev>.json
Must be started with ISO_MODEL_DTYPE=fp32 ISO_BODY_DTYPE=fp32 ISO_BANK_DIRS=cache/round6/p3/fp32_banks (the sbatch sets them).
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import torch

assert os.environ.get("ISO_MODEL_DTYPE") == "fp32" and os.environ.get("ISO_BODY_DTYPE") == "fp32", "set the fp32 env vars"
BANKS = os.environ.get("ISO_BANK_DIRS", "cache/round6/p3/fp32_banks").split(",")[0]
OUT = os.environ.get("ISO_P3_AUDIT_OUT", "results/round6/p3/fp32_audit")  # smoke tests use a separate directory


def dump(obj, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    tmp = fn + f".tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, fn)


def rows_of(spec):
    idx = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    return idx


def ensure_dyn(repo, rev):
    """fp32 dyn measure + fp32 bank (skip if present). The bank is written by measure() itself (torch.save)."""
    from dyn_round4 import measure
    name = repo.split("/")[1]
    fn = f"{OUT}/dyn/{name}/{rev}.json"
    if os.path.exists(fn) and os.path.exists(f"{BANKS}/{name}/{rev}.pt"):
        return json.load(open(fn))
    dargs = SimpleNamespace(data="data/samples.jsonl", max_len=256, bank_per_seq=10, n_pairs=20000, n_fold=20, n_est=500,
                            seed=0, cache=BANKS + "/")
    torch.manual_seed(0)
    res = measure(repo, rev, dargs)
    res["dtype"] = "fp32"
    dump(res, fn)
    f = res["function"]
    print(f"[{name}@{rev}] dyn fp32: loss={f['loss_base']:.4f} dl_all={f['sets']['all']['dloss']:+.4f} "
          f"dl_top8={f['sets']['top8']['dloss']:+.4f} ({res['elapsed_s']:.0f}s)", flush=True)
    return res


def run_dyn(args):
    from download_round4 import ckpt_dir
    from round5_gpu import logit_metrics
    rows = [l.split("\t")[:2] for l in open("configs/round6_p3_audit_ckpts.tsv").read().splitlines() if l.strip()]
    for i in rows_of(args.rows):
        repo, rev = rows[i]
        name = repo.split("/")[1]
        ensure_dyn(repo, rev)
        fn = f"{OUT}/logit/{name}/{rev}.json"
        if os.path.exists(fn):
            continue
        delta = torch.load(f"{BANKS}/{name}/{rev}.pt", weights_only=False)["delta"]
        la = SimpleNamespace(data="data/samples.jsonl", max_len=256, n_est=500)
        res = logit_metrics(ckpt_dir(repo, rev), delta, la)
        res["dtype"] = "fp32"
        dump(res, fn)
        print(f"[{name}@{rev}] logit fp32: loss={res['loss_base']:.4f} "
              + " ".join(f"{n}={v['half_E_var']:.3f}" for n, v in res["sets"].items()) + f" ({res['elapsed_s']:.0f}s)", flush=True)


@torch.no_grad()
def token_shares(model_name, rev):
    """round6_explore_s4.one with the fp32 body and the fp32 bank (same definitions)."""
    import round6_causal as rc
    from round3_lib import normalize
    C = rc.Ckpt(model_name, rev)
    X, Y, FL, _ = C.eval_cache()
    Wf = C.W * C.gamma
    bf = C.W @ C.beta
    b = torch.load(f"{BANKS}/{model_name}/{rev}.pt", weights_only=False)
    V1b = normalize(b["x"].to(rc.DEV).float(), b["norm_type"], b["eps"])
    u = torch.nn.functional.normalize(V1b, dim=-1).mean(0)
    u = u / u.norm()
    c = (V1b @ u).mean().item()
    U, S, Vh = torch.linalg.svd(Wf.double(), full_matrices=False)
    w_tr = -(S[0] - S[1]).item() * U[:, 0].float()
    v1 = Vh[0].float()
    wu = Wf @ u
    dfix, dtr, dev_p = [], [], []
    for s in range(0, len(Y), 4096):
        V1 = normalize(X[s:s + 4096].float(), C.ntype, C.eps)
        y = Y[s:s + 4096]
        L = V1 @ Wf.T + bf
        Ly = L.gather(1, y[:, None])[:, 0]
        base = torch.logsumexp(L, -1) - Ly
        p = V1 @ u
        a = c - p
        dfix.append(torch.logsumexp(L + a[:, None] * wu, -1) - (Ly + a * wu[y]) - base)
        at = V1 @ v1
        at = at - at.mean()
        dtr.append(torch.logsumexp(L + at[:, None] * w_tr, -1) - (Ly + at * w_tr[y]) - base)
        dev_p.append(p - c)
    out = {"dtype": "fp32", "n_tokens": len(Y), "c": c}
    for name, D in (("fix", torch.cat(dfix)), ("trunc_meancomp", torch.cat(dtr))):
        D = D.double()
        tot = D.sum()
        srt = torch.sort(D, descending=True).values
        n = len(D)
        out[name] = {"mean": (tot / n).item(), "share_top0.1pct": (srt[:n // 1000].sum() / tot).item(),
                     "share_top1pct": (srt[:n // 100].sum() / tot).item(),
                     "share_top10pct": (srt[:n // 10].sum() / tot).item(),
                     "median": D.median().item(), "frac_negative": (D < 0).double().mean().item()}
    Pa = torch.cat(dev_p).abs().float().cpu()
    out["abs_p_minus_c"] = {str(q): Pa.quantile(q).item() for q in (0.5, 0.9, 0.99, 0.999)}
    out["abs_p_minus_c"]["max"] = Pa.max().item()
    del C
    torch.cuda.empty_cache()
    return out


def run_s4(args):
    import round6_causal as rc
    rows = [l.split("\t")[:2] for l in open("configs/round6_causal.tsv").read().splitlines() if l.strip()]
    for i in rows_of(args.rows):
        model_name, rev = rows[i]
        ensure_dyn(f"EleutherAI/{model_name}", rev)
        fn = f"{OUT}/s4/{model_name}/{rev}.json"
        if not os.path.exists(fn):
            res = rc.ablate(model_name, rev, SimpleNamespace(max_tokens=0, bias_steps=200))
            res["dtype"] = "fp32"
            dump(res, fn)
        fn = f"{OUT}/s4_tokens/{model_name}/{rev}.json"
        if not os.path.exists(fn):
            t0 = time.time()
            res = token_shares(model_name, rev)
            res["elapsed_s"] = time.time() - t0
            dump(res, fn)
            print(f"[{model_name}@{rev}] tokens fp32: fix mean {res['fix']['mean']:+.4f} top0.1% {res['fix']['share_top0.1pct']:.3f}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dyn", "s4"])
    ap.add_argument("--rows", required=True)
    args = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    (run_dyn if args.mode == "dyn" else run_s4)(args)


if __name__ == "__main__":
    main()
