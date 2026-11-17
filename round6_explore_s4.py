"""Round 6 S4, exploratory (not preregistered): where along the eval set does the cost of 'fix the ubar component'
(intervention 2) and of the sigma_1 truncation sit? Per-token Delta-nll concentration, the |p_t - c| distribution,
and the most frequent input tokens among the top-1% contributors. Writes results/round6/causal/explore_tokens.json.
"""
import collections
import json
import os

import torch

import round6_causal as rc
from round3_lib import normalize


@torch.no_grad()
def one(model_name, rev):
    C = rc.Ckpt(model_name, rev)
    X, Y, FL, _ = C.eval_cache()
    rows = [json.loads(l) for l in open(rc.EVAL)]
    inp = []
    for r in rows:
        ids = C.tok(r["text"], add_special_tokens=True, truncation=True, max_length=rc.CHUNK)["input_ids"]
        inp.extend(ids[:-1])
    inp = torch.as_tensor(inp, device=rc.DEV)
    assert len(inp) == len(Y)
    Wf = C.W * C.gamma
    bf = C.W @ C.beta
    b = torch.load(f"cache/round4_banks/{model_name}/{rev}.pt", weights_only=False)
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
        base = torch.logsumexp(L, -1) - L.gather(1, y[:, None])[:, 0]
        p = V1 @ u
        a = c - p
        Lf = L + a[:, None] * wu
        dfix.append(torch.logsumexp(Lf, -1) - (L.gather(1, y[:, None])[:, 0] + a * wu[y]) - base)
        at = V1 @ v1
        at = at - at.mean()  # truncation with the eval-mean compensated (b0 analogue)
        Lt = L + at[:, None] * w_tr
        dtr.append(torch.logsumexp(Lt, -1) - (L.gather(1, y[:, None])[:, 0] + at * w_tr[y]) - base)
        dev_p.append(p - c)
    out = {}
    for name, D in (("fix", torch.cat(dfix)), ("trunc_meancomp", torch.cat(dtr))):
        D = D.double()
        tot = D.sum()
        srt = torch.sort(D, descending=True).values
        n = len(D)
        top = torch.topk(D, n // 100).indices
        cnt = collections.Counter(inp[top].tolist())
        out[name] = {"mean": (tot / n).item(), "share_top0.1pct": (srt[:n // 1000].sum() / tot).item(),
                     "share_top1pct": (srt[:n // 100].sum() / tot).item(),
                     "share_top10pct": (srt[:n // 10].sum() / tot).item(),
                     "median": D.median().item(), "frac_negative": (D < 0).double().mean().item(),
                     "top1pct_input_tokens": [(C.tok.decode([t]), k) for t, k in cnt.most_common(15)]}
    P = torch.cat(dev_p).double()
    out["p_minus_c"] = {"sd": P.std().item(), "q01": P.quantile(0.01).item() if len(P) < 16_000_000 else None,
                        "q50": P.median().item(), "q99": None, "absmax": P.abs().max().item(),
                        "bank_sd": (V1b @ u).std().item()}
    Pa = P.abs().float().cpu()
    out["p_minus_c"]["abs_q"] = {q: Pa.quantile(q).item() for q in (0.5, 0.9, 0.99, 0.999)} if len(Pa) < 16_000_000 else None
    del C
    torch.cuda.empty_cache()
    return out


def main():
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    res = {f"{m}/{r}": one(m, r) for m, r in (("pythia-70m", "step143000"), ("pythia-160m", "step143000"),
                                               ("pythia-70m", "step80000"))}
    os.makedirs("results/round6/causal", exist_ok=True)
    json.dump(res, open("results/round6/causal/explore_tokens.json", "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False)[:1500], flush=True)


if __name__ == "__main__":
    main()
