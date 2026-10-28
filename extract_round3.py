"""Round 3, GPU part (one model per call): step 0 (bank of final-norm inputs + checks 0a/0b),
step 1 (gamma-folding test; A/B/RankMe for V0-V3) and step 3 (mean-removal intervention on next-token loss).

Writes results/round3/<slug>.json and cache/banks/<slug>.pt (fp32 bank, gamma/beta, intervention means,
dimension ranking, logit-offset vectors for step 3b).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from measure import final_norm_module, load_samples, resolve_model, self_similarity
from round3_lib import (ROUND2_KEY, VNAMES, bank_metrics_r3, bank_self_sim, decompose, variants)

KS = [1, 2, 4, 8, 16, 32]
N_RAND = 5


def norm_params(norm):
    eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon", None) or 1e-6
    ntype = "LayerNorm" if isinstance(norm, torch.nn.LayerNorm) else "RMSNorm"
    beta = norm.bias.detach().float() if getattr(norm, "bias", None) is not None else None
    return norm.weight.detach().float(), beta, float(eps), ntype


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bank_per_seq", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--n_fold", type=int, default=20)
    ap.add_argument("--n_est", type=int, default=500, help="sequences [0, n_est) estimate m; the rest evaluate loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/round3/")
    ap.add_argument("--cache", default="cache/banks/")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    torch.backends.cuda.matmul.allow_tf32 = False  # the folding test compares fp32 logits at 1e-4
    torch.set_float32_matmul_precision("highest")
    path = resolve_model(args.model)
    slug = os.path.basename(path.rstrip("/"))
    dev = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev,
                                                 local_files_only=True).eval()
    norm = final_norm_module(model)
    gamma, beta, eps, ntype = norm_params(norm)
    head = model.get_output_embeddings()
    W = head.weight.detach().float()  # [V, d] fp32 copy; for GPT-2 this is also the input embedding (tied)
    b_head = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0]))
    rows = load_samples(args.data, 0)
    rng = np.random.default_rng(args.seed)  # same stream as measure.py / diag_final_norm.py -> same bank positions
    print(f"[{slug}] loaded {time.time() - t0:.0f}s; norm={type(norm).__name__} ({ntype}) eps={eps} "
          f"beta={'yes' if beta is not None else 'no'} head_bias={'yes' if b_head is not None else 'no'}", flush=True)

    bank_x, bank_seq, bank_pos = [], [], []
    full_ss = {v: {"all": [], "no_pos0": []} for v in VNAMES}
    rec_err, fold = [], {"rel_err": [], "agree": 0, "n": 0, "ref_vs_model_maxabs": 0.0}
    m_sum, m_n = torch.zeros(W.shape[1], dtype=torch.float64, device=dev), 0
    eval_h, eval_y = [], []
    W_fold = W * gamma  # W_U diag(gamma): unties GPT-2's shared embedding (only the head copy is changed)
    b_fold = (W @ beta if beta is not None else torch.zeros(W.shape[0], device=dev)) + (
        b_head if b_head is not None else 0)

    for sid, row in enumerate(rows):
        ids = tok(row["text"], add_special_tokens=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt")["input_ids"].to(dev)
        T = ids.shape[1]
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        h = out.hidden_states[-1][0].float()
        V = variants(cap["x"][0], gamma, beta, ntype, eps)
        rec_err.append(((V["V3"] - h).norm() / h.norm()).item())  # check 0a
        for v in VNAMES:
            full_ss[v]["all"].append(self_similarity(V[v]))
            full_ss[v]["no_pos0"].append(self_similarity(V[v][1:]))
        pos = rng.choice(T, size=min(args.bank_per_seq, T), replace=False)
        bank_x.append(V["V0"][torch.as_tensor(pos, device=dev)].cpu())
        bank_seq.extend([sid] * len(pos))
        bank_pos.append(pos)
        if sid < args.n_fold:  # step 1: gamma folding, fp32
            ref = V["V3"] @ W.T + (b_head if b_head is not None else 0)
            fl = V["V1"] @ W_fold.T + b_fold
            fold["rel_err"].append(((ref - fl).abs().max() / ref.abs().max()).item())
            fold["agree"] += int((ref.argmax(-1) == fl.argmax(-1)).sum())
            fold["n"] += T
            fold["ref_vs_model_maxabs"] = max(fold["ref_vs_model_maxabs"],
                                              (ref - out.logits[0].float()).abs().max().item())
        if sid < args.n_est:  # step 3: mean of h without position 0
            m_sum += h[1:].sum(0).double()
            m_n += T - 1
        else:
            eval_h.append(out.hidden_states[-1][0].cpu())  # bf16, exact to upcast later
            eval_y.append(ids[0, 1:].cpu())
        if (sid + 1) % 200 == 0:
            print(f"[{slug}] {sid + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)

    X0 = torch.cat(bank_x).to(dev)
    seq = np.asarray(bank_seq)
    pos = np.concatenate(bank_pos)
    BV = variants(X0, gamma, beta, ntype, eps)
    keep = pos != 0
    keep_t = torch.as_tensor(keep, device=dev)

    # ---- check 0b and step-1 metrics
    r2 = json.load(open(f"results/diag/{slug}.json"))["variants"]
    step1, check0b = {}, {}
    for v in VNAMES:
        step1[v] = {}
        for tag, Xb, sq, ps, ss in (("all", BV[v], seq, pos, full_ss[v]["all"]),
                                    ("no_pos0", BV[v][keep_t], seq[keep], pos[keep], full_ss[v]["no_pos0"])):
            r = {"isotropy_full": float(1 - np.mean(ss)), "isotropy_bank": bank_self_sim(Xb, sq)}
            r.update(bank_metrics_r3(Xb, sq, ps, args.n_pairs, args.seed))
            step1[v][tag] = r
        a, ref2 = step1[v]["all"], r2[ROUND2_KEY[v]]["all"]
        check0b[v] = {"A_bank": a["isotropy_bank"], "A_round2_full": ref2["isotropy"],
                      "dA": abs(a["isotropy_bank"] - ref2["isotropy"]),
                      "ubar_sq": a["ubar_sq"], "B_round2": ref2["cross_cos"],
                      "dB": abs(a["ubar_sq"] - ref2["cross_cos"])}
        check0b[v]["pass"] = check0b[v]["dA"] < 0.01 and check0b[v]["dB"] < 0.01
    check0a = {"max_rel_err": float(np.max(rec_err)), "pass": float(np.max(rec_err)) < 1e-2}
    fold_res = {"max_rel_err": float(np.max(fold["rel_err"])), "argmax_agree": fold["agree"] / fold["n"],
                "n_positions": fold["n"], "ref_vs_model_logits_maxabs": fold["ref_vs_model_maxabs"],
                "unties_embeddings": bool(model.config.tie_word_embeddings)}
    fold_res["pass"] = fold_res["max_rel_err"] < 1e-4 and fold_res["argmax_agree"] >= 0.9999
    print(f"[{slug}] check 0a: {check0a}", flush=True)
    for v in VNAMES:
        print(f"[{slug}] check 0b {v}: {check0b[v]}", flush=True)
    print(f"[{slug}] fold test: {fold_res}", flush=True)
    for v in VNAMES:
        a = step1[v]["all"]
        print(f"[{slug}] {v}: A={a['isotropy_full']:.3f} B={a['cross_cos']:.3f} Brc={a['cross_cos_robust_centered']:.3f} "
              f"RankMe/d(norm)={a['rankme_over_d_norm']:.3f}", flush=True)

    # ---- step 2 ranking (same function as analyze_round3.py): contribution of each dim to ||ubar||^2 on V3
    ub2, contrib = decompose(BV["V1"], gamma, beta)
    order = torch.argsort(contrib, descending=True)
    top32 = order[:32].tolist()

    # ---- step 3: remove the mean of h on a set of dims S -> logits shift by -W[:, S] m_S
    m = (m_sum / m_n).float()
    sets = {f"top{k}": order[:k].tolist() for k in KS}
    sets["all"] = list(range(W.shape[1]))
    rest = np.setdiff1d(np.arange(W.shape[1]), top32)
    for k in KS:
        for r in range(N_RAND):
            sets[f"rand{k}_s{r}"] = np.random.default_rng(1000 + r).choice(rest, size=k, replace=False).tolist()
    names = list(sets)
    D = torch.stack([W[:, sets[n]] @ m[sets[n]] for n in names])  # [n_var, V] logit offsets
    tot = {"base": 0.0, **{n: 0.0 for n in names}}
    n_tok = 0
    direct = None
    for e, (hb, y) in enumerate(zip(eval_h, eval_y)):
        hf = hb.to(dev).float()[:-1]
        y = y.to(dev)
        L = hf @ W.T + (b_head if b_head is not None else 0)  # [T-1, V]
        tgt = L.gather(1, y[:, None])[:, 0]
        tot["base"] += (torch.logsumexp(L, -1) - tgt).sum().item()
        for c in range(0, len(names), 8):
            Dc = D[c:c + 8]
            Lv = L[None] - Dc[:, None, :]
            nll = torch.logsumexp(Lv, -1) - (tgt[None] - Dc[:, y])
            for n, val in zip(names[c:c + 8], nll.sum(-1).tolist()):
                tot[n] += val
        n_tok += len(y)
        if e < 5:  # verify the linear shortcut once, directly for top8
            S = sets["top8"]
            h2 = hf.clone()
            h2[:, S] -= m[S]
            L2 = h2 @ W.T + (b_head if b_head is not None else 0)
            Ls = L - D[names.index("top8")]
            d_ = (L2 - Ls).abs().max().item() / L.abs().max().item()
            direct = max(direct or 0.0, d_)
    loss = {n: v / n_tok for n, v in tot.items()}
    dloss = {n: loss[n] - loss["base"] for n in names}
    step3 = {"n_eval_tokens": n_tok, "loss_base": loss["base"], "dloss": dloss,
             "shortcut_vs_direct_max_rel_logit_diff": direct, "sets": {n: sets[n] for n in names},
             "m_norm": m.norm().item(), "m_top8_share": (m[sets["top8"]].pow(2).sum() / m.pow(2).sum()).item()}
    for k in KS:
        rnd = [dloss[f"rand{k}_s{r}"] for r in range(N_RAND)]
        print(f"[{slug}] dloss top{k}={dloss[f'top{k}']:+.4f}  rand mean={np.mean(rnd):+.4f} "
              f"[{min(rnd):+.4f}, {max(rnd):+.4f}]", flush=True)
    print(f"[{slug}] dloss all={dloss['all']:+.4f}  base={loss['base']:.4f}  shortcut check={direct:.2e}", flush=True)

    os.makedirs(args.cache, exist_ok=True)
    torch.save({"x": X0.cpu(), "seq": seq, "pos": pos, "gamma": gamma.cpu(),
                "beta": beta.cpu() if beta is not None else None, "eps": eps, "norm_type": ntype,
                "head_bias": b_head is not None, "m": m.cpu(), "order": order.cpu(),
                "delta": {n: D[names.index(n)].cpu() for n in names if not n.startswith("rand")}},
               os.path.join(args.cache, f"{slug}.pt"))
    res = {"model": slug, "path": path, "n_params": sum(p.numel() for p in model.parameters()),
           "d": W.shape[1], "vocab": W.shape[0], "norm_module": type(norm).__name__, "norm_type": ntype,
           "eps": eps, "has_beta": beta is not None, "head_bias": b_head is not None,
           "tied": bool(model.config.tie_word_embeddings),
           "check0a": check0a, "check0b": check0b, "fold": fold_res, "step1": step1,
           "ubar_sq_V3": ub2, "top32_dims": top32, "step3": step3,
           "seed": args.seed, "elapsed_s": time.time() - t0}
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(os.path.join(args.out, f"{slug}.json"), "w"), indent=1)
    print(f"[{slug}] wrote {args.out}{slug}.json ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
