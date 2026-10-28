"""Round 4 B (GPU): training dynamics on intermediate checkpoints (Pythia, OLMo-2).

For each checkpoint (rows of configs/round4_ckpts.tsv, local copies under /dataset1/zailong/models/ckpts/):
  checks 0a (V3 reconstruction) and 1a (gamma folding); last-layer V0-V3 metrics; gamma / beta statistics;
  per-dimension decomposition on V3 and V1; mean-removal intervention (top-1/2/4/8/16/32, all) with offset spread,
  additive share and cancellation correlation over the whole vocabulary; massive activations on the round-1 layers.
Same data, positions, bank and estimation / evaluation split as extract_round3.py.
Writes results/round4/dyn/<model>/<revision>.json and cache/round4_banks/<model>/<revision>.pt.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from download_round4 import ckpt_dir
from extract_round3 import KS, norm_params
from measure import final_norm_module, load_samples, select_layers, self_similarity
from round3_lib import VNAMES, bank_metrics_r3, decompose, variants


def pct_rank(x):
    """Percentile rank in [0, 1] with ties averaged (at step 0 gamma is all ones -> every dim gets 0.5)."""
    s, idx = torch.sort(x)
    _, inv, counts = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0)
    r = torch.empty(x.shape, device=x.device)
    r[idx] = ((ends - counts + ends - 1).float() / 2)[inv]
    return r / (len(x) - 1)


def additive_share(a, total):
    a, total = a.double(), total.double()
    return (((a - a.mean()) * (total - total.mean())).mean() / total.var(unbiased=False)).item()


@torch.no_grad()
def measure(repo, rev, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = ckpt_dir(repo, rev)
    name = repo.split("/")[1]
    dev = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev,
                                                 local_files_only=True).eval()
    norm = final_norm_module(model)
    gamma, beta, eps, ntype = norm_params(norm)
    head = model.get_output_embeddings()
    W = head.weight.detach().float()
    b_head = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
    n_voc = min(len(tok), W.shape[0])  # rows past the tokenizer are padding (Pythia 50304 vs 50277)
    L = model.config.num_hidden_layers
    layers = select_layers(L)
    d = W.shape[1]
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0]))
    rows = load_samples(args.data, 0)
    rng = np.random.default_rng(args.seed)

    bank_x, bank_seq, bank_pos = [], [], []
    full_ss = {v: [] for v in VNAMES}
    rec_err, fold = [], {"rel_err": [], "agree": 0, "n": 0}
    m_sum, m_n = torch.zeros(d, dtype=torch.float64, device=dev), 0
    eval_h, eval_y = [], []
    W_fold = W * gamma
    b_fold = (W @ beta if beta is not None else torch.zeros(W.shape[0], device=dev)) + (
        b_head if b_head is not None else 0)
    absm = {l: torch.zeros(d, dtype=torch.float64, device=dev) for l in layers}
    pos0 = {l: [] for l in layers}
    n_tok, first_ids = 0, []
    for sid, row in enumerate(rows):
        ids = tok(row["text"], add_special_tokens=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt")["input_ids"].to(dev)
        T = ids.shape[1]
        first_ids.append(int(ids[0, 0]))
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        h = out.hidden_states[-1][0].float()
        V = variants(cap["x"][0], gamma, beta, ntype, eps)
        rec_err.append(((V["V3"] - h).norm() / h.norm()).item())
        for v in VNAMES:
            full_ss[v].append(self_similarity(V[v]))
        pos = rng.choice(T, size=min(args.bank_per_seq, T), replace=False)
        bank_x.append(V["V0"][torch.as_tensor(pos, device=dev)].cpu())
        bank_seq.extend([sid] * len(pos))
        bank_pos.append(pos)
        for l in layers:
            hl = out.hidden_states[l][0].float()
            absm[l] += hl.abs().sum(0).double()
            nr = hl.norm(dim=-1)
            pos0[l].append((nr[0] / nr[1:].median()).item())
        n_tok += T
        if sid < args.n_fold:
            ref = V["V3"] @ W.T + (b_head if b_head is not None else 0)
            fl = V["V1"] @ W_fold.T + b_fold
            fold["rel_err"].append(((ref - fl).abs().max() / ref.abs().max()).item())
            fold["agree"] += int((ref.argmax(-1) == fl.argmax(-1)).sum())
            fold["n"] += T
        if sid < args.n_est:
            m_sum += h[1:].sum(0).double()
            m_n += T - 1
        else:
            eval_h.append(out.hidden_states[-1][0].cpu())
            eval_y.append(ids[0, 1:].cpu())

    X0 = torch.cat(bank_x).to(dev)
    seq, pos = np.asarray(bank_seq), np.concatenate(bank_pos)
    BV = variants(X0, gamma, beta, ntype, eps)
    keep_t = torch.as_tensor(pos != 0, device=dev)
    step1 = {}
    for v in VNAMES:
        step1[v] = {}
        for tag, Xb, sq, ps, ss in (("all", BV[v], seq, pos, full_ss[v]),
                                    ("no_pos0", BV[v][keep_t], seq[pos != 0], pos[pos != 0], None)):
            r = bank_metrics_r3(Xb, sq, ps, args.n_pairs, args.seed)
            if ss is not None:
                r["isotropy_full"] = float(1 - np.mean(ss))
            step1[v][tag] = r
    check0a = {"max_rel_err": float(np.max(rec_err)), "pass": float(np.max(rec_err)) < 1e-2}
    fold_res = {"max_rel_err": float(np.max(fold["rel_err"])), "argmax_agree": fold["agree"] / fold["n"]}
    fold_res["pass"] = fold_res["max_rel_err"] < 1e-4 and fold_res["argmax_agree"] >= 0.9999

    # gamma / beta
    ag = gamma.abs()
    gb = {"gamma_max_over_median": (ag.max() / ag.median()).item(), "gamma_norm": gamma.norm().item(),
          "gamma_mean": gamma.mean().item(), "gamma_abs_median": ag.median().item(),
          "beta_norm": beta.norm().item() if beta is not None else None,
          "beta_over_gamma": (beta.norm() / gamma.norm()).item() if beta is not None else None}

    # per-dimension decomposition (V3 and V1)
    ub3, c3 = decompose(BV["V1"], gamma, beta)
    ub1, c1 = decompose(BV["V1"])
    o3, o1 = torch.argsort(c3, descending=True), torch.argsort(c1, descending=True)
    mu, sd = BV["V1"].mean(0), BV["V1"].std(0, unbiased=False)
    share_pct, gamma_pct = pct_rank(mu.pow(2) / (mu.pow(2) + sd.pow(2))), pct_rank(ag)
    cum = lambda c, o: {k: c[o[:k]].sum().item() for k in (1, 2, 4, 8, 10, 32)}  # noqa: E731
    decomp = {"ubar_sq_V3": ub3, "ubar_sq_V1": ub1, "top10_V3": o3[:10].tolist(), "top10_V1": o1[:10].tolist(),
              "cum_V3": cum(c3, o3), "cum_V1": cum(c1, o1), "contrib_V3_top10": c3[o3[:10]].tolist(),
              "v1_share_pct_of_V3_top10": share_pct[o3[:10]].tolist(),
              "gamma_pct_of_V3_top10": gamma_pct[o3[:10]].tolist()}

    # mean removal: logits shift by -W[:, S] m_S
    m = (m_sum / m_n).float()
    sets = {f"top{k}": o3[:k].tolist() for k in KS}
    sets["all"] = list(range(d))
    names = list(sets)
    D = torch.stack([W[:, sets[n]] @ m[sets[n]] for n in names])
    tot = {"base": 0.0, **{n: 0.0 for n in names}}
    nt = 0
    for hb, y in zip(eval_h, eval_y):
        hf = hb.to(dev).float()[:-1]
        y = y.to(dev)
        Lg = hf @ W.T + (b_head if b_head is not None else 0)
        tgt = Lg.gather(1, y[:, None])[:, 0]
        tot["base"] += (torch.logsumexp(Lg, -1) - tgt).sum().item()
        for c in range(0, len(names), 8):
            Dc = D[c:c + 8]
            nll = torch.logsumexp(Lg[None] - Dc[:, None, :], -1) - (tgt[None] - Dc[:, y])
            for n, val in zip(names[c:c + 8], nll.sum(-1).tolist()):
                tot[n] += val
        nt += len(y)
    loss = {n: v / nt for n, v in tot.items()}
    Dv = D[:, :n_voc].double()
    dall = Dv[names.index("all")]
    func = {"n_eval_tokens": nt, "loss_base": loss["base"], "n_vocab_used": n_voc,
            "sd_all": dall.std(unbiased=False).item(), "sets": {}}
    for n in names:
        dS = Dv[names.index(n)]
        e = {"dloss": loss[n] - loss["base"], "offset_sd": dS.std(unbiased=False).item(),
             "offset_absmean": dS.mean().abs().item()}
        if n != "all":
            dR = dall - dS
            e.update({"sd_R": dR.std(unbiased=False).item(), "additive_share": additive_share(dS, dall),
                      "corr_S_R": torch.corrcoef(torch.stack([dS, dR]))[0, 1].item()})
        func["sets"][n] = e
    per_dim = []
    for k in o3[:10].tolist():
        col = W[:n_voc, k].double()
        dk = col * m[k]
        per_dim.append({"dim": k, "contrib_V3": c3[k].item(), "m": m[k].item(), "gamma": gamma[k].item(),
                        "offset_sd": dk.std(unbiased=False).item(), "offset_absmean": dk.mean().abs().item(),
                        "additive_share": additive_share(dk, dall), "col_mean": col.mean().item(),
                        "col_sd": col.std(unbiased=False).item()})
    func["per_dim_top10"] = per_dim

    # massive activations on the round-1 layers
    massive = []
    for l in layers:
        a = (absm[l] / n_tok).float()
        top = torch.topk(a, 5)
        pr = np.asarray(pos0[l])
        massive.append({"layer": l, "outlier_ratio": (a.max() / a.median()).item(),
                        "top5_dims": top.indices.tolist(), "top5_absmean": top.values.tolist(),
                        "top5_gamma_pct": gamma_pct[top.indices].tolist(),
                        "pos0_ratio_mean": float(pr.mean()), "pos0_ratio_median": float(np.median(pr))})

    res = {"model": name, "repo": repo, "revision": rev, "path": path, "n_layers": L, "d": d,
           "vocab_head": W.shape[0], "vocab_tok": len(tok), "norm_module": type(norm).__name__,
           "norm_type": ntype, "eps": eps, "has_beta": beta is not None, "head_bias": b_head is not None,
           "tied": bool(model.config.tie_word_embeddings),
           "head_is_input_embedding": head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(),
           "bos_token_id": tok.bos_token_id, "first_token_is_bos_frac": float(np.mean(
               [tok.bos_token_id is not None and f == tok.bos_token_id for f in first_ids])),
           "check0a": check0a, "fold": fold_res, "step1": step1, "gamma_beta": gb, "decomp": decomp,
           "function": func, "massive": massive, "seed": args.seed, "elapsed_s": time.time() - t0}
    os.makedirs(os.path.join(args.cache, name), exist_ok=True)
    torch.save({"x": X0.cpu(), "seq": seq, "pos": pos, "gamma": gamma.cpu(),
                "beta": beta.cpu() if beta is not None else None, "eps": eps, "norm_type": ntype, "m": m.cpu(),
                "order": o3.cpu(), "delta": {n: D[names.index(n)].cpu() for n in names}},
               os.path.join(args.cache, name, f"{rev}.pt"))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round4_ckpts.tsv")
    ap.add_argument("--rows", required=True, help="row indices of --list, e.g. 0-18 or 3,7")
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--bank_per_seq", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--n_fold", type=int, default=20)
    ap.add_argument("--n_est", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/round4/dyn/")
    ap.add_argument("--cache", default="cache/round4_banks/")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    idx = []
    for part in args.rows.split(","):
        a, _, b = part.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    for i in idx:
        repo, rev = rows[i]
        fn = os.path.join(args.out, repo.split("/")[1], f"{rev}.json")
        if os.path.exists(fn):
            print(f"{fn} exists, skipping", flush=True)
            continue
        res = measure(repo, rev, args)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        json.dump(res, open(fn, "w"), indent=1)
        s, f = res["step1"], res["function"]["sets"]
        print(f"[{res['model']}@{rev}] 0a={res['check0a']['max_rel_err']:.1e} fold={res['fold']['max_rel_err']:.1e}/"
              f"{res['fold']['argmax_agree']:.5f} A V1={s['V1']['all']['isotropy_full']:.3f} "
              f"V3={s['V3']['all']['isotropy_full']:.3f} |γ|max/med={res['gamma_beta']['gamma_max_over_median']:.1f} "
              f"loss={res['function']['loss_base']:.3f} dl_all={f['all']['dloss']:+.3f} dl_top8={f['top8']['dloss']:+.3f} "
              f"pos0max={max(x['pos0_ratio_median'] for x in res['massive']):.1f} ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
