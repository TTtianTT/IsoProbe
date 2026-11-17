"""Round 6 phase 2 T2: is late-training saturation concentrated on the high-deviation tokens, and what are they?

dump (GPU, one checkpoint per row of configs/round6_p2_tokens.tsv): per eval token of data/eval_loss_round6.jsonl
    nll, predictive entropy, p_t = ubar . V1_t (ubar, c = bank mean of p from that checkpoint's round-4/5 bank),
    input / target token id, paragraph index, position -> cache/round6/p2/tokens/<model>/<rev>.npz (not in git);
    plus the first left singular vector u1 of the folded head W diag(gamma) (all rows).
analyze (CPU): 2a split of sum_t (loss_143000 - loss_before) into H (top q of |p_t - c| at step 143000, q = 0.1 / 0.5 / 1 %)
    and its complement, paragraph bootstrap (1000); 2b Spearman of |p_t - c| with -log input / target frequency and entropy,
    top-30 input / target tokens in H, base loss on H vs complement, top-30 vocabulary entries by |u1|.
Writes results/round6/p2/tokens/summary.json (+ .md).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

PAIRS = {"pythia-70m": ("step50000", "step143000"), "pythia-160m": ("step80000", "step143000"),
         "pythia-410m": ("step80000", "step143000")}
CACHE = "cache/round6/p2/tokens"


@torch.no_grad()
def dump(model_name, rev, cache):
    import round6_causal as rc
    from round3_lib import normalize
    out = os.path.join(cache, model_name, f"{rev}.npz")
    if os.path.exists(out):
        print(f"{out} exists", flush=True)
        return
    C = rc.Ckpt(model_name, rev)
    X, Y, FL, chk = C.eval_cache()
    rows = [json.loads(l) for l in open(rc.EVAL)]
    inp, para = [], []
    for i, r in enumerate(rows):
        ids = C.tok(r["text"], add_special_tokens=True, truncation=True, max_length=rc.CHUNK)["input_ids"]
        inp.extend(ids[:-1])
        para.extend([i] * (len(ids) - 1))
    assert len(inp) == len(Y)
    bank_fn = next(f for f in (f"cache/round4_banks/{model_name}/{rev}.pt", f"cache/round5_banks/{model_name}/{rev}.pt",
                               f"cache/round6/poly_banks/{model_name}/{rev}.pt")  # last: PolyPythias seeds (fallback 2)
                   if os.path.exists(f))
    b = torch.load(bank_fn, weights_only=False)
    V1b = normalize(b["x"].to(rc.DEV).float(), b["norm_type"], b["eps"])
    u = torch.nn.functional.normalize(V1b, dim=-1).mean(0)
    u = u / u.norm()
    c = (V1b @ u).mean().item()
    Wf = C.W * C.gamma
    bf = C.W @ C.beta if C.beta is not None else None
    nll, ent, p = [], [], []
    for s in range(0, len(Y), 4096):
        V1 = normalize(X[s:s + 4096].float(), C.ntype, C.eps)
        L = C.logits(V1, Wf, bf)
        lp = torch.log_softmax(L, -1)
        y = Y[s:s + 4096]
        nll.append(-lp.gather(1, y[:, None])[:, 0])
        ent.append(-(lp.exp() * lp).sum(-1))
        p.append(V1 @ u)
    U, S, Vh = torch.linalg.svd(Wf.double(), full_matrices=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, nll=torch.cat(nll).cpu().numpy(), ent=torch.cat(ent).cpu().numpy(),
                        p=torch.cat(p).cpu().numpy(), c=c, inp=np.asarray(inp, dtype=np.int32),
                        tgt=Y.cpu().numpy().astype(np.int32), para=np.asarray(para, dtype=np.int32),
                        pos=C.pos.cpu().numpy().astype(np.int32), in_samples=FL.cpu().numpy(),
                        u1=U[:, 0].float().cpu().numpy(), s1=S[0].item(), s2=S[1].item(), check=chk)
    print(f"[{model_name}@{rev}] dumped {len(Y)} tokens; mean nll {torch.cat(nll).mean().item():.4f}; c {c:.3f}", flush=True)


def spearman_fast(a, b):
    from round3_lib import spearman
    return float(spearman(a, b)[0])


def analyze(cache, out_dir, n_boot=1000, seed=0, models=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/dataset1/zailong/models/ckpts/pythia-70m/step143000", local_files_only=True)
    counts = np.load("cache/unigrams/8fed1dc1b157.npy").astype(np.float64)
    nlf = -np.log((counts + 1) / (counts.sum() + len(counts)))  # add-1 smoothed -log frequency
    res = {}
    rng = np.random.default_rng(seed)
    pairs = dict(PAIRS, **{f"pythia-70m-seed{k}": ("step50000", "step143000") for k in (2, 3)}) if models else PAIRS
    for m, (r0, r1) in pairs.items():
        if models and m not in models:
            continue
        f0, f1 = (os.path.join(cache, m, f"{r}.npz") for r in (r0, r1))
        if not (os.path.exists(f0) and os.path.exists(f1)):
            continue
        A, B = np.load(f0), np.load(f1)
        assert (A["tgt"] == B["tgt"]).all() and (A["para"] == B["para"]).all()
        d = B["nll"].astype(np.float64) - A["nll"].astype(np.float64)
        dev = np.abs(B["p"] - B["c"]).astype(np.float64)
        para = B["para"]
        n_par = int(para.max()) + 1
        W = rng.multinomial(n_par, np.full(n_par, 1 / n_par), size=n_boot).astype(np.float64)  # paragraph weights
        r = {"before": r0, "after": r1, "n_tokens": int(len(d)), "mean_d": float(d.mean()), "sum_d": float(d.sum()),
             "loss_before": float(A["nll"].mean()), "loss_after": float(B["nll"].mean()), "split": {}}
        order = np.argsort(-dev)
        for q in (0.001, 0.005, 0.01):
            k = int(round(q * len(d)))
            H = np.zeros(len(d), bool)
            H[order[:k]] = True
            sH = np.bincount(para, weights=d * H, minlength=n_par)
            sC = np.bincount(para, weights=d * ~H, minlength=n_par)
            tot = sH + sC
            share_C = sC.sum() / tot.sum()
            bs = (W @ sC) / (W @ tot)
            cls = "diffuse" if share_C >= 0.8 else ("concentrated" if share_C <= 0.2 else "mixed")
            r["split"][f"{q * 100:g}%"] = {
                "k": k, "share_H": float(sH.sum() / tot.sum()), "share_complement": float(share_C),
                "share_complement_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "mean_d_H": float(d[H].mean()), "mean_d_complement": float(d[~H].mean()),
                "dev_threshold": float(dev[order[k - 1]]), "class": cls}
        # 2b on the late checkpoint
        k = int(round(0.001 * len(d)))
        H = np.zeros(len(d), bool)
        H[order[:k]] = True
        r["2b"] = {
            "spearman_dev_vs_neglogfreq_input": spearman_fast(dev, nlf[B["inp"]]),
            "spearman_dev_vs_neglogfreq_target": spearman_fast(dev, nlf[B["tgt"]]),
            "spearman_dev_vs_entropy": spearman_fast(dev, B["ent"]),
            "loss_H": float(B["nll"][H].mean()), "loss_complement": float(B["nll"][~H].mean()),
            "entropy_H": float(B["ent"][H].mean()), "entropy_complement": float(B["ent"][~H].mean()),
            "frac_H_pos0": float((B["pos"][H] == 0).mean()),
            "median_neglogfreq_input_H": float(np.median(nlf[B["inp"][H]])),
            "median_neglogfreq_input_all": float(np.median(nlf[B["inp"]])),
            "top30_input_H": [(tok.decode([int(t)]), n) for t, n in collections.Counter(B["inp"][H].tolist()).most_common(30)],
            "top30_target_H": [(tok.decode([int(t)]), n) for t, n in collections.Counter(B["tgt"][H].tolist()).most_common(30)],
            "sign_of_p_minus_c_H": {"pos": int(((B["p"] - B["c"])[H] > 0).sum()), "neg": int(((B["p"] - B["c"])[H] < 0).sum())},
        }
        # exploratory: share of tokens that decode to an incomplete UTF-8 byte sequence (U+FFFD), i.e. byte-level pieces
        vocab_ids = np.unique(np.concatenate([B["inp"], B["tgt"]]))
        bad = np.zeros(int(vocab_ids.max()) + 1, bool)
        for t in vocab_ids:
            bad[t] = "\ufffd" in tok.decode([int(t)])
        r["2b"]["explore_incomplete_utf8"] = {
            "input_H": float(bad[B["inp"][H]].mean()), "input_all": float(bad[B["inp"]].mean()),
            "target_H": float(bad[B["tgt"][H]].mean()), "target_all": float(bad[B["tgt"]].mean()),
            "share_of_all_incomplete_inputs_in_H": float((bad[B["inp"]] & H).sum() / max(bad[B["inp"]].sum(), 1))}
        r["2b"]["top30_input_H_raw"] = [(tok.convert_ids_to_tokens(int(t)), n) for t, n in
                                        collections.Counter(B["inp"][H].tolist()).most_common(30)]
        u1 = B["u1"]
        top = np.argsort(-np.abs(u1))[:30]
        r["2b"]["top30_u1"] = [(tok.decode([int(t)]) if t < len(tok) else f"<pad {t}>", float(u1[t])) for t in top]
        r["2b"]["u1_sign_note"] = "SVD sign is arbitrary; the logit change of token v is (s1 * (v1 . dV1)) * u1[v]"
        res[m] = r
        print(m, json.dumps({k: v for k, v in r.items() if k != "2b"})[:600], flush=True)
        print("  2b", {k: v for k, v in r["2b"].items() if not k.startswith("top")}, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dump", "analyze"])
    ap.add_argument("--rows", default="0")
    ap.add_argument("--list", default="configs/round6_p2_tokens.tsv")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out", default="results/round6/p2/tokens/")
    ap.add_argument("--models", default="", help="comma list for analyze (fallback 2: pythia-70m-seed2,pythia-70m-seed3)")
    args = ap.parse_args()
    if args.mode == "analyze":
        analyze(args.cache, args.out, models=[m for m in args.models.split(",") if m] or None)
        return
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    idx = []
    for part in args.rows.split(","):
        a, _, b = part.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    for i in idx:
        dump(*rows[i], args.cache)


if __name__ == "__main__":
    main()
