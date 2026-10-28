"""Measure isotropy / anisotropy / spectral metrics of a causal LM's hidden states.

One JSON per model in --out. Metrics per selected layer, each computed on all
positions ("all") and with position 0 dropped ("no_pos0"):
  A  Su & Collier (2023) within-sequence self-similarity and isotropy = 1 - mean s(x)
  B  cross-sequence mean cosine over random token pairs, raw and mean-centered
  C  spectrum of the centered token bank: RankMe, top-1 / top-10 variance share
  D  outlier dimensions (max / median of per-dim mean |activation|) and
     norm of position 0 relative to the median norm of the other positions
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

MODEL_ROOT = "/dataset1/zailong/models"


# ---------------------------------------------------------------- metrics

def self_similarity(h):
    """Mean pairwise cosine over i != j for one sequence h [T, d]: O(T d)."""
    T = h.shape[0]
    u = torch.nn.functional.normalize(h.float(), dim=-1)
    return ((u.sum(0).pow(2).sum() - T) / (T * (T - 1))).item()


def self_similarity_bruteforce(h):
    u = torch.nn.functional.normalize(h.float(), dim=-1)
    c = u @ u.T
    T = h.shape[0]
    return ((c.sum() - c.diagonal().sum()) / (T * (T - 1))).item()


def sample_cross_pairs(seq_ids, n_pairs, rng):
    """n_pairs index pairs (i, j) into the bank with seq_ids[i] != seq_ids[j]."""
    seq_ids = np.asarray(seq_ids)
    out_i, out_j = [], []
    need = n_pairs
    while need > 0:
        i = rng.integers(0, len(seq_ids), size=2 * need)
        j = rng.integers(0, len(seq_ids), size=2 * need)
        keep = seq_ids[i] != seq_ids[j]
        out_i.append(i[keep][:need])
        out_j.append(j[keep][:need])
        need -= len(out_i[-1])
    return np.concatenate(out_i), np.concatenate(out_j)


def pair_cosine(X, i, j):
    u = torch.nn.functional.normalize(X, dim=-1)
    return (u[i] * u[j]).sum(-1).mean().item()


def spectrum(Xc):
    """Spectral metrics of an already-centered matrix Xc [N, d]."""
    s = torch.linalg.svdvals(Xc).double()
    p = s / s.sum()
    p = p[p > 0]
    rankme = torch.exp(-(p * p.log()).sum()).item()
    var = s.pow(2) / s.pow(2).sum()
    return {
        "rankme": rankme,
        "rankme_over_d": rankme / Xc.shape[1],
        "top1_var": var[0].item(),
        "top10_var": var[:10].sum().item(),
        "max_rank": min(Xc.shape),
    }


def bank_metrics(X, seq_ids, n_pairs, seed, device):
    """B and C on a token bank X [N, d] (CPU fp16) -> dict."""
    X = X.to(device).float()
    i, j = sample_cross_pairs(seq_ids, n_pairs, np.random.default_rng(seed))
    i, j = torch.as_tensor(i, device=device), torch.as_tensor(j, device=device)
    Xc = X - X.mean(0, keepdim=True)
    out = {
        "n_tokens": X.shape[0],
        "cross_cos": pair_cosine(X, i, j),
        "cross_cos_centered": pair_cosine(Xc, i, j),
    }
    out.update(spectrum(Xc))
    return out


# ---------------------------------------------------------------- helpers

def resolve_model(name):
    if os.path.isdir(name):
        return name
    p = os.path.join(MODEL_ROOT, name)
    if os.path.isdir(p):
        return p
    raise FileNotFoundError(f"model {name!r} not found locally")


def select_layers(L):
    """hidden_states indices: embedding (0), ~every 1/8 depth, last (L)."""
    return sorted({0, L} | {int(round(k * L / 8)) for k in range(1, 8)})


def final_norm_module(model):
    base = model.base_model
    for name in ("norm", "ln_f", "final_layernorm", "final_layer_norm"):  # final_layer_norm: GPT-NeoX (Pythia)
        if hasattr(base, name):
            return getattr(base, name)
    return None


def load_samples(path, n):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[:n] if n else rows


# ---------------------------------------------------------------- main

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_seqs", type=int, default=0, help="0 = all")
    ap.add_argument("--bank_per_seq", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/")
    args = ap.parse_args()

    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    path = resolve_model(args.model)
    slug = os.path.basename(path.rstrip("/"))
    device = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device, local_files_only=True)
    model.eval()
    print(f"[{slug}] loaded in {time.time() - t0:.0f}s, dtype={model.dtype}", flush=True)

    cfg = model.config
    L = cfg.num_hidden_layers
    d = cfg.hidden_size
    layers = select_layers(L)
    n_params = sum(p.numel() for p in model.parameters())
    rows = load_samples(args.data, args.n_seqs)

    acc = {l: {"ss_all": [], "ss_no0": [], "pos0_ratio": [],
               "absmean_all": torch.zeros(d, dtype=torch.float64, device=device),
               "absmean_no0": torch.zeros(d, dtype=torch.float64, device=device),
               "bank": [], "bank_pos": []} for l in layers}
    bank_seq = []
    n_tok_all = n_tok_no0 = 0
    lengths, bos_flags, first_ids = [], [], []
    norm_check = None
    apply_norm = False
    fnorm = final_norm_module(model)
    rng = np.random.default_rng(args.seed)

    for sid, row in enumerate(rows):
        enc = tok(row["text"], add_special_tokens=True, truncation=True,
                  max_length=args.max_len, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        T = ids.shape[1]
        lengths.append(T)
        first_ids.append(int(ids[0, 0]))
        bos_flags.append(tok.bos_token_id is not None and int(ids[0, 0]) == tok.bos_token_id)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        hs = out.hidden_states
        assert len(hs) == L + 1, (len(hs), L)

        if sid == 0:
            # is hidden_states[-1] already the final-norm output fed to the LM head?
            head = model.get_output_embeddings()
            logits = out.logits.float()
            diff_raw = (head(hs[-1]).float() - logits).abs().max().item()
            diff_normed = (head(fnorm(hs[-1])).float() - logits).abs().max().item() if fnorm else None
            apply_norm = diff_normed is not None and diff_normed < diff_raw
            norm_check = {"final_norm_module": type(fnorm).__name__ if fnorm else None,
                          "lmhead(hs[-1]) vs logits maxabs": diff_raw,
                          "lmhead(norm(hs[-1])) vs logits maxabs": diff_normed,
                          "logit_absmax": logits.abs().max().item(),
                          "applied_final_norm_manually": apply_norm}
            print(f"[{slug}] final-norm check: {norm_check}", flush=True)
        if apply_norm:
            hs = hs[:-1] + (fnorm(hs[-1]),)

        pos = rng.choice(T, size=min(args.bank_per_seq, T), replace=False)
        pos_t = torch.as_tensor(pos, device=device)
        bank_seq.extend([sid] * len(pos))
        n_tok_all += T
        n_tok_no0 += T - 1
        for l in layers:
            h = hs[l][0].float()  # [T, d], fp32 from here on
            a = acc[l]
            a["ss_all"].append(self_similarity(h))
            a["ss_no0"].append(self_similarity(h[1:]))
            norms = h.norm(dim=-1)
            a["pos0_ratio"].append((norms[0] / norms[1:].median()).item())
            ah = h.abs()
            a["absmean_all"] += ah.sum(0).double()
            a["absmean_no0"] += ah[1:].sum(0).double()
            a["bank"].append(h[pos_t].half().cpu())
            a["bank_pos"].append(pos)
        if (sid + 1) % 100 == 0:
            print(f"[{slug}] {sid + 1}/{len(rows)} seqs, {time.time() - t0:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()
    bank_seq = np.asarray(bank_seq)
    layer_out = []
    for l in layers:
        a = acc[l]
        X = torch.cat(a["bank"])
        bpos = np.concatenate(a["bank_pos"])
        keep = bpos != 0
        res = {"layer": l, "rel_depth": l / L,
               "label": "embedding" if l == 0 else ("final (post-norm)" if l == L else f"block {l}")}
        for tag, ss, absm, ntok, bank_kw in (
                ("all", a["ss_all"], a["absmean_all"], n_tok_all, (X, bank_seq)),
                ("no_pos0", a["ss_no0"], a["absmean_no0"], n_tok_no0,
                 (X[torch.as_tensor(keep)], bank_seq[keep]))):
            ss = np.asarray(ss)
            absm = (absm / ntok).float()
            top = torch.topk(absm, 5)
            r = {"self_sim_mean": float(ss.mean()), "self_sim_std": float(ss.std()),
                 "isotropy": float(1 - ss.mean()),
                 "outlier_ratio": (absm.max() / absm.median()).item(),
                 "outlier_top5_dims": top.indices.tolist(),
                 "outlier_top5_absmean": top.values.tolist()}
            r.update(bank_metrics(*bank_kw, args.n_pairs, args.seed, device))
            res[tag] = r
        pr = np.asarray(a["pos0_ratio"])
        res["pos0_norm_ratio_mean"] = float(pr.mean())
        res["pos0_norm_ratio_median"] = float(np.median(pr))
        layer_out.append(res)
        print(f"[{slug}] layer {l:3d}: iso={res['all']['isotropy']:.3f} "
              f"iso_no0={res['no_pos0']['isotropy']:.3f} cross={res['all']['cross_cos']:.3f} "
              f"rankme/d={res['all']['rankme_over_d']:.3f} pos0/med={res['pos0_norm_ratio_median']:.1f}",
              flush=True)

    first_tok = {str(k): int(v) for k, v in zip(*np.unique(first_ids, return_counts=True))}
    result = {
        "model": slug, "path": path, "n_params": n_params, "n_layers": L, "d": d,
        "model_type": cfg.model_type,
        "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id,
        "bos_added_frac": float(np.mean(bos_flags)), "bos_added": bool(np.mean(bos_flags) > 0.5),
        "first_token_counts": dict(sorted(first_tok.items(), key=lambda kv: -kv[1])[:5]),
        "first_token_text": tok.convert_ids_to_tokens(int(max(first_tok, key=first_tok.get))),
        "n_seqs": len(rows), "mean_len": float(np.mean(lengths)), "min_len": int(np.min(lengths)),
        "max_len": args.max_len, "bank_per_seq": args.bank_per_seq, "n_pairs": args.n_pairs,
        "final_norm_check": norm_check,
        "layers": layer_out,
        "seed": args.seed, "dtype": "bfloat16 forward, float32 metrics",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "host": platform.node(),
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(args.out, exist_ok=True)
    suffix = f"_n{args.n_seqs}" if args.n_seqs else ""
    fn = os.path.join(args.out, f"{slug}{suffix}.json")
    with open(fn, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{slug}] wrote {fn} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
