"""Decompose the final norm: which step creates the last-layer anisotropy?

The final norm computes  norm(x) = gamma * xhat + beta  (beta only for LayerNorm), where
xhat = (x - mean) / std for LayerNorm and xhat = x / rms(x) for RMSNorm. For every
variant below A / B / C are computed on the same sequences and bank positions as measure.py:
  pre        x, the input of the final norm (= output of the last block)
  xhat       normalized, no affine
  no_bias    gamma * xhat
  post       gamma * xhat + beta  (= hidden_states[-1])
  post-beta  post minus beta (only if the norm has a bias; equals no_bias up to bf16 rounding)
Output: results/diag/<slug>.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from measure import bank_metrics, final_norm_module, load_samples, resolve_model, self_similarity


def decompose(norm, x):
    """fp32 recomputation of the norm's steps from its input x [T, d]."""
    x = x.float()
    eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon", None) or 1e-6
    if isinstance(norm, torch.nn.LayerNorm):
        xhat = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    else:  # RMSNorm variants used by Llama / Qwen / Mistral / Olmo
        xhat = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    gamma = norm.weight.float()
    beta = norm.bias.float() if getattr(norm, "bias", None) is not None else None
    v = {"pre": x, "xhat": xhat, "no_bias": gamma * xhat}
    v["post_recomputed"] = v["no_bias"] + beta if beta is not None else v["no_bias"]
    return v, beta


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_seqs", type=int, default=0)
    ap.add_argument("--bank_per_seq", type=int, default=10)
    ap.add_argument("--n_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/diag/")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    path = resolve_model(args.model)
    slug = os.path.basename(path.rstrip("/"))
    device = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device,
                                                 local_files_only=True).eval()
    norm = final_norm_module(model)
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0], y=out))
    rows = load_samples(args.data, args.n_seqs)
    rng = np.random.default_rng(args.seed)  # same stream as measure.py -> same bank positions

    names = None
    ss, bank, bank_pos, bank_seq = {}, {}, [], []
    beta = None
    post_check, pos0_ratio_pre = [], []
    for sid, row in enumerate(rows):
        enc = tok(row["text"], add_special_tokens=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt")
        ids = enc["input_ids"].to(device)
        T = ids.shape[1]
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        v, beta = decompose(norm, cap["x"][0])
        v["post"] = out.hidden_states[-1][0].float()
        if beta is not None:
            v["post_minus_beta"] = v["post"] - beta
        # recomputation must match the module's own output (= hidden_states[-1])
        post_check.append((v.pop("post_recomputed") - v["post"]).norm().item() / v["post"].norm().item())
        if names is None:
            names = list(v)
            ss = {k: {"all": [], "no_pos0": []} for k in names}
            bank = {k: [] for k in names}
        norms = v["pre"].norm(dim=-1)
        pos0_ratio_pre.append((norms[0] / norms[1:].median()).item())
        pos = rng.choice(T, size=min(args.bank_per_seq, T), replace=False)
        bank_pos.append(pos)
        bank_seq.extend([sid] * len(pos))
        for k in names:
            ss[k]["all"].append(self_similarity(v[k]))
            ss[k]["no_pos0"].append(self_similarity(v[k][1:]))
            bank[k].append(v[k][torch.as_tensor(pos, device=device)].half().cpu())

    bank_pos = np.concatenate(bank_pos)
    bank_seq = np.asarray(bank_seq)
    keep = bank_pos != 0
    variants = {}
    for k in names:
        X = torch.cat(bank[k])
        variants[k] = {}
        for tag, Xb, sb, s in (("all", X, bank_seq, ss[k]["all"]),
                               ("no_pos0", X[torch.as_tensor(keep)], bank_seq[keep], ss[k]["no_pos0"])):
            r = {"isotropy": float(1 - np.mean(s))}
            r.update(bank_metrics(Xb, sb, args.n_pairs, args.seed, device))
            variants[k][tag] = r
        a = variants[k]["all"]
        print(f"[{slug}] {k:16s} iso={a['isotropy']:.3f} cross={a['cross_cos']:.3f} "
              f"cross_c={a['cross_cos_centered']:.3f} rankme/d={a['rankme_over_d']:.3f} top1={a['top1_var']:.3f}",
              flush=True)

    # how large is beta relative to the tokens, and does it point along their mean?
    post_bank = torch.cat(bank["post"]).float()
    beta_info = None
    if beta is not None:
        b = beta.cpu()
        mu = post_bank.mean(0)
        beta_info = {"beta_norm": b.norm().item(),
                     "median_post_token_norm": post_bank.norm(dim=-1).median().item(),
                     "cos(beta, mean post token)": torch.nn.functional.cosine_similarity(b, mu, dim=0).item(),
                     "beta_norm / mean_post_token_norm": (b.norm() / mu.norm()).item()}
        print(f"[{slug}] beta: {beta_info}", flush=True)

    result = {
        "model": slug, "path": path, "n_params": sum(p.numel() for p in model.parameters()),
        "norm_type": type(norm).__name__, "norm_has_bias": beta is not None,
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "embeddings_actually_shared": model.get_input_embeddings().weight.data_ptr()
                                      == model.get_output_embeddings().weight.data_ptr(),
        "post_recompute_rel_err_max": float(np.max(post_check)),
        "pre_pos0_norm_ratio_median": float(np.median(pos0_ratio_pre)),
        "beta": beta_info, "variants": variants, "n_seqs": len(rows), "seed": args.seed,
        "elapsed_s": time.time() - t0,
    }
    print(f"[{slug}] tie={result['tie_word_embeddings']} shared={result['embeddings_actually_shared']} "
          f"recompute_err={result['post_recompute_rel_err_max']:.2e}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"{slug}{f'_n{args.n_seqs}' if args.n_seqs else ''}.json")
    json.dump(result, open(fn, "w"), indent=1)
    print(f"[{slug}] wrote {fn} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
