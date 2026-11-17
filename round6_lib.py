"""Round 6 shared helpers (fp64, device-agnostic): banks, heads, bank metrics, gauge transforms, gauge metric h*.

Gauge: logits z = W h + b. For invertible A, (A h, W A^-1, b) gives the same logits. Base point (V1, W diag(gamma), W beta).
h* = Mt^(1/2) [h; 1] with Mt = [W, b]^T P [W, b]: its inner products are those of the centered logits (bias included), so
h* is invariant (up to an orthogonal map) under every linear gauge transform, translations and the gamma / beta folding.
"""
import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open

from measure import resolve_model

MODELS12 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "Qwen2.5-7B-Instruct", "Llama-3.1-8B",
            "Llama-3.1-8B-Instruct", "Qwen3-8B-Base", "Qwen3-8B", "Mistral-Small-24B-Instruct-2501",
            "Olmo-3.1-32B-Instruct", "Qwen3-32B"]
DT = torch.float64


# ---------------------------------------------------------------- data

def load_bank(slug, root="cache/banks"):
    """Round-3 bank -> dict with fp64 V1, V3, gamma, beta (None for RMSNorm), seq, pos and the raw bank."""
    b = torch.load(os.path.join(root, f"{slug}.pt"), weights_only=False)
    x = b["x"].to(DT)
    g = b["gamma"].to(DT)
    be = b["beta"].to(DT) if b["beta"] is not None else None
    v1 = _normalize64(x, b["norm_type"], b["eps"])
    v3 = g * v1 + (be if be is not None else 0)
    return {"V1": v1, "V3": v3, "gamma": g, "beta": be, "seq": np.asarray(b["seq"]), "pos": np.asarray(b["pos"]),
            "raw": b}


def _normalize64(x, norm_type, eps):
    if norm_type == "LayerNorm":
        return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def weight_map(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(path, f) for k, f in json.load(open(idx))["weight_map"].items()}
    f = os.path.join(path, "model.safetensors")
    with safe_open(f, "pt") as sf:
        return {k: f for k in sf.keys()}


def load_head(slug):
    """W_U [V, d] from safetensors (lm_head, else tied wte), rounded to bf16 as in the GPU jobs, returned in fp64."""
    wm = weight_map(resolve_model(slug))
    key = next(k for k in ("lm_head.weight", "embed_out.weight", "wte.weight", "transformer.wte.weight",
                           "model.embed_tokens.weight") if k in wm)
    with safe_open(wm[key], "pt") as sf:
        return sf.get_tensor(key).to(torch.bfloat16).to(DT), key


def unigram_pi(slug, n_rows):
    g = json.load(open("results/round3/unigram_groups.json"))["model_group"][slug]
    c = np.load(f"cache/unigrams/{g}.npy").astype(np.float64)
    pi = np.zeros(n_rows)
    n = min(n_rows, len(c))
    pi[:n] = c[:n]
    return torch.as_tensor(pi / pi.sum(), dtype=DT), g


# ---------------------------------------------------------------- metrics (all differentiable where it matters)

def seq_index(seq):
    _, inv = np.unique(seq, return_inverse=True)
    return torch.as_tensor(inv)


def isotropy_bank(X, sidx):
    """1 - mean over sequences of the mean pairwise cosine among that sequence's bank tokens (needs >= 2 tokens)."""
    U = torch.nn.functional.normalize(X, dim=-1)
    ns = int(sidx.max()) + 1
    S = torch.zeros(ns, X.shape[1], dtype=X.dtype, device=X.device).index_add(0, sidx.to(X.device), U)
    n = torch.bincount(sidx, minlength=ns).to(X.device, X.dtype)
    ok = n >= 2
    s = (S[ok].pow(2).sum(-1) - n[ok]) / (n[ok] * (n[ok] - 1))
    return 1 - s.mean()


def cross_cos_exact(X, sidx):
    U = torch.nn.functional.normalize(X, dim=-1)
    ns = int(sidx.max()) + 1
    S = torch.zeros(ns, X.shape[1], dtype=X.dtype, device=X.device).index_add(0, sidx.to(X.device), U)
    n = torch.bincount(sidx, minlength=ns).to(X.dtype)
    N = X.shape[0]
    return ((U.sum(0).pow(2).sum() - S.pow(2).sum()) / (N * N - n.pow(2).sum())).item()


def spectrum64(Xc, d):
    """RankMe/d and top-1 variance share of a centered matrix, from the eigenvalues of Xc^T Xc (fp64)."""
    C = Xc.T @ Xc
    ev = torch.linalg.eigvalsh(C).clamp_min(0).flip(0)
    s = ev.sqrt()
    p = s / s.sum()
    p = p[p > 0]
    rankme = math.exp(-(p * p.log()).sum().item())
    return {"rankme_over_d": rankme / d, "top1": (ev[0] / ev.sum()).item()}


_GREF = {}


def gauss_ref(N, d):
    if (N, d) not in _GREF:
        g = torch.Generator().manual_seed(0)
        Z = torch.randn(N, d, generator=g, dtype=DT)
        _GREF[(N, d)] = spectrum64(Z - Z.mean(0), d)["rankme_over_d"]
    return _GREF[(N, d)]


def metrics(X, sidx, d=None):
    """A (bank isotropy), B_raw / B_c (exact cross-sequence mean cosine, raw / bank-mean centered), RankMe/d, top1."""
    d = d or X.shape[1]
    Xc = X - X.mean(0)
    sp = spectrum64(Xc, d)
    return {"A": isotropy_bank(X, sidx).item(), "B_raw": cross_cos_exact(X, sidx), "B_c": cross_cos_exact(Xc, sidx),
            "rankme_over_d": sp["rankme_over_d"], "rankme_over_d_norm": sp["rankme_over_d"] / gauss_ref(X.shape[0], d),
            "top1": sp["top1"]}


METRIC_KEYS = ["A", "B_raw", "B_c", "rankme_over_d", "top1"]


# ---------------------------------------------------------------- gauge transforms

def rng_for(family, kappa, seed):
    code = {"diag": 1, "orth": 2, "lin": 3}[family]
    return np.random.default_rng([6, code, int(kappa), int(seed)])


def haar(d, rng):
    Z = torch.as_tensor(rng.standard_normal((d, d)), dtype=DT)
    Q, R = torch.linalg.qr(Z)
    return Q * torch.sign(torch.diagonal(R))


def log_uniform_s(d, kappa, rng):
    return torch.as_tensor(np.exp(rng.uniform(-0.5 * math.log(kappa), 0.5 * math.log(kappa), d)), dtype=DT)


def sample_transform(family, d, kappa, seed):
    """Returns (apply(X) -> X A^T, A_inv_apply(W) -> W A^-1, description); kappa ignored for orth."""
    rng = rng_for(family, kappa if family != "orth" else 0, seed)
    if family == "diag":
        s = log_uniform_s(d, kappa, rng)
        return (lambda X: X * s), (lambda W: W / s), {"s": s}
    if family == "orth":
        Q = haar(d, rng)
        return (lambda X: X @ Q.T), (lambda W: W @ Q.T), {"Q": Q}  # Q^-1 = Q^T: W Q^-1 = W Q^T
    Q1, Q2 = haar(d, rng), haar(d, rng)
    s = log_uniform_s(d, kappa, rng)
    A = (Q1 * s) @ Q2  # Q1 diag(s) Q2
    Ainv = Q2.T @ (Q1.T / s[:, None])  # Q2^T diag(1/s) Q1^T
    return (lambda X: X @ A.T), (lambda W: W @ Ainv), {"A": A}


# ---------------------------------------------------------------- gauge metric

def gauge_root(W, b, weights=None):
    """R = Mt^(1/2), Mt = Wt^T P Wt, Wt = [W, b]; P = I - 11^T/V (weights None) or diag(pi) - pi pi^T."""
    V = W.shape[0]
    Wt = torch.cat([W, (b if b is not None else torch.zeros(V, dtype=DT))[:, None]], 1)
    if weights is None:
        c = Wt.sum(0)
        M = Wt.T @ Wt - torch.outer(c, c) / V
    else:
        c = weights @ Wt
        M = (Wt * weights[:, None]).T @ Wt - torch.outer(c, c)
    M = (M + M.T) / 2
    ev, Q = torch.linalg.eigh(M)
    neg = float((ev < 0).sum())
    R = (Q * ev.clamp_min(0).sqrt()) @ Q.T
    return R, {"n_neg_eig": neg, "min_eig": ev[0].item(), "max_eig": ev[-1].item()}


def hstar(H, R):
    """h* = R [h; 1] for rows of H (R symmetric)."""
    return H @ R[:-1] + R[-1]


def rel_diff(a, b):
    return abs(a - b) / max(abs(b), 1e-3)
