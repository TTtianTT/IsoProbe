"""Shared, device-agnostic helpers for round 3 (used by extract_round3.py on GPU and analyze_round3.py on CPU).

Representations at the final norm, from its input x [N, d]:
  V0 = x
  V1 = x / rms(x)  (RMSNorm)  or  (x - mean_x) / std_x  (LayerNorm)
  V2 = gamma * V1
  V3 = V2 + beta  (= hidden_states[-1]; V3 = V2 for RMSNorm)
"""
import math

import numpy as np
import torch

from measure import sample_cross_pairs, self_similarity, spectrum

VNAMES = ["V0", "V1", "V2", "V3"]
ROUND2_KEY = {"V0": "pre", "V1": "xhat", "V2": "no_bias", "V3": "post"}  # names in results/diag/*.json


def normalize(x, norm_type, eps):
    x = x.float()
    if norm_type == "LayerNorm":
        return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def variants(x, gamma, beta, norm_type, eps):
    v1 = normalize(x, norm_type, eps)
    v2 = gamma * v1
    return {"V0": x.float(), "V1": v1, "V2": v2, "V3": v2 + beta if beta is not None else v2}


# ---------------------------------------------------------------- step 2: exact per-dimension decomposition

def mean_unit(H):
    """ubar = mean of the L2-normalized rows of H [N, d]."""
    return torch.nn.functional.normalize(H.float(), dim=-1).mean(0)


def decompose(V1, gamma=None, beta=None):
    """Exact: H = gamma*V1 + beta; cross-sequence cosine ~= ||ubar||^2, dim k contributes ubar_k^2 / ||ubar||^2."""
    H = V1 if gamma is None else gamma * V1
    if beta is not None:
        H = H + beta
    ub = mean_unit(H)
    tot = ub.pow(2).sum()
    return tot.item(), (ub.pow(2) / tot)


def cross_cos_exact(H, seq):
    """Mean cosine over all pairs from different sequences (no sampling), via per-sequence sums."""
    U = torch.nn.functional.normalize(H.float(), dim=-1)
    seq = torch.as_tensor(seq, device=U.device)
    n = U.shape[0]
    S = torch.zeros(int(seq.max()) + 1, U.shape[1], device=U.device, dtype=U.dtype).index_add_(0, seq, U)
    counts = torch.bincount(seq).double()
    tot = U.sum(0).double().pow(2).sum()
    same = S.double().pow(2).sum()  # includes the diagonal i == j
    return ((tot - same) / (n * n - counts.pow(2).sum())).item()


def approx_cos(V1, gamma=None, beta=None):
    """cos ~= sum m_k^2 / sum (m_k^2 + s_k^2), m_k = gamma_k mu_k + beta_k, s_k = |gamma_k| sigma_k."""
    mu, sd = V1.float().mean(0), V1.float().std(0, unbiased=False)
    g = torch.ones_like(mu) if gamma is None else gamma.float()
    m = g * mu + (beta.float() if beta is not None else 0)
    s = g.abs() * sd
    return (m.pow(2).sum() / (m.pow(2) + s.pow(2)).sum()).item()


# ---------------------------------------------------------------- step 1: metrics per representation

def robust_mean(X, pos, thr=10.0):
    """Mean over tokens that are not position 0 and whose norm is <= thr x median norm."""
    nrm = X.norm(dim=-1)
    keep = (torch.as_tensor(pos, device=X.device) != 0) & (nrm <= thr * nrm.median())
    return X[keep].mean(0), int(keep.sum())


_GAUSS = {}


def gauss_rankme_over_d(N, d, device, seed=0):
    """RankMe/d of a centered isotropic Gaussian N x d matrix (normalizer for RankMe/d)."""
    if (N, d) not in _GAUSS:
        g = torch.Generator(device=device).manual_seed(seed)
        Z = torch.randn(N, d, generator=g, device=device)
        _GAUSS[(N, d)] = spectrum(Z - Z.mean(0))["rankme_over_d"]
    return _GAUSS[(N, d)]


def bank_metrics_r3(X, seq, pos, n_pairs, seed):
    """Raw B (sampled pairs), ||ubar||^2, exact cross cos, robust-centered B, RankMe/d (raw and Gaussian-normalized)."""
    X = X.float()
    i, j = sample_cross_pairs(seq, n_pairs, np.random.default_rng(seed))
    i, j = torch.as_tensor(i, device=X.device), torch.as_tensor(j, device=X.device)
    U = torch.nn.functional.normalize(X, dim=-1)
    mu, n_used = robust_mean(X, pos)
    Xc = X - mu
    Uc = torch.nn.functional.normalize(Xc, dim=-1)
    sp = spectrum(Xc)
    gref = gauss_rankme_over_d(X.shape[0], X.shape[1], X.device, seed)
    return {
        "n_tokens": X.shape[0],
        "cross_cos": (U[i] * U[j]).sum(-1).mean().item(),
        "ubar_sq": U.mean(0).pow(2).sum().item(),
        "cross_cos_exact": cross_cos_exact(X, seq),
        "cross_cos_robust_centered": (Uc[i] * Uc[j]).sum(-1).mean().item(),
        "robust_mean_n_tokens": n_used,
        "rankme_over_d": sp["rankme_over_d"],
        "gauss_rankme_over_d": gref,
        "rankme_over_d_norm": sp["rankme_over_d"] / gref,
        "top1_var": sp["top1_var"],
    }


def bank_self_sim(X, seq):
    """Isotropy estimated only from bank tokens (<= 10 per sequence): 1 - mean over sequences of s(x)."""
    seq = np.asarray(seq)
    vals = [self_similarity(X[torch.as_tensor(np.flatnonzero(seq == s), device=X.device)])
            for s in np.unique(seq) if (seq == s).sum() >= 2]
    return 1 - float(np.mean(vals))


# ---------------------------------------------------------------- misc

def spearman(a, b):
    """Spearman rho (Pearson on average ranks) and a two-sided normal-approximation p-value."""
    import pandas as pd
    ra, rb = pd.Series(np.asarray(a)).rank().to_numpy(), pd.Series(np.asarray(b)).rank().to_numpy()
    rho = float(np.corrcoef(ra, rb)[0, 1])
    n = len(ra)
    if n < 4 or abs(rho) >= 1:
        return rho, 0.0 if abs(rho) >= 1 else float("nan"), n
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    return rho, math.erfc(abs(t) / math.sqrt(2)), n
