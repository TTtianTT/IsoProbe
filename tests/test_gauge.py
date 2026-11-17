"""Round 6: the gauge metric h* = Mt^(1/2)[h; 1] is invariant under output-preserving reparametrizations."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from round6_lib import (DT, METRIC_KEYS, gauge_root, hstar, metrics, rel_diff,  # noqa: E402
                        sample_transform, seq_index)


def _setup(seed, V=60, d=10, N=200):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(V, d, generator=g, dtype=DT)
    b = torch.randn(V, generator=g, dtype=DT)
    H = torch.randn(N, d, generator=g, dtype=DT) + 2 * torch.randn(d, generator=g, dtype=DT)
    sidx = seq_index(np.repeat(np.arange(N // 10), 10))
    pi = torch.rand(V, generator=g, dtype=DT)
    return W, b, H, sidx, pi / pi.sum()


def _centered_logits(W, b, H, P):
    return (H @ W.T + b) @ P


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hstar_gram_equals_centered_logits(seed, weights):
    W, b, H, _, pi = _setup(seed)
    w = pi if weights else None
    R, _ = gauge_root(W, b, w)
    Z = hstar(H, R)
    V = W.shape[0]
    P = torch.diag(pi) - torch.outer(pi, pi) if weights else torch.eye(V, dtype=DT) - 1.0 / V
    L = H @ W.T + b
    assert torch.allclose(Z @ Z.T, L @ P @ L.T, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize("weights", [False, True])
@pytest.mark.parametrize("family", ["diag", "orth", "lin"])
def test_hstar_metrics_invariant_under_gauge(family, weights):
    W, b, H, sidx, pi = _setup(3)
    w = pi if weights else None
    m0 = metrics(hstar(H, gauge_root(W, b, w)[0]), sidx, H.shape[1])
    for s in range(3):
        fx, fw, _ = sample_transform(family, H.shape[1], 100, s)
        m1 = metrics(hstar(fx(H), gauge_root(fw(W), b, w)[0]), sidx, H.shape[1])
        assert max(rel_diff(m1[k], m0[k]) for k in METRIC_KEYS) < 1e-6


def test_hstar_invariant_under_translation_and_beta_folding():
    W, b, H, sidx, _ = _setup(4)
    g = torch.Generator().manual_seed(9)
    gamma = torch.rand(H.shape[1], generator=g, dtype=DT) * 5 + 0.1
    beta = torch.randn(H.shape[1], generator=g, dtype=DT)
    V3 = gamma * H + beta  # original head (W, b) on V3  ==  folded head (W diag g, b + W beta) on H
    m3 = metrics(hstar(V3, gauge_root(W, b)[0]), sidx, H.shape[1])
    m1 = metrics(hstar(H, gauge_root(W * gamma, b + W @ beta)[0]), sidx, H.shape[1])
    assert max(rel_diff(m1[k], m3[k]) for k in METRIC_KEYS) < 1e-6


def test_raw_metrics_orth_invariant_but_not_diag():
    _, _, H, sidx, _ = _setup(5)
    m0 = metrics(H, sidx)
    fx, _, _ = sample_transform("orth", H.shape[1], 0, 0)
    m1 = metrics(fx(H), sidx)
    assert max(abs(m1[k] - m0[k]) for k in METRIC_KEYS) < 1e-10
    fx, _, _ = sample_transform("diag", H.shape[1], 100, 0)
    assert abs(metrics(fx(H), sidx)["A"] - m0["A"]) > 1e-3
