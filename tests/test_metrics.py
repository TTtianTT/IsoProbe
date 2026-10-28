import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measure import (sample_cross_pairs, select_layers, self_similarity,  # noqa: E402
                     self_similarity_bruteforce, spectrum)


@pytest.mark.parametrize("T,d,shift", [(2, 8, 0.0), (17, 64, 0.0), (256, 768, 0.0), (128, 512, 3.0)])
def test_self_similarity_matches_bruteforce(T, d, shift):
    g = torch.Generator().manual_seed(T * d)
    h = torch.randn(T, d, generator=g) * torch.rand(T, 1, generator=g) * 5 + shift
    assert self_similarity(h) == pytest.approx(self_similarity_bruteforce(h), abs=1e-5)


def test_self_similarity_extremes():
    v = torch.randn(1, 32)
    assert self_similarity(v.repeat(50, 1) * torch.rand(50, 1) + 0) == pytest.approx(1.0, abs=1e-5)
    # random high-dim Gaussian vectors are near-orthogonal
    assert abs(self_similarity(torch.randn(200, 4096))) < 0.01


def test_self_similarity_bf16_input():
    h = torch.randn(64, 256) + 1.0
    assert self_similarity(h.bfloat16()) == pytest.approx(self_similarity_bruteforce(h.bfloat16()), abs=1e-5)


def test_cross_pairs_never_same_sequence():
    seq = np.repeat(np.arange(100), 10)
    i, j = sample_cross_pairs(seq, 20000, np.random.default_rng(0))
    assert len(i) == len(j) == 20000
    assert (seq[i] != seq[j]).all()


def test_spectrum_rank_and_variance():
    g = torch.Generator().manual_seed(0)
    # rank-3 data with variances 9:4:1 along orthogonal axes
    Z = torch.randn(5000, 3, generator=g) * torch.tensor([3.0, 2.0, 1.0])
    Q, _ = torch.linalg.qr(torch.randn(64, 3, generator=g))
    X = Z @ Q.T
    s = spectrum(X - X.mean(0))
    assert 2.5 < s["rankme"] < 3.0
    assert s["top10_var"] == pytest.approx(1.0, abs=1e-6)
    assert s["top1_var"] == pytest.approx(9 / 14, abs=0.03)
    # isotropic Gaussian: RankMe close to d
    X = torch.randn(20000, 64, generator=g)
    assert spectrum(X - X.mean(0))["rankme_over_d"] > 0.95


def test_select_layers():
    assert select_layers(12) == [0, 2, 3, 4, 6, 8, 9, 10, 12]
    assert select_layers(64) == list(range(0, 65, 8))
    for L in (12, 28, 32, 36, 40, 64):
        ls = select_layers(L)
        assert ls[0] == 0 and ls[-1] == L and len(ls) == 9
