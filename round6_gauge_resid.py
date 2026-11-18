"""Round 6 phase 3 U1: numerical check of the residual-stream gauge group (configs/round6_prereg.md, U1; docs/theory_round6.md).

fp64 throughout (CPU). For each case the model weights are rewritten so that the residual stream becomes x' = A x
(column convention) at every layer, and we compare the logits and per-layer residual-stream metrics before / after.

  pre-norm (gpt2 LayerNorm, Qwen2.5 RMSNorm): fold every norm's gain (and LN bias) into the linear maps that read it,
      then reads W -> W R^T with R = P A P (P = centering for LN, I for RMSNorm; R is orthogonal on range(P) for A in
      the group), writes (token / position embeddings, attention-output and MLP-output weights and biases) -> A W,
      head -> W_U diag(gamma_f) R^T with bias W_U beta_f.
  post-sublayer norm (OLMo-2: x + gamma * n(f(x)), linear reads): reads W -> W A^{-1}; o_proj / down_proj -> A W;
      post-norm gains gamma -> |A| gamma (A a signed permutation); final norm folded into the head, head -> W_U diag(g_f) A^T.
      The Haar-Q control applies the same recipe with gamma unchanged (not in the group).

Metrics (each layer position l = 0..L: block inputs and the final-norm input), sequences 0..N-1 of data/samples.jsonl,
max_len 256, position 0 dropped: A = 1 - mean within-sequence pairwise cosine; centred B = mean cross-sequence cosine
after subtracting the mean over all tokens; RankMe of the centred matrix. On n(x) (= V1 at the last position) and on raw x.
Writes results/round6/p3/gauge_resid/<case>.json.
"""
import argparse
import json
import math
import os
import time
import types

import torch

from measure import load_samples

torch.set_default_dtype(torch.float64)
DT = torch.float64
OUT = "results/round6/p3/gauge_resid"
MODELS = {"gpt2": "/dataset1/zailong/models/gpt2", "qwen2.5-7b": "/dataset1/zailong/models/Qwen2.5-7B-Instruct",
          "olmo2-1b": "/dataset1/zailong/models/ckpts/OLMo-2-0425-1B/main"}


# ------------------------------------------------------------------ transforms

def haar(d, seed):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(d, d, generator=g, dtype=DT)
    Q, R = torch.linalg.qr(Z)
    return Q * torch.sign(torch.diagonal(R))[None, :]


def haar_keep_ones(d, seed):
    """Orthogonal Q with Q 1 = 1: Haar on the complement of 1, identity on 1."""
    one = torch.ones(d, dtype=DT) / math.sqrt(d)
    B = torch.linalg.qr(torch.cat([one[:, None], torch.eye(d, dtype=DT)[:, :d - 1]], 1))[0]  # B[:,0] = +-one
    B = B * torch.sign(B[:, :1].sum())  # make the first column +one
    Qs = haar(d - 1, seed)
    M = torch.eye(d, dtype=DT)
    M[1:, 1:] = Qs
    return B @ M @ B.T


def shear(d, seed, c=3.0):
    """A x = x + (w.x) 1/sqrt(d) + (c - 1)(1.x / d) 1 with w orthogonal to 1: A 1 = c 1 and P A P = P."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(d, generator=g, dtype=DT)
    w = w - w.mean()
    one = torch.ones(d, dtype=DT)
    return torch.eye(d, dtype=DT) + torch.outer(one, w) / math.sqrt(d) + (c - 1) * torch.outer(one, one) / d


def signed_perm(d, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(d, generator=g)
    s = torch.where(torch.rand(d, generator=g) < 0.5, -1.0, 1.0).to(DT)
    A = torch.zeros(d, d, dtype=DT)
    A[torch.arange(d), p] = s
    return A


# ------------------------------------------------------------------ models

def rms_forward64(self, x):
    eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps")
    return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


class Runner:
    """Holds an fp64 HF model, its folded head (W, b) and hooks on the final-norm input."""

    def __init__(self, key):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.key = key
        path = MODELS[key]
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float64, local_files_only=True,
                                                          attn_implementation="sdpa").eval()
        for m in self.model.modules():  # HF RMSNorm upcasts to fp32 internally; keep everything in fp64
            if type(m).__name__.endswith("RMSNorm"):
                m.forward = types.MethodType(rms_forward64, m)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.arch = {"gpt2": "ln", "qwen2": "rms", "olmo2": "post"}[cfg.model_type]
        self.d = cfg.hidden_size if hasattr(cfg, "hidden_size") else cfg.n_embd
        base = self.model.base_model
        self.final_norm = base.ln_f if self.arch == "ln" else base.norm
        self.cap = {}
        self.final_norm.register_forward_hook(lambda m, i, o: self.cap.update(x=i[0]))
        # fold the final norm into an untied head copy
        W = self.model.get_output_embeddings().weight.detach().clone()
        g = self.final_norm.weight.detach().clone()
        bta = self.final_norm.bias.detach().clone() if getattr(self.final_norm, "bias", None) is not None else None
        self.W = W * g[None, :]
        self.b = W @ bta if bta is not None else torch.zeros(W.shape[0], dtype=DT)
        self.eps = getattr(self.final_norm, "eps", None) or getattr(self.final_norm, "variance_epsilon")
        self.fold_pre_norms()

    def fold_pre_norms(self):
        """pre-norm models: fold every block-norm gain (and LN bias) into the reads; set the norm to identity gain."""
        base = self.model.base_model
        if self.arch == "ln":
            for blk in base.h:
                for ln, lin in ((blk.ln_1, blk.attn.c_attn), (blk.ln_2, blk.mlp.c_fc)):  # Conv1D: y = x W + b, W [in, out]
                    g, bt = ln.weight.detach().clone(), ln.bias.detach().clone()
                    lin.bias.copy_(bt @ lin.weight + lin.bias)
                    lin.weight.copy_(g[:, None] * lin.weight)
                    ln.weight.fill_(1.0)
                    ln.bias.zero_()
        elif self.arch == "rms":
            for blk in base.layers:
                for ln, lins in ((blk.input_layernorm, (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj)),
                                 (blk.post_attention_layernorm, (blk.mlp.gate_proj, blk.mlp.up_proj))):
                    g = ln.weight.detach().clone()
                    for lin in lins:  # Linear: y = W x + b, W [out, in]
                        lin.weight.copy_(lin.weight * g[None, :])
                    ln.weight.fill_(1.0)
        self.final_norm.weight.fill_(1.0)
        if getattr(self.final_norm, "bias", None) is not None:
            self.final_norm.bias.zero_()
        # untie the input embedding from the head (the head is used only through self.W, self.b)
        emb = self.model.get_input_embeddings()
        emb.weight = torch.nn.Parameter(emb.weight.detach().clone(), requires_grad=False)

    def transform(self, A, post_mode="perm"):
        base = self.model.base_model
        d = self.d
        P = torch.eye(d, dtype=DT) - torch.ones(d, d, dtype=DT) / d if self.arch == "ln" else torch.eye(d, dtype=DT)
        emb = self.model.get_input_embeddings()
        emb.weight.copy_(emb.weight @ A.T)
        if self.arch == "ln":
            R = P @ A @ P
            base.wpe.weight.copy_(base.wpe.weight @ A.T)
            for blk in base.h:
                for lin in (blk.attn.c_attn, blk.mlp.c_fc):  # read: W_col -> W_col R^T  <=>  W -> R W
                    lin.weight.copy_(R @ lin.weight)
                for lin in (blk.attn.c_proj, blk.mlp.c_proj):  # write: W_col -> A W_col  <=>  W -> W A^T
                    lin.weight.copy_(lin.weight @ A.T)
                    lin.bias.copy_(lin.bias @ A.T)
            self.W = self.W @ R.T
        elif self.arch == "rms":
            for blk in base.layers:
                for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.mlp.gate_proj, blk.mlp.up_proj):
                    lin.weight.copy_(lin.weight @ A.T)
                for lin in (blk.self_attn.o_proj, blk.mlp.down_proj):
                    lin.weight.copy_(A @ lin.weight)
            self.W = self.W @ A.T
        else:  # post-sublayer norm (OLMo-2)
            Ainv = torch.linalg.inv(A)
            absA = A.abs()
            for blk in base.layers:
                for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.mlp.gate_proj, blk.mlp.up_proj):
                    lin.weight.copy_(lin.weight @ Ainv)
                for lin in (blk.self_attn.o_proj, blk.mlp.down_proj):
                    lin.weight.copy_(A @ lin.weight)
                if post_mode == "perm":
                    for ln in (blk.post_attention_layernorm, blk.post_feedforward_layernorm):
                        ln.weight.copy_(absA @ ln.weight)
            self.W = self.W @ A.T  # final norm already folded; n(A x) = A n(x) for orthogonal A

    @torch.no_grad()
    def run(self, seqs):
        """logits (folded head) and residual streams [L+1] for each sequence."""
        logits, resid = [], []
        for ids in seqs:
            out = self.model.base_model(input_ids=ids[None], output_hidden_states=True)
            assert len(out.hidden_states) == self.model.config.num_hidden_layers + 1
            hs = [h[0] for h in out.hidden_states[:-1]] + [self.cap["x"][0]]
            if self.arch == "ln":
                xL = self.cap["x"][0]
                xc = xL - xL.mean(-1, keepdim=True)
                v1 = xc / torch.sqrt(xc.pow(2).mean(-1, keepdim=True) + self.eps)
            else:
                xL = self.cap["x"][0]
                v1 = xL / torch.sqrt(xL.pow(2).mean(-1, keepdim=True) + self.eps)
            logits.append(v1 @ self.W.T + self.b)
            resid.append(torch.stack(hs))
        return logits, resid


# ------------------------------------------------------------------ metrics

def metrics(Xs, center_tokens):
    """Xs: list of [T_s, d] (position 0 already dropped). center_tokens: LN-style per-token centring + normalisation."""
    if center_tokens:
        Xs = [x - x.mean(-1, keepdim=True) for x in Xs]
    U = [torch.nn.functional.normalize(x, dim=-1) for x in Xs]
    within = []
    for u in U:
        n = len(u)
        s = u.sum(0)
        within.append(((s @ s) - n) / (n * (n - 1)))
    A = 1 - torch.stack(within).mean().item()
    X = torch.cat(Xs)
    mu = X.mean(0)
    Uc = [torch.nn.functional.normalize(x - mu, dim=-1) for x in Xs]
    tot = torch.cat(Uc).sum(0)
    ns = torch.tensor([len(u) for u in Uc], dtype=DT)
    same = sum((u.sum(0) @ u.sum(0)) for u in Uc)
    N = ns.sum()
    Bc = ((tot @ tot - same) / (N * N - (ns * ns).sum())).item()
    s = torch.linalg.svdvals(X - mu)
    p = s / s.sum()
    p = p[p > 0]
    rankme = math.exp(-(p * p.log()).sum().item())
    return {"A": A, "Bc": Bc, "RankMe": rankme}


def all_metrics(resid, center):
    L1 = resid[0].shape[0]
    return [metrics([r[l, 1:] for r in resid], center) for l in range(L1)]


# ------------------------------------------------------------------ main

CASES = {  # case -> (model, transform kind, seed, expected invariant)
    "gpt2_q0": ("gpt2", "keep1", 0, True), "gpt2_q1": ("gpt2", "keep1", 1, True), "gpt2_q2": ("gpt2", "keep1", 2, True),
    "gpt2_shear": ("gpt2", "shear", 0, True), "gpt2_haar_ctrl": ("gpt2", "haar", 0, False),
    "qwen2.5-7b_q0": ("qwen2.5-7b", "haar", 0, True),
    "olmo2-1b_perm0": ("olmo2-1b", "perm", 0, True), "olmo2-1b_perm1": ("olmo2-1b", "perm", 1, True),
    "olmo2-1b_perm2": ("olmo2-1b", "perm", 2, True), "olmo2-1b_haar_ctrl": ("olmo2-1b", "haar_post", 0, False),
}
NSEQ = {"gpt2": 200, "qwen2.5-7b": 50, "olmo2-1b": 200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--nseq", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    cases = [c for c, v in CASES.items() if v[0] == args.model]
    nseq = args.nseq or NSEQ[args.model]
    t0 = time.time()
    R0 = Runner(args.model)
    rows = load_samples("data/samples.jsonl", 0)[:nseq]
    seqs = [R0.tok(r["text"], add_special_tokens=True, truncation=True, max_length=256, return_tensors="pt")["input_ids"][0]
            for r in rows]
    seqs = [s for s in seqs if len(s) >= 3]
    lg0, rs0 = R0.run(seqs)
    is_ln = R0.arch == "ln"
    m0_n = all_metrics(rs0, True) if is_ln else None
    m0_raw = all_metrics(rs0, False)
    state0 = {k: v.clone() for k, v in R0.model.state_dict().items()} if len(cases) > 1 else None
    W0, b0 = R0.W.clone(), R0.b.clone()
    print(f"[{args.model}] base run: {len(seqs)} seqs, {sum(len(s) for s in seqs)} tokens, {time.time() - t0:.0f}s", flush=True)
    for case in cases:
        fn = os.path.join(args.out, f"{case}.json")
        if os.path.exists(fn):
            continue
        _, kind, seed, expect = CASES[case]
        if state0 is not None:
            R0.model.load_state_dict(state0)
        R0.W, R0.b = W0.clone(), b0.clone()
        d = R0.d
        A = {"keep1": lambda: haar_keep_ones(d, seed), "shear": lambda: shear(d, seed), "haar": lambda: haar(d, seed),
             "perm": lambda: signed_perm(d, seed), "haar_post": lambda: haar(d, seed)}[kind]()
        R0.transform(A, post_mode="perm" if kind == "perm" else "none")
        lg1, rs1 = R0.run(seqs)
        rel = max(((a - b).abs().max() / b.abs().max()).item() for a, b in zip(lg1, lg0))
        argmax = sum(int((a.argmax(-1) == b.argmax(-1)).sum()) for a, b in zip(lg1, lg0)) / sum(len(b) for b in lg0)
        res = {"case": case, "model": args.model, "arch": R0.arch, "transform": kind, "seed": seed, "expected_invariant": expect,
               "dtype": "fp64", "n_seq": len(seqs), "n_tokens": sum(len(s) for s in seqs), "d": d,
               "A_orth_err": (A.T @ A - torch.eye(d, dtype=DT)).abs().max().item(),
               "A_ones": ((A @ torch.ones(d, dtype=DT)) / math.sqrt(d)).tolist()[:1] if is_ln else None,
               "logits_max_rel_err": rel, "argmax_agree": argmax}
        m1_raw = all_metrics(rs1, False)
        res["metrics_raw_before"], res["metrics_raw_after"] = m0_raw, m1_raw
        res["max_abs_diff_raw"] = {k: max(abs(a[k] - b[k]) for a, b in zip(m0_raw, m1_raw)) for k in ("A", "Bc", "RankMe")}
        if is_ln:
            m1_n = all_metrics(rs1, True)
            res["metrics_n_before"], res["metrics_n_after"] = m0_n, m1_n
            res["max_abs_diff_n"] = {k: max(abs(a[k] - b[k]) for a, b in zip(m0_n, m1_n)) for k in ("A", "Bc", "RankMe")}
        key = "max_abs_diff_n" if is_ln else "max_abs_diff_raw"
        res["judged_on"] = "n(x)" if is_ln else "raw x"
        res["P8_3_pass"] = bool(rel < 1e-6) if expect else None
        res["P8_4_pass"] = bool(max(res[key].values()) < 1e-6) if expect else None
        res["elapsed_s"] = time.time() - t0
        os.makedirs(args.out, exist_ok=True)
        json.dump(res, open(fn, "w"), indent=1)
        print(f"[{case}] logits rel {rel:.2e} argmax {argmax:.5f} {key} {res[key]}"
              + (f" raw {res['max_abs_diff_raw']}" if is_ln else "") + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
