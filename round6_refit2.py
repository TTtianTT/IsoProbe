"""Round 6 phase 2 T5: output-head refit, full version (body frozen; train final-norm gamma, beta and W_U in fp32).

Arms (configs/round6_p2_refit.tsv: model, rev, arm, seed):
  random : gamma = 1, beta = 0, W_U ~ N(0, 0.02^2) (seeded); data order = chunk permutation with the same seed
  own    : start from the checkpoint's own gamma, beta, W_U
  trunc  : own head with sigma_1 of W diag(gamma) cut to sigma_2 (W' = W - (s1 - s2) u1 (v1 / gamma)^T)
Optimizer: AdamW lr 1e-3, wd 0, betas (0.9, 0.999), linear warmup 500 steps then cosine to 0 at 10000 steps; batch 16 x 512
wikitext-103 train tokens (cache/round6/p2/train_chunks_full.pt). Eval every 1000 steps on data/eval_loss_round6.jsonl.
Extension: if eval loss fell by > 0.005 over the last 1000 steps, run 10000 more steps with a cosine from 1e-4 to 0 (no warmup).
Saves head + optimizer + step every 1000 steps under <--cache>/<model>/<rev>/<arm>_s<seed>/ (resumable).
At the end: spectrum of W diag(gamma) (first 50277 rows): s1^2/sum, s1/s2, |cos(ubar_V1, v1)| (ubar from the checkpoint bank).
Writes results/round6/p2/refit/<model>/<rev>/<arm>_s<seed>.json.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

import round6_causal as rc
from round3_lib import normalize

FULL = "cache/round6/p2/train_chunks_full.pt"
N_VOC = 50277


def prepare_full():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rc.TOK_PATH, local_files_only=True)
    files = [hf_hub_download("Salesforce/wikitext", f, repo_type="dataset", local_files_only=True) for f in rc.TRAIN_FILES]
    lines = [t for t in load_dataset("parquet", data_files=files, split="train")["text"] if t.strip()]
    parts = []
    for i in range(0, len(lines), 5000):
        ids = tok(lines[i:i + 5000], add_special_tokens=False)["input_ids"]
        parts.append(np.concatenate([np.asarray(x, dtype=np.int32) for x in ids if x]))
    ids = np.concatenate(parts)
    nc = len(ids) // rc.CHUNK
    chunks = torch.as_tensor(ids[:nc * rc.CHUNK].reshape(nc, rc.CHUNK))
    os.makedirs(os.path.dirname(FULL), exist_ok=True)
    torch.save({"chunks": chunks, "n_lines": len(lines), "n_tokens": int(len(ids))}, FULL)
    print(f"{len(lines)} lines, {len(ids)} tokens -> {nc} chunks of {rc.CHUNK}", flush=True)


def lr_at(step, total=10000, peak=1e-3, ext=False):
    warm = min(500, total // 20)  # 500 at the preregistered total of 10000
    if ext:  # extension phase: cosine from 1e-4 to 0 over another `total` (10000) steps
        return 1e-4 * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))
    if step < warm:
        return peak * (step + 1) / warm
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))


def head_spec(W, g, u):
    Wf = (W * g)[:N_VOC].double()
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    return {"s1sq_over_sum": (S[0] ** 2 / (S ** 2).sum()).item(), "s1_over_s2": (S[0] / S[1]).item(),
            "abs_cos_ubar_v1": abs(float(Vh[0].float() @ u))}


def run(model_name, rev, arm, seed, args):
    t0 = time.time()
    tag = f"{arm}_s{seed}"
    fn = os.path.join(args.out, model_name, rev, f"{tag}.json")
    cdir = os.path.join(args.cache, model_name, rev, tag)
    os.makedirs(cdir, exist_ok=True)
    C = rc.Ckpt(model_name, rev)
    X, Y, FL, _ = C.eval_cache(args.max_tokens)
    b = torch.load(next(f for f in (f"cache/round4_banks/{model_name}/{rev}.pt", f"cache/round5_banks/{model_name}/{rev}.pt")
                        if os.path.exists(f)), weights_only=False)
    V1b = normalize(b["x"].to(rc.DEV).float(), b["norm_type"], b["eps"])
    u = torch.nn.functional.normalize(V1b, dim=-1).mean(0)
    u = u / u.norm()
    chunks = torch.load(FULL, weights_only=False)["chunks"]
    perm = torch.as_tensor(np.random.default_rng(seed).permutation(len(chunks)))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    beta0 = C.beta if C.beta is not None else torch.zeros_like(C.gamma)
    if arm == "random":
        g = torch.ones_like(C.gamma)
        bt = torch.zeros_like(C.gamma)
        W = (torch.randn(C.W.shape, generator=gen) * 0.02).to(rc.DEV)
    else:
        g, bt, W = C.gamma.clone(), beta0.clone(), C.W.clone()
        if arm == "trunc":
            U, S, Vh = torch.linalg.svd((C.W * C.gamma).double(), full_matrices=False)
            W = (C.W.double() - (S[0] - S[1]) * torch.outer(U[:, 0], Vh[0] / C.gamma.double())).float()
    g, bt, W = (t.clone().requires_grad_(True) for t in (g, bt, W))
    opt = torch.optim.AdamW([g, bt, W], lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999))
    res = {"model": model_name, "revision": rev, "arm": arm, "seed": seed, "evals": {}, "train_trace": [],
           "spec_start": head_spec(W.detach(), g.detach(), u), "extended": False, "complete": False}
    T, E = args.total, args.total // 10  # T = 10000 steps, eval every E = 1000 (smoke tests only change T)
    step, total = 0, T
    saves = sorted(glob.glob(os.path.join(cdir, "step*.pt")), key=lambda f: int(f.split("step")[-1][:-3]))
    if saves:
        s = torch.load(saves[-1], weights_only=False, map_location=rc.DEV)
        with torch.no_grad():
            g.copy_(s["g"]); bt.copy_(s["bt"]); W.copy_(s["W"])
        opt.load_state_dict(s["opt"])
        res, step = s["res"], s["step"]
        total = 2 * T if res["extended"] else T
        print(f"[{model_name}@{rev} {tag}] resumed at step {step}", flush=True)

    @torch.no_grad()
    def evaluate():
        tot, n = 0.0, 0
        for c in range(0, len(Y), 4096):
            V1 = normalize(X[c:c + 4096].float(), C.ntype, C.eps)
            L = (V1 * g + bt) @ W.T
            y = Y[c:c + 4096]
            tot += (torch.logsumexp(L, -1) - L.gather(1, y[:, None])[:, 0]).double().sum().item()
            n += len(y)
        return tot / n

    if step == 0:
        res["evals"]["0"] = evaluate()
    while step < total:
        ext = step >= T
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step - T if ext else step, total=T, ext=ext)
        idx = perm[(step * 16) % len(chunks): (step * 16) % len(chunks) + 16]
        ids = chunks[idx].long().to(rc.DEV)
        with torch.no_grad():
            V1 = normalize(C.x_of(ids).float(), C.ntype, C.eps)[:, :-1]
        L = (V1 * g + bt) @ W.T
        loss = torch.nn.functional.cross_entropy(L.reshape(-1, L.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            res["train_trace"].append((step, loss.item()))
        if step % E == 0:
            res["evals"][str(step)] = evaluate()
            if step == T and not res["extended"]:
                drop = res["evals"][str(T - E)] - res["evals"][str(T)]
                res["drop_last_1000"] = drop
                if drop > 0.005:
                    res["extended"], total = True, 2 * T
            if step == total:
                res["drop_last_1000_final"] = res["evals"][str(step - E)] - res["evals"][str(step)]
                res["spec_end"] = head_spec(W.detach(), g.detach(), u)
                res["complete"] = True
            res["elapsed_s"] = res.get("elapsed_s", 0) + time.time() - t0
            t0 = time.time()
            torch.save({"g": g.detach(), "bt": bt.detach(), "W": W.detach(), "opt": opt.state_dict(), "step": step,
                        "res": res}, os.path.join(cdir, f"step{step}.pt"))
            for f in glob.glob(os.path.join(cdir, "step*.pt")):  # keep only the latest save
                if int(f.split("step")[-1][:-3]) < step:
                    os.remove(f)
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            json.dump(res, open(fn, "w"), indent=1)
            print(f"[{model_name}@{rev} {tag}] step {step}: train {loss.item():.4f} eval {res['evals'][str(step)]:.4f} "
                  f"({res['elapsed_s']:.0f}s)", flush=True)
    print(f"[{model_name}@{rev} {tag}] done: {res.get('spec_end')}", flush=True)
    del C
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "run"])
    ap.add_argument("--list", default="configs/round6_p2_refit.tsv")
    ap.add_argument("--rows", default="0")
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--out", default="results/round6/p2/refit/")
    ap.add_argument("--cache", default="cache/round6/p2/refit/")
    ap.add_argument("--total", type=int, default=10000, help="steps of the main phase; change only for smoke tests")
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare_full()
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
        m, rev, arm, seed = rows[i]
        fn = os.path.join(args.out, m, rev, f"{arm}_s{seed}.json")
        if os.path.exists(fn) and json.load(open(fn)).get("complete"):
            print(f"{fn} complete, skipping", flush=True)
            continue
        run(m, rev, arm, int(seed), args)


if __name__ == "__main__":
    main()
