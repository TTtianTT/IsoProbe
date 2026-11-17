"""Round 6 phase 2 T6 (exploratory pilot): does cheap continued training of pythia-70m push or reverse the collapse?

Full-model fine-tuning on wikitext-103 train (cache/round6/p2/train_chunks_full.pt, seed-0 permutation), batch 16 x 512,
AdamW lr 1e-4 (constant, wd 0), 3000 steps, bf16 weights are upcast to fp32 for training.
Arms (configs/round6_p2_ft.tsv: rev, arm, lambda):
  plain   : next-token loss only
  rpen    : + lambda * || mean_t normalize(x_t) ||^2, x = final-norm input, mean over all positions of the batch
  ctrlpen : + lambda * (r . mean_t normalize(x_t))^2 for a fixed random unit direction r (seed 0)
Every 500 steps: V1 ||ubar||^2 on the round-3 bank positions (data/samples.jsonl, 10 positions per sequence, same RNG as
rounds 3-5), W_U s1^2/sum (first 50277 rows), and the loss on data/eval_loss_round6.jsonl (batched, as round6_poly.eval_loss).
Weights are not saved (only metrics). Writes results/round6/p2/ft/<rev>_<arm>_l<lambda>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from download_round4 import ckpt_dir
from measure import final_norm_module, load_samples
from round3_lib import normalize

FULL = "cache/round6/p2/train_chunks_full.pt"
DEV = os.environ.get("ISO_DEV", "cuda:0")  # ISO_DEV=cpu only for smoke tests
N_VOC = 50277


@torch.no_grad()
def measure(model, tok, norm, cap, eval_enc, eval_flag):
    model.eval()
    rows = load_samples("data/samples.jsonl", 0)
    rng = np.random.default_rng(0)
    eps = getattr(norm, "eps", 1e-5)
    xs = []
    for r in rows:
        ids = tok(r["text"], add_special_tokens=True, truncation=True, max_length=256, return_tensors="pt")["input_ids"].to(DEV)
        model(input_ids=ids)
        T = ids.shape[1]
        pos = rng.choice(T, size=min(10, T), replace=False)
        xs.append(cap["x"][0, torch.as_tensor(pos, device=DEV)].float())
    V1 = normalize(torch.cat(xs), "LayerNorm", eps)
    ub = torch.nn.functional.normalize(V1, dim=-1).mean(0).pow(2).sum().item()
    W = model.get_output_embeddings().weight.float()[:N_VOC]
    S = torch.linalg.svdvals(W.double())
    tot, n = 0.0, 0
    order = sorted(range(len(eval_enc)), key=lambda i: len(eval_enc[i]))
    for s in range(0, len(order), 32):
        idx = order[s:s + 32]
        T = max(len(eval_enc[i]) for i in idx)
        ids = torch.zeros(len(idx), T, dtype=torch.long)
        am = torch.zeros(len(idx), T, dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(eval_enc[i])] = torch.as_tensor(eval_enc[i])
            am[j, :len(eval_enc[i])] = 1
        ids, am = ids.to(DEV), am.to(DEV)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=am).logits.float(), -1)[:, :-1]
        nll = -lp.gather(2, ids[:, 1:, None])[..., 0]
        v = am[:, 1:].bool()
        tot += (nll * v).sum().item()
        n += int(v.sum())
    model.train()
    return {"ubar_sq_V1": ub, "s1sq_over_sum": (S[0] ** 2 / (S ** 2).sum()).item(), "eval_loss": tot / n}


def run(rev, arm, lam, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    fn = os.path.join(args.out, f"{rev}_{arm}_l{lam:g}.json")
    if os.path.exists(fn):
        print(f"{fn} exists", flush=True)
        return
    t0 = time.time()
    path = ckpt_dir("EleutherAI/pythia-70m", rev)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map=DEV, local_files_only=True)
    model.train()
    for mod in model.modules():  # deterministic forward: no dropout (Pythia has none, but be explicit)
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    norm = final_norm_module(model)
    cap = {}
    norm.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0]))
    eval_rows = [json.loads(l) for l in open("data/eval_loss_round6.jsonl")]
    eval_enc = [tok(r["text"], add_special_tokens=True, truncation=True, max_length=512)["input_ids"] for r in eval_rows]
    chunks = torch.load(FULL, weights_only=False)["chunks"]
    perm = torch.as_tensor(np.random.default_rng(0).permutation(len(chunks)))
    r = torch.randn(norm.weight.shape[0], generator=torch.Generator().manual_seed(0)).to(DEV)
    r = r / r.norm()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    res = {"revision": rev, "arm": arm, "lambda": lam, "metrics": {"0": measure(model, tok, norm, cap, eval_enc, None)},
           "trace": []}
    print(f"[{rev} {arm} l={lam:g}] step 0 {res['metrics']['0']}", flush=True)
    for step in range(1, args.steps + 1):
        ids = chunks[perm[(step - 1) * 16: step * 16]].long().to(DEV)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        m = torch.nn.functional.normalize(cap["x"].float(), dim=-1).reshape(-1, r.shape[0]).mean(0)
        pen = m.pow(2).sum() if arm == "rpen" else ((m @ r) ** 2 if arm == "ctrlpen" else torch.zeros((), device=DEV))
        total = loss + lam * pen
        opt.zero_grad()
        total.backward()
        opt.step()
        if step % 50 == 0:
            res["trace"].append((step, loss.item(), pen.item()))
        if step % args.every == 0:
            res["metrics"][str(step)] = measure(model, tok, norm, cap, eval_enc, None)
            print(f"[{rev} {arm} l={lam:g}] step {step} {res['metrics'][str(step)]} pen {pen.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    res["elapsed_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    json.dump(res, open(fn, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round6_p2_ft.tsv")
    ap.add_argument("--rows", default="0")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=500, help="measurement interval; change only for smoke tests")
    ap.add_argument("--out", default="results/round6/p2/ft/")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    idx = []
    for part in args.rows.split(","):
        a, _, b = part.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    for i in idx:
        rev, arm, lam = rows[i]
        run(rev, arm, float(lam), args)


if __name__ == "__main__":
    main()
