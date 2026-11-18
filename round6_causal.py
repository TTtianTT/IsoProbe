"""Round 6 S4 (ablate) and S5 (refit) on local Pythia checkpoints (configs/round6_causal.tsv); see configs/round6_prereg.md.

All interventions act at the final norm: V1 = normalize(x), logits = W_f V1 + b', W_f = W_U diag(gamma), b' = W_U beta.
  ablate (S4): 4b mean / fix / project along ubar(V1) and 5 random unit directions; 4c truncate sigma_1 -> sigma_2 of
               W_f, then fit a logit bias (init b0 = mean compensation, Adam lr 1e-2, 200 steps of 8 x 512 train tokens);
               control: the same bias fit on the untruncated head (init 0).
  refit (S5):  freeze the body, train gamma, beta, W_U (fp32) with AdamW lr 1e-4, wd 0, 1000 steps of 16 x 512 tokens,
               saving head + optimizer every 250 steps (cache/round6/refit/, resumable) and evaluating the loss.
Loss: data/eval_loss_round6.jsonl (also reported without the in_samples paragraphs).
Train data: wikitext-103 train, Pythia tokens in 512-token chunks shuffled with seed 0 (refit: chunks [0, 16000),
            ablate bias fit: chunks [16000, 17600)), cached in cache/round6/train_chunks_pythia.pt.
Writes results/round6/causal/{ablate,refit}/<model>/<rev>.json.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from download_round4 import ckpt_dir
from extract_round3 import norm_params
from measure import final_norm_module
from round3_lib import normalize

EVAL = "data/eval_loss_round6.jsonl"
TRAIN_CACHE = "cache/round6/train_chunks_pythia.pt"
TRAIN_FILES = [f"wikitext-103-raw-v1/train-0000{i}-of-00002.parquet" for i in range(2)]
TOK_PATH = ckpt_dir("EleutherAI/pythia-70m", "step143000")
CHUNK, N_CHUNKS = 512, 18000
REFIT_OFF, BIAS_OFF = 0, 16000
DEV = "cuda:0"
# body precision: bf16 (default, all phase-1 and phase-2 results) or fp32 (phase 2 precision check, ISO_BODY_DTYPE=fp32)
BODY_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32}[os.environ.get("ISO_BODY_DTYPE", "bf16")]


def prepare_train(max_chars=60_000_000):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, local_files_only=True)
    files = [hf_hub_download("Salesforce/wikitext", f, repo_type="dataset", local_files_only=True) for f in TRAIN_FILES]
    lines, n = [], 0
    for t in load_dataset("parquet", data_files=files, split="train")["text"]:
        if t.strip():
            lines.append(t)
            n += len(t)
            if n >= max_chars:
                break
    ids = []
    for i in range(0, len(lines), 2000):
        for x in tok(lines[i:i + 2000], add_special_tokens=False)["input_ids"]:
            ids.extend(x)
    nc = len(ids) // CHUNK
    chunks = torch.as_tensor(np.asarray(ids[:nc * CHUNK], dtype=np.int64).reshape(nc, CHUNK))
    perm = torch.as_tensor(np.random.default_rng(0).permutation(nc))
    chunks = chunks[perm][:N_CHUNKS].to(torch.int32)
    os.makedirs(os.path.dirname(TRAIN_CACHE), exist_ok=True)
    torch.save({"chunks": chunks, "n_chars": n, "n_lines": len(lines), "n_tokens": len(ids)}, TRAIN_CACHE)
    print(f"{len(lines)} lines, {n} chars, {len(ids)} tokens -> {nc} chunks, kept {len(chunks)}", flush=True)


class Ckpt:
    """Pythia checkpoint on GPU with a hook on the final norm input."""

    def __init__(self, model_name, rev):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.path = ckpt_dir(f"EleutherAI/{model_name}", rev)
        self.tok = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=BODY_DTYPE, device_map=DEV,
                                                          local_files_only=True).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.norm = final_norm_module(self.model)
        self.gamma, self.beta, self.eps, self.ntype = norm_params(self.norm)
        head = self.model.get_output_embeddings()
        assert getattr(head, "bias", None) is None
        self.W = head.weight.detach().float()
        self.cap = {}
        self.norm.register_forward_hook(lambda m, inp, out: self.cap.update(x=inp[0]))

    @torch.no_grad()
    def x_of(self, ids):
        self.model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids))
        return self.cap["x"]

    @torch.no_grad()
    def eval_cache(self, max_tokens=0):
        """x (bf16) and targets for every eval position; checks the fp32 recomputed logits on the first 5 sequences."""
        rows = [json.loads(l) for l in open(EVAL)]
        xs, ys, fl, ps = [], [], [], []
        chk = 0.0
        n = 0
        for i, r in enumerate(rows):
            ids = self.tok(r["text"], add_special_tokens=True, truncation=True, max_length=CHUNK,
                           return_tensors="pt")["input_ids"].to(DEV)
            if i < 5:
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids))
                x = self.cap["x"]
                L = self.logits(normalize(x[0].float(), self.ntype, self.eps), self.W * self.gamma,
                                self.W @ self.beta if self.beta is not None else None)
                ref = out.logits[0].float()
                chk = max(chk, ((L - ref).abs().max() / ref.abs().max()).item())
            else:
                x = self.x_of(ids)
            xs.append(x[0, :-1].to(BODY_DTYPE))
            ys.append(ids[0, 1:])
            fl.append(torch.full((ids.shape[1] - 1,), bool(r["in_samples"]), device=DEV))
            ps.append(torch.arange(ids.shape[1] - 1, device=DEV))
            n += ids.shape[1] - 1
            if max_tokens and n >= max_tokens:
                break
        self.pos = torch.cat(ps)  # position of the predicting token (0 = first token of the paragraph)
        return torch.cat(xs), torch.cat(ys), torch.cat(fl), chk

    @staticmethod
    def logits(V1, Wf, bf):
        L = V1 @ Wf.T
        return L + bf if bf is not None else L


def train_batch(chunks, i, bs, off):
    return chunks[off + i * bs: off + (i + 1) * bs].long().to(DEV)


# ---------------------------------------------------------------- S4

@torch.no_grad()
def ablate(model_name, rev, args):
    t0 = time.time()
    C = Ckpt(model_name, rev)
    X, Y, FL, chk = C.eval_cache(args.max_tokens)
    Wf = C.W * C.gamma
    bf = C.W @ C.beta if C.beta is not None else None
    bank_dirs = os.environ.get("ISO_BANK_DIRS", "cache/round4_banks,cache/round5_banks").split(",")  # phase 3 U0: fp32 banks
    bank_fn = next(f for f in (f"{bd}/{model_name}/{rev}.pt" for bd in bank_dirs) if os.path.exists(f))
    b = torch.load(bank_fn, weights_only=False)
    V1b = normalize(b["x"].to(DEV).float(), b["norm_type"], b["eps"])
    assert (b["gamma"].to(DEV) - C.gamma).abs().max().item() == 0
    d = V1b.shape[1]
    ubar = torch.nn.functional.normalize(V1b, dim=-1).mean(0)
    dirs = {"ubar": ubar / ubar.norm()}
    for k in range(5):
        g = torch.Generator().manual_seed(k)
        r = torch.randn(d, generator=g).to(DEV)
        dirs[f"rand{k}"] = r / r.norm()
    cvals = {k: (V1b @ u).mean().item() for k, u in dirs.items()}
    # 4c: SVD of the folded head
    U, S, Vh = torch.linalg.svd(Wf.double(), full_matrices=False)
    s1, s2 = S[0].item(), S[1].item()
    u1, v1 = U[:, 0].float(), Vh[0].float()
    w_tr = -(s1 - s2) * u1  # truncated logits: L + (v1 . V1) * w_tr

    # bias fits on train chunks
    chunks = torch.load(TRAIN_CACHE, weights_only=False)["chunks"]
    proj_sum, proj_n = 0.0, 0
    for i in range(20):
        V1 = normalize(C.x_of(train_batch(chunks, i, 8, BIAS_OFF)).float(), C.ntype, C.eps)
        proj_sum += (V1 @ v1).sum().item()
        proj_n += V1.shape[0] * V1.shape[1]
    b0 = -(proj_sum / proj_n) * w_tr  # = (s1 - s2) E[v1 . V1] u1
    fits = {}
    with torch.enable_grad():
        for tag, trunc, init in (("trunc", True, b0), ("control", False, torch.zeros_like(b0))):
            bias = init.clone().requires_grad_(True)
            opt = torch.optim.Adam([bias], lr=1e-2)
            trace = []
            for i in range(args.bias_steps):
                ids = train_batch(chunks, i, 8, BIAS_OFF)
                with torch.no_grad():
                    V1 = normalize(C.x_of(ids).float(), C.ntype, C.eps)[:, :-1].reshape(-1, d)
                    L = C.logits(V1, Wf, bf)
                    if trunc:
                        L = L + (V1 @ v1)[:, None] * w_tr
                y = ids[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(L + bias, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if i % 20 == 0 or i == args.bias_steps - 1:
                    trace.append((i, loss.item()))
            fits[tag] = {"bias": bias.detach(), "trace": trace}

    # one pass over the evaluation set for every variant: logits L + alpha_t * w
    names = ["base"]
    for k in dirs:
        names += [f"{k}/mean", f"{k}/fix", f"{k}/proj"]
    names += ["trunc", "trunc+b0", "trunc+bias", "control_bias"]
    wd = {k: Wf @ u for k, u in dirs.items()}
    tot = torch.zeros(len(names), dtype=torch.float64, device=DEV)
    tot_out = torch.zeros(len(names), dtype=torch.float64, device=DEV)
    n_all, n_out = 0, 0
    buckets = {"pos0": (0, 1), "pos1-15": (1, 16), "pos16+": (16, 10 ** 9)}  # exploratory breakdown
    tot_b = {k: torch.zeros(len(names), dtype=torch.float64, device=DEV) for k in buckets}
    n_b = {k: 0 for k in buckets}
    for c in range(0, len(Y), 4096):
        V1 = normalize(X[c:c + 4096].float(), C.ntype, C.eps)
        y = Y[c:c + 4096]
        keep = ~FL[c:c + 4096]
        L = C.logits(V1, Wf, bf)
        Ly = L.gather(1, y[:, None])[:, 0]
        out = []

        def nll(alpha=None, w=None, bias=None):
            Lv, Lvy = L, Ly
            if w is not None:
                Lv = Lv + alpha[:, None] * w
                Lvy = Lvy + alpha * w[y]
            if bias is not None:
                Lv = Lv + bias
                Lvy = Lvy + bias[y]
            return torch.logsumexp(Lv, -1) - Lvy

        out.append(nll())
        for k, u in dirs.items():
            p = V1 @ u
            cc = cvals[k]
            out.append(nll(torch.full_like(p, -cc), wd[k]))
            out.append(nll(cc - p, wd[k]))
            out.append(nll(-p, wd[k]))
        a_tr = V1 @ v1
        out.append(nll(a_tr, w_tr))
        out.append(nll(a_tr, w_tr, b0))
        out.append(nll(a_tr, w_tr, fits["trunc"]["bias"]))
        out.append(nll(bias=fits["control"]["bias"]))
        O = torch.stack(out).double()
        tot += O.sum(1)
        tot_out += O[:, keep].sum(1)
        pc = C.pos[c:c + 4096]
        for kb, (lo, hi) in buckets.items():
            mb = (pc >= lo) & (pc < hi)
            tot_b[kb] += O[:, mb].sum(1)
            n_b[kb] += int(mb.sum())
        n_all += len(y)
        n_out += int(keep.sum())
    loss = {n: (tot[i] / n_all).item() for i, n in enumerate(names)}
    loss_out = {n: (tot_out[i] / n_out).item() for i, n in enumerate(names)}
    dl = {n: loss[n] - loss["base"] for n in names}
    dl_out = {n: loss_out[n] - loss_out["base"] for n in names}
    res = {"model": model_name, "revision": rev, "bank": bank_fn, "n_eval_tokens": n_all, "n_eval_tokens_out": n_out,
           "check_logits_max_rel": chk, "loss": loss, "dloss": dl, "loss_out": loss_out, "dloss_out": dl_out,
           "c": cvals, "ubar_sq_V1_bank": torch.nn.functional.normalize(V1b, dim=-1).mean(0).pow(2).sum().item(),
           "sd_proj_ubar_bank": (V1b @ dirs["ubar"]).std().item(),
           "svd": {"s1": s1, "s2": s2, "s1sq_over_sum": (S[0] ** 2 / (S ** 2).sum()).item(),
                   "abs_cos_v1_ubar": (v1 @ dirs["ubar"]).abs().item(), "E_train_v1_proj": proj_sum / proj_n},
           "bias_fit": {k: {"trace": v["trace"], "bias_norm": v["bias"].norm().item()} for k, v in fits.items()},
           "trunc_bias_minus_control": loss["trunc+bias"] - loss["control_bias"],
           "by_pos": {kb: {"n": n_b[kb], "dloss": {n: ((tot_b[kb][i] - tot_b[kb][0]) / max(n_b[kb], 1)).item()
                                                     for i, n in enumerate(names)},
                           "sum_share_of_total_dloss": {n: ((tot_b[kb][i] - tot_b[kb][0]) / (tot[i] - tot[0])).item()
                                                        if n != "base" and (tot[i] - tot[0]).abs() > 0 else None
                                                        for i, n in enumerate(names)}} for kb in buckets},
           "elapsed_s": time.time() - t0}
    print(f"[{model_name}@{rev}] base={loss['base']:.4f} (n={n_all}, chk {chk:.1e}) ubar: mean {dl['ubar/mean']:+.4f} "
          f"fix {dl['ubar/fix']:+.4f} proj {dl['ubar/proj']:+.4f}; rand0 mean {dl['rand0/mean']:+.4f} fix "
          f"{dl['rand0/fix']:+.4f}; trunc {dl['trunc']:+.4f} +b0 {dl['trunc+b0']:+.4f} +bias {dl['trunc+bias']:+.4f} "
          f"control {dl['control_bias']:+.4f} ({res['elapsed_s']:.0f}s)", flush=True)
    del C
    torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- S5

def refit(model_name, rev, args, fn):
    t0 = time.time()
    C = Ckpt(model_name, rev)
    X, Y, FL, chk = C.eval_cache(args.max_tokens)
    chunks = torch.load(TRAIN_CACHE, weights_only=False)["chunks"]
    cdir = os.path.join(args.refit_cache, model_name, rev)
    os.makedirs(cdir, exist_ok=True)
    g = C.gamma.clone().requires_grad_(True)
    bt = (C.beta if C.beta is not None else torch.zeros_like(C.gamma)).clone().requires_grad_(True)
    Wp = C.W.clone().requires_grad_(True)
    opt = torch.optim.AdamW([g, bt, Wp], lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999))
    res = {"model": model_name, "revision": rev, "check_logits_max_rel": chk, "evals": {}, "train_trace": [],
           "complete": False}
    start = 0
    saves = sorted(glob.glob(os.path.join(cdir, "step*.pt")), key=lambda f: int(f.split("step")[-1][:-3]))
    if saves:
        s = torch.load(saves[-1], weights_only=False)
        with torch.no_grad():
            g.copy_(s["gamma"].to(DEV))
            bt.copy_(s["beta"].to(DEV))
            Wp.copy_(s["W"].to(DEV))
        opt.load_state_dict(s["opt"])
        res = s["res"]
        start = s["step"]
        print(f"[{model_name}@{rev}] resumed from step {start}", flush=True)

    @torch.no_grad()
    def evaluate():
        tot, tot_out, n, n_out = 0.0, 0.0, 0, 0
        for c in range(0, len(Y), 4096):
            V1 = normalize(X[c:c + 4096].float(), C.ntype, C.eps)
            L = (V1 * g + bt) @ Wp.T
            y = Y[c:c + 4096]
            l = torch.logsumexp(L, -1) - L.gather(1, y[:, None])[:, 0]
            keep = ~FL[c:c + 4096]
            tot += l.double().sum().item()
            tot_out += l[keep].double().sum().item()
            n += len(y)
            n_out += int(keep.sum())
        return {"loss": tot / n, "loss_out": tot_out / n_out, "n": n, "n_out": n_out}

    if start == 0:
        res["evals"]["0"] = evaluate()
        print(f"[{model_name}@{rev}] step 0 eval {res['evals']['0']['loss']:.4f}", flush=True)
    for step in range(start, args.refit_steps):
        ids = train_batch(chunks, step, 16, REFIT_OFF)
        with torch.no_grad():
            V1 = normalize(C.x_of(ids).float(), C.ntype, C.eps)[:, :-1]
        L = (V1 * g + bt) @ Wp.T
        loss = torch.nn.functional.cross_entropy(L.reshape(-1, L.shape[-1]), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            res["train_trace"].append((step, loss.item()))
        if (step + 1) % 250 == 0:
            res["evals"][str(step + 1)] = evaluate()
            res["elapsed_s"] = res.get("elapsed_s", 0) + time.time() - t0
            t0 = time.time()
            res["complete"] = step + 1 >= args.refit_steps
            torch.save({"gamma": g.detach().cpu(), "beta": bt.detach().cpu(), "W": Wp.detach().cpu(),
                        "opt": opt.state_dict(), "step": step + 1, "res": res},
                       os.path.join(cdir, f"step{step + 1}.pt"))
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            json.dump(res, open(fn, "w"), indent=1)
            print(f"[{model_name}@{rev}] step {step + 1}: train {loss.item():.4f} eval "
                  f"{res['evals'][str(step + 1)]['loss']:.4f} ({res['elapsed_s']:.0f}s)", flush=True)
    del C
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "ablate", "refit"])
    ap.add_argument("--list", default="configs/round6_causal.tsv")
    ap.add_argument("--rows", default="0", help="row indices, e.g. 0-4 or 3,7")
    ap.add_argument("--max_tokens", type=int, default=0, help="truncate the eval set (0 = all)")
    ap.add_argument("--bias_steps", type=int, default=200)
    ap.add_argument("--refit_steps", type=int, default=1000)
    ap.add_argument("--out", default="results/round6/causal/")
    ap.add_argument("--refit_cache", default="cache/round6/refit/",
                    help="where refit saves/resumes head + optimizer (use a separate dir for smoke tests)")
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare_train()
        return
    torch.manual_seed(0)
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    idx = []
    for part in args.rows.split(","):
        a, _, b = part.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    for i in idx:
        model_name, rev = rows[i]
        fn = os.path.join(args.out, args.mode, model_name, f"{rev}.json")
        if os.path.exists(fn) and (args.mode == "ablate" or json.load(open(fn)).get("complete")):
            print(f"{fn} exists, skipping", flush=True)
            continue
        if args.mode == "ablate":
            res = ablate(model_name, rev, args)
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            json.dump(res, open(fn, "w"), indent=1)
        else:
            refit(model_name, rev, args, fn)


if __name__ == "__main__":
    main()
