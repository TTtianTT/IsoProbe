"""Round 6 S6 (GPU): PolyPythias final checkpoints (configs/round6_polypythia.tsv, pytorch_model.bin only).

For each row: all dyn_round4.measure metrics (same data, bank positions and splits as rounds 3-5; bank to
cache/round6/poly_banks/), plus the round-5 W_U spectrum and direction alignment (round5_cpu.spec_stats / align on the
first 50277 head rows, bf16 head as loaded). Collapse (descriptive only): V1 ||ubar||^2 >= 0.9 and W_U s1^2/sum >= 0.5.
Writes results/round6/poly/<model>-seed<k>/<rev>.json.

Phase 2 (T4): --list configs/round6_poly_ckpts.tsv; --eval adds the next-token loss on data/eval_loss_round6.jsonl
(batched forward, right padding, fp32 log-softmax of the model's own logits; paragraphs truncated to 512 tokens) written
to results/round6/p2/poly_eval/<model>/<rev>.json; --eval_only skips the dyn / spectrum part (original-seed windows).
Rows whose checkpoint is not downloaded yet are skipped, so the same command can be re-run as downloads finish.
"""
import argparse
import json
import math
import os
from types import SimpleNamespace

import torch

from download_round4 import ckpt_dir
from dyn_round4 import measure
from round3_lib import variants
from round5_cpu import N_VOC, align, spec_stats


def head_from_bin(path):
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    key = next(k for k in ("embed_out.weight", "lm_head.weight") if k in sd)
    return sd[key].to(torch.bfloat16).float()


@torch.no_grad()
def eval_loss(path, bs=32, max_len=512, dtype="bf16"):
    """dtype bf16: the first T4 path (bf16 logits; inflated by up to ~1.8 nats on collapsed checkpoints, superseded).
    dtype fp32: whole model in fp32 (phase 2 T4 amendment). Device from ISO_DEV (default cuda:0)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("ISO_DEV", "cuda:0")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype],
                                                 device_map=dev, local_files_only=True).eval()
    rows = [json.loads(l) for l in open("data/eval_loss_round6.jsonl")]
    enc = [tok(r["text"], add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"] for r in rows]
    flag = [bool(r["in_samples"]) for r in rows]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    tot, n, tot_out, n_out = 0.0, 0, 0.0, 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        T = max(len(enc[i]) for i in idx)
        ids = torch.zeros(len(idx), T, dtype=torch.long)
        am = torch.zeros(len(idx), T, dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(enc[i])] = torch.as_tensor(enc[i])
            am[j, :len(enc[i])] = 1
        ids, am = ids.to(dev), am.to(dev)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=am).logits.float(), -1)[:, :-1]
        nll = -lp.gather(2, ids[:, 1:, None])[..., 0]
        valid = am[:, 1:].bool()
        per = (nll * valid).sum(1).double()
        cnt = valid.sum(1)
        out = torch.as_tensor([not flag[i] for i in idx], device=dev)
        tot += per.sum().item()
        n += int(cnt.sum())
        tot_out += per[out].sum().item()
        n_out += int(cnt[out].sum())
    del model
    torch.cuda.empty_cache()
    return {"loss": tot / n, "n_tokens": n, "loss_out": tot_out / n_out, "n_tokens_out": n_out, "batch": bs, "dtype": dtype}


def dump_atomic(obj, fn):
    tmp = fn + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round6_polypythia.tsv")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="results/round6/poly/")
    ap.add_argument("--eval", action="store_true", help="also compute the eval_loss_round6 loss (phase 2 T4)")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_out", default="results/round6/p2/poly_eval/")
    ap.add_argument("--cache", default="cache/round6/poly_banks/")
    ap.add_argument("--eval_dtype", default="bf16", choices=["bf16", "fp32"], help="fp32 with --eval_out results/round6/p2/poly_eval_fp32/")
    args = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    idx = []
    for part in args.rows.split(","):
        a, _, b = part.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    for i in idx:
        repo, rev = rows[i]
        name = repo.split("/")[1]
        fn = os.path.join(args.out, name, f"{rev}.json")
        if not os.path.exists(os.path.join(ckpt_dir(repo, rev), ".complete")):
            print(f"{repo}@{rev} not downloaded, skipping", flush=True)
            continue
        efn = os.path.join(args.eval_out, name, f"{rev}.json")
        if (args.eval or args.eval_only) and not os.path.exists(efn):
            e = eval_loss(ckpt_dir(repo, rev), dtype=args.eval_dtype)
            os.makedirs(os.path.dirname(efn), exist_ok=True)
            dump_atomic({"model": name, "revision": rev, **e}, efn)
            print(f"[{name}@{rev}] eval loss {e['loss']:.4f} ({e['n_tokens']} tokens)", flush=True)
        if args.eval_only or os.path.exists(fn):
            continue
        torch.manual_seed(0)
        dargs = SimpleNamespace(data="data/samples.jsonl", max_len=256, bank_per_seq=10, n_pairs=20000, n_fold=20,
                                n_est=500, seed=0, cache=args.cache)
        res = measure(repo, rev, dargs)
        b = torch.load(os.path.join(dargs.cache, name, f"{rev}.pt"), weights_only=False)
        W = head_from_bin(ckpt_dir(repo, rev))[:N_VOC]
        st, Vh = spec_stats(W)
        stf, Vhf = spec_stats(W * b["gamma"].float())
        V = variants(b["x"], b["gamma"], b["beta"], b["norm_type"], b["eps"])
        u3 = torch.nn.functional.normalize(V["V3"], dim=-1).mean(0)
        u1 = torch.nn.functional.normalize(V["V1"], dim=-1).mean(0)
        res["spec"] = {"W_U": st, "W_U_folded": stf, "align_V3_WU": align(u3, Vh), "align_V1_WUgamma": align(u1, Vhf),
                       "random_abs_cos": math.sqrt(2 / (math.pi * W.shape[1]))}
        ub1 = res["step1"]["V1"]["all"]["ubar_sq"]
        res["collapsed"] = bool(ub1 >= 0.9 and st["s1sq_over_sum"] >= 0.5)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        dump_atomic(res, fn)
        print(f"[{name}@{rev}] V1 ubar^2={ub1:.3f} A V1={res['step1']['V1']['all']['isotropy_full']:.3f} "
              f"V3={res['step1']['V3']['all']['isotropy_full']:.3f} s1^2/sum={st['s1sq_over_sum']:.3f} "
              f"cos1={res['spec']['align_V1_WUgamma']['abs_cos_v1']:.3f} loss={res['function']['loss_base']:.3f} "
              f"collapsed={res['collapsed']} ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
