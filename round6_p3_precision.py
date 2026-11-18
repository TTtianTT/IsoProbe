"""Round 6 phase 3 U4: half precision and the loss of collapsed models (configs/round6_prereg.md, U4).

  scatter       : (CPU) bf16 - fp32 eval loss of the 1300 phase-2 T4 checkpoints vs V1 ||ubar||^2 and W_U s1^2/sum; P8-8.
  mech          : (GPU) pythia-70m step 143000 / 50000, eval loss with per-segment precision (body = embedding + layers,
                  final norm, head); activations are cast to the next segment's dtype; log-softmax in fp32.
  big --model M : (GPU) bf16 vs fp32 whole-model loss and demeaned dloss on the first >= 100k predicted tokens of
                  data/eval_loss_round6.jsonl; m = mean final-norm output over data/samples.jsonl 0-499 (max_len 256, pos >= 1).
Writes results/round6/p3/precision/{scatter.json, mech.json, big/<model>.json}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

OUT = "results/round6/p3/precision"
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def dump(obj, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(obj, open(fn + ".tmp", "w"), indent=1)
    os.replace(fn + ".tmp", fn)


def scatter():
    from round3_lib import spearman
    from round6_p2_summary import seed_series
    names = sorted(os.listdir("results/round6/p2/poly_eval_fp32"))
    pts = []
    for name in names:
        ser = seed_series(name)
        for st, v in ser.items():
            fb = f"results/round6/p2/poly_eval/{name}/step{st}.json"
            ff = f"results/round6/p2/poly_eval_fp32/{name}/step{st}.json"
            if os.path.exists(fb) and os.path.exists(ff):
                lb, lf = json.load(open(fb))["loss"], json.load(open(ff))["loss"]
                pts.append({"model": name, "step": st, "ubar": v["ub"], "s1": v["s1"], "loss_bf16": lb, "loss_fp32": lf, "diff": lb - lf})
    d = [p["diff"] for p in pts]
    r_u, r_s = spearman(d, [p["ubar"] for p in pts]), spearman(d, [p["s1"] for p in pts])
    by_size = {}
    for sz in ("70m", "160m"):
        q = [p for p in pts if p["model"].startswith(f"pythia-{sz}")]
        by_size[sz] = {"n": len(q), "rho_ubar": spearman([p["diff"] for p in q], [p["ubar"] for p in q])[0],
                       "rho_s1": spearman([p["diff"] for p in q], [p["s1"] for p in q])[0]}
    bins = {}
    for lo, hi in ((0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)):
        q = [p["diff"] for p in pts if lo <= p["ubar"] < hi]
        if q:
            bins[f"{lo}-{hi}"] = {"n": len(q), "median": float(np.median(q)), "max": float(np.max(q))}
    res = {"n": len(pts), "P8_8": {"rho_ubar": r_u[0], "pass": bool(r_u[0] > 0.8)}, "rho_s1": r_s[0], "by_size": by_size,
           "diff_by_ubar_bin": bins, "diff_min": min(d), "diff_max": max(d), "points": pts}
    dump(res, f"{OUT}/scatter.json")
    print(json.dumps({k: v for k, v in res.items() if k != "points"}, indent=1))


def eval_set(tok, max_len=512, min_tokens=0):
    rows = [json.loads(l) for l in open("data/eval_loss_round6.jsonl")]
    enc, n = [], 0
    for r in rows:
        e = tok(r["text"], add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
        enc.append(e)
        n += len(e) - 1
        if min_tokens and n >= min_tokens:
            break
    return enc


def batches(enc, bs, dev):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        T = max(len(enc[i]) for i in idx)
        ids = torch.zeros(len(idx), T, dtype=torch.long)
        am = torch.zeros(len(idx), T, dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(enc[i])] = torch.as_tensor(enc[i])
            am[j, :len(enc[i])] = 1
        yield ids.to(dev), am.to(dev)


@torch.no_grad()
def mech():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from download_round4 import ckpt_dir
    dev = "cuda:0"
    grid = [("fp32", "fp32", "fp32"), ("bf16", "bf16", "bf16"), ("fp32", "bf16", "bf16"), ("bf16", "fp32", "bf16"),
            ("bf16", "bf16", "fp32"), ("bf16", "fp32", "fp32"), ("fp16", "fp16", "fp16"), ("fp16", "fp32", "fp32")]
    fn = f"{OUT}/mech.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for rev in ("step143000", "step50000"):
        path = ckpt_dir("EleutherAI/pythia-70m", rev)
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        enc = eval_set(tok)
        for body, norm, head in grid:
            key = f"{rev}/{body}-{norm}-{head}"
            if key in res:
                continue
            t0 = time.time()
            m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map=dev, local_files_only=True).eval()
            gx = m.gpt_neox
            gx.embed_in.to(DT[body])
            gx.layers.to(DT[body])
            gx.final_layer_norm.to(DT[norm])
            m.embed_out.to(DT[head])
            gx.final_layer_norm.register_forward_pre_hook(lambda mod, a, _d=DT[norm]: (a[0].to(_d),) + tuple(a[1:]))
            m.embed_out.register_forward_pre_hook(lambda mod, a, _d=DT[head]: (a[0].to(_d),) + tuple(a[1:]))
            tot, n = 0.0, 0
            for ids, am in batches(enc, 32, dev):
                lp = torch.log_softmax(m(input_ids=ids, attention_mask=am).logits.float(), -1)[:, :-1]
                nll = -lp.gather(2, ids[:, 1:, None])[..., 0]
                valid = am[:, 1:].bool()
                tot += (nll * valid).sum().double().item()
                n += int(valid.sum())
            res[key] = {"body": body, "final_norm": norm, "head": head, "loss": tot / n, "n_tokens": n, "elapsed_s": time.time() - t0}
            print(f"[70m {rev}] body {body} norm {norm} head {head}: loss {tot / n:.4f}", flush=True)
            dump(res, fn)
            del m
            torch.cuda.empty_cache()


@torch.no_grad()
def big(model_arg, min_tokens=100_000, bs=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from measure import load_samples, resolve_model
    dev = "cuda:0"
    path = resolve_model(model_arg)
    slug = os.path.basename(path.rstrip("/"))
    fn = f"{OUT}/big/{slug}.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {"model": slug}
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    enc = eval_set(tok, min_tokens=min_tokens)
    est = load_samples("data/samples.jsonl", 0)[:500]
    for prec in ("fp32", "bf16"):
        if prec in res:
            continue
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(path, dtype=DT[prec], device_map=dev, local_files_only=True).eval()
        base, head = m.base_model, m.get_output_embeddings()
        msum, mn = None, 0
        for r in est:
            ids = tok(r["text"], add_special_tokens=True, truncation=True, max_length=256, return_tensors="pt")["input_ids"].to(dev)
            h = base(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, 1:].double()
            msum = h.sum(0) if msum is None else msum + h.sum(0)
            mn += h.shape[0]
        mvec = (msum / mn).to(DT[prec])
        tot, tot_dm, n = 0.0, 0.0, 0
        for ids, am in batches(enc, bs, dev):
            h = base(input_ids=ids, attention_mask=am).last_hidden_state
            y = ids[:, 1:, None]
            valid = am[:, 1:].bool()
            for hh, acc in ((h, "base"), (h - mvec, "dm")):
                lp = torch.log_softmax(head(hh).float(), -1)[:, :-1]
                s = ((-lp.gather(2, y)[..., 0]) * valid).sum().double().item()
                if acc == "base":
                    tot += s
                else:
                    tot_dm += s
                del lp
            n += int(valid.sum())
        res[prec] = {"loss": tot / n, "loss_demeaned": tot_dm / n, "dloss_demeaned": (tot_dm - tot) / n, "n_tokens": n,
                     "n_paragraphs": len(enc), "m_norm": mvec.float().norm().item(), "elapsed_s": time.time() - t0}
        print(f"[{slug} {prec}] loss {tot / n:.4f} demeaned dloss {(tot_dm - tot) / n:+.4f} (n={n}, {time.time() - t0:.0f}s)", flush=True)
        dump(res, fn)
        del m, base, head
        torch.cuda.empty_cache()
    if "fp32" in res and "bf16" in res:
        res["loss_bf16_minus_fp32"] = res["bf16"]["loss"] - res["fp32"]["loss"]
        res["dloss_bf16_minus_fp32"] = res["bf16"]["dloss_demeaned"] - res["fp32"]["dloss_demeaned"]
        dump(res, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scatter", "mech", "big"])
    ap.add_argument("--model")
    args = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.mode == "scatter":
        scatter()
    elif args.mode == "mech":
        mech()
    else:
        big(args.model)


if __name__ == "__main__":
    main()
