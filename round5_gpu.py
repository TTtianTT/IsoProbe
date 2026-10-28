"""Round 5 items 2-3 (GPU). Rows of configs/round5_gpu.tsv: kind <TAB> repo-or-model <TAB> revision.

  new : a checkpoint not measured in round 4 -> run dyn_round4.measure (all round-4 metrics; JSON to
        results/round5/dyn/<model>/<rev>.json, bank to cache/round5_banks/), then the logit metrics below
  old : a round-4 Pythia checkpoint -> logit metrics only (bank from cache/round4_banks/)
  r3  : a round-3 model (models.txt name) -> logit metrics only (bank from cache/banks/)

Logit metrics (one forward pass over the evaluation sequences 500-999, exactly as the round-3/4 delta-loss):
  loss_base; for each saved delta set: 1/2 E_t Var_{p_t}(delta), the first-order term E_t[delta_y - E_{p_t} delta],
  and the delta's sd over the vocabulary; S_const = ||mean_t Lc_t||^2 / mean_t ||Lc_t||^2 with Lc_t = L_t - mean_v L_t
  over the tokenizer's vocabulary. Writes results/round5/logit/<model>/<rev>.json (r3: results/round5/logit/round3/<slug>.json).
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from download_round4 import ckpt_dir
from measure import load_samples, resolve_model


@torch.no_grad()
def logit_metrics(path, delta, args, check_direct=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda:0"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=dev,
                                                 local_files_only=True).eval()
    head = model.get_output_embeddings()
    W = head.weight.detach().float()
    b_head = head.bias.detach().float() if getattr(head, "bias", None) is not None else None
    n_voc = min(len(tok), W.shape[0])
    names = list(delta)
    D = torch.stack([delta[n].to(dev).double() for n in names])  # [k, V]
    Dc = D - D.mean(1, keepdim=True)  # Var_p is shift invariant; centering avoids cancellation in E[d^2] - E[d]^2
    rows = load_samples(args.data, 0)
    acc_var = torch.zeros(len(names), dtype=torch.float64, device=dev)
    acc_first = torch.zeros(len(names), dtype=torch.float64, device=dev)
    sum_lc = torch.zeros(n_voc, dtype=torch.float64, device=dev)
    sum_lc2 = 0.0
    loss_sum, nt = 0.0, 0
    direct = None
    for sid in range(args.n_est, len(rows)):
        ids = tok(rows[sid]["text"], add_special_tokens=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt")["input_ids"].to(dev)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
        hf = out.hidden_states[-1][0].float()[:-1]
        y = ids[0, 1:]
        L = hf @ W.T + (b_head if b_head is not None else 0)  # [T-1, V], identical to the round-3/4 delta-loss logits
        logp = torch.log_softmax(L, -1)
        loss_sum += -logp.gather(1, y[:, None]).sum().item()
        p = logp.double().exp()
        e1 = p @ Dc.T  # [T-1, k]
        e2 = p @ Dc.pow(2).T
        acc_var += (e2 - e1.pow(2)).sum(0)
        acc_first += (Dc[:, y].T - e1).sum(0)
        if check_direct and direct is None:  # direct per-position sum for 2 sets on the first 5 positions
            direct = {}
            for j in range(min(2, len(names))):
                v_direct = [(p[t] * (D[j] - (p[t] * D[j]).sum()).pow(2)).sum().item() for t in range(min(5, len(y)))]
                v_formula = (e2 - e1.pow(2))[:5, j].tolist()
                direct[names[j]] = {"direct": v_direct, "formula": v_formula,
                                    "max_rel_diff": max(abs(a - b) / max(abs(a), 1e-30) for a, b in zip(v_direct, v_formula))}
        Lc = L[:, :n_voc].double()
        Lc = Lc - Lc.mean(1, keepdim=True)
        sum_lc += Lc.sum(0)
        sum_lc2 += Lc.pow(2).sum().item()
        nt += len(y)
    s_const = (sum_lc / nt).pow(2).sum().item() / (sum_lc2 / nt)
    res = {"path": path, "n_eval_tokens": nt, "loss_base": loss_sum / nt, "n_vocab_used": n_voc, "S_const": s_const,
           "sets": {n: {"half_E_var": 0.5 * acc_var[i].item() / nt, "first_order": acc_first[i].item() / nt,
                        "sd_vocab": D[i, :n_voc].std(unbiased=False).item()} for i, n in enumerate(names)},
           "direct_check": direct, "elapsed_s": time.time() - t0}
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round5_gpu.tsv")
    ap.add_argument("--rows", required=True, help="0-based row range a-b")
    ap.add_argument("--check_direct", action="store_true")
    ap.add_argument("--data", default="data/samples.jsonl")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--n_est", type=int, default=500)
    args = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDPA has no execution plan here on B300
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = [l.rstrip("\n").split("\t") for l in open(args.list) if l.strip()]
    a, _, b = args.rows.partition("-")
    for i in range(int(a), int(b or a) + 1):
        kind, repo, rev = rows[i]
        if kind == "r3":
            slug = os.path.basename(resolve_model(repo).rstrip("/"))
            out_fn = f"results/round5/logit/round3/{slug}.json"
            path, bank = resolve_model(repo), f"cache/banks/{slug}.pt"
        else:
            name = repo.split("/")[1]
            out_fn = f"results/round5/logit/{name}/{rev}.json"
            path = ckpt_dir(repo, rev)
            bank = f"cache/{'round5' if kind == 'new' else 'round4'}_banks/{name}/{rev}.pt"
            if kind == "new":
                dyn_fn = f"results/round5/dyn/{name}/{rev}.json"
                if not os.path.exists(dyn_fn):
                    from dyn_round4 import measure
                    dargs = SimpleNamespace(data=args.data, max_len=args.max_len, bank_per_seq=10, n_pairs=20000,
                                            n_fold=20, n_est=args.n_est, seed=0, cache="cache/round5_banks/")
                    torch.manual_seed(0)
                    res = measure(repo, rev, dargs)
                    os.makedirs(os.path.dirname(dyn_fn), exist_ok=True)
                    json.dump(res, open(dyn_fn, "w"), indent=1)
                    print(f"[{name}@{rev}] dyn: 0a={res['check0a']['max_rel_err']:.1e} fold={res['fold']['max_rel_err']:.1e} "
                          f"A V1={res['step1']['V1']['all']['isotropy_full']:.3f} V3={res['step1']['V3']['all']['isotropy_full']:.3f} "
                          f"loss={res['function']['loss_base']:.3f} ({res['elapsed_s']:.0f}s)", flush=True)
        if os.path.exists(out_fn):
            print(f"{out_fn} exists, skipping", flush=True)
            continue
        delta = torch.load(bank, weights_only=False)["delta"]
        res = logit_metrics(path, delta, args, check_direct=args.check_direct)
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        json.dump(res, open(out_fn, "w"), indent=1)
        print(f"[{repo}@{rev}] logit: loss={res['loss_base']:.4f} S_const={res['S_const']:.3f} "
              + " ".join(f"{n}={v['half_E_var']:.3f}" for n, v in res["sets"].items())
              + (f" direct={ {k: round(v['max_rel_diff'], 9) for k, v in res['direct_check'].items()} }"
                 if res["direct_check"] else "") + f" ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
