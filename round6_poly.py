"""Round 6 S6 (GPU): PolyPythias final checkpoints (configs/round6_polypythia.tsv, pytorch_model.bin only).

For each row: all dyn_round4.measure metrics (same data, bank positions and splits as rounds 3-5; bank to
cache/round6/poly_banks/), plus the round-5 W_U spectrum and direction alignment (round5_cpu.spec_stats / align on the
first 50277 head rows, bf16 head as loaded). Collapse (descriptive only): V1 ||ubar||^2 >= 0.9 and W_U s1^2/sum >= 0.5.
Writes results/round6/poly/<model>-seed<k>/<rev>.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="configs/round6_polypythia.tsv")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="results/round6/poly/")
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
        if os.path.exists(fn):
            print(f"{fn} exists, skipping", flush=True)
            continue
        if not os.path.exists(os.path.join(ckpt_dir(repo, rev), ".complete")):
            print(f"{repo}@{rev} not downloaded, skipping", flush=True)
            continue
        torch.manual_seed(0)
        dargs = SimpleNamespace(data="data/samples.jsonl", max_len=256, bank_per_seq=10, n_pairs=20000, n_fold=20,
                                n_est=500, seed=0, cache="cache/round6/poly_banks/")
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
        json.dump(res, open(fn, "w"), indent=1)
        print(f"[{name}@{rev}] V1 ubar^2={ub1:.3f} A V1={res['step1']['V1']['all']['isotropy_full']:.3f} "
              f"V3={res['step1']['V3']['all']['isotropy_full']:.3f} s1^2/sum={st['s1sq_over_sum']:.3f} "
              f"cos1={res['spec']['align_V1_WUgamma']['abs_cos_v1']:.3f} loss={res['function']['loss_base']:.3f} "
              f"collapsed={res['collapsed']} ({res['elapsed_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
