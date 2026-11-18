"""Round 6 phase 3 U2 diagnostic (CPU; configs/round6_prereg.md, "U2 诊断"): for every model x dataset x pool,
(a) max |cos_rep - cos_V1| of the whitened cosines (SVD + numerical rank tolerance, main fit set) over the 7 representations,
(b) the 7-representation score range after rounding the cosines to 1e-12, and the number of identical-sentence pairs.
Reads cache/round6/p3/whiten_emb/<model>.pt; writes results/round6/p3/whiten_wide/diag/<model>.json.
"""
import json
import os
import sys

import torch

from round6_lib import sample_transform
from round6_p3_whiten import DATASETS, POOLS, load_all_sts
from round6_whiten import zca_svd
from sts_round4 import spearman100


@torch.no_grad()
def main(model):
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    slug = os.path.basename(model.rstrip("/"))
    fn = f"results/round6/p3/whiten_wide/diag/{slug}.json"
    if os.path.exists(fn):
        return
    e = torch.load(f"cache/round6/p3/whiten_emb/{slug}.pt", weights_only=False)
    G = torch.load(f"cache/round6/gauge/{slug}.pt", weights_only=False)
    sid = {s: i for i, s in enumerate(e["sents"])}
    fit = [json.loads(l)["text"] for l in open("data/fit_sentences_round6.jsonl")]
    fi = torch.as_tensor([sid[s] for s in fit])
    sts = load_all_sts()
    pairs = {k: (torch.as_tensor([sid[s] for s in df.sentence1]), torch.as_tensor([sid[s] for s in df.sentence2])) for k, df in sts.items()}
    gold = {k: torch.as_tensor(df.score.to_numpy().astype("float64")) for k, df in sts.items()}
    d = e["mean"].shape[1]
    g = e["gamma"].double()
    b = e["beta"].double() if e["beta"] is not None else torch.zeros_like(g)
    res = {"model": slug, "identical_pairs": {k: int((df.sentence1 == df.sentence2).sum()) for k, df in sts.items()}, "pools": {}}
    for pool in POOLS:
        V1 = e[pool].double()
        assert not torch.isnan(V1).any(), "NaN rows: run round6_p3_whiten.py first (it repairs the cache)"
        reps = {"V1": V1, "V3": V1 * g + b, "D_Amax": V1 * G["worst_s"]["k100_A_max"], "D_Amin": V1 * G["worst_s"]["k100_A_min"]}
        for s in range(3):
            fx, _, _ = sample_transform("lin", d, 100, s)
            reps[f"lin{s}"] = fx(V1)
        cos = {}
        for r, E in reps.items():
            mu, Wh, _ = zca_svd(E[fi], rank_tol=True)
            Z = torch.nn.functional.normalize((E - mu) @ Wh, dim=-1)
            for ds, (i1, i2) in pairs.items():
                cos[(r, ds)] = (Z[i1] * Z[i2]).sum(-1)
            del Z, E
        out = {}
        for ds in DATASETS:
            dc = max((cos[(r, ds)] - cos[("V1", ds)]).abs().max().item() for r in reps)
            sc = [spearman100(cos[(r, ds)], gold[ds]) for r in reps]
            sr = [spearman100(cos[(r, ds)].round(decimals=12), gold[ds]) for r in reps]
            out[ds] = {"max_abs_cos_diff": dc, "score_range": max(sc) - min(sc), "score_range_rounded": max(sr) - min(sr)}
        res["pools"][pool] = out
        print(f"[{slug} {pool}] max|dcos| {max(v['max_abs_cos_diff'] for v in out.values()):.1e} range {max(v['score_range'] for v in out.values()):.1e} "
              f"rounded {max(v['score_range_rounded'] for v in out.values()):.1e}", flush=True)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(res, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
