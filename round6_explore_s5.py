"""Round 6 S5, exploratory: spectrum of the refitted folded head W diag(gamma) (first 50277 rows) at steps 250 and 1000.
CPU only; reads cache/round6/refit/. Writes results/round6/causal/explore_refit_spec.json."""
import torch, json
out = {}
for m, revs in (("pythia-70m", ["step50000", "step80000", "step110000", "step130000", "step143000"]),
                ("pythia-160m", ["step80000", "step110000", "step130000", "step143000"]), ("pythia-410m", ["step80000", "step143000"])):
    for r in revs:
        row = {}
        for s in ("step250", "step1000"):
            d = torch.load(f"cache/round6/refit/{m}/{r}/{s}.pt", weights_only=False, map_location="cpu")
            W = (d["W"] * d["gamma"]).double()[:50277]
            sv = torch.linalg.svdvals(W)
            row[s] = {"s1": sv[0].item(), "s1sq_over_sum": (sv[0] ** 2 / (sv ** 2).sum()).item()}
        out[f"{m}/{r}"] = row
        print(m, r, {k: (round(v["s1"], 1), round(v["s1sq_over_sum"], 3)) for k, v in row.items()}, flush=True)
json.dump(out, open("results/round6/causal/explore_refit_spec.json", "w"), indent=1)
