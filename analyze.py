"""Aggregate results/*.json into figures (figs/) and summary tables (results/summary.{md,csv}).

Runs on the login node (CPU only).
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# instruct-tuning may change representation geometry, so every model is labelled
KIND = {
    "gpt2": "base", "gpt2-medium": "base", "gpt2-large": "base", "gpt2-xl": "base",
    "Llama-3.1-8B": "base", "Qwen3-8B-Base": "base",
    "Qwen2.5-7B-Instruct": "instruct",
    "Llama-3.1-8B-Instruct": "instruct",
    "Qwen3-8B": "post-trained",
    "Mistral-Small-24B-Instruct-2501": "instruct",
    "Olmo-3.1-32B-Instruct": "instruct",
    "Qwen3-32B": "post-trained",
}
# categorical slots in fixed order (dataviz reference palette, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def load(results_dir="results"):
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(results_dir, "*.json")))]
    rs = [r for r in rs if r["n_seqs"] >= 1000]
    return sorted(rs, key=lambda r: r["n_params"])


def label(r):
    return f"{r['model']} ({r['n_params'] / 1e9:.1f}B, {KIND.get(r['model'], '?')})"


def fig_last_layer_isotropy(rs, path):
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(rs) + 1.4))
    names = [label(r) for r in rs][::-1]
    vals = [r["layers"][-1]["all"]["isotropy"] for r in rs][::-1]
    # bar colour encodes base vs post-trained (the comparison of interest), not model identity
    cols = [COLORS[0] if KIND.get(r["model"]) == "base" else COLORS[1] for r in rs][::-1]
    ax.barh(names, vals, color=cols, height=0.6)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[0], label="base"),
                       plt.Rectangle((0, 0), 1, 1, color=COLORS[1], label="instruct / post-trained")],
              loc="lower right", fontsize=8.5)
    for y, v in enumerate(vals):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.axvline(0.25, color=INK2, lw=1.2, ls="--")
    ax.text(0.26, -0.75, "0.25 reference", color=INK2, fontsize=8, va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("isotropy = 1 − mean within-sequence self-similarity")
    ax.set_ylim(-1, len(rs) - 0.5)
    fig.suptitle("Last-layer isotropy (hidden_states[-1], after final norm; all positions)",
                 x=0.01, ha="left", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_depth(rs, path, metrics, title):
    variants = [("all", "all positions"), ("no_pos0", "position 0 removed")]
    fig, axes = plt.subplots(len(variants), len(metrics), figsize=(4.2 * len(metrics), 3.4 * len(variants)),
                             sharex=True, squeeze=False)
    for row, (var, vname) in enumerate(variants):
        for col, (key, mname, logy) in enumerate(metrics):
            ax = axes[row][col]
            for i, r in enumerate(rs):
                xs = [l["rel_depth"] for l in r["layers"]]
                ys = [l[var][key] if var in l and key in l[var] else l[key] for l in r["layers"]]
                ax.plot(xs, ys, color=COLORS[i], lw=2, marker="o", ms=4, label=label(r))
                # final point is post final-norm: mark it distinctly
                ax.plot(xs[-1], ys[-1], marker="D", ms=7, color=COLORS[i], mec="white", mew=1.2)
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{mname} — {vname}", color=INK, fontsize=10, loc="left")
            if row == len(variants) - 1:
                ax.set_xlabel("relative depth")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(rs)), fontsize=8.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title}   (0 = embedding output, ◆ = last layer, after final norm)", x=0.01, ha="left",
                 color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_pos0(rs, path):
    """Position-0 norm ratio and outlier ratio are not per-variant for the former."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, r in enumerate(rs):
        xs = [l["rel_depth"] for l in r["layers"]]
        axes[0].plot(xs, [l["pos0_norm_ratio_median"] for l in r["layers"]], color=COLORS[i], lw=2,
                     marker="o", ms=4, label=label(r))
        axes[1].plot(xs, [l["no_pos0"]["outlier_ratio"] for l in r["layers"]], color=COLORS[i], lw=2,
                     marker="o", ms=4)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].axhline(1, color=INK2, lw=1, ls="--")
    axes[0].set_title("‖h₀‖ / median‖hₜ, t>0‖ (median over sequences)", fontsize=10, loc="left", color=INK)
    axes[1].set_title("outlier dims: max / median of per-dim mean |h| (pos 0 removed)", fontsize=10,
                      loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("relative depth")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


STAGES = [("pre", "x (pre-norm)"), ("xhat", "x̂ (normalized)"), ("no_bias", "γ⊙x̂"), ("post", "γ⊙x̂+β (post)")]


def load_diag(rs, diag_dir="results/diag"):
    order = {r["model"]: i for i, r in enumerate(rs)}
    ds = [json.load(open(f)) for f in glob.glob(os.path.join(diag_dir, "*.json"))]
    ds = [d for d in ds if d["n_seqs"] >= 1000 and d["model"] in order]
    return sorted(ds, key=lambda d: order[d["model"]])


def fig_diag(groups, path):
    """Each metric across the steps of the final norm, one panel row per model group."""
    metrics = [("isotropy", "A isotropy"), ("cross_cos", "B cross-seq cos"), ("top1_var", "C top-1 PC var")]
    fig, axes = plt.subplots(len(groups), len(metrics), figsize=(4.4 * len(metrics), 3.3 * len(groups)),
                             squeeze=False)
    for row, (gname, ds) in enumerate(groups):
        for col, (key, mname) in enumerate(metrics):
            ax = axes[row][col]
            for i, d in enumerate(ds):
                ys = [d["variants"][k]["all"][key] for k, _ in STAGES]
                ax.plot(range(len(STAGES)), ys, color=COLORS[i], lw=2, marker="o", ms=5,
                        label=f"{d['model']} ({KIND.get(d['model'], '?')})")
            ax.set_xticks(range(len(STAGES)), [n for _, n in STAGES], fontsize=8.5)
            ax.set_title(f"{mname} — {gname}", fontsize=10, loc="left", color=INK)
        axes[row][-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Final norm decomposed: last block output → normalize → scale γ → add β (all positions)",
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def diag_table(ds):
    lines = ["\n**Final-norm decomposition** (all positions; A isotropy / B raw cross cos / C top-1 var)\n",
             "| model | norm | bias | tied emb | " + " | ".join(f"{n} A / B / top1" for _, n in STAGES)
             + " | post−β A / B | ‖β‖/‖mean post‖ | cos(β, mean post) | ‖h₀‖/med pre-norm |",
             "|---" * (8 + len(STAGES)) + "|"]
    for d in ds:
        v = d["variants"]
        cells = [f"{v[k]['all']['isotropy']:.3f} / {v[k]['all']['cross_cos']:.3f} / {v[k]['all']['top1_var']:.2f}"
                 for k, _ in STAGES]
        pmb = (f"{v['post_minus_beta']['all']['isotropy']:.3f} / {v['post_minus_beta']['all']['cross_cos']:.3f}"
               if "post_minus_beta" in v else "–")
        b = d["beta"] or {}
        lines.append(f"| {d['model']} | {d['norm_type']} | {'yes' if d['norm_has_bias'] else 'no'} | "
                     f"{'yes' if d['embeddings_actually_shared'] else 'no'} | " + " | ".join(cells) + f" | {pmb} | "
                     + (f"{b['beta_norm / mean_post_token_norm']:.3f} | {b['cos(beta, mean post token)']:.3f}"
                        if b else "– | –") + f" | {d['pre_pos0_norm_ratio_median']:.1f} |")
    return lines


def tables(rs, md_path, csv_path, ds=()):
    cols = [("isotropy", "A iso"), ("cross_cos", "B cos"), ("cross_cos_centered", "B cos (centered)"),
            ("rankme_over_d", "RankMe/d"), ("top1_var", "top-1 var"), ("top10_var", "top-10 var")]
    lines, csv = [], ["model,params_B,kind,bos_added,variant," + ",".join(c for c, _ in cols)]
    for var, vname in (("all", "all positions"), ("no_pos0", "position 0 removed")):
        lines += [f"\n**Last layer, {vname}**\n",
                  "| model | params | kind | BOS | " + " | ".join(n for _, n in cols) + " |",
                  "|---|---|---|---|" + "---|" * len(cols)]
        for r in rs:
            m = r["layers"][-1][var]
            vals = [m[c] for c, _ in cols]
            lines.append(f"| {r['model']} | {r['n_params'] / 1e9:.2f}B | {KIND.get(r['model'], '?')} | "
                         f"{'yes' if r['bos_added'] else 'no'} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
            csv.append(f"{r['model']},{r['n_params'] / 1e9:.3f},{KIND.get(r['model'], '?')},{r['bos_added']},{var},"
                       + ",".join(f"{v:.5f}" for v in vals))
    # per-layer detail
    lines += ["\n**All selected layers** (iso / B cos / B cos centered / RankMe/d; all → no_pos0; "
              "pos0 norm ratio; outlier ratio)\n"]
    for r in rs:
        lines += [f"\n{label(r)}, L={r['n_layers']}, d={r['d']}, first token `{r['first_token_text']}`\n",
                  "| layer | rel | iso | iso no0 | B cos | B cos no0 | Bc cos | Bc cos no0 | RankMe/d | "
                  "RankMe/d no0 | ‖h0‖/med | outlier | outlier no0 |",
                  "|---" * 13 + "|"]
        for l in r["layers"]:
            a, b = l["all"], l["no_pos0"]
            lines.append(
                f"| {l['layer']} ({l['label']}) | {l['rel_depth']:.2f} | {a['isotropy']:.3f} | {b['isotropy']:.3f} | "
                f"{a['cross_cos']:.3f} | {b['cross_cos']:.3f} | {a['cross_cos_centered']:.3f} | "
                f"{b['cross_cos_centered']:.3f} | {a['rankme_over_d']:.3f} | {b['rankme_over_d']:.3f} | "
                f"{l['pos0_norm_ratio_median']:.1f} | {a['outlier_ratio']:.0f} | {b['outlier_ratio']:.0f} |")
    if ds:
        lines += diag_table(ds)
    open(md_path, "w").write("\n".join(lines) + "\n")
    open(csv_path, "w").write("\n".join(csv) + "\n")


def main():
    rs = load()
    print(f"{len(rs)} models: {[r['model'] for r in rs]}")
    os.makedirs("figs", exist_ok=True)
    for f in glob.glob("figs/*.png"):
        os.remove(f)
    fig_last_layer_isotropy(rs, "figs/fig1_last_layer_isotropy.png")
    # at most 8 series per chart: facet into the GPT-2 family and the modern models
    groups = [("gpt2", [r for r in rs if r["model"].startswith("gpt2")]),
              ("modern", [r for r in rs if not r["model"].startswith("gpt2")])]
    for g, grs in groups:
        assert len(grs) <= len(COLORS), g
        fig_depth(grs, f"figs/fig2_depth_{g}.png",
                  [("isotropy", "A isotropy", False), ("cross_cos", "B cross-seq cos", False),
                   ("rankme_over_d", "C RankMe/d", True)],
                  f"Metrics vs relative depth — {g}")
        fig_depth(grs, f"figs/fig3_centered_spectrum_{g}.png",
                  [("cross_cos_centered", "B cross-seq cos, centered", False), ("top1_var", "C top-1 PC var", False)],
                  f"Centered-bank metrics vs relative depth — {g}")
        fig_pos0(grs, f"figs/fig4_pos0_outliers_{g}.png")
    ds = load_diag(rs)
    if ds:
        fig_diag([(g, [d for d in ds if d["model"] in {r["model"] for r in grs}]) for g, grs in groups],
                 "figs/fig5_final_norm.png")
    tables(rs, "results/summary.md", "results/summary.csv", ds)
    print("wrote figs/*.png, results/summary.{md,csv}")


if __name__ == "__main__":
    main()
