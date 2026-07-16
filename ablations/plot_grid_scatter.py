#!/usr/bin/env python3
"""Latent-grid scatter: channels (x) vs latent grid (y), colour = full-composite score.

Each point is one (channels, spatial) configuration placed on a log2-log2 lattice. Selection is the
same TWO-STAGE scheme as the other latent-grid figures (ablation_metrics.two_stage_select): Stage A
picks each config's checkpoint by the DICE+lesion-MAE trade-off; Stage B scores/ranks configs by the
full composite over DICE/SSIM/LPIPS/PSNR/lesion-MAE at that checkpoint. The point colour (and printed
number) is that Stage-B composite (dark = good); the overall best cell is ringed in red.

Cells not yet trained -- or trained but missing a ranking metric at their checkpoint -- are drawn as
small hollow 'n/a' markers (use --rank-on-common to keep the latter). Run on the node holding the run
dirs; point --runs at them (defaults to ablations/runs).

Output: ablations/figures/latent_grid_scatter.{pdf,png}
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_metrics import (GRID_CHANNELS, GRID_SPATIAL, grid_run_name,
                              epoch_series, two_stage_select, SELECT_KEYS_SEG, RANK_KEYS)


def _load_grid(runs_root):
    series_by_cell = {}
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            rd = os.path.join(runs_root, grid_run_name(C, S))
            s = epoch_series(rd) if os.path.isdir(rd) else {}
            if s:
                series_by_cell[(C, S)] = s
    return series_by_cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--weights", default="1,1",
                    help="Stage-A checkpoint weights DICE,areaMAE (default equal)")
    ap.add_argument("--rank-weights", default=None,
                    help="Stage-B ranking weights DICE,SSIM,LPIPS,PSNR,areaMAE (default equal)")
    ap.add_argument("--rank-on-common", action="store_true",
                    help="rank ALL configs on the metric subset present everywhere")
    ap.add_argument("--annotate", default="composite", choices=["composite", "metrics", "none"],
                    help="text on each point: composite score / DICE+SSIM+MAE / nothing")
    args = ap.parse_args()
    weights = tuple(float(x) for x in args.weights.split(","))
    assert len(weights) == len(SELECT_KEYS_SEG), "--weights needs 2 values (DICE,areaMAE)"
    rank_weights = tuple(float(x) for x in args.rank_weights.split(",")) if args.rank_weights else None
    assert rank_weights is None or len(rank_weights) == len(RANK_KEYS), \
        f"--rank-weights needs {len(RANK_KEYS)} values {RANK_KEYS}"

    series_by_cell = _load_grid(args.runs)
    # two-stage: Stage-A checkpoint by DICE+areaMAE, Stage-B full composite -> point colour + ring
    _, best, _, _ = two_stage_select(series_by_cell, seg_weights=weights,
                                     rank_weights=rank_weights, rank_on_common=args.rank_on_common)
    total = len(GRID_CHANNELS) * len(GRID_SPATIAL)
    print(f"[grid] {len(best)}/{total} cells ranked (Stage-B full composite).")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.2, 6.6))

    xs, ys, cs = [], [], []
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            x, y = np.log2(C), np.log2(S)
            if (C, S) in best:
                xs.append(x); ys.append(y); cs.append(best[(C, S)]["composite"])
            else:
                ax.scatter([x], [y], s=520, marker="s", facecolors="none",
                           edgecolors="0.75", linewidths=1.0, zorder=2)
                ax.text(x, y, "n/a", ha="center", va="center", fontsize=7.5, color="0.6", zorder=3)

    if cs:
        vmin, vmax = min(cs), max(cs)
        sc = ax.scatter(xs, ys, c=cs, s=560, marker="s", cmap="viridis_r",
                        vmin=vmin, vmax=vmax, edgecolors="white", linewidths=1.2, zorder=4)
        # per-point annotation (dark = good: high composite -> dark -> white text)
        for x, y, c in zip(xs, ys, cs):
            if args.annotate == "composite":
                rel = (c - vmin) / (vmax - vmin + 1e-12)
                ax.text(x, y, f"{c:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if rel > 0.5 else "black", zorder=5)
        if args.annotate == "metrics":
            for (C, S), e in best.items():
                ax.annotate(f"D{e['DICE']:.2f}\nS{e['SSIM']:.2f}\nM{e['areaMAE']:.2f}",
                            (np.log2(C), np.log2(S)), fontsize=6.2, ha="center", va="center",
                            color="0.15", zorder=5)
        # ring the overall best composite
        (bC, bS), be = max(best.items(), key=lambda kv: kv[1]["composite"])
        ax.scatter([np.log2(bC)], [np.log2(bS)], s=1150, marker="s", facecolors="none",
                   edgecolors="#d62728", linewidths=2.6, zorder=6)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("full composite at best checkpoint  $\\uparrow$", fontsize=10)
        print(f"[grid] best cell: {bC} ch x {bS} px  composite={be['composite']:.3f}  "
              f"(DICE {be['DICE']:.3f}, SSIM {be['SSIM']:.3f}, MAE {be['areaMAE']:.3f}, "
              f"epoch {be['best_epoch']})")

    ax.set_xticks([np.log2(c) for c in GRID_CHANNELS]); ax.set_xticklabels(GRID_CHANNELS)
    ax.set_yticks([np.log2(s) for s in GRID_SPATIAL]); ax.set_yticklabels(GRID_SPATIAL)
    ax.set_xlabel("latent channels $C$", fontsize=11)
    ax.set_ylabel("latent grid  $H{=}W$  [px]", fontsize=11)
    ax.set_xlim(np.log2(GRID_CHANNELS[0]) - 0.6, np.log2(GRID_CHANNELS[-1]) + 0.6)
    ax.set_ylim(np.log2(GRID_SPATIAL[0]) - 0.6, np.log2(GRID_SPATIAL[-1]) + 0.6)
    # no in-figure title (goes in the LaTeX \caption); the red ring marks the best config
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "latent_grid_scatter." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
