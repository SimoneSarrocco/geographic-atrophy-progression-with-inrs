#!/usr/bin/env python3
"""Per-metric latent-grid scatter: latent grid (x) vs latent channels (y), one PANEL per metric.

Unlike plot_grid_scatter.py (a single panel coloured by the composite), this draws one scatter
panel PER displayed metric (default DICE, SSIM, lesion-area MAE). In EVERY panel each point is the
value of that metric at the SAME, single best-chosen checkpoint for that configuration -- so the
panels are mutually consistent (they all read the one epoch that best balances the objectives).

Selection is the SAME TWO-STAGE scheme as plot_grid_metric_row.py (ablation_metrics.two_stage_select):
  * Stage A (checkpoint per config): best_epoch = argmax of the DICE+lesion-MAE composite -- the
    segmentation-focused checkpoint. Every panel value is read at THIS epoch, so panels are mutually
    consistent; PSNR/SSIM/LPIPS are reported here, not used to pick it.
  * Stage B (rank across configs): the full composite over DICE/SSIM/LPIPS/PSNR/lesion-MAE at that
    checkpoint -> the config with the best full composite is ringed in red in every panel.
Metrics are min-max normalized (bright-good inverted so DARK = good) and equal-weighted; tune with
--weights (Stage A) / --rank-weights (Stage B). Colour is dark = good in every panel.

Axes: x = latent grid H=W [px], y = latent channels C (both log2). Untrained/missing cells are
drawn as hollow 'n/a' markers. Point run on the node holding the run dirs.

    python plot_grid_metric_scatter.py --runs runs_r2 [--metrics DICE SSIM areaMAE LPIPS]
                                       [--weights 1,1] [--rank-weights 1,1,1,1,1] [--rank-on-common]

Output: ablations/figures/latent_grid_metric_scatter.{pdf,png}
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_metrics import (GRID_CHANNELS, GRID_SPATIAL, grid_run_name, epoch_series,
                              two_stage_select, SELECT_KEYS_SEG, RANK_KEYS, LOWER_BETTER)

# display metadata per metric (label, lower-is-better?, number format)
META = {
    "DICE":    dict(label=r"DICE  $\uparrow$",                    lower=False, fmt="{:.3f}"),
    "SSIM":    dict(label=r"SSIM  $\uparrow$",                    lower=False, fmt="{:.3f}"),
    "areaMAE": dict(label=r"lesion-area MAE [mm$^2$]  $\downarrow$", lower=True, fmt="{:.2f}"),
    "LPIPS":   dict(label=r"LPIPS  $\downarrow$",                 lower=True,  fmt="{:.3f}"),
    "PSNR":    dict(label=r"PSNR [dB]  $\uparrow$",               lower=False, fmt="{:.1f}"),
}


def _load_grid(runs_root):
    series_by_cell = {}
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            rd = os.path.join(runs_root, grid_run_name(C, S))
            s = epoch_series(rd) if os.path.isdir(rd) else {}
            if s:
                series_by_cell[(C, S)] = s
    return series_by_cell


def _panel(ax, metric, best, series_by_cell, best_cell):
    """Draw one metric panel: each cell coloured by `metric` at its shared best checkpoint."""
    lower = META[metric]["lower"]
    cmap = "viridis" if lower else "viridis_r"          # dark = good in every panel
    fmt = META[metric]["fmt"].format

    xs, ys, cs = [], [], []
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            x, y = np.log2(S), np.log2(C)               # x = grid, y = channels
            val = None
            if (C, S) in best:
                ep = best[(C, S)]["best_epoch"]
                val = series_by_cell[(C, S)].get(ep, {}).get(metric)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                ax.scatter([x], [y], s=520, marker="s", facecolors="none",
                           edgecolors="0.75", linewidths=1.0, zorder=2)
                ax.text(x, y, "n/a", ha="center", va="center", fontsize=7, color="0.6", zorder=3)
            else:
                xs.append(x); ys.append(y); cs.append(float(val))

    if cs:
        vmin, vmax = min(cs), max(cs)
        sc = ax.scatter(xs, ys, c=cs, s=560, marker="s", cmap=cmap,
                        vmin=vmin, vmax=vmax, edgecolors="white", linewidths=1.2, zorder=4)
        for x, y, c in zip(xs, ys, cs):
            rel = (c - vmin) / (vmax - vmin + 1e-12)     # relative to THIS panel's range
            good = (1 - rel) if lower else rel           # good cells are now DARK -> white text
            ax.text(x, y, fmt(c), ha="center", va="center", fontsize=8.2,
                    color="white" if good > 0.5 else "black", zorder=5)
        cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    # ring the shared composite-best cell in every panel
    if best_cell is not None:
        bC, bS = best_cell
        ax.scatter([np.log2(bS)], [np.log2(bC)], s=1150, marker="s", facecolors="none",
                   edgecolors="#d62728", linewidths=2.6, zorder=6)

    ax.set_xticks([np.log2(s) for s in GRID_SPATIAL]); ax.set_xticklabels(GRID_SPATIAL)
    ax.set_yticks([np.log2(c) for c in GRID_CHANNELS]); ax.set_yticklabels(GRID_CHANNELS)
    ax.set_xlabel(r"latent grid  $H{=}W$  [px]", fontsize=11)
    ax.set_xlim(np.log2(GRID_SPATIAL[0]) - 0.6, np.log2(GRID_SPATIAL[-1]) + 0.6)
    ax.set_ylim(np.log2(GRID_CHANNELS[0]) - 0.6, np.log2(GRID_CHANNELS[-1]) + 0.6)
    ax.set_title(META[metric]["label"], fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--metrics", nargs="+", default=["DICE", "SSIM", "areaMAE"],
                    choices=list(META), help="which metric panels to draw (value at shared best ckpt)")
    ap.add_argument("--weights", default="1,1",
                    help="Stage-A checkpoint weights DICE,areaMAE (default equal)")
    ap.add_argument("--rank-weights", default=None,
                    help="Stage-B ranking weights DICE,SSIM,LPIPS,PSNR,areaMAE (default equal)")
    ap.add_argument("--rank-on-common", action="store_true",
                    help="rank ALL configs on the metric subset present everywhere")
    args = ap.parse_args()
    weights = tuple(float(x) for x in args.weights.split(","))
    assert len(weights) == len(SELECT_KEYS_SEG), "--weights needs 2 values (DICE,areaMAE)"
    rank_weights = tuple(float(x) for x in args.rank_weights.split(",")) if args.rank_weights else None
    assert rank_weights is None or len(rank_weights) == len(RANK_KEYS), \
        f"--rank-weights needs {len(RANK_KEYS)} values {RANK_KEYS}"

    series_by_cell = _load_grid(args.runs)
    # TWO-STAGE (same as plot_grid_metric_row.py): Stage A picks each config's checkpoint by
    # DICE+areaMAE; Stage B ranks configs by the full composite (adds PSNR/SSIM/LPIPS) at that
    # checkpoint. Panels read values at the Stage-A checkpoint; the ring is the Stage-B winner.
    stageA, stageB, _, rank_keys_used = two_stage_select(
        series_by_cell, seg_weights=weights, rank_weights=rank_weights,
        rank_on_common=args.rank_on_common)
    best = stageA
    total = len(GRID_CHANNELS) * len(GRID_SPATIAL)
    print(f"[grid] {len(stageA)}/{total} cells (Stage-A checkpoint = DICE+areaMAE; "
          f"Stage-B ranking over {rank_keys_used}).")
    best_cell = max(stageB, key=lambda k: stageB[k]["composite"]) if stageB else None
    if best_cell is not None:
        print(f"[grid] Stage-B best cell: {best_cell[0]} ch x {best_cell[1]} px  "
              f"composite={stageB[best_cell]['composite']:.3f}  "
              f"epoch={stageA[best_cell]['best_epoch']}")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    n = len(args.metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4), squeeze=False)
    for ax, metric in zip(axes[0], args.metrics):
        _panel(ax, metric, best, series_by_cell, best_cell)
    axes[0][0].set_ylabel(r"latent channels  $C$", fontsize=11)

    # no overall in-figure title (goes in the LaTeX \caption); per-panel metric labels are kept as
    # column headers and the red ring marks the shared best config.
    fig.tight_layout()

    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "latent_grid_metric_scatter." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
