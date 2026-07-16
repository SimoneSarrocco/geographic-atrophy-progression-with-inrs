#!/usr/bin/env python3
"""Full latent-grid heatmaps: channels x spatial, value at the two-stage best checkpoint.

For every (channels, spatial) cell of the 6x5 latent grid it reads the held-out validation curves
(ablation_metrics.epoch_series -- the verified checkpoint-consistent reader) and applies the same
TWO-STAGE selection as the other latent-grid figures (ablation_metrics.two_stage_select): Stage A
picks each config's checkpoint by the DICE+lesion-MAE trade-off; Stage B scores configs by the full
composite over DICE/SSIM/LPIPS/PSNR/lesion-MAE at that checkpoint. It renders four heatmaps of the
value AT that checkpoint: DICE, SSIM, lesion-area MAE, and the Stage-B composite. Colour is
dark = good in every panel; the best config is ringed in red.

Cells not yet trained show as hatched 'n/a'. Run on the node holding the run dirs; point
--runs at them (defaults to ablations/runs).

Output: ablations/figures/latent_grid_heatmap.{pdf,png}
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_metrics import (GRID_CHANNELS, GRID_SPATIAL, grid_run_name,
                              epoch_series, two_stage_select, inject_reeval_lpips,
                              SELECT_KEYS_SEG, RANK_KEYS)


def _load_grid(runs_root):
    series_by_cell = {}
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            rd = os.path.join(runs_root, grid_run_name(C, S))
            s = epoch_series(rd) if os.path.isdir(rd) else {}
            if s:
                series_by_cell[(C, S)] = s
    return series_by_cell


# metric key, title, lower-is-better, colormap (DARK = better in all panels)
PANELS = [("DICE", r"DICE $\uparrow$", False, "viridis_r"),
          ("SSIM", r"SSIM $\uparrow$", False, "viridis_r"),
          ("areaMAE", r"lesion-area MAE [mm$^2$] $\downarrow$", True, "viridis"),
          ("composite", "full composite $\\uparrow$", False, "magma_r")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--weights", default="1,1",
                    help="Stage-A checkpoint weights DICE,areaMAE (default equal)")
    ap.add_argument("--rank-weights", default=None,
                    help="Stage-B ranking weights DICE,SSIM,LPIPS,PSNR,areaMAE (default equal)")
    ap.add_argument("--rank-on-common", action="store_true",
                    help="rank ALL configs on the metric subset present everywhere")
    ap.add_argument("--no-lpips-fallback", action="store_true",
                    help="do NOT backfill missing per-epoch LPIPS from the reeval_loo summary "
                         "(by default it IS backfilled so configs whose training-curve LPIPS was "
                         "disabled still get LPIPS in the ranking + composite panel)")
    args = ap.parse_args()
    weights = tuple(float(x) for x in args.weights.split(","))
    assert len(weights) == len(SELECT_KEYS_SEG), "--weights needs 2 values (DICE,areaMAE)"
    rank_weights = tuple(float(x) for x in args.rank_weights.split(",")) if args.rank_weights else None
    assert rank_weights is None or len(rank_weights) == len(RANK_KEYS), \
        f"--rank-weights needs {len(RANK_KEYS)} values {RANK_KEYS}"

    series_by_cell = _load_grid(args.runs)
    if not args.no_lpips_fallback:
        patched = inject_reeval_lpips(series_by_cell, args.runs, weights)
        if patched:
            print(f"[LPIPS fallback] backfilled reeval_loo LPIPS for {len(patched)} config(s) whose "
                  f"training-curve LPIPS was missing:")
            for (C, S), ep, lp in patched:
                print(f"    {C}x{S}  (Stage-A ep {ep})  LPIPS={lp:.3f}")
    # two-stage: Stage-A checkpoint by DICE+areaMAE, Stage-B full composite (cells + composite panel)
    _, best, _, _ = two_stage_select(series_by_cell, seg_weights=weights,
                                     rank_weights=rank_weights, rank_on_common=args.rank_on_common)
    n_have = len(best)
    best_cell = max(best, key=lambda k: best[k]["composite"]) if best else None
    print(f"[grid] {n_have}/{len(GRID_CHANNELS) * len(GRID_SPATIAL)} cells ranked (Stage-B composite).")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.4))

    nC, nS = len(GRID_CHANNELS), len(GRID_SPATIAL)
    for ax, (mkey, title, lower, cmap) in zip(axes.ravel(), PANELS):
        M = np.full((nS, nC), np.nan)          # rows = spatial (y), cols = channels (x)
        for j, C in enumerate(GRID_CHANNELS):
            for i, S in enumerate(GRID_SPATIAL):
                if (C, S) in best:
                    M[i, j] = best[(C, S)][mkey]
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        # annotate every cell (value, or 'n/a' hatched for untrained)
        for j, C in enumerate(GRID_CHANNELS):
            for i, S in enumerate(GRID_SPATIAL):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                               facecolor="0.92", edgecolor="0.75",
                                               hatch="///", lw=0))
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.5")
                else:
                    # readable text colour against the cell
                    lo, hi = np.nanmin(M), np.nanmax(M)
                    rel = (M[i, j] - lo) / (hi - lo + 1e-12)
                    tc = "white" if (rel < 0.45) ^ (cmap.endswith("_r")) else "black"
                    fmt = f"{M[i, j]:.3f}" if mkey != "areaMAE" else f"{M[i, j]:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=8, color=tc)
        # ring the Stage-B best config in every panel
        if best_cell is not None:
            bj, bi = GRID_CHANNELS.index(best_cell[0]), GRID_SPATIAL.index(best_cell[1])
            ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                       edgecolor="#d62728", linewidth=2.4, zorder=5))
        ax.set_xticks(range(nC)); ax.set_xticklabels(GRID_CHANNELS)
        ax.set_yticks(range(nS)); ax.set_yticklabels(GRID_SPATIAL)
        ax.set_xlabel("latent channels $C$")
        ax.set_ylabel("latent grid  $H{=}W$ [px]")
        ax.set_title(title, pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ringtxt = (f"  |  red = best config {best_cell[0]}ch x {best_cell[1]}px" if best_cell else "")
    fig.suptitle("Latent-grid ablation — value at the DICE+lesion-MAE checkpoint, "
                 "ranked by full composite" + ringtxt,
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "latent_grid_heatmap." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
