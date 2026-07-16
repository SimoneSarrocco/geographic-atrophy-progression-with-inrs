#!/usr/bin/env python3
"""MedFuncta-Fig.4-style SINGLE-ROW heatmap panel of the latent-grid ablation.

One compact row, one heatmap per metric -- PSNR, SSIM, LPIPS, DICE, lesion-area MAE -- over the
latent grid (x = latent grid H=W [px], y = latent channels C). TWO-STAGE selection:
  * Stage A (checkpoint per config): best_epoch = argmax of the DICE(up)+lesion-MAE(down) composite
    (segmentation only). Every cell in every panel is read at THIS epoch, so the numbers are mutually
    consistent; PSNR/SSIM/LPIPS are reported here, not used to pick the epoch.
  * Stage B (rank across configs): the SAME (previous) composite criterion applied over ALL five
    metrics (adds reconstruction PSNR/SSIM/LPIPS) at each config's Stage-A checkpoint -> the red ring
    marks the config with the best full composite.
Each panel is min-max normalized with a bright=good colormap (lower-better panels use the reversed
map) so the best region is obvious at a glance.

    python plot_grid_metric_row.py --runs runs_r2 [--weights 1,1] [--rank-weights 1,1,1,1,1]

Prints (stdout only, NOT in the figure) the Stage-A best epoch + metrics AND the Stage-B ranking for
every configuration. Output: ablations/figures/latent_grid_metric_row.{pdf,png}
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_metrics import (GRID_CHANNELS, GRID_SPATIAL, grid_run_name, discover_grid, epoch_series,
                              two_stage_select, inject_reeval_lpips,
                              SELECT_KEYS_SEG, RANK_KEYS, LOWER_BETTER)

# panels in display order (MedFuncta-style: recon metrics first, then segmentation)
PANELS = [
    ("PSNR",    r"PSNR [dB]  $\uparrow$",               False, "{:.1f}"),
    ("SSIM",    r"SSIM  $\uparrow$",                    False, "{:.3f}"),
    ("LPIPS",   r"LPIPS  $\downarrow$",                 True,  "{:.3f}"),
    ("DICE",    r"DICE  $\uparrow$",                    False, "{:.3f}"),
    ("areaMAE", r"lesion-area MAE [mm$^2$]  $\downarrow$", True, "{:.2f}"),
]


def _load_grid(runs_root, debug=False):
    out = {}
    missing_dir, empty = [], []
    # Prefer CONFIG-based discovery (reads each run's real latent_dim) so the figure is correct
    # regardless of dir naming; fall back to the name convention for any cell not discovered.
    disc = discover_grid(runs_root)
    if debug and disc:
        print(f"[debug] config-discovered {len(disc)} (C,S) cells from latent_dim: "
              + ", ".join(f"{C}x{S}->{n}" for (C, S), n in sorted(disc.items())))
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            name = disc.get((C, S)) or grid_run_name(C, S)
            rd = os.path.join(runs_root, name)
            is_dir = os.path.isdir(rd)
            s = epoch_series(rd) if is_dir else {}
            if s:
                out[(C, S)] = s
            elif not is_dir:
                missing_dir.append(name)
            else:
                empty.append(name)
    if debug:
        print(f"\n[debug] runs_root={runs_root}  -> {len(out)}/"
              f"{len(GRID_CHANNELS)*len(GRID_SPATIAL)} cells loaded")
        if missing_dir:
            print(f"[debug] {len(missing_dir)} cells: NO DIR named this under runs_root "
                  f"(-> rename or fix grid_run_name/_EXISTING):\n         " + ", ".join(missing_dir))
        if empty:
            print(f"[debug] {len(empty)} cells: dir EXISTS but no per-epoch val-eval metrics found "
                  f"(need val_eval*holdout*/*_metrics_ep=*.json + lesion_areas*_epoch_*.csv):\n         "
                  + ", ".join(empty))
        print(f"[debug] actually present under {runs_root}: "
              + ", ".join(sorted(os.listdir(runs_root))[:60]) + "\n")
    return out


def _panel(ax, metric, lower, fmt, best, series, best_cell):
    nC, nS = len(GRID_CHANNELS), len(GRID_SPATIAL)
    M = np.full((nC, nS), np.nan)                       # rows = channels, cols = spatial
    for i, C in enumerate(GRID_CHANNELS):
        for j, S in enumerate(GRID_SPATIAL):
            if (C, S) in best:
                ep = best[(C, S)]["best_epoch"]
                v = series[(C, S)].get(ep, {}).get(metric)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    M[i, j] = float(v)
    Mm = np.ma.masked_invalid(M)
    # light/yellow = good, dark/violet = bad, in EVERY panel: higher-better metrics use plain viridis
    # (high value -> yellow), lower-better metrics use reversed viridis (low value -> yellow).
    cmap = plt.get_cmap("viridis_r" if lower else "viridis").copy()
    cmap.set_bad("#ececec")
    vmin, vmax = (np.nanmin(M), np.nanmax(M)) if np.isfinite(M).any() else (0, 1)
    im = ax.imshow(Mm, cmap=cmap, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(nC):
        for j in range(nS):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color="0.55")
            else:
                rel = (M[i, j] - vmin) / (vmax - vmin + 1e-12)
                good = (1 - rel) if lower else rel      # good cells are now LIGHT/yellow -> black text
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                        color="black" if good > 0.5 else "white")

    # blue ring = BEST cell for THIS metric (argmax for higher-better, argmin for lower-better),
    # nested just inside the red ring so both are visible when they coincide.
    if np.isfinite(M).any():
        flat = np.nanargmin(M) if lower else np.nanargmax(M)
        bmi, bmj = np.unravel_index(flat, M.shape)
        ax.add_patch(plt.Rectangle((bmj - 0.42, bmi - 0.42), 0.84, 0.84, fill=False,
                                   edgecolor="#1f77b4", linewidth=2.4, zorder=6))

    if best_cell is not None:
        bi, bj = GRID_CHANNELS.index(best_cell[0]), GRID_SPATIAL.index(best_cell[1])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                                   edgecolor="#d62728", linewidth=2.6, zorder=5))

    ax.set_xticks(range(nS)); ax.set_xticklabels(GRID_SPATIAL)
    ax.set_yticks(range(nC)); ax.set_yticklabels(GRID_CHANNELS)
    ax.set_xlabel(r"latent grid $H{=}W$", fontsize=9.5)
    ax.set_title(metric_title(metric), fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(length=0, labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.16)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_visible(False)


def metric_title(metric):
    return {k: lbl for k, lbl, _, _ in PANELS}[metric]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--weights", default="1,1", help="Stage-A checkpoint weights DICE,areaMAE (equal)")
    ap.add_argument("--rank-weights", default=None,
                    help="Stage-B ranking weights DICE,SSIM,LPIPS,PSNR,areaMAE (default equal)")
    ap.add_argument("--rank-on-common", action="store_true",
                    help="rank ALL configs on the metric subset present everywhere (drop any metric "
                         "missing for even one config) instead of excluding incomplete configs")
    ap.add_argument("--csv", default=None,
                    help="path for the Stage-A+Stage-B table (default figures/latent_grid_ranking.csv)")
    ap.add_argument("--no-lpips-fallback", action="store_true",
                    help="do NOT backfill missing per-epoch LPIPS from the reeval_loo summary "
                         "(by default it IS backfilled so configs whose training-curve LPIPS was "
                         "disabled still get LPIPS in the ranking + panel)")
    ap.add_argument("--debug", action="store_true",
                    help="print, per grid cell, whether its run dir was found and had per-epoch "
                         "val-eval metrics -- to diagnose an all-NaN figure (name/location mismatch).")
    args = ap.parse_args()
    weights = tuple(float(x) for x in args.weights.split(","))
    assert len(weights) == len(SELECT_KEYS_SEG), "--weights needs 2 values (DICE,areaMAE)"
    rank_weights = tuple(float(x) for x in args.rank_weights.split(",")) if args.rank_weights else None
    assert rank_weights is None or len(rank_weights) == len(RANK_KEYS), \
        f"--rank-weights needs {len(RANK_KEYS)} values {RANK_KEYS}"

    series = _load_grid(args.runs, debug=args.debug)
    if not args.no_lpips_fallback:
        patched = inject_reeval_lpips(series, args.runs, weights)
        if patched:
            print(f"[LPIPS fallback] backfilled reeval_loo LPIPS for {len(patched)} config(s) whose "
                  f"training-curve LPIPS was missing:")
            for (C, S), ep, lp in patched:
                print(f"    {C}x{S}  (Stage-A ep {ep})  LPIPS={lp:.3f}")
    # Stage A: checkpoint per config by DICE+areaMAE. Stage B: full composite (adds PSNR/SSIM/LPIPS).
    stageA, stageB, _, rank_keys_used = two_stage_select(
        series, seg_weights=weights, rank_weights=rank_weights, rank_on_common=args.rank_on_common)
    best = stageA                                         # heatmap cells read at Stage-A best epoch
    total = len(GRID_CHANNELS) * len(GRID_SPATIAL)
    best_cell = max(stageB, key=lambda k: stageB[k]["composite"]) if stageB else None

    # ---- report to the user (stdout ONLY): Stage-A checkpoints, then Stage-B ranking ----
    print(f"\n[Stage A] {len(stageA)}/{total} configs -- best checkpoint by DICE+areaMAE "
          f"(weights {weights}). Every panel value is read at this epoch.")
    hdr = f"{'config (C x S)':>14} | {'best_ep':>7} | {'DICE':>6} {'areaMAE':>7} | " \
          f"{'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}"
    print(hdr); print("-" * len(hdr))
    for C in GRID_CHANNELS:
        for S in GRID_SPATIAL:
            if (C, S) not in stageA:
                continue
            ep = stageA[(C, S)]["best_epoch"]; r = series[(C, S)][ep]
            print(f"{C:>4} x {S:<3}{'':>4} | {ep:>7} | "
                  f"{r.get('DICE', float('nan')):>6.3f} {r.get('areaMAE', float('nan')):>7.3f} | "
                  f"{r.get('PSNR', float('nan')):>6.2f} {r.get('SSIM', float('nan')):>6.3f} "
                  f"{r.get('LPIPS', float('nan')):>6.3f}")

    # transparency: never silently drop a config from the ranking. A config is absent from Stage B
    # iff some RANK_KEYS metric is missing/NaN at its Stage-A checkpoint -> say which, loudly.
    dropped = [c for c in stageA if c not in stageB]
    if dropped:
        print(f"\n[WARNING] {len(dropped)} config(s) EXCLUDED from the Stage-B ranking because a "
              f"ranking metric is missing at their Stage-A checkpoint (composite needs all of "
              f"{RANK_KEYS}):")
        for C in GRID_CHANNELS:
            for S in GRID_SPATIAL:
                if (C, S) not in dropped:
                    continue
                r = series[(C, S)][stageA[(C, S)]["best_epoch"]]
                miss = [k for k in RANK_KEYS
                        if k not in r or (isinstance(r[k], float) and np.isnan(r[k]))]
                print(f"          {C:>4} x {S:<3} (ep {stageA[(C, S)]['best_epoch']}): missing {miss}")
        print("          -> recompute those metrics, or re-run with --rank-on-common to rank ALL "
              "configs on the metric subset present everywhere.")

    print(f"\n[Stage B] configs ranked by the full composite over {rank_keys_used} "
          f"(weights {rank_weights or 'equal'}), at each config's Stage-A checkpoint:")
    print(f"{'rank':>4}  {'config':>10}  {'ep':>3}  {'composite':>9}")
    for i, (cell, e) in enumerate(sorted(stageB.items(), key=lambda kv: -kv[1]["composite"]), 1):
        tag = "  <-- BEST (red ring)" if cell == best_cell else ""
        print(f"{i:>4}  {cell[0]:>4} x {cell[1]:<3}  {e['best_epoch']:>3}  {e['composite']:>9.3f}{tag}")
    if best_cell is not None:
        print(f"\n[result] OVERALL BEST config: {best_cell[0]} ch x {best_cell[1]} px "
              f"(Stage-A epoch {stageB[best_cell]['best_epoch']}, Stage-B composite "
              f"{stageB[best_cell]['composite']:.3f}) -> ringed in every panel.\n")

    # ---- dump Stage-A + Stage-B tables to CSV (one row per config) ----
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    rank_of = {cell: i for i, (cell, _) in
               enumerate(sorted(stageB.items(), key=lambda kv: -kv[1]["composite"]), 1)}
    csv_path = args.csv or os.path.join(outdir, "latent_grid_ranking.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["latent_channels", "latent_grid", "run_name", "stageA_best_epoch",
                    "DICE", "areaMAE", "PSNR", "SSIM", "LPIPS",
                    "in_stageB", "stageB_rank", "stageB_composite", "is_best"])
        for C in GRID_CHANNELS:
            for S in GRID_SPATIAL:
                if (C, S) not in stageA:
                    continue
                ep = stageA[(C, S)]["best_epoch"]; r = series[(C, S)][ep]
                inB = (C, S) in stageB
                w.writerow([
                    C, S, grid_run_name(C, S), ep,
                    f"{r.get('DICE', float('nan')):.4f}", f"{r.get('areaMAE', float('nan')):.4f}",
                    f"{r.get('PSNR', float('nan')):.4f}", f"{r.get('SSIM', float('nan')):.4f}",
                    f"{r.get('LPIPS', float('nan')):.4f}",
                    int(inB), rank_of.get((C, S), ""),
                    f"{stageB[(C, S)]['composite']:.4f}" if inB else "",
                    int((C, S) == best_cell)])
    print("wrote", csv_path)

    # ---- the figure ----
    plt.rcParams.update({"font.size": 10})
    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(2.75 * n, 3.5), squeeze=False)
    for ax, (metric, _lbl, lower, fmt) in zip(axes[0], PANELS):
        _panel(ax, metric, lower, fmt, best, series, best_cell)
    axes[0][0].set_ylabel(r"latent channels $C$", fontsize=9.5)

    # legend: blue = best cell for THAT metric; red = overall best config (same cell in all panels)
    handles = [plt.Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor="#1f77b4",
                          markeredgewidth=2.2, markersize=12, linestyle="none", label="best per metric"),
               plt.Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor="#d62728",
                          markeredgewidth=2.2, markersize=12, linestyle="none", label="best overall (subjective judgment)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    # no in-figure title (goes in the LaTeX \caption); rings mark the best cells
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "latent_grid_metric_row." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
