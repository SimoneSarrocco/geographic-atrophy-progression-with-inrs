#!/usr/bin/env python3
"""
Round-2 per-epoch validation curves (publication-ready).

For every config in the two latent sweeps it reads the per-epoch held-out
validation metrics and plots score-vs-epoch, one line per configuration, with
the BEST CHECKPOINT (epoch of highest mean held-out DICE -- the rule
build_atlas uses to save checkpoint_best.pth) marked with a star.

Extraction is byte-for-byte consistent with build_atlas's checkpoint rule and
summarize_eval (the trusted paper reader), verified against the writing code:

  * DICE / SSIM  per epoch = mean over hold-out POSITIONS of (mean over eyes of the
    first-modality value). Read only from the held-out leave-one-out dirs
    'val_eval_holdout_V{p}/' and 'val_eval_holdout_last/'. This reconstructs
    build_atlas's  _last_val_eval_dice  (build_atlas.py:947) EXACTLY, so the DICE
    curve's argmax IS the epoch checkpoint_best.pth was saved at (build_atlas.py:979).
    Deliberately EXCLUDES the observed-visit TTO sets 'val_opt_*' (trivially easy)
    and the pooled aggregate 'val_eval_loo_avg' (would inflate the position mean).
  * lesion-MAE  per epoch = mean |Pred_Area_mm2 - GT_Area_mm2| over the lesion CSV
    rows whose Set starts with 'val_eval' (summarize_eval.py:216).

  * best checkpoint = argmax over epochs of the DICE curve above (== checkpoint_best).

Layout: 2 sweep rows (channels @ 32x32 | spatial @ 64ch) x 3 metric columns
        (DICE, SSIM, lesion-area MAE). Colour encodes latent size
        (sequential), so the capacity ordering is legible at a glance.

Output: ablations/figures/round2_epoch_curves.{pdf,png}

Run on the node that holds ablations/runs/.  Override the run root with --runs.
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Verified metric extraction lives in ONE place (unit-tested + cross-checked vs raw files).
from ablation_metrics import epoch_series as _epoch_series

# ----- the two sweeps: run-name -> swept value (label) -------------------------
CHAN = [("r2_a9_chan16", 16), ("r2_a9_chan32", 32), ("r2_a9_base", 64),
        ("r2_a9_chan128", 128), ("r2_a9_chan256", 256), ("r2_a9_chan512", 512)]
SPAT = [("r2_a9_latent8", 8), ("r2_a9_latent16", 16), ("r2_a9_base", 32),
        ("r2_a9_latent64", 64), ("r2_a9_latent128", 128)]

METRICS = [("DICE", r"DICE $\uparrow$", False),
           ("SSIM", r"SSIM $\uparrow$", False),
           ("areaMAE", r"lesion-area MAE [mm$^2$] $\downarrow$", True)]


# ----------------------------------------------------------------------- plotting
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"),
                    help="root containing <config_name>/ run dirs")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 120,
    })

    sweeps = [("Latent channels", CHAN, "ch", plt.cm.viridis),
              ("Latent grid", SPAT, "px", plt.cm.plasma)]

    fig, axes = plt.subplots(2, len(METRICS), figsize=(11.4, 6.6), sharex=True)

    for row, (title, sweep, unit, cmap) in enumerate(sweeps):
        colors = cmap(np.linspace(0.12, 0.88, len(sweep)))
        loaded = []
        for (name, val), c in zip(sweep, colors):
            rd = os.path.join(args.runs, name)
            series = _epoch_series(rd) if os.path.isdir(rd) else {}
            if not series:
                # Self-diagnose: show which metric dirs DO exist so a naming mismatch is obvious.
                found = sorted({os.path.basename(os.path.dirname(p)) for p in
                                glob.glob(os.path.join(rd, "**", "*_metrics_ep=*.json"), recursive=True)})
                print(f"[warn] no held-out val data for {name} ({rd}); "
                      f"metric dirs present: {found or 'none'}")
            loaded.append((name, val, c, series))

        for col, (mkey, mlabel, lower_better) in enumerate(METRICS):
            ax = axes[row, col]
            for name, val, c, series in loaded:
                eps = sorted(e for e in series if mkey in series[e])
                if not eps:
                    continue
                ys = [series[e][mkey] for e in eps]
                ax.plot(eps, ys, "-", color=c, lw=1.6, alpha=0.9, zorder=2)
                # highlight this config's OWN best value for THIS metric
                # (max for higher-better DICE/SSIM, min for lower-better lesion-area MAE).
                best_e = (min if lower_better else max)(eps, key=lambda e: series[e][mkey])
                ax.plot(best_e, series[best_e][mkey], "*", ms=13, color=c,
                        mec="white", mew=0.8, zorder=5)
            if row == 0:                                   # metric name as a bold column header
                ax.set_title(mlabel, fontsize=13, fontweight="bold", pad=10)
            if col == 0:                                   # sweep name as a rotated row header
                ax.annotate(title, xy=(0, 0.5), xytext=(-46, 0),
                            xycoords="axes fraction", textcoords="offset points",
                            rotation=90, ha="center", va="center",
                            fontsize=12, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Epoch")
            ax.margins(x=0.02)

        # per-row legend mapping colour -> swept value, placed JUST OUTSIDE the right of the
        # last panel so it is large and clear yet never overlaps any curve.
        handles = [Line2D([], [], color=c, lw=2.8, label=f"{val} {unit}")
                   for (name, val), c in zip(sweep, colors)]
        handles.append(Line2D([], [], color="0.3", marker="*", ms=13, lw=0,
                              mec="white", label="best value"))
        leg = axes[row, -1].legend(handles=handles, fontsize=10, ncol=1,
                                   loc="center left", bbox_to_anchor=(1.03, 0.5),
                                   frameon=True, fancybox=True, framealpha=0.95,
                                   edgecolor="0.7", handlelength=1.8,
                                   labelspacing=0.6, borderpad=0.7)
        leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, "round2_epoch_curves." + ext)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
