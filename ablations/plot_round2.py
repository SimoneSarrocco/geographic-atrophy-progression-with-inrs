#!/usr/bin/env python3
"""
Round-2 latent-capacity ablation figure (publication-ready).

Two independent one-knob sweeps that cross at the base latent [64, 32, 32]:
  * CHANNEL sweep  : channels in {16,32,64,128,256,512}, spatial FIXED 32x32
  * SPATIAL sweep  : grid side in {8,16,32,64,128},      channels FIXED 64

Renders a (metrics x sweeps) grid of line plots with +/-1 SD error bars, log2 x-axis,
the shared anchor [64,32,32] drawn as a hollow marker and the selected config (chan128)
starred.  Numbers below are the best-checkpoint reeval (holdout='last', per-eye SD).

Output: ablations/figures/round2_latent_sweep.{pdf,png}

NOTE: areaMAE per-eye SD was not in the ranked summary; set to None -> no error bar.
      To add it, read omega ... per-eye CSV and fill AREA_SD.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- data
# x = channels (spatial fixed 32x32)
CH_X      = [16,    32,    64,    128,   256,   512]
CH = {
    "DICE":   ([0.887, 0.881, 0.875, 0.875, 0.883, 0.883], [0.018,0.024,0.023,0.019,0.022,0.020]),
    "SSIM":   ([0.526, 0.525, 0.556, 0.653, 0.607, 0.599], [0.041,0.040,0.037,0.037,0.031,0.039]),
    "LPIPS":  ([0.429, 0.389, 0.457, 0.323, 0.363, 0.337], [0.032,0.022,0.029,0.032,0.027,0.026]),
    "areaMAE":([0.275, 0.254, 0.237, 0.251, 0.275, 0.272], None),
}
# x = spatial grid side (channels fixed 64)
SP_X      = [8,     16,    32,    64,    128]
SP = {
    "DICE":   ([0.865, 0.874, 0.875, 0.874, 0.875], [0.020,0.021,0.023,0.029,0.031]),
    "SSIM":   ([0.452, 0.550, 0.556, 0.595, 0.605], [0.031,0.035,0.037,0.033,0.041]),
    "LPIPS":  ([0.758, 0.514, 0.457, 0.327, 0.272], [0.035,0.031,0.029,0.023,0.027]),
    "areaMAE":([0.341, 0.211, 0.237, 0.337, 0.359], None),
}

# metric -> (row label, arrow direction)
METRICS = [("DICE", r"DICE $\uparrow$"),
           ("SSIM", r"SSIM $\uparrow$"),
           ("LPIPS", r"LPIPS $\downarrow$"),
           ("areaMAE", r"lesion-area MAE [mm$^2$] $\downarrow$")]

ANCHOR_CH, ANCHOR_SP = 64, 32   # the shared base [64,32,32]
STAR_CH = 128                   # selected config chan128

plt.rcParams.update({"font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

fig, axes = plt.subplots(len(METRICS), 2, figsize=(6.6, 8.2), sharex="col")

def _panel(ax, X, series, anchor, star=None):
    mu, sd = series
    ax.errorbar(X, mu, yerr=sd, marker="o", ms=4, lw=1.4, capsize=2.5,
                color="#1f4e79", ecolor="#9bb8d3", zorder=3)
    # hollow ring on the shared anchor [64,32,32]
    ai = X.index(anchor)
    ax.plot(anchor, mu[ai], "o", ms=10, mfc="none", mec="#444", mew=1.4, zorder=4)
    if star is not None and star in X:
        si = X.index(star)
        ax.plot(star, mu[si], "*", ms=15, color="#c0392b", zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(X); ax.set_xticklabels([str(v) for v in X])
    ax.margins(x=0.08)

for r, (key, ylab) in enumerate(METRICS):
    _panel(axes[r, 0], CH_X, CH[key], ANCHOR_CH, star=STAR_CH)
    _panel(axes[r, 1], SP_X, SP[key], ANCHOR_SP, star=None)
    axes[r, 0].set_ylabel(ylab)

axes[0, 0].set_title("Latent channels  (spatial fixed $32{\\times}32$)")
axes[0, 1].set_title("Latent grid  (channels fixed $64$)")
axes[-1, 0].set_xlabel("channels  $C$")
axes[-1, 1].set_xlabel("grid side  $H{=}W$")

# shared legend
from matplotlib.lines import Line2D
handles = [Line2D([], [], color="#1f4e79", marker="o", ms=4, lw=1.4, label="sweep ($\\pm$1 SD)"),
           Line2D([], [], color="#444", marker="o", ms=9, mfc="none", lw=0, label="base [64,32,32]"),
           Line2D([], [], color="#c0392b", marker="*", ms=12, lw=0, label="selected (128 ch)")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(rect=(0, 0, 1, 0.97))

outdir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    p = os.path.join(outdir, "round2_latent_sweep." + ext)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print("wrote", p)
