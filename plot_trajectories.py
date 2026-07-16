"""Multi-patient GA-area trajectory figures for the GAP-INR paper.

Reads a ``lesion_areas{label}_epoch_{N}.csv`` produced by build_atlas
(analyze_and_plot_lesion_sizes) and renders, for one split (test by default):

  1. OVERLAY  -- all eyes on one GA-area(mm^2)-vs-weeks axes, GT = markers,
     prediction = line, eyes COLORED BY THEIR PROGRESSION RATE (GT slope), so a
     single figure shows the model tracks both slow and fast progressors.
  2. SMALL-MULTIPLES -- one panel per eye (GT vs predicted), shared axes.

CSV columns: Patient_Eye, Set ({split}_opt | {split}_eval), Weeks, GT_Area_mm2,
Pred_Area_mm2, Dice. "_eval" rows are the HELD-OUT predictions (the real forecast);
"_opt" rows are the fit/observed visits. Per (eye, week) we prefer the held-out
prediction when available.

Pure pandas/matplotlib -- no GAP-INR/torch deps, no GPU. Re-runnable on any epoch's CSV.
"""
import argparse
import os

import numpy as np
import pandas as pd


def _per_eye(df, split):
    """-> dict eye_id -> {'weeks','gt','pred','heldout'(bool per pt),'slope'} sorted by week."""
    sets = [f"{split}_opt", f"{split}_eval"]
    sub = df[df["Set"].isin(sets)].copy()
    if sub.empty:  # fall back: any set whose name starts with split
        sub = df[df["Set"].astype(str).str.startswith(split)].copy()
    out = {}
    for eye, g in sub.groupby("Patient_Eye"):
        weeks = sorted(g["Weeks"].unique())
        rec = {"weeks": [], "gt": [], "pred": [], "heldout": []}
        for w in weeks:
            rows = g[g["Weeks"] == w]
            ev = rows[rows["Set"] == f"{split}_eval"]
            op = rows[rows["Set"] == f"{split}_opt"]
            chosen = ev if not ev.empty else op
            gt = float(chosen["GT_Area_mm2"].iloc[0]) if pd.notna(chosen["GT_Area_mm2"].iloc[0]) else np.nan
            pr = float(chosen["Pred_Area_mm2"].iloc[0]) if pd.notna(chosen["Pred_Area_mm2"].iloc[0]) else np.nan
            rec["weeks"].append(float(w)); rec["gt"].append(gt); rec["pred"].append(pr)
            rec["heldout"].append(not ev.empty)
        # GT progression rate (mm^2/week) via least-squares slope over the real GT points.
        ww = np.asarray(rec["weeks"]); gg = np.asarray(rec["gt"])
        m = np.isfinite(ww) & np.isfinite(gg)
        rec["slope"] = float(np.polyfit(ww[m], gg[m], 1)[0]) if m.sum() >= 2 else 0.0
        out[eye] = rec
    return out


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 220, "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
    })
    return plt


def plot_overlay(eyes, out_path, title=""):
    plt = _style()
    from matplotlib import cm
    from matplotlib.colors import Normalize
    slopes = np.array([e["slope"] for e in eyes.values()])
    norm = Normalize(vmin=float(slopes.min()), vmax=float(max(slopes.max(), slopes.min() + 1e-6)))
    cmap = plt.get_cmap("coolwarm")   # blue = slow progressor, red = fast
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for eye, rec in sorted(eyes.items(), key=lambda kv: kv[1]["slope"]):
        c = cmap(norm(rec["slope"]))
        w = np.asarray(rec["weeks"]); gt = np.asarray(rec["gt"]); pr = np.asarray(rec["pred"])
        ax.plot(w, pr, "-", color=c, lw=2.0, zorder=3,
                label=f"{eye}  ({rec['slope']:+.2f} mm²/wk)")
        ax.plot(w, gt, "o", color=c, mfc="white", mec=c, mew=1.4, ms=6, zorder=4)
    # legend proxies for line styles
    ax.plot([], [], "-", color="0.3", lw=2.0, label="— predicted")
    ax.plot([], [], "o", color="0.3", mfc="white", label="○ ground truth")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("GT progression rate (mm²/week)")
    ax.set_xlabel("Weeks from baseline"); ax.set_ylabel("GA area (mm²)")
    ax.set_title(title or "Predicted vs GT GA-area trajectories (colored by progression rate)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=3)
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def plot_small_multiples(eyes, out_path, title=""):
    plt = _style()
    items = sorted(eyes.items(), key=lambda kv: kv[1]["slope"], reverse=True)  # fast -> slow
    n = len(items); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    # shared y-range for comparability
    allv = [v for rec in eyes.values() for v in (rec["gt"] + rec["pred"]) if np.isfinite(v)]
    ymin, ymax = (min(allv), max(allv)) if allv else (0, 1)
    pad = 0.08 * (ymax - ymin + 1e-6)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    for k, (eye, rec) in enumerate(items):
        ax = axes[k // ncol][k % ncol]
        w = np.asarray(rec["weeks"]); gt = np.asarray(rec["gt"]); pr = np.asarray(rec["pred"])
        ho = np.asarray(rec["heldout"])
        ax.plot(w, gt, "--o", color="0.35", mfc="white", ms=5, lw=1.4, label="GT")
        ax.plot(w, pr, "-", color="#c0392b", lw=2.0, label="Pred")
        if ho.any():  # held-out existing visit prediction -- distinct (red star)
            ax.plot(w[ho], pr[ho], "*", color="#D32F2F", ms=14, mec="k", mew=0.5,
                    zorder=6, label="Pred (held-out visit)")
        ax.set_title(f"{eye}  ({rec['slope']:+.2f} mm²/wk)")
        ax.set_ylim(ymin - pad, ymax + pad)
        if k % ncol == 0: ax.set_ylabel("GA area (mm²)")
        if k // ncol == nrow - 1: ax.set_xlabel("Weeks")
    for k in range(n, nrow * ncol):  # hide empty cells
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title or "Per-patient GA-area trajectories (fast → slow)", y=1.06, fontweight="bold")
    fig.tight_layout(); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Multi-patient GA-area trajectory figures.")
    ap.add_argument("--csv", required=True, help="lesion_areas{label}_epoch_{N}.csv")
    ap.add_argument("--split", default="test", help="which split to plot (test|val|train)")
    ap.add_argument("--out_dir", default=None, help="output dir (default: alongside the CSV)")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    eyes = _per_eye(df, a.split)
    if not eyes:
        avail = sorted(df["Set"].astype(str).unique())
        raise SystemExit(f"No eyes for split '{a.split}'. Sets present in CSV: {avail}")
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.csv)), "trajectory_figures")
    tag = os.path.splitext(os.path.basename(a.csv))[0]
    p1 = plot_overlay(eyes, os.path.join(out_dir, f"{tag}_{a.split}_overlay.png"),
                      title=f"{a.split.upper()} GA-area trajectories — predicted vs GT (n={len(eyes)} eyes)")
    p2 = plot_small_multiples(eyes, os.path.join(out_dir, f"{tag}_{a.split}_grid.png"),
                              title=f"{a.split.upper()} per-patient GA-area trajectories")
    print(f"{len(eyes)} eyes ({a.split}); slopes mm²/wk: " +
          ", ".join(f"{k}={v['slope']:+.2f}" for k, v in sorted(eyes.items(), key=lambda kv: kv[1]['slope'])))
    print("saved:\n ", p1, "\n ", p2)


if __name__ == "__main__":
    main()
