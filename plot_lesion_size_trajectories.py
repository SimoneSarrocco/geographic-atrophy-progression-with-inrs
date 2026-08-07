"""Per-eye GA-area progression panel. This is the per-patient figure in the paper.

One panel per eye of the chosen split, laid out on a single row. Ground truth is a
black line with filled dots, the model's continuous estimate is a smooth crimson
line, and a shaded band marks the extrapolated region. Prediction types are told
apart by marker shape rather than by colour, so all predictions keep one colour.

    python plot_lesion_size_trajectories.py --csv <lesion_areas_*.csv> --split test \\
        [--holdout-dir DIR] [--new-csv NEW.csv] [--ncols N] [--out-dir DIR]

INPUT

  --csv          Required. A lesion_areas{label}_epoch_{N}.csv from build_model
                 (analyze_and_plot_lesion_sizes), in
                 runs/faf_ga/<run>/evaluation_*/lesion_analysis/. Columns:
                 Patient_Eye, Set ("<split>_opt" or "<split>_eval"), Weeks,
                 GT_Area_mm2, Pred_Area_mm2, Dice.

  --holdout-dir  Optional, but use it for a paper figure. A build_model
                 holdout_timeline_arrays/ directory with one <eye>.npz per eye.
                 With it, every existing visit is plotted as its leave-one-out
                 prediction: that visit held out, the latent fitted on the others.
                 Without it, the observed visits come from --csv, where they are
                 reconstructions of visits the latent already saw. Those look better
                 than the model's real forecasting ability.

  --new-csv      Optional predictions at times that have no ground truth. Columns:
                 Patient_Eye, Weeks, Pred_Area_mm2, Kind ("interp" or "extrap").

  --ncols        Panels per row. Default is all eyes on one row.
  --out-dir      Output directory. Default is alongside --csv.

OUTPUT

  <out-dir>/lesion_size_trajectories.png
  <out-dir>/lesion_size_trajectories.pdf

Uses pandas, numpy, scipy and matplotlib only. No torch, no checkpoint, no GPU.
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Publication palette (CVD-safe: black + red + blue + slate). Identity is by COLOUR, and OUR four
# prediction TYPES are distinguished by SHAPE only (not four hues) -> clean, not chaotic.
C_GT     = "#1a1a1a"   # ground truth (reference ink)
C_PRED   = "#d1495b"   # Ours (highlighted) -- ALL Our marks share this red
C_HOLDEDGE = "#7a1324"  # darker red rim for the hold-out star so it pops
C_RECON  = C_PRED; C_HOLD = C_PRED; C_INTERP = C_PRED; C_EXTRAP = C_PRED
# classical comparators -- CVD-safe distinct hues. All are fit on ALL observed GT visits (the full
# patient history) and drawn as ONE curve spanning both regimes.
#   Linear            : ONE ordinary least-squares line, np.polyfit(w, gt, 1) -- the `linear_regression`
#                       floor of the comparison table (_classical.py:39-47).
#   Cubic-spline      : ONE scipy CubicSpline through every observed visit, continued past the last
#                       knot. CLAMPED to the visible y-band (see _panel) so its blow-up never distorts
#                       the scale; the point where it leaves the band is flagged.
#   Copy-forward      : last observed area (no progression).
CLASSICAL = [("Linear",            "#2a78d6", (0, (5, 2))),
             ("Cubic-spline",      "#009e73", (0, (1, 1.6))),
             ("Copy-forward",      "#6b7280", (0, (3, 2, 1, 2)))]


def _classical_area(w_obs, a_obs, w_q, method):
    """Predict GA area at query weeks w_q from the support visits (w_obs, a_obs):

      Linear         ONE ordinary least-squares line, both coefficients from np.polyfit(w, a, 1),
                     valid over the whole time axis -- interpolation and extrapolation are the SAME
                     line, no regime switch. This is the `linear_regression` floor of the comparison
                     table (_classical.py:39-47), which likewise fits one OLS line over its support
                     and evaluates it at the target whether that target is interior or in the future.
      Cubic-spline   one scipy CubicSpline through all support visits (scipy defaults: not-a-knot,
                     extrapolate=True), evaluated at w_q; past the last knot this is the same spline continued (see below).
      Copy-forward   last observed area (no progression). Areas clamped >= 0 (GA cannot be negative)
                     for the regression line and Copy-forward; the spline is deliberately NOT clamped.

    Cubic-spline is returned RAW (no >=0 clamp), matching ImageFlowNet's `_cubic_spline_interp`, which
    calls scipy `CubicSpline` with the defaults (bc_type='not-a-knot', extrapolate=True) and does not
    clip. With only 4 visits not-a-knot degenerates to a SINGLE global cubic, so past the last knot the
    curve diverges as (t - t_last)^3 -- often downwards through zero. Clamping that to 0 here would
    disguise the divergence as a flat, plausible-looking floor; _panel clamps it to the plot band
    instead and marks the exit point.
    """
    w_obs = np.asarray(w_obs, float); a_obs = np.asarray(a_obs, float); w_q = np.asarray(w_q, float)
    if method == "Copy-forward" or len(w_obs) < 2:
        return np.clip(np.full_like(w_q, a_obs[-1]), 0.0, None)
    if method == "Cubic-spline":
        from scipy.interpolate import CubicSpline
        return CubicSpline(w_obs, a_obs)(w_q)                              # full history: interp + extrap, unclipped
    if method == "Linear":
        slope, intercept = np.polyfit(w_obs, a_obs, 1)                     # one OLS fit, no regime switch
        return np.clip(slope * w_q + intercept, 0.0, None)
    raise ValueError(method)


def _per_eye(df, split):
    sub = df[df["Set"].astype(str).str.startswith(split)].copy()
    out = {}
    for eye, g in sub.groupby("Patient_Eye"):
        rec = {"weeks": [], "gt": [], "pred": [], "heldout": []}
        for w in sorted(g["Weeks"].unique()):
            rows = g[g["Weeks"] == w]
            ev = rows[rows["Set"] == f"{split}_eval"]
            chosen = ev if not ev.empty else rows
            rec["weeks"].append(float(w)); rec["gt"].append(float(chosen["GT_Area_mm2"].iloc[0]))
            rec["pred"].append(float(chosen["Pred_Area_mm2"].iloc[0])); rec["heldout"].append(not ev.empty)
        out[eye] = {k: np.asarray(v) for k, v in rec.items()}
    return out


def _per_eye_holdout(holdout_dir, df, split):
    """Like _per_eye, but each existing visit's predicted area is the LEAVE-ONE-OUT prediction obtained
    when THAT visit was held out and the latent optimised on all the OTHER visits (the genuine per-visit
    Scenario-2 score) -- read from build_model' holdout_timeline_arrays/<eye>.npz. Weeks + GT come from
    the CSV; only the predictions are overridden. Visits are matched chronologically (both sorted by
    time) and cross-checked on GT area. Only the LAST visit is flagged held-out (=> star / forecast)."""
    base = _per_eye(df, split)
    out = {}
    for eye, rec in base.items():
        f = os.path.join(holdout_dir, f"{eye}.npz")
        pred = rec["pred"].copy()
        if os.path.exists(f):
            z = np.load(f)
            positions = sorted({int(k[1:k.index("_")]) for k in z.files})
            loo = [float(z[f"v{p}_pred_area"]) for p in positions if f"v{p}_pred_area" in z.files]
            if len(loo) == len(pred):                        # chronological 1:1 with the CSV visits
                pred = np.asarray(loo)
            else:
                print(f"[holdout] {eye}: {len(loo)} LOO preds vs {len(pred)} CSV visits -> keeping CSV preds")
        ho = np.zeros(len(pred), bool)
        if len(ho):
            ho[-1] = True                                    # last visit = the forecast target (star)
        out[eye] = {"weeks": rec["weeks"], "gt": rec["gt"], "pred": pred, "heldout": ho}
    return out


def _new_by_eye(new_csv):
    if not new_csv or not os.path.exists(new_csv):
        return {}
    nd = pd.read_csv(new_csv); out = {}
    for eye, g in nd.groupby("Patient_Eye"):
        d = {}
        for kind in ("interp", "extrap"):
            gk = g[g["Kind"].astype(str).str.lower() == kind].sort_values("Weeks")
            if not gk.empty:
                d[kind] = (gk["Weeks"].to_numpy(float), gk["Pred_Area_mm2"].to_numpy(float))
        out[eye] = d
    return out


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, length=0)


def _panel(ax, eye, rec, new, title_fs=10):
    w, gt, pr = rec["weeks"], rec["gt"], rec["pred"]
    # ALL observed visits are the patient history; classical comparators interpolate between existing
    # visits within this range and extrapolate the future points from the full history (see _classical_area).
    last = w.max() if len(w) else 0
    # light-pink shaded region marking the EXTRAPOLATION zone (everything after the last observed visit)
    allw = list(w) + [x for kind in ("interp", "extrap") if new and kind in new for x in new[kind][0]]
    if allw and max(allw) > last:
        ax.axvspan(last, max(allw) * 1.02, color="#e05a70", alpha=0.09, zorder=0)
        ax.axvline(last, color="0.72", lw=1.0, ls=(0, (4, 3)), zorder=1)
    # continuous model estimate (observed + held-out + new), sorted
    pw, pa = list(w), list(pr)
    for kind in ("interp", "extrap"):
        if new and kind in new:
            pw += list(new[kind][0]); pa += list(new[kind][1])
    pw = np.asarray(pw); pa = np.asarray(pa); o = np.argsort(pw)
    # Classical comparators, both fit on ALL observed GT visits (the full patient history) and drawn
    # as ONE curve across the whole panel -- interpolation and extrapolation come from the same fit:
    #   * Linear       -> one ordinary least-squares line, np.polyfit(w, gt, 1).
    #   * Cubic-spline -> one scipy CubicSpline through every observed visit (not-a-knot, extrapolate),
    #                     continued past the last knot. Being an interpolating spline it reproduces the
    #                     GT exactly at the observed weeks, so inside the observed range it rides on the
    #                     ground-truth line; it separates only at the midpoints and after the last visit.
    #   * Copy-forward -> flat at the last observed area.
    q_all = np.linspace(float(min(pw)), float(max(pw)), 240)
    # y-limits fixed by the STABLE curves (GT, Ours, the regression line, copy-forward), so the
    # diverging spline cannot blow up the scale.
    yv = list(gt) + list(pa) + list(_classical_area(w, gt, q_all, "Linear")) + [float(gt[-1])]
    lo, hi = min(yv), max(yv); pad = 0.15 * (hi - lo + 1e-6)
    ylo, yhi = lo - pad, hi + pad
    for name, col, dash in CLASSICAL:
        qa = _classical_area(w, gt, q_all, name)
        # The not-a-knot spline diverges past the last knot (see _classical_area). The curve is drawn up
        # to the FIRST sample that leaves the plot band and STOPS there, marked with a triangle on the
        # axis: everything beyond is off-scale, and continuing the clamped curve along the band edge
        # would draw a long horizontal line that no estimator produced.
        outside = (qa < ylo) | (qa > yhi)
        if outside.any():
            k = int(np.argmax(outside))
            yk = ylo if qa[k] < ylo else yhi
            ax.plot(np.append(q_all[:k], q_all[k]), np.append(qa[:k], yk),
                    ls=dash, color=col, lw=1.3, alpha=0.9, zorder=2)
            ax.plot([q_all[k]], [yk], ls="none", color=col, marker="v" if qa[k] < ylo else "^",
                    ms=6.5, mec="white", mew=0.7, zorder=7)
        else:
            ax.plot(q_all, qa, ls=dash, color=col, lw=1.3, alpha=0.9, zorder=2)
    # OUR estimate -- highlighted with a translucent halo + solid crimson line on top
    ax.plot(pw[o], pa[o], "-", color=C_PRED, lw=5.0, alpha=0.16, zorder=3, solid_capstyle="round")
    ax.plot(pw[o], pa[o], "-", color=C_PRED, lw=2.2, alpha=0.95, zorder=4)
    # GT trajectory
    ax.plot(w, gt, "-", color=C_GT, lw=1.7, zorder=5)
    ax.scatter(w, gt, s=42, facecolor=C_GT, edgecolor="white", linewidths=0.8, zorder=6)
    # predicted markers -- every existing visit is a leave-one-out (hold-out) prediction, one red dot
    ax.scatter(w, pr, s=42, marker="o", facecolor=C_RECON, edgecolor="white",
               linewidths=0.8, zorder=6)
    if new and "interp" in new:
        ax.scatter(*new["interp"], s=52, marker="D", facecolor=C_INTERP, edgecolor="white",
                   linewidths=0.8, zorder=6)
    if new and "extrap" in new:
        ax.scatter(*new["extrap"], s=60, marker="^", facecolor=C_EXTRAP, edgecolor="white",
                   linewidths=0.8, zorder=6)
    ax.set_title(eye, fontsize=title_fs, fontweight="bold", color=C_GT, pad=4)
    ax.set_ylim(ylo, yhi)
    ax.margins(x=0.06)
    _despine(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--holdout-dir", default=None,
                    help="build_model holdout_timeline_arrays dir. If given, each existing visit's "
                         "predicted area is the LEAVE-ONE-OUT prediction (that visit held out, latent "
                         "optimised on the others) instead of the in-sample reconstruction from --csv.")
    ap.add_argument("--new-csv", default=None)
    ap.add_argument("--ncols", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    plt.rcParams.update({"font.family": ["Nimbus Sans", "DejaVu Sans"], "axes.edgecolor": "0.35",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    df = pd.read_csv(args.csv)
    # per-visit predictions: leave-one-out (each visit held out) if --holdout-dir given, else --csv rows
    per_eye = _per_eye_holdout(args.holdout_dir, df, args.split) if args.holdout_dir else _per_eye(df, args.split)
    if not per_eye:
        raise SystemExit(f"no '{args.split}' data ({args.holdout_dir or args.csv})")
    new = _new_by_eye(args.new_csv)
    eyes = sorted(per_eye)

    # Layout: all test eyes on a SINGLE row (cohort-average panel removed) to minimise vertical space.
    ne = len(eyes)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(2.9 * ne + 0.3, 4.0))
    gs = GridSpec(1, ne, figure=fig, wspace=0.34, left=0.05, right=0.995, top=0.76, bottom=0.16)
    for k, eye in enumerate(eyes):
        _panel(fig.add_subplot(gs[0, k]), f"Test eye {k + 1}", per_eye[eye], new.get(eye))   # anonymised

    fig.supxlabel("weeks from baseline", fontsize=11, color=C_GT)
    fig.supylabel("GA area  [mm$^2$]", fontsize=11, color=C_GT)
    handles = [
        Line2D([0], [0], marker="o", color=C_GT, lw=1.8, label="Ground truth"),
        Line2D([0], [0], color=C_PRED, lw=2.6, label="Ours"),
        Line2D([0], [0], marker="o", color=C_RECON, ls="none", ms=8, mec="white", label="Ours: hold-out prediction"),
        Line2D([0], [0], marker="D", color=C_INTERP, ls="none", ms=8, mec="white", label="Ours: interpolation"),
        Line2D([0], [0], marker="^", color=C_EXTRAP, ls="none", ms=9, mec="white", label="Ours: extrapolation"),
    ] + [Line2D([0], [0], color=col, ls=dash, lw=1.4, label=name)
         for name, col, dash in CLASSICAL] + [
        Line2D([0], [0], marker="^", color=dict((n, c) for n, c, _ in CLASSICAL)["Cubic-spline"],
               ls="none", ms=7, mec="white", label="off-scale (clamped for display)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.995), columnspacing=1.4, handletextpad=0.5)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "lesion_size_trajectories.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "(%d eyes, single row)" % len(eyes))


if __name__ == "__main__":
    main()
