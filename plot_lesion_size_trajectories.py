"""Modern per-eye GA-area TRAJECTORY figure for the GAP-INR paper (test set).

One clean panel per test eye + a prominent AVERAGE panel, showing how GAP-INR tracks GA (lesion) size
over time vs the ground truth, with reconstructed / held-out / interpolated / extrapolated points
colour-coded to MATCH the GAP-INR timeline figure (blue=observed recon, gold=hold-out, teal=new
interpolation, purple=new extrapolation). Ground truth is a black line with filled dots; the model's
continuous estimate is a smooth crimson line. A shaded band marks the extrapolated (future) region.

Inputs:
  --csv      lesion_areas{label}_epoch_{N}.csv  (build_model): cols Patient_Eye, Set
             ({split}_opt | {split}_eval), Weeks, GT_Area_mm2, Pred_Area_mm2, Dice.
  --new-csv  OPTIONAL new-time-point predictions: cols Patient_Eye, Weeks, Pred_Area_mm2,
             Kind ({interp|extrap}).

    python plot_lesion_size_trajectories.py --csv <lesion_areas.csv> --split test [--new-csv <new.csv>]

Output: <out_dir>/lesion_size_trajectories.{png,pdf}
"""
import argparse, math, os
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
# classical comparators -- CVD-safe distinct hues, point+line. Computed from ALL observed GT visits
# (the full patient history). Cubic-spline is drawn but MASKED to the visible y-band (see _panel) so
# its extrapolation blow-up never distorts the scale.
CLASSICAL = [("Linear",       "#2a78d6", (0, (5, 2))),
             ("Cubic-spline", "#009e73", (0, (1, 1.6))),
             ("Copy-forward", "#6b7280", (0, (3, 2, 1, 2)))]


def _classical_area(w_obs, a_obs, w_q, method):
    """Predict GA area at query weeks w_q from ALL observed GT visits (w_obs, a_obs) = the FULL patient
    history. The computation is regime-aware, matching how the numbers are scored:

      INTERPOLATION  (query week within the observed range) is computed BETWEEN THE TWO BRACKETING
                     existing visits -- linear: straight interpolation of the two neighbours
                     (``np.interp``); cubic: the natural cubic spline through all observed visits.
      EXTRAPOLATION  (query week beyond the LAST observed visit) uses the WHOLE history -- linear: ONE
                     straight line, the full-history least-squares SLOPE anchored at the last observed
                     visit (so it is a single straight ray with no kink and continuous with the
                     observed trajectory); cubic: the same spline extended past the last knot.
      Copy-forward   = last observed area (no progression). Areas clamped >= 0 (GA cannot be negative).
    """
    w_obs = np.asarray(w_obs, float); a_obs = np.asarray(a_obs, float); w_q = np.asarray(w_q, float)
    last = w_obs.max()
    if method == "Copy-forward" or len(w_obs) < 2:
        return np.clip(np.full_like(w_q, a_obs[-1]), 0.0, None)
    if method == "Cubic-spline":
        from scipy.interpolate import CubicSpline
        return np.clip(CubicSpline(w_obs, a_obs)(w_q), 0.0, None)          # full history: interp + extrap
    if method == "Linear":
        slope = np.polyfit(w_obs, a_obs, 1)[0]                             # full-history least-squares slope
        y = np.where(w_q <= last + 1e-9,
                     np.interp(w_q, w_obs, a_obs),                         # interp: between bracketing visits
                     a_obs[-1] + slope * (w_q - last))                     # extrap: single straight ray, full-history slope
        return np.clip(y, 0.0, None)
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
    w, gt, pr, ho = rec["weeks"], rec["gt"], rec["pred"], rec["heldout"]
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
    # y-limits fixed by the STABLE curves (GT, Ours, Linear, copy-forward). Cubic-spline is then drawn
    # but MASKED to this band -> it shows where it is sensible and just stops (NaN) where it explodes,
    # so it never distorts the scale and leaves no near-vertical exit streak.
    yv = list(gt) + list(pa) + list(_classical_area(w, gt, pw[o], "Linear")) + [float(gt[-1])]
    lo, hi = min(yv), max(yv); pad = 0.15 * (hi - lo + 1e-6)
    ylo, yhi = lo - pad, hi + pad
    # classical comparators (behind): interpolate between observed visits, extrapolate from full history
    for name, col, dash in CLASSICAL:
        qa = _classical_area(w, gt, pw[o], name)
        if name == "Cubic-spline":
            qa = np.where((qa >= ylo) & (qa <= yhi), qa, np.nan)      # keep the y-scale sane
        ax.plot(pw[o], qa, ls=dash, color=col, lw=1.3, alpha=0.9, zorder=2)
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


def _avg_panel(ax, per_eye):
    maxv = max(len(r["weeks"]) for r in per_eye.values())
    gtM, prM, wM = [], [], []
    for i in range(maxv):
        gts = [r["gt"][i] for r in per_eye.values() if len(r["weeks"]) > i]
        prs = [r["pred"][i] for r in per_eye.values() if len(r["weeks"]) > i]
        wks = [r["weeks"][i] for r in per_eye.values() if len(r["weeks"]) > i]
        gtM.append((np.mean(gts), np.std(gts, ddof=1) / math.sqrt(len(gts)) if len(gts) > 1 else 0))
        prM.append((np.mean(prs), np.std(prs, ddof=1) / math.sqrt(len(prs)) if len(prs) > 1 else 0))
        wM.append(np.mean(wks))
    wM = np.asarray(wM); gm, gse = np.array(gtM).T; pm, pse = np.array(prM).T
    # No SE band here: the between-eye lesion-size variance is huge (0.7-13 mm^2) and would swamp the
    # mean-curve DIFFERENCES. Show mean curves only, tight y-axis -> Ours-vs-GT-vs-baselines is visible.
    # fit each comparator on the cohort-mean HISTORY (all but the last visit) and extrapolate over wM
    classical = {name: _classical_area(wM[:-1], gm[:-1], wM, name) for name, _, _ in CLASSICAL}
    for name, col, dash in CLASSICAL:
        ax.plot(wM, classical[name], ls=dash, color=col, lw=1.6, alpha=0.95, zorder=2, marker="o", ms=4, label=name)
    ax.plot(wM, pm, "-", color=C_PRED, lw=5.0, alpha=0.16, zorder=3, solid_capstyle="round")
    ax.plot(wM, pm, "-", color=C_PRED, lw=2.4, zorder=4, label="Ours")
    ax.scatter(wM, pm, s=40, color=C_PRED, edgecolor="white", linewidths=0.8, zorder=5)
    ax.plot(wM, gm, "-", color=C_GT, lw=2.2, zorder=5, label="Ground truth")
    ax.scatter(wM, gm, s=40, color=C_GT, edgecolor="white", linewidths=0.8, zorder=6)
    ax.set_title("Cohort average (n=%d)" % len(per_eye), fontsize=10, fontweight="bold", color=C_GT)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right", ncol=1)   # empty region (curves rise L->R)
    yv = list(gm) + list(pm) + [v for a in classical.values() for v in a]
    lo, hi = min(yv), max(yv); pad = 0.12 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.margins(x=0.06); _despine(ax)


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
    ] + [Line2D([0], [0], color=col, ls=dash, lw=1.4, label=name) for name, col, dash in CLASSICAL]
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
