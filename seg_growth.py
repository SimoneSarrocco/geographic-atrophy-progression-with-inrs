"""
Segmentation-growth visualisation for GA progression.

Given an eye's PREDICTED GA masks at a sequence of timepoints (observed +
interpolated + extrapolated), render two publication figures that make the
spatial progression legible:

  1. Contour overlay  -- the GA boundary at each week drawn on one panel, colored
     by week (a single colormap). Shows the lesion expanding outward and in which
     directions ("growth rings").
  2. Onset / "volume" map -- per pixel, the EARLIEST week at which GA is predicted
     there, as a single color-coded image (the temporal stack collapsed to one
     map). This is the compact "volume" view: warm = late onset (recent growth),
     cool = early/baseline atrophy.

Pure function (no GAP-INR deps) so it can be unit-tested on any checkpoint.
"""
import os
import numpy as np


def save_seg_growth_figure(weeks, masks, out_path, faf_bg=None, title="", cmap="viridis",
                           area_per_px_mm2=None):
    """Three-panel publication figure of predicted GA progression for one eye:

      (1) GA boundary contour at each week, colored by week ("growth rings");
      (2) per-pixel onset map: earliest week each pixel becomes GA (compact 'volume');
      (3) GA-area-vs-week curve quantifying the rate alongside the spatial views.

    weeks : list[float] ascending (weeks-from-baseline of each frame)
    masks : list[np.ndarray(H,W) bool/0-1], predicted GA mask per week
    faf_bg: optional (H,W) grayscale in [0,1] drawn under the overlays (e.g. baseline FAF)
    area_per_px_mm2: mm^2 per pixel; if given, panel (3) is in mm^2 (else pixel count).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    weeks = [float(w) for w in weeks]
    masks = [np.asarray(m).astype(bool) for m in masks]
    H, W = masks[0].shape
    norm = Normalize(vmin=min(weeks), vmax=max(weeks) if max(weeks) > min(weeks) else min(weeks) + 1)
    cmapf = plt.get_cmap(cmap)

    # width_ratios: two square image panels + a wider curve panel
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2),
                           gridspec_kw={"width_ratios": [1, 1, 1.15]})

    # ---- (1) contour overlay ("growth rings") ----
    ax[0].imshow(faf_bg if faf_bg is not None else np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
    for w, m in zip(weeks, masks):
        if m.any():
            ax[0].contour(m.astype(float), levels=[0.5], colors=[cmapf(norm(w))], linewidths=2.0)
    ax[0].set_title("GA boundary over time"); ax[0].axis("off")

    # ---- (2) onset / volume map: earliest week each pixel is GA ----
    onset = np.full((H, W), np.nan, dtype=float)
    for w, m in zip(weeks, masks):                 # weeks ascending -> first hit wins
        onset[m & np.isnan(onset)] = w
    if faf_bg is not None:
        ax[1].imshow(faf_bg, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(np.ma.masked_invalid(onset), cmap=cmap, norm=norm, alpha=0.9)
    ax[1].set_title("Predicted GA onset week (per pixel)"); ax[1].axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmapf); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[:2], fraction=0.046, pad=0.02, location="bottom")
    cbar.set_label("Weeks from baseline")

    # ---- (3) GA area vs week ----
    areas = np.array([float(m.sum()) for m in masks])
    unit = "mm²"
    if area_per_px_mm2 is not None:
        areas = areas * float(area_per_px_mm2)
    else:
        unit = "pixels"
    pts = ax[2].scatter(weeks, areas, c=weeks, cmap=cmap, norm=norm, s=55, zorder=3,
                        edgecolors="k", linewidths=0.6)
    ax[2].plot(weeks, areas, "-", color="0.4", lw=1.6, zorder=2)
    if len(weeks) >= 2:
        slope = float(np.polyfit(weeks, areas, 1)[0])
        ax[2].annotate(f"rate ≈ {slope:+.3f} {unit}/wk", xy=(0.04, 0.92),
                       xycoords="axes fraction", fontsize=10,
                       bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax[2].set_xlabel("Weeks from baseline"); ax[2].set_ylabel(f"Predicted GA area ({unit})")
    ax[2].set_title("GA area over time"); ax[2].grid(alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # quick self-test on a GAP-INR checkpoint: decode one eye's GA mask across weeks.
    import argparse, glob, torch
    from models.inr_decoder import INR_Decoder
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eye", type=int, default=0)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--out", default="/tmp/seg_growth_test.png")
    a = p.parse_args()
    d = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = d["args"]; idec = args["inr_decoder"]
    dec = INR_Decoder(args, "cpu"); dec.load_state_dict(d["inr_decoder"]); dec.eval()
    sr = int(sum(idec["out_dim"][:-1]))
    cons = args["dataset"]["constraints"]["weeks_from_baseline"]; wmin, wmax = cons["min"], cons["max"]
    r = a.res
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, r), torch.linspace(-1, 1, r), indexing="ij")
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], -1); N = coords.shape[0]
    idcs = torch.zeros(N, 1, dtype=torch.int32); z = d["latents"][a.eye:a.eye + 1]
    weeks = [0, 12, 24, 36, 48, 60, 72]
    masks, recon0 = [], None
    with torch.no_grad():
        for wk in weeks:
            tnorm = 2.0 * (wk - wmin) / max(wmax - wmin, 1e-6) - 1.0
            out = dec(coords, z, torch.full((N, 1), float(tnorm)), idcs_df=idcs, time_vals=None)
            seg = torch.softmax(out[:, sr:sr + 2], -1)[:, 1].reshape(r, r) > 0.5
            masks.append(seg.numpy())
            if recon0 is None:
                recon0 = out[:, 0].reshape(r, r).clamp(0, 1).numpy()
    save_seg_growth_figure(weeks, masks, a.out, faf_bg=recon0,
                           title=f"Eye {a.eye} predicted GA growth")
    areas = [int(m.sum()) for m in masks]
    print("weeks:", weeks, "\nGA pixel counts:", areas, "\nsaved:", a.out)
