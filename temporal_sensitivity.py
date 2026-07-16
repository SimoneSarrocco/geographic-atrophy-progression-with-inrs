"""
temporal_sensitivity.py — "Is the model actually doing anything over time?"

The single most important sanity check for a longitudinal INR: hold an eye's
representation fixed and sweep ONLY the temporal condition (weeks). If the
predicted FAF / GA do not change, the time pathway is dead (the static-collapse
failure) and NO number of training epochs will help — you must change the design
(FiLM reaching the path, seg branch, smaller latent, monotonicity prior, ...).

This is deliberately cheap: on the TRAIN split the per-eye latents are restored
straight from the checkpoint, so NO test-time optimisation (TTA) is needed — it
isolates the decoder + conditioning pathway. On val/test the latents are first
fit on ALL visits (one TTA round, holdout='none') then probed.

For each eye we decode the predicted FAF + GA mask at a dense grid of weeks and
report, per eye:
  - area_slope_mm2_per_wk : linear-fit slope of predicted GA area vs week
  - area_range_mm2        : max - min predicted GA area over the sweep
  - area_rel_range        : area_range / mean area (scale-free)
  - seg_change_frac       : fraction of ever-GA pixels that flip membership over t
  - faf_temporal_std      : mean over pixels of the FAF std across t
  - verdict               : STATIC (collapsed) or RESPONSIVE

and a cohort verdict (fraction of static eyes + medians). Outputs a CSV, a JSON
summary, a per-eye area-vs-week small-multiples grid and a cohort overlay.

Usage:
  python temporal_sensitivity.py --checkpoint <run>/checkpoint_epoch_99.pth          # train, no TTA (fast)
  python temporal_sensitivity.py --checkpoint <...> --split test                     # test (fits latents first)
  python temporal_sensitivity.py --checkpoint <...> --horizon_weeks 96 --n_t 16      # longer / denser sweep
"""
import os
import json
import argparse
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from build_atlas import AtlasBuilder
from utils import generate_world_grid, typecheck_img
from data_loading.dataset import validate_splits


def parse_args():
    p = argparse.ArgumentParser(description="GAP-INR temporal-sensitivity / static-collapse diagnostic")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="Which eyes to probe. 'train' needs NO TTA (latents from checkpoint); "
                        "val/test fit latents on ALL visits first (one TTA round).")
    p.add_argument("--horizon_weeks", type=float, default=48.0,
                   help="Extra weeks to extrapolate beyond each eye's last observed visit.")
    p.add_argument("--n_t", type=int, default=12, help="Number of timepoints in the dense sweep.")
    p.add_argument("--epochs_val", type=int, default=None,
                   help="Override TTA epochs when --split is val/test (default: config value).")
    p.add_argument("--area_eps_mm2", type=float, default=0.02,
                   help="An eye is STATIC if its GA area range is below this AND seg_change_frac < frac_eps.")
    p.add_argument("--frac_eps", type=float, default=0.01,
                   help="Seg-change-fraction threshold for the STATIC flag.")
    p.add_argument("--output_dir", default=None, help="Override output dir (default: alongside checkpoint).")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    return p.parse_args()


def probe_eye(builder, split, sub_id, grid_coords, grid_shape, weeks):
    """Decode an eye across `weeks`. Returns (t_used, areas_mm2, masks, faf_stack, area_per_px)."""
    dec = builder.inr_decoder.get(split)
    df = builder.datasets[split].df
    tkey = builder._temporal_key
    sub_df = df[df["sub_id_int"] == sub_id].sort_values(tkey)
    if sub_df.empty:
        return None
    actual_rows = [r.to_dict() for _, r in sub_df.iterrows()]
    area_per_px = (float(actual_rows[0].get("ScaleXSlo", 1.0))
                   * float(actual_rows[0].get("ScaleYSlo", 1.0)))
    extrap = True  # the whole point is to probe beyond the observed range
    t_used, areas, masks, faf_stack = [], [], [], []
    for w in weeks:
        new_row = builder._get_interpolated_row_dict(actual_rows, float(w))
        try:
            vol = builder._reconstruct_visit(new_row, int(sub_id), grid_coords, grid_shape,
                                             split=split, allow_extrapolation=extrap)
        except Exception:
            continue
        pred_np = typecheck_img(vol)
        mask = pred_np[..., dec.sr_dims] > 0.5            # hard GA channel (same as lesion CSV)
        t_used.append(float(w))
        areas.append(float(mask.sum()) * area_per_px)
        masks.append(mask)
        if dec.sr_dims > 0:
            faf_stack.append(np.clip(pred_np[..., 0], 0, 1))
    if len(t_used) < 2:
        return None
    return t_used, np.asarray(areas), np.asarray(masks), np.asarray(faf_stack), area_per_px


def eye_metrics(t, areas, masks, faf_stack, area_eps, frac_eps):
    t = np.asarray(t, dtype=float)
    slope = float(np.polyfit(t, areas, 1)[0]) if len(t) >= 2 else 0.0
    area_range = float(areas.max() - areas.min())
    mean_area = float(areas.mean())
    rel_range = area_range / (mean_area + 1e-6)
    # fraction of ever-GA pixels that change membership across the sweep
    union = masks.any(axis=0)
    inter = masks.all(axis=0)
    changed = union & (~inter)
    seg_change_frac = float(changed.sum()) / max(int(union.sum()), 1)
    faf_temporal_std = float(faf_stack.std(axis=0).mean()) if faf_stack.size else float("nan")
    is_static = (area_range < area_eps) and (seg_change_frac < frac_eps)
    return {
        "area_slope_mm2_per_wk": slope,
        "area_range_mm2": area_range,
        "mean_area_mm2": mean_area,
        "area_rel_range": rel_range,
        "seg_change_frac": seg_change_frac,
        "faf_temporal_std": faf_temporal_std,
        "n_t": int(len(t)),
        "verdict": "STATIC" if is_static else "RESPONSIVE",
    }


def main():
    a = parse_args()
    if not os.path.exists(a.checkpoint):
        raise FileNotFoundError(a.checkpoint)

    print(f"Loading checkpoint: {a.checkpoint}")
    chkp = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = chkp["args"]
    epoch = chkp.get("epoch", 0)

    args["device"] = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if a.epochs_val is not None:
        args["epochs"]["val"] = a.epochs_val
    args["epochs"]["train"] = 0
    args["validation"]["activate"] = False
    args.setdefault("test", {})["activate"] = False  # don't trigger test() inside __init__

    # Same split-integrity + leakage guard as evaluate.py.
    sets = validate_splits(args)
    lat = chkp.get("latents")
    indep = args.get("dataset", {}).get("independent_visits", False)
    if lat is not None and not indep and len(sets.get("train", [])) and lat.shape[0] != len(sets["train"]):
        print("\n*** LEAKAGE WARNING: checkpoint has %d per-eye latents but config resolves %d train eyes; "
              "this checkpoint was trained on a DIFFERENT split. ***\n" % (lat.shape[0], len(sets["train"])))

    out_dir = a.output_dir or os.path.join(
        os.path.dirname(a.checkpoint),
        f"temporal_sensitivity_{a.split}_ep{epoch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    chkp_dir = os.path.dirname(a.checkpoint)
    args["load_model"] = {"path": chkp_dir, "epoch": epoch}
    args["output_dir"] = out_dir

    print("Initialising model / data ...")
    builder = AtlasBuilder(args)
    grid_coords, grid_shape = generate_world_grid(args, device=builder.device)

    df = builder.datasets[a.split].df
    if df is None or len(df) == 0:
        raise RuntimeError(f"No {a.split} eyes found — check the split.")
    subs = sorted(df["sub_id_int"].unique())
    print(f"{a.split}: {len(subs)} eyes")

    # val/test latents must be fit before probing (one TTA round on ALL visits).
    if a.split in ("val", "test"):
        print(f"Fitting {a.split} latents on ALL visits (holdout='none', {args['epochs']['val']} epochs) ...")
        builder._run_validation_round(epoch, args.get("tb_writer"), grid_coords, grid_shape,
                                      subs, holdout_position="none", tag_suffix="tsens", split=a.split)

    rows = []
    per_eye_curves = []  # (eye_id, t, areas, observed_weeks)
    id_col = args["dataset"].get("id_column", "subject_id")
    tkey = builder._temporal_key
    for sub_id in subs:
        sub_df = df[df["sub_id_int"] == sub_id].sort_values(tkey)
        eye_id = str(sub_df.iloc[0].get(id_col, sub_id))
        obs_weeks = [float(r.get(tkey, 0.0)) for _, r in sub_df.iterrows()]
        w_max = max(obs_weeks) if obs_weeks else 48.0
        weeks = np.linspace(0.0, w_max + a.horizon_weeks, a.n_t)
        res = probe_eye(builder, a.split, sub_id, grid_coords, grid_shape, weeks)
        if res is None:
            print(f"  [skip] {eye_id}: could not decode")
            continue
        t_used, areas, masks, faf_stack, _ = res
        m = eye_metrics(t_used, areas, masks, faf_stack, a.area_eps_mm2, a.frac_eps)
        m.update({"eye_id": eye_id, "sub_id_int": int(sub_id)})
        rows.append(m)
        per_eye_curves.append((eye_id, np.asarray(t_used), areas, obs_weeks))
        print(f"  {eye_id:18s} slope={m['area_slope_mm2_per_wk']:+.4f} mm2/wk  "
              f"range={m['area_range_mm2']:.3f}  relrange={m['area_rel_range']:.3f}  "
              f"segchg={m['seg_change_frac']:.3f}  -> {m['verdict']}")

    if not rows:
        raise RuntimeError("No eyes could be probed.")

    # ---- write CSV ----
    import csv
    csv_path = os.path.join(out_dir, f"temporal_sensitivity_{a.split}_ep{epoch}.csv")
    cols = ["eye_id", "sub_id_int", "n_t", "area_slope_mm2_per_wk", "area_range_mm2",
            "mean_area_mm2", "area_rel_range", "seg_change_frac", "faf_temporal_std", "verdict"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"\nWrote {csv_path}")

    # ---- cohort summary + verdict ----
    n = len(rows)
    n_static = sum(r["verdict"] == "STATIC" for r in rows)
    med_abs_slope = float(np.median([abs(r["area_slope_mm2_per_wk"]) for r in rows]))
    med_rel_range = float(np.median([r["area_rel_range"] for r in rows]))
    med_seg_chg = float(np.median([r["seg_change_frac"] for r in rows]))
    cohort_static = (n_static / n > 0.5) or (med_rel_range < 0.05)
    summary = {
        "split": a.split, "epoch": epoch, "n_eyes": n, "n_static": n_static,
        "frac_static": n_static / n, "median_abs_slope_mm2_per_wk": med_abs_slope,
        "median_area_rel_range": med_rel_range, "median_seg_change_frac": med_seg_chg,
        "cohort_verdict": "COLLAPSED (time pathway not used)" if cohort_static
                          else "RESPONSIVE (predictions vary with time)",
        "checkpoint": a.checkpoint, "horizon_weeks": a.horizon_weeks, "n_t": a.n_t,
    }
    with open(os.path.join(out_dir, f"temporal_sensitivity_{a.split}_ep{epoch}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print(f"TEMPORAL SENSITIVITY  [{a.split}, epoch {epoch}]")
    print("-" * 64)
    print(f"  eyes probed                 : {n}")
    print(f"  STATIC (collapsed) eyes     : {n_static}/{n}  ({100*n_static/n:.0f}%)")
    print(f"  median |area slope|         : {med_abs_slope:.4f} mm2/week")
    print(f"  median area relative range  : {med_rel_range:.3f}")
    print(f"  median seg-change fraction  : {med_seg_chg:.3f}")
    print(f"  >>> COHORT VERDICT          : {summary['cohort_verdict']}")
    print("=" * 64)

    # ---- cohort overlay (area vs week, colored by slope) ----
    slopes = np.array([r["area_slope_mm2_per_wk"] for r in rows])
    vmax = max(abs(slopes).max(), 1e-6)
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(7, 5))
    for (eye_id, t, areas, obs), s in zip(per_eye_curves, slopes):
        ax.plot(t, areas, "-", color=cmap(0.5 + 0.5 * s / vmax), alpha=0.8, lw=1.5)
    ax.set_xlabel("weeks from baseline")
    ax.set_ylabel("predicted GA area (mm$^2$)")
    ax.set_title(f"Temporal response — {a.split} (n={n})\n"
                 f"{summary['cohort_verdict']}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="GA growth rate (mm$^2$/wk)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"overlay_{a.split}_ep{epoch}.png"), dpi=150)
    plt.close(fig)

    # ---- per-eye small multiples ----
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow), squeeze=False)
    order = np.argsort(-slopes)  # fastest growers first
    for ax_i, idx in enumerate(order):
        ax = axes[ax_i // ncol][ax_i % ncol]
        eye_id, t, areas, obs = per_eye_curves[idx]
        r = rows[idx]
        ax.plot(t, areas, "-o", ms=3, color="#1565C0" if r["verdict"] == "RESPONSIVE" else "#9E9E9E")
        for ow in obs:
            ax.axvline(ow, color="#BBBBBB", lw=0.6, ls=":")
        ax.set_title(f"{eye_id}\n{r['area_slope_mm2_per_wk']:+.3f} mm2/wk [{r['verdict']}]", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Predicted GA area vs week — {a.split}, epoch {epoch} "
                 f"(dotted = observed visits)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"per_eye_{a.split}_ep{epoch}.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures + CSV + summary in: {out_dir}")
    if args.get("tb_writer") is not None:
        args["tb_writer"].close()


if __name__ == "__main__":
    main()
