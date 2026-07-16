"""Diagnostic: do predicted FAF, predicted segmentation, and GT line up?

For a few eyes (one representative visit each) it decodes the predicted FAF + GA mask,
loads the GT FAF + GT mask (center-cropped to the prediction like the figures do), and
builds a row:
  [GT FAF | GT FAF + GT-seg contour(green) | Pred FAF | Pred FAF + Pred-seg contour(red) |
   Pred FAF + BOTH contours]
so we can directly see (a) whether the predicted seg sits on the dark lesion in the
predicted FAF, and (b) whether the predicted FOV matches the GT (landmarks/contour align).

Prints per eye: GT seg area, Pred seg area (mm^2), and the IoU between the predicted seg and
a simple 'dark region' proxy on the predicted FAF (low-intensity inside the field) to quantify
the 'seg bigger than the FAF lesion' impression.

Usage: python analyze_faf_seg.py --checkpoint <...>.pth --eyes EYE07_OD,EYE23_OD --split val
"""
import os, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from build_atlas import AtlasBuilder
from utils import generate_world_grid, typecheck_img, load_2d_modality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--eyes", default=None, help="comma list of eye ids; default: first 4 in split")
    ap.add_argument("--visit", default="last", choices=["last", "first"])
    ap.add_argument("--out", default=None)
    args_cmd = ap.parse_args()

    chkp = torch.load(args_cmd.checkpoint, map_location="cpu", weights_only=False)
    args = chkp["args"]; epoch = chkp.get("epoch", 0)
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    args["epochs"]["train"] = 0
    args["validation"]["activate"] = False
    args.setdefault("test", {})["activate"] = False
    args["load_model"] = {"path": os.path.dirname(args_cmd.checkpoint), "epoch": epoch}
    out = args_cmd.out or os.path.join(os.path.dirname(args_cmd.checkpoint),
                                       f"faf_seg_alignment_{args_cmd.split}_ep{epoch}.png")
    args["output_dir"] = os.path.dirname(out)

    b = AtlasBuilder(args)
    split = args_cmd.split
    gc, gs = generate_world_grid(args, device=b.device)
    df = b.datasets[split].df
    sr_dims = sum(args["inr_decoder"]["out_dim"][:-1])  # FAF channels; seg hard channel is at this index
    tkey = b._temporal_key
    id_col = args["dataset"].get("id_column", "subject_id")
    mods = args["dataset"]["modalities"]

    # val/test need fitted latents first (one TTA round on all visits)
    subs_all = sorted(df["sub_id_int"].unique())
    if split in ("val", "test"):
        print(f"Fitting {split} latents (holdout none, {args['epochs']['val']} steps)...")
        b._run_validation_round(epoch, args.get("tb_writer"), gc, gs, subs_all,
                                holdout_position="none", tag_suffix="align", split=split)

    # choose eyes
    want = [e.strip() for e in args_cmd.eyes.split(",")] if args_cmd.eyes else None
    rows = []
    for sub_id in subs_all:
        sdf = df[df["sub_id_int"] == sub_id].sort_values(tkey)
        eye = str(sdf.iloc[0].get(id_col, sub_id))
        if want and eye not in want:
            continue
        rows.append((sub_id, eye, sdf))
        if not want and len(rows) >= 4:
            break

    per_px_mm2 = None
    fig, axes = plt.subplots(len(rows), 5, figsize=(20, 4 * len(rows)), squeeze=False)
    col_titles = ["GT FAF", "GT FAF + GT seg", "Pred FAF", "Pred FAF + Pred seg", "Pred FAF + both"]
    for ri, (sub_id, eye, sdf) in enumerate(rows):
        row = (sdf.iloc[-1] if args_cmd.visit == "last" else sdf.iloc[0]).to_dict()
        per_px_mm2 = float(row.get("ScaleXSlo", 1.0)) * float(row.get("ScaleYSlo", 1.0))
        vol = b._reconstruct_visit(row, int(sub_id), gc, gs, split=split, allow_extrapolation=False)
        pred = typecheck_img(vol)
        pred_faf = np.clip(pred[..., 0], 0, 1)
        pred_seg = (pred[..., sr_dims] > 0.5).astype(np.float32)

        # GT, center-cropped to pred shape (mirrors build_atlas figure alignment)
        gt_faf = load_2d_modality(b.datasets[split].resolve_path(row, mods[0]), is_seg=False,
                                  patient_stats=b._get_patient_stats(split, sub_id), args=b.args)
        gt_seg = load_2d_modality(b.datasets[split].resolve_path(row, mods[1]), is_seg=True,
                                  patient_stats=b._get_patient_stats(split, sub_id), args=b.args)
        Hp, Wp = pred_faf.shape
        if gt_faf.shape != pred_faf.shape:
            hs = max(0, (gt_faf.shape[0] - Hp) // 2); ws = max(0, (gt_faf.shape[1] - Wp) // 2)
            gt_faf = gt_faf[hs:hs + Hp, ws:ws + Wp]; gt_seg = gt_seg[hs:hs + Hp, ws:ws + Wp]
        gt_seg = (gt_seg > 0.5).astype(np.float32)

        # quantify: areas + overlap of pred seg with a 'dark region' proxy on pred FAF
        gt_a = float(gt_seg.sum()) * per_px_mm2
        pr_a = float(pred_seg.sum()) * per_px_mm2
        # dark proxy: bottom-20th-percentile intensity inside a central field (exclude corners)
        yy, xx = np.mgrid[0:Hp, 0:Wp]
        field = ((yy - Hp / 2) ** 2 + (xx - Wp / 2) ** 2) < (0.48 * min(Hp, Wp)) ** 2
        thr = np.percentile(pred_faf[field], 20)
        dark = ((pred_faf <= thr) & field).astype(np.float32)
        inter = float((pred_seg * dark).sum()); union = float(((pred_seg + dark) > 0).sum())
        iou_dark = inter / max(union, 1)
        print(f"{eye:14s} GT_area={gt_a:.3f}  Pred_area={pr_a:.3f} mm^2  "
              f"pred_seg vs FAF-dark IoU={iou_dark:.3f}  (pred_seg px={int(pred_seg.sum())})")

        def show(ax, img, t):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=9); ax.axis("off")
        show(axes[ri][0], gt_faf, f"{eye}  GT FAF")
        show(axes[ri][1], gt_faf, "GT FAF + GT seg")
        axes[ri][1].contour(gt_seg, levels=[0.5], colors="#00E676", linewidths=1.2)
        show(axes[ri][2], pred_faf, "Pred FAF")
        show(axes[ri][3], pred_faf, f"Pred FAF + Pred seg ({pr_a:.2f} mm²)")
        axes[ri][3].contour(pred_seg, levels=[0.5], colors="#FF1744", linewidths=1.2)
        show(axes[ri][4], pred_faf, "Pred FAF + GT(green)/Pred(red)")
        axes[ri][4].contour(gt_seg, levels=[0.5], colors="#00E676", linewidths=1.0)
        axes[ri][4].contour(pred_seg, levels=[0.5], colors="#FF1744", linewidths=1.0)
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(f"{t}", fontsize=10)
    fig.suptitle(f"FAF/seg alignment — {split}, epoch {epoch}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nSaved {out}")
    if args.get("tb_writer") is not None:
        args["tb_writer"].close()


if __name__ == "__main__":
    main()
