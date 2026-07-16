"""Classical interpolation FLOOR for the missing-visit comparison (GAP-INR paper).

For each canonical TEST eye, LEAVE-ONE-VISIT-OUT (the shared spec's folds, identical to GAP-INR /
gliomagrowth / NISF / MetaSeg): hold out one visit, reconstruct its FAF from the OTHER usable visits
by classical interpolation, and score it with the SAME protocol as every model --
    DICE = dice(segmentor(interpolated_FAF), REAL GA mask)          [shared frozen segmentor]
plus PSNR / SSIM (segmentor-free) and lesion-area MAE (mm^2), all on the native 512 grid.

Two methods:
  linear        -- linear interp between the two BRACKETING support visits (held-out in the middle);
                   if the held-out visit is an endpoint, linear EXTRAPOLATION from the nearest two.
  cubic_spline  -- cubic spline through ALL support visits, evaluated at the held-out timestamp.

Buckets mirror the shared spec (eval_spec.loo_folds): INTERPOLATION (held-out is NOT the last
visit -- bracketed both sides for interior positions) vs EXTRAPOLATION (held-out IS the last visit --
forward only). A COPY-FORWARD reference (= the temporally-nearest support visit) is scored the same
way. This replaces the previous version, which extrapolated-to-last and scored pseudo-label DICE
(segmentor(GT_image)); now it is true LOO interpolation with REAL masks, directly comparable.

Run (env with monai + the ImageFlowNet src on path):
    python run_baseline_interp.py --segmentor-ckpt <SEG.pty> [--methods linear cubic_spline]
        [--gpu-id 0] [--dump-root <DUMPS>]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import monai

# ImageFlowNet src (exact loader geometry + metric fns) and the shared comparison utils.
_IFN_ROOT = "/".join(os.path.realpath(__file__).split("/")[:-3])   # .../baselines/imageflownet
_IFN_SRC = os.path.join(_IFN_ROOT, "src")
sys.path.insert(0, _IFN_SRC)
sys.path.insert(0, _IFN_ROOT)                                     # holds eval_spec / common_preproc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _classical import interp_pixelwise, fit_predict_area                 # noqa: E402

from datasets.retina_faf_ga import CROP_SIZE, _load_raw, normalize_image   # noqa: E402
from utils.metrics import psnr, ssim, dice_coeff, hausdorff               # noqa: E402
import eval_spec as spec                                           # noqa: E402
try:
    import dump_io                                                       # noqa: E402
except Exception:
    dump_io = None

EVAL = 512
RF = (CROP_SIZE / EVAL) ** 2
_DEFAULT_SEG = os.path.join(_IFN_ROOT, "checkpoints", "segment_retina_faf_ga_512_seed1.pty")


def _faf01(x):                       # [-1,1] -> [0,1]
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


_LPIPS_MODEL = None
def _lpips(pred01, gt01, device):
    """LPIPS(AlexNet) between two [0,1] grayscale FAFs, matching eval_faf_ga's LPIPS. NaN if unavailable."""
    global _LPIPS_MODEL
    try:
        if _LPIPS_MODEL is None:
            import lpips
            _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        a = torch.from_numpy(pred01[None, None]).float().to(device) * 2 - 1
        b = torch.from_numpy(gt01[None, None]).float().to(device) * 2 - 1
        with torch.no_grad():
            return float(_LPIPS_MODEL(a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)).item())
    except Exception:
        return float("nan")


def _seg_mask(segmentor, faf_norm, device):
    x = torch.from_numpy(faf_norm[None, None, ...]).float().to(device)
    return (segmentor(x) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)


def _area(mask, sx, sy):
    return float((mask > 0.5).sum()) * float(sx) * float(sy) * RF


def _interp_faf(method, t_sup, faf_sup, t_h):
    """Reconstruct the held-out FAF at time t_h from support (t_sup, faf_sup), both sorted by t.
    Thin wrapper over the shared primitive so FAF-space and mask-space interpolate identically."""
    return interp_pixelwise(method, t_sup, faf_sup, t_h)


@torch.no_grad()
def main():
    global EVAL, RF                    # declared before any use of EVAL (incl. the --eval-dim default)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--segmentor-ckpt", default=_DEFAULT_SEG)
    ap.add_argument("--methods", nargs="+", default=["linear", "linear_regression", "cubic_spline", "growth_rate"],
                    choices=["linear", "linear_regression", "cubic_spline", "growth_rate"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--eval-dim", type=int, default=EVAL,
                    help="scoring grid (crop620 -> resize this). Default 512; 256 for Scenario-2, "
                         "620 for Scenario-3. Use a separate --out dir per resolution.")
    ap.add_argument("--dump-root", default=None,
                    help="if set, write dump_io cases under scenario='interp' (method=linear|cubic_spline)")
    args = ap.parse_args()
    EVAL = int(args.eval_dim); RF = (CROP_SIZE / EVAL) ** 2   # scoring grid + mm^2 pitch follow --eval-dim
    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")

    segmentor = torch.nn.Sequential(
        monai.networks.nets.DynUNet(spatial_dims=2, in_channels=1, out_channels=1,
                                    kernel_size=[5, 5, 5, 5], filters=[16, 32, 64, 128],
                                    strides=[1, 1, 1, 1], upsample_kernel_size=[1, 1, 1, 1]),
        torch.nn.Sigmoid()).to(device)
    segmentor.load_state_dict(torch.load(args.segmentor_ckpt, map_location=device))
    segmentor.eval()

    rows_df = spec.usable_visit_rows(args.split)
    spec.assert_split_parity(rows_df["Eye_ID"].unique(), args.split, source="interp_baseline")

    # per-eye visit records (sorted by visit): t(weeks), faf[-1,1], real mask, sx, sy
    eyes = {}
    for eye, g in rows_df.groupby("Eye_ID"):
        g = g.sort_values("Visit_Number")
        vd = g["visit_date"].astype(float).values
        wk = (vd - vd.min()) / 7.0
        recs = {}
        for k, (_, r) in enumerate(g.iterrows()):
            recs[int(r["Visit_Number"])] = dict(
                t=float(wk[k]),
                faf=normalize_image(_load_raw(r["faf_path"], (EVAL, EVAL), CROP_SIZE, is_mask=False)),
                mask=(_load_raw(r["ga_mask_path"], (EVAL, EVAL), CROP_SIZE, is_mask=True) > 128).astype(np.uint8),
                sx=float(r["ScaleXSlo"]), sy=float(r["ScaleYSlo"]))
        eyes[eye] = recs

    folds = list(spec.loo_folds(args.split))  # (eye, holdout_visit, kind)

    for method in args.methods:
        rows, cf_rows = [], []
        for eye, hv, kind in folds:
            recs = eyes[eye]
            if hv not in recs:
                continue
            sup_v = sorted(v for v in recs if v != hv)
            t_sup = np.array([recs[v]["t"] for v in sup_v], dtype=np.float64)
            h = recs[hv]
            ga = _area(h["mask"], h["sx"], h["sy"])
            is_extrap = int(kind == "extrapolation")

            if method == "growth_rate":
                # AREA-ONLY: fit the OBSERVED (GT) support GA areas vs time, predict the held-out
                # area (interp for interior, extrap for last visit). No FAF/segmentor -> the
                # intensity+seg metrics (DICE/HD/PSNR/SSIM) are NaN; only area-MAE is scored.
                a_sup = [_area(recs[v]["mask"], recs[v]["sx"], recs[v]["sy"]) for v in sup_v]
                pa = fit_predict_area(t_sup, a_sup, h["t"])
                rows.append(dict(Patient_Eye=eye, Set="test_eval", holdout=hv, is_extrap=is_extrap,
                                 Dice=np.nan, HD=np.nan, PSNR=np.nan, SSIM=np.nan, LPIPS=np.nan,
                                 GT_Area_mm2=ga, Pred_Area_mm2=pa, area_MAE_mm2=abs(pa - ga)))
                nn = sup_v[int(np.argmin(np.abs(t_sup - h["t"])))]
                cfa = _area(recs[nn]["mask"], recs[nn]["sx"], recs[nn]["sy"])
                cf_rows.append(dict(Patient_Eye=eye, Set="copyforward", holdout=hv, is_extrap=is_extrap,
                                    Dice=np.nan, HD=np.nan, PSNR=np.nan, SSIM=np.nan, LPIPS=np.nan,
                                    GT_Area_mm2=ga, Pred_Area_mm2=cfa, area_MAE_mm2=abs(cfa - ga)))
                continue

            faf_sup = np.stack([recs[v]["faf"] for v in sup_v]).astype(np.float32)
            pred = _interp_faf(method, t_sup, faf_sup, h["t"])
            pmask = _seg_mask(segmentor, pred, device)
            d = dice_coeff(label_pred=pmask, label_true=h["mask"])
            hd = hausdorff(label_pred=pmask, label_true=h["mask"])
            p = psnr(_faf01(pred), _faf01(h["faf"]), max_value=1)
            s = ssim(_faf01(pred), _faf01(h["faf"]), data_range=1)
            lp = _lpips(_faf01(pred), _faf01(h["faf"]), device)
            pa = _area(pmask, h["sx"], h["sy"])
            rows.append(dict(Patient_Eye=eye, Set="test_eval", holdout=hv, is_extrap=is_extrap,
                             Dice=d, HD=hd, PSNR=p, SSIM=s, LPIPS=lp, GT_Area_mm2=ga, Pred_Area_mm2=pa,
                             area_MAE_mm2=abs(pa - ga)))

            # copy-forward = temporally NEAREST support visit
            nn = sup_v[int(np.argmin(np.abs(t_sup - h["t"])))]
            cf = recs[nn]
            cfm = _seg_mask(segmentor, cf["faf"], device)
            cfa = _area(cfm, h["sx"], h["sy"])
            cf_rows.append(dict(Patient_Eye=eye, Set="copyforward", holdout=hv, is_extrap=is_extrap,
                                Dice=dice_coeff(label_pred=cfm, label_true=h["mask"]),
                                HD=hausdorff(label_pred=cfm, label_true=h["mask"]),
                                PSNR=psnr(_faf01(cf["faf"]), _faf01(h["faf"]), max_value=1),
                                SSIM=ssim(_faf01(cf["faf"]), _faf01(h["faf"]), data_range=1),
                                LPIPS=_lpips(_faf01(cf["faf"]), _faf01(h["faf"]), device),
                                GT_Area_mm2=ga, Pred_Area_mm2=cfa, area_MAE_mm2=abs(cfa - ga)))

            if dump_io is not None and args.dump_root:
                dump_io.write_case(args.dump_root, method=method, scenario="interp",
                                   eye_id=eye, tgt_visit=hv,
                                   pred_faf=_faf01(pred), gt_faf=_faf01(h["faf"]),
                                   pred_mask=pmask, gt_mask=h["mask"], weeks=h["t"],
                                   dice=d, hd=hd, psnr=p, ssim=s,
                                   gt_area_mm2=ga, pred_area_mm2=pa, is_extrap=is_extrap)

        def summarize(rs):
            def bucket(name, sel):
                sub = [r for r in rs if sel(r)]
                o = {"Set": name, "n": len(sub)}
                for col, key in (("DICE_mean", "Dice"), ("HD_mean", "HD"), ("PSNR_mean", "PSNR"),
                                 ("SSIM_mean", "SSIM"), ("LPIPS_mean", "LPIPS"), ("area_MAE_mm2", "area_MAE_mm2")):
                    a = np.array([r.get(key, np.nan) for r in sub], float)
                    a = a[~np.isnan(a)]
                    o[col] = float(a.mean()) if a.size else float("nan")
                    o[col + "_se"] = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
                return o
            return [bucket("interpolation", lambda r: not r["is_extrap"]),
                    bucket("extrapolation", lambda r: r["is_extrap"]),
                    bucket("ALL", lambda r: True)]

        summ_cols = ["Set", "DICE_mean", "HD_mean", "PSNR_mean", "SSIM_mean", "LPIPS_mean", "area_MAE_mm2", "n"]
        out_dir = os.path.join(args.out, "results_interp_%s" % method)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "leave_one_out_summary_test.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction="ignore")
            w.writeheader(); w.writerows(summarize(rows))
        with open(os.path.join(out_dir, "leave_one_out_summary_test_copyforward.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction="ignore")
            w.writeheader(); w.writerows(summarize(cf_rows))
        with open(os.path.join(out_dir, "folds_test.csv"), "w", newline="") as f:
            cols = ["Patient_Eye", "Set", "holdout", "is_extrap", "Dice", "HD", "PSNR", "SSIM", "LPIPS",
                    "GT_Area_mm2", "Pred_Area_mm2", "area_MAE_mm2"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows); w.writerows(cf_rows)

        print("\n==== interpolation floor: %s (real-mask DICE, %d grid) ====" % (method, EVAL))
        for r in summarize(rows):
            print("  %-14s n=%-3d DICE %.3f  PSNR %.2f  SSIM %.3f  areaMAE %.4f mm^2" %
                  (r["Set"], r["n"], r["DICE_mean"], r["PSNR_mean"], r["SSIM_mean"], r["area_MAE_mm2"]))
        print("  -> %s" % out_dir)
        try:                                    # eval-time TensorBoard (scalar per bucket/metric)
            from torch.utils.tensorboard import SummaryWriter
            _w = SummaryWriter(os.path.join(out_dir, "tb_eval"))
            _w.add_text("provenance", "classical image-floor %s | GT-FAF interp | crop%d -> score@%d | split=test"
                        % (method, CROP_SIZE, EVAL))
            for r in summarize(rows):
                for m in ("DICE_mean", "HD_mean", "PSNR_mean", "SSIM_mean", "LPIPS_mean", "area_MAE_mm2"):
                    v = r.get(m)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        _w.add_scalar("%s/%s/%s" % (method, r["Set"], m.replace("_mean", "")), float(v), 0)
            _w.flush(); _w.close()
        except Exception as _e:
            print("[warn] TB skipped:", _e)


if __name__ == "__main__":
    main()
