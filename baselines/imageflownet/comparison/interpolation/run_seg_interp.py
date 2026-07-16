"""Segmentor-free classical SEGMENTATION floor for the GAP-INR comparison.

GAP-INR is segmentation-native (no external image->mask segmentor; the IFN-family UCSF segmentor
is OOD on this cohort, ceiling DICE ~0.26). So the fair classical floor for GAP-INR does NOT interpolate
FAF images + segment; instead it interpolates the OBSERVED GA MASKS directly and scores the result
against the real held-out mask -- exactly how GAP-INR's own DICE is computed.

For each canonical TEST eye, LEAVE-ONE-VISIT-OUT (shared spec = identical folds/split to GAP-INR):
hold out one visit, reconstruct its GA MASK from the OTHER visits' GT masks by a classical method,
and score DICE / HD / IoU / lesion-area MAE (mm^2) vs the real held-out mask on the native 512 grid.
DICE/HD/area use GAP-INR's EXACT conventions (utils.py) so the floor is directly comparable.

Methods:
  linear        -- per-pixel linear interp of the two BRACKETING support masks (interior); linear
                   EXTRAPOLATION from the nearest two for a last-visit hold-out; threshold 0.5.
  cubic_spline  -- per-pixel cubic spline through all support masks at the held-out time; threshold 0.5.
  growth_rate   -- AREA-ONLY: fit observed GA area ~ a0 + b*t, predict the held-out area (interp or
                   extrap). No mask -> DICE/HD NaN; only area-MAE is scored.
Buckets: INTERPOLATION (interior hold-out) vs EXTRAPOLATION (last visit). COPY-FORWARD reference
(= temporally nearest support mask) is scored the same way. Geometry: 620 center-crop -> 512 resize
(GA-preserving), identical to the shared spec / GAP-INR faf_ga_512.

Run (only needs numpy/scipy/PIL + the shared spec + the clinical CSV -- NO GPU, NO segmentor):
    python run_seg_interp.py [--methods linear cubic_spline growth_rate] [--split test] [--dump-root D]
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_IFN_ROOT = os.path.dirname(os.path.dirname(_HERE))   # .../baselines/imageflownet (holds eval_spec)
sys.path.insert(0, _IFN_ROOT)
from _classical import interp_pixelwise, fit_predict_area                 # noqa: E402
import eval_spec as spec                                           # noqa: E402
try:
    import dump_io                                                       # noqa: E402
except Exception:
    dump_io = None

CROP_SIZE = spec.CROP_SIZE                 # 620 (native ~768 center-crop, GA-preserving)
EVAL = spec.EVAL_DIM                       # 512
RF = (CROP_SIZE / EVAL) ** 2               # mm^2 area pitch: 512-grid -> native pixel pitch


def _load_mask(path):
    """GA mask on the SAME grid as GAP-INR/the spec: center-crop 620 (native), resize 512 NEAREST,
    binarise > 127. Identical to summarize_eval._load_mask_512 and the IFN mask loader geometry."""
    m = Image.open(path).convert("L")
    W, H = m.size
    l, t = (W - CROP_SIZE) // 2, (H - CROP_SIZE) // 2
    m = m.crop((l, t, l + CROP_SIZE, t + CROP_SIZE)).resize((EVAL, EVAL), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


# ---- GAP-INR's EXACT metric conventions (utils.py) so the floor DICE/HD match GAP-INR's ----
def _tpfpfn(pm, rm):
    pm, rm = pm > 0.5, rm > 0.5
    return (int(np.count_nonzero(pm & rm)), int(np.count_nonzero(pm & ~rm)),
            int(np.count_nonzero(~pm & rm)))


def _dice(pm, rm):
    tp, fp, fn = _tpfpfn(pm, rm)
    return 1.0 if (2 * tp + fp + fn) == 0 else (2.0 * tp) / (2 * tp + fp + fn)


def _iou(pm, rm):
    tp, fp, fn = _tpfpfn(pm, rm)
    return 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)


def _hd(pm, rm):
    pm, rm = pm > 0.5, rm > 0.5
    if not pm.any() and not rm.any():
        return 0.0
    if not pm.any() or not rm.any():
        return float(np.sqrt(sum(float(d) ** 2 for d in pm.shape)))   # one-empty -> array diagonal
    from skimage.metrics import hausdorff_distance
    return float(hausdorff_distance(pm, rm))


def _area(mask, sx, sy):
    return float((mask > 0.5).sum()) * float(sx) * float(sy) * RF


def main():
    global EVAL, RF                    # declared before any use of EVAL (incl. the --eval-dim default)
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["linear", "linear_regression", "cubic_spline", "growth_rate"],
                    choices=["linear", "linear_regression", "cubic_spline", "growth_rate"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--eval-dim", type=int, default=EVAL,
                    help="scoring grid (crop620 -> resize this). Default spec 512; use 256 for the "
                         "Scenario-2 track or 620 for the Scenario-3 (missing-visits) track. Put each "
                         "resolution in its OWN --out dir so results_seg_interp_* don't clobber.")
    ap.add_argument("--dump-root", default=None,
                    help="if set, write dump_io cases under scenario='interp' (method=<name>_seg)")
    args = ap.parse_args()
    EVAL = int(args.eval_dim); RF = (CROP_SIZE / EVAL) ** 2   # scoring grid + mm^2 pitch follow --eval-dim

    rows_df = spec.usable_visit_rows(args.split)
    spec.assert_split_parity(rows_df["Eye_ID"].unique(), args.split, source="seg_interp_floor")

    # per-eye visit records (sorted by visit): t(weeks), GT mask, per-visit mm/px scales
    eyes = {}
    for eye, g in rows_df.groupby("Eye_ID"):
        g = g.sort_values("Visit_Number")
        vd = g["visit_date"].astype(float).values
        wk = (vd - vd.min()) / 7.0
        recs = {}
        for k, (_, r) in enumerate(g.iterrows()):
            recs[int(r["Visit_Number"])] = dict(
                t=float(wk[k]), mask=_load_mask(r["ga_mask_path"]),
                sx=float(r["ScaleXSlo"]), sy=float(r["ScaleYSlo"]))
        eyes[eye] = recs

    folds = list(spec.loo_folds(args.split))   # (eye, holdout_visit, kind)

    for method in args.methods:
        rows, cf_rows = [], []
        for eye, hv, kind in folds:
            recs = eyes[eye]
            if hv not in recs:
                continue
            sup_v = sorted(v for v in recs if v != hv)
            t_sup = np.array([recs[v]["t"] for v in sup_v], dtype=np.float64)
            h = recs[hv]
            gt = h["mask"]
            ga = _area(gt, h["sx"], h["sy"])
            is_extrap = int(kind == "extrapolation")

            # copy-forward = temporally NEAREST support mask (shared by all methods)
            nn = sup_v[int(np.argmin(np.abs(t_sup - h["t"])))]
            cfm = recs[nn]["mask"]

            if method == "growth_rate":
                a_sup = [_area(recs[v]["mask"], recs[v]["sx"], recs[v]["sy"]) for v in sup_v]
                pa = fit_predict_area(t_sup, a_sup, h["t"])
                rows.append(dict(Patient_Eye=eye, Set="test_eval", holdout=hv, is_extrap=is_extrap,
                                 Dice=np.nan, HD=np.nan, IoU=np.nan,
                                 GT_Area_mm2=ga, Pred_Area_mm2=pa, area_MAE_mm2=abs(pa - ga)))
                cfa = _area(cfm, h["sx"], h["sy"])
                cf_rows.append(dict(Patient_Eye=eye, Set="copyforward", holdout=hv, is_extrap=is_extrap,
                                    Dice=np.nan, HD=np.nan, IoU=np.nan,
                                    GT_Area_mm2=ga, Pred_Area_mm2=cfa, area_MAE_mm2=abs(cfa - ga)))
                continue

            # per-pixel interpolate the GT support masks, threshold at 0.5
            msup = np.stack([recs[v]["mask"] for v in sup_v]).astype(np.float32)
            soft = interp_pixelwise(method, t_sup, msup, h["t"])
            pmask = (soft > 0.5).astype(np.uint8)
            d, hd, iou = _dice(pmask, gt), _hd(pmask, gt), _iou(pmask, gt)
            pa = _area(pmask, h["sx"], h["sy"])
            rows.append(dict(Patient_Eye=eye, Set="test_eval", holdout=hv, is_extrap=is_extrap,
                             Dice=d, HD=hd, IoU=iou,
                             GT_Area_mm2=ga, Pred_Area_mm2=pa, area_MAE_mm2=abs(pa - ga)))

            cfa = _area(cfm, h["sx"], h["sy"])
            cf_rows.append(dict(Patient_Eye=eye, Set="copyforward", holdout=hv, is_extrap=is_extrap,
                                Dice=_dice(cfm, gt), HD=_hd(cfm, gt), IoU=_iou(cfm, gt),
                                GT_Area_mm2=ga, Pred_Area_mm2=cfa, area_MAE_mm2=abs(cfa - ga)))

            if dump_io is not None and args.dump_root:
                dump_io.write_case(args.dump_root, method="%s_seg" % method, scenario="interp",
                                   eye_id=eye, tgt_visit=hv, pred_mask=pmask, gt_mask=gt,
                                   weeks=h["t"], dice=d, hd=hd,
                                   gt_area_mm2=ga, pred_area_mm2=pa, is_extrap=is_extrap)

        def summarize(rs):
            def bucket(name, sel):
                sub = [r for r in rs if sel(r)]
                o = {"Set": name, "n": len(sub)}
                for col, key in (("DICE_mean", "Dice"), ("HD_mean", "HD"), ("IoU_mean", "IoU"),
                                 ("area_MAE_mm2", "area_MAE_mm2")):
                    a = np.array([r[key] for r in sub], float)
                    a = a[~np.isnan(a)]                       # growth_rate leaves DICE/HD/IoU NaN
                    o[col] = float(a.mean()) if a.size else float("nan")
                    o[col + "_se"] = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
                return o
            return [bucket("interpolation", lambda r: not r["is_extrap"]),
                    bucket("extrapolation", lambda r: r["is_extrap"]),
                    bucket("ALL", lambda r: True)]

        summ_cols = ["Set", "DICE_mean", "HD_mean", "IoU_mean", "area_MAE_mm2", "n"]
        out_dir = os.path.join(args.out, "results_seg_interp_%s" % method)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "leave_one_out_summary_test.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction="ignore")
            w.writeheader(); w.writerows(summarize(rows))
        with open(os.path.join(out_dir, "leave_one_out_summary_test_copyforward.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction="ignore")
            w.writeheader(); w.writerows(summarize(cf_rows))
        with open(os.path.join(out_dir, "folds_test.csv"), "w", newline="") as f:
            cols = ["Patient_Eye", "Set", "holdout", "is_extrap", "Dice", "HD", "IoU",
                    "GT_Area_mm2", "Pred_Area_mm2", "area_MAE_mm2"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows); w.writerows(cf_rows)

        print("\n==== GAP-INR seg-space floor: %s (real-mask DICE, %d grid, no segmentor) ====" % (method, EVAL))
        for r in summarize(rows):
            print("  %-14s n=%-3d DICE %.3f  IoU %.3f  HD %.1f  areaMAE %.4f mm^2" %
                  (r["Set"], r["n"], r["DICE_mean"], r["IoU_mean"], r["HD_mean"], r["area_MAE_mm2"]))
        print("  -> %s" % out_dir)
        try:                                    # eval-time TensorBoard (scalar per bucket/metric)
            from torch.utils.tensorboard import SummaryWriter
            _w = SummaryWriter(os.path.join(out_dir, "tb_eval"))
            _w.add_text("provenance", "classical seg-floor %s | GT-mask interp | crop%d -> score@%d | split=test"
                        % (method, CROP_SIZE, EVAL))
            for r in summarize(rows):
                for m in ("DICE_mean", "HD_mean", "IoU_mean", "area_MAE_mm2"):
                    v = r.get(m)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        _w.add_scalar("%s/%s/%s" % (method, r["Set"], m.replace("_mean", "")), float(v), 0)
            _w.flush(); _w.close()
        except Exception as _e:
            print("[warn] TB skipped:", _e)


if __name__ == "__main__":
    main()
