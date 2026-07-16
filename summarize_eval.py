"""
Summarise a GAP-INR evaluation directory into a paper-ready leave-one-out table.

Reads the per-held-out-position metric JSONs written during validation/test
(``*_holdout_V*/*.json`` and ``*_eval*/*.json``), and the lesion-area CSV, then
reports per-position and cohort held-out performance with mean ± standard error,
split into INTERPOLATION (a non-final visit held out) vs EXTRAPOLATION (the last
visit held out), plus the lesion-area MAE.

Usage:
    python summarize_eval.py --eval_dir tmp/<run>/eval_test
    # -> writes <eval_dir>/leave_one_out_summary.csv and prints the table.
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd


def _load_metric_json(path):
    """Return (metrics, subjects): metric -> list of per-eye scalars, and the aligned list of
    Subject (eye) ids (same order), from a GAP-INR metrics JSON."""
    d = json.load(open(path))
    out = {}
    subjects = []
    for entry in d:
        subjects.append(str(entry.get("Subject", "")))
        for k, v in entry.items():
            if k == "Subject":
                continue
            # Append NaN (not skip) for empty/None/non-numeric so every metric list stays length-aligned
            # with `subjects` -- the growth buckets index per-eye by position. _mean_se drops the NaNs.
            if isinstance(v, list):
                val = v[0] if len(v) else np.nan
            else:
                val = v
            try:
                out.setdefault(k, []).append(float(val) if val is not None else np.nan)
            except (TypeError, ValueError):
                out.setdefault(k, []).append(np.nan)
    return out, subjects


def _load_mask_512(path, crop, res):
    """Load a GA mask the SAME way eval_omega does: center-crop `crop` (620) then resize to
    `res` (512) NEAREST, binarise >127. So GAP-INR's growth uses identical masks to ImageFlowNet."""
    from PIL import Image
    m = Image.open(path).convert("L")
    W, H = m.size
    l, t = (W - crop) // 2, (H - crop) // 2
    m = m.crop((l, t, l + crop, t + crop)).resize((res, res), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def _dsc(a, b):
    inter = float(np.logical_and(a, b).sum())
    return (2.0 * inter + 1e-12) / (float(a.sum()) + float(b.sum()) + 1e-12)


def _eye_growth(data_csv, crop, res):
    """{eye_id: [growth per chronological visit]} where growth[k] = 1 - DSC(mask[k-1], mask[k])
    from the RAW GT masks (growth[0] uses mask[0]<->mask[1]). GT-only, so identical across methods
    and matched to ImageFlowNet's eval_omega definition. The k-th value aligns with the eye's k-th
    held-out position (by RANK), which is robust to 0- vs 1-based holdout numbering."""
    df = pd.read_csv(data_csv).dropna(subset=["ga_mask_path"])
    sort_col = "Visit_Number" if "Visit_Number" in df.columns else None
    out = {}
    for eye, g in df.groupby("Eye_ID"):
        g = g.sort_values(sort_col) if sort_col else g
        masks = [_load_mask_512(p, crop, res) for p in g["ga_mask_path"]
                 if isinstance(p, str) and os.path.exists(p)]
        if len(masks) < 2:
            out[str(eye)] = [0.0] * len(masks)
            continue
        gr = [1.0 - _dsc(masks[max(k - 1, 0)], masks[k]) for k in range(len(masks))]
        gr[0] = 1.0 - _dsc(masks[0], masks[1])
        out[str(eye)] = gr
    return out


def _mean_se(vals):
    a = np.asarray(vals, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, np.nan, 0
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), se, int(a.size)


def _select_best_epoch(rows, criterion="DICE"):
    """Keep only the rows from the single best validation epoch, per split. The best epoch is the one
    whose AGGREGATE held-out DICE (mean over ALL held-out eyes and positions of that split at that
    epoch) is highest -- exactly the criterion build_atlas uses to save checkpoint_best.pth. So the
    reported numbers are the SELECTED checkpoint's, not an average over the whole training trajectory.

    The chosen epoch is picked by DICE; every other metric (PSNR/SSIM/area) is then read off that same
    epoch, so only DICE carries the (standard early-stopping) selection bias.

    Falls back to keeping a split's rows unchanged (pooled) if its filenames carry no 'ep=' tag.

    Returns (kept_rows, best_epoch_by_split) -- the latter is used to read the SAME epoch's
    lesion-area / monotonicity CSVs, so EVERY reported metric comes from the one selected checkpoint.
    """
    out, best_by_split = [], {}
    for split in sorted({r["split"] for r in rows}):
        srows = [r for r in rows if r["split"] == split]
        epochs = sorted({r["epoch"] for r in srows if r["epoch"] is not None})
        if not epochs:
            print(f"[best_checkpoint] {split}: no 'ep=' in filenames -> keeping pooled rows.")
            out.extend(srows)
            continue
        best_ep, best_agg = None, None
        for ep in epochs:
            vals = [v for r in srows if r["epoch"] == ep for v in r["metrics"].get(criterion, [])]
            if not vals:
                continue
            agg = float(np.mean(vals))
            if best_agg is None or agg > best_agg:
                best_agg, best_ep = agg, ep
        if best_ep is None:           # criterion absent in all JSONs -> keep pooled
            print(f"[best_checkpoint] {split}: '{criterion}' not found -> keeping pooled rows.")
            out.extend(srows)
            continue
        kept = [r for r in srows if r["epoch"] == best_ep]
        best_by_split[split] = best_ep
        npos = len({r["position"] for r in kept})
        print(f"[best_checkpoint] {split}: best epoch = {best_ep} "
              f"(mean {criterion} {best_agg:.4f} over {npos} held-out position(s)).")
        out.extend(kept)
    return out, best_by_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--metrics", nargs="+",
                    default=["DICE", "Precision", "Recall", "IoU", "HD", "PSNR", "SSIM", "LPIPS", "LOSS"])
    # Optional minor/major GA-growth stratification (ImageFlowNet Table 1 framing). Pass the data
    # CSV (with ga_mask_path) to enable it; growth = 1 - DSC(prev visit, held-out visit) from the GT
    # masks, identical to eval_omega. Off by default -> existing summary is unchanged.
    ap.add_argument("--data_csv", default=None, help="data CSV to enable minor/major-growth buckets")
    ap.add_argument("--crop_size", type=int, default=620)
    ap.add_argument("--score_res", type=int, default=512)
    ap.add_argument("--growth_thr", type=float, default=0.1, help="major growth if 1-DSC > thr")
    ap.add_argument("--best_checkpoint", action="store_true",
                    help="Report metrics from the SINGLE best validation epoch (reconstructs "
                         "checkpoint_best.pth) instead of POOLING every validation epoch. Per split, "
                         "picks the epoch with the highest aggregate held-out DICE -- the same "
                         "criterion build_atlas uses to save checkpoint_best.pth -- then reports every "
                         "held-out position's metrics AT that one epoch. Falls back to pooled for any "
                         "split whose metric filenames lack an 'ep=' tag.")
    args = ap.parse_args()

    # 1. Gather held-out metric JSONs by (split, position).
    rows = []  # one row per (split, position) with per-eye metric arrays
    for jp in glob.glob(os.path.join(args.eval_dir, "**", "*metrics*.json"), recursive=True):
        name = os.path.basename(os.path.dirname(jp))
        # held-out sets only: val_eval_holdout_V{p}, test_eval(_test), *_eval*
        m = re.search(r"holdout_V(\d+)", name)
        if "eval" not in name:
            continue
        if "opt" in name:        # skip the observed-visit (opt) sets
            continue
        split = "val" if name.startswith("val") else ("test" if name.startswith("test") else "other")
        pos = int(m.group(1)) if m else None     # None = single hold-out (e.g. test 'last')
        ep_m = re.search(r"ep=(\d+)", os.path.basename(jp))   # epoch is in the FILENAME, not the dir
        epoch = int(ep_m.group(1)) if ep_m else None
        metrics, subjects = _load_metric_json(jp)
        rows.append({"split": split, "position": pos, "epoch": epoch,
                     "metrics": metrics, "subjects": subjects})

    if not rows:
        print(f"No held-out metric JSONs found under {args.eval_dir}")
        return

    # --best_checkpoint: collapse each split to its single best validation epoch (reconstructs
    # checkpoint_best.pth) instead of pooling all epochs into the running training-average.
    best_by_split = {}
    if args.best_checkpoint:
        rows, best_by_split = _select_best_epoch(rows)

    # 2. Determine extrapolation = the largest held-out position within each split.
    pos_by_split = {}
    for r in rows:
        if r["position"] is not None:
            pos_by_split.setdefault(r["split"], set()).add(r["position"])
    last_pos = {s: (max(ps) if ps else None) for s, ps in pos_by_split.items()}

    def kind_of(split, pos):
        if pos is None:
            return "extrapolation"      # single hold-out = last visit = extrapolation
        return "extrapolation" if pos == last_pos.get(split) else "interpolation"

    # 3. lesion-area MAE + lesion-size MONOTONICITY (held-out), epoch-consistent with the metrics
    #    above. In --best_checkpoint mode we read ONLY the CSV at each split's selected epoch (so every
    #    reported number is the SAME checkpoint's); otherwise we pool every epoch's rows.
    def _csv_epoch(path):
        mm = re.search(r"epoch_(\d+)", os.path.basename(path))
        return int(mm.group(1)) if mm else None

    def _epoch_consistent_frames(pattern, split):
        """All CSV frames for `split`, restricted to that split's best epoch in --best_checkpoint mode."""
        frames = []
        for cp in glob.glob(os.path.join(args.eval_dir, "**", pattern), recursive=True):
            if args.best_checkpoint and split in best_by_split and _csv_epoch(cp) != best_by_split[split]:
                continue                                  # wrong epoch -> not this checkpoint
            try:
                frames.append(pd.read_csv(cp))
            except Exception:
                pass
        return frames

    area_mae = {}
    for split in ("val", "test"):
        sub = []
        for df in _epoch_consistent_frames("lesion_areas*.csv", split):
            if "Set" not in df.columns or "GT_Area_mm2" not in df.columns:
                continue                                  # e.g. lesion_areas_newvisits_*.csv (no 'Set')
            s = df[df["Set"].astype(str).str.startswith(f"{split}_eval")].dropna(
                subset=["GT_Area_mm2", "Pred_Area_mm2"])
            if len(s):
                sub.append(s)
        if sub:
            alls = pd.concat(sub, ignore_index=True)
            area_mae[split] = float((alls["Pred_Area_mm2"] - alls["GT_Area_mm2"]).abs().mean())

    # lesion-size monotonicity: mean total predicted GA-area DECREASE (mm^2) over each eye's FULL
    # predicted trajectory ('{split}_full' rows). 0 = perfectly non-decreasing (the clinical ideal).
    mono = {}
    for split in ("val", "test"):
        sub = []
        for df in _epoch_consistent_frames("lesion_monotonicity*.csv", split):
            if "Set" not in df.columns or "pred_decrease_mm2" not in df.columns:
                continue
            s = df[df["Set"].astype(str) == f"{split}_full"]
            if len(s):
                sub.append(s)
        if sub:
            alls = pd.concat(sub, ignore_index=True)
            mono[split] = float(alls["pred_decrease_mm2"].astype(float).mean())

    # 4. Build summary rows: per (split, position), and aggregated by kind + overall per split.
    out_rows = []

    def _by_eye(recs, get):
        """{eye_id: [values]} collected across recs; `get(r)` -> (subjects, values) aligned lists.
        The Subject field is per HELD-OUT VISIT (e.g. 'EYE09_OS_V2'); strip the trailing '_V<n>'
        so values are grouped by PATIENT-EYE, giving n = #eyes (=6 on test), not #held-out visits."""
        d = {}
        for r in recs:
            subs, vals = get(r)
            for sid, v in zip(subs, vals):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    eye = re.sub(r"_V\d+$", "", str(sid))
                    d.setdefault(eye, []).append(float(v))
        return d

    def summarize(label, split, recs):
        # PER-PATIENT aggregation: average each metric WITHIN an eye (over that eye's held-out visits
        # in this bucket), THEN mean +/- se OVER EYES. So n = #eyes (=6 on the test set), and no eye
        # with more held-out visits is over-weighted vs one with fewer.
        row = {"split": split, "group": label}
        n = 0
        for m in args.metrics:
            per_eye = [float(np.mean(v)) for v in
                       _by_eye(recs, lambda r, m=m: (r["subjects"], r["metrics"].get(m, []))).values()]
            mu, se, n = _mean_se(per_eye)
            row[f"{m}_mean"], row[f"{m}_se"] = mu, se
        row["n_heldout_visits"] = n          # per-patient aggregation -> this is the eye count (=6)
        # area-MAE + monotonicity are per-split (trajectory-level): attach to ALL / extrapolation only.
        if label in ("ALL", "extrapolation"):
            if split in area_mae:
                row["area_MAE_mm2"] = area_mae[split]
            if split in mono:
                row["mono_decrease_mm2"] = mono[split]
        # Per-BUCKET area-MAE, ALSO per-eye: mean over each eye's held-out |pred-gt| areas, then over eyes.
        eye_amae = _by_eye(recs, lambda r: (r["subjects"],
                                            [abs(pi - gi) for gi, pi in zip(r["metrics"].get("GT_Area_mm2", []),
                                                                            r["metrics"].get("Pred_Area_mm2", []))]))
        eye_amae = [float(np.mean(v)) for v in eye_amae.values()]
        if eye_amae:
            a_mu, a_se, _ = _mean_se(eye_amae)
            row["areaMAE_mean"], row["areaMAE_se"] = a_mu, a_se
        return row

    for split in sorted({r["split"] for r in rows}):
        srecs = [r for r in rows if r["split"] == split]
        # per position
        for pos in sorted({r["position"] for r in srecs}, key=lambda x: (x is None, x)):
            precs = [r for r in srecs if r["position"] == pos]
            out_rows.append(summarize(f"V{pos} ({kind_of(split, pos)})" if pos else "last (extrapolation)",
                                      split, precs))
        # grouped interpolation / extrapolation + overall. Prefer the PER-POSITION records; the
        # pos=None 'last' record (leave-one-out only) is a redundant pool of ALL held-out visits and
        # would DOUBLE-COUNT into extrapolation/ALL. Fall back to srecs when there are no per-position
        # records (e.g. holdout_strategy=last, which only has the single pos=None record).
        posrecs = [r for r in srecs if r["position"] is not None]
        grouprecs = posrecs if posrecs else srecs
        for kind in ("interpolation", "extrapolation"):
            krecs = [r for r in grouprecs if kind_of(split, r["position"]) == kind]
            if krecs:
                out_rows.append(summarize(kind, split, krecs))
        # overall
        out_rows.append(summarize("ALL", split, grouprecs))

    # 4b. OPT-IN minor/major GA-growth buckets (ImageFlowNet framing). Isolated + fail-soft so it
    # can never break the main interp/extrap summary. growth = 1 - DSC(prev, held-out) from GT masks.
    if args.data_csv:
        try:
            grw = _eye_growth(args.data_csv, args.crop_size, args.score_res)
            for split in sorted({r["split"] for r in rows}):
                srecs = [r for r in rows if r["split"] == split]
                # rank each eye's held-out positions chronologically -> index into its growth list
                eye_positions = {}
                for r in srecs:
                    for s in r["subjects"]:
                        eye_positions.setdefault(s, set()).add(r["position"])
                eye_rank = {e: {p: i for i, p in enumerate(sorted(ps, key=lambda x: (x is None, x)))}
                            for e, ps in eye_positions.items()}
                # collect per-metric values into minor/major bins
                bins = {"minor_growth": {m: [] for m in args.metrics},
                        "major_growth": {m: [] for m in args.metrics}}
                ncount = {"minor_growth": 0, "major_growth": 0}
                for r in srecs:
                    subs = r["subjects"]
                    for i, eye in enumerate(subs):
                        gl = grw.get(eye)
                        if not gl:
                            continue
                        rank = eye_rank[eye].get(r["position"], 0)
                        g = gl[rank] if rank < len(gl) else gl[-1]
                        b = "major_growth" if g > args.growth_thr else "minor_growth"
                        ncount[b] += 1
                        for mname in args.metrics:
                            vals = r["metrics"].get(mname, [])
                            if i < len(vals):
                                bins[b][mname].append(vals[i])
                for b in ("minor_growth", "major_growth"):
                    row = {"split": split, "group": b}
                    nn = 0
                    for mname in args.metrics:
                        mu, se, nn2 = _mean_se(bins[b][mname])
                        row[f"{mname}_mean"], row[f"{mname}_se"] = mu, se
                        nn = max(nn, nn2)
                    row["n_heldout_visits"] = ncount[b]
                    out_rows.append(row)
        except Exception as e:
            print(f"[growth] skipped minor/major-growth buckets: {type(e).__name__}: {e}")

    sdf = pd.DataFrame(out_rows)
    # --best_checkpoint writes a SEPARATE file so the pooled (training-trajectory) summary is preserved.
    out_name = "leave_one_out_summary_best.csv" if args.best_checkpoint else "leave_one_out_summary.csv"
    out_csv = os.path.join(args.eval_dir, out_name)
    sdf.to_csv(out_csv, index=False)

    # 5. Pretty print
    print(f"\n==== Leave-one-out held-out summary ({args.eval_dir}) ====")
    for split in sorted(set(sdf["split"])):
        print(f"\n[{split.upper()}]")
        for _, r in sdf[sdf["split"] == split].iterrows():
            parts = [f"{m}={r[f'{m}_mean']:.3f}±{r[f'{m}_se']:.3f}" for m in args.metrics
                     if not np.isnan(r[f'{m}_mean'])]
            def _has(c):
                return c in r and not pd.isna(r.get(c))
            extra = (f"  areaMAE={r['area_MAE_mm2']:.3f}mm²" if _has("area_MAE_mm2") else "")
            extra += (f"  areaMAE*={r['areaMAE_mean']:.3f}±{r['areaMAE_se']:.3f}mm²" if _has("areaMAE_mean") else "")
            extra += (f"  monoDec={r['mono_decrease_mm2']:.3f}mm²" if _has("mono_decrease_mm2") else "")
            print(f"  {r['group']:<26} n={int(r['n_heldout_visits']):<3} " + "  ".join(parts) + extra)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
