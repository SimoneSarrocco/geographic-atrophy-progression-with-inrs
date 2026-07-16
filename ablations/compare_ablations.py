#!/usr/bin/env python
"""Aggregate every ablation run into ONE comparison table.

Reads ablations/runs/index.json (written by run_ablations.py), loads each run's
leave_one_out_summary.csv (written by summarize_eval.py), and emits a master table
of held-out performance per ablation, split into interpolation / extrapolation / ALL,
with the best-checkpoint path. Writes CSV + a LaTeX table you can paste.

IMPORTANT — ablations are compared on the VALIDATION split (default). The test set is
reserved and touched ONCE, at the very end, for the final headline comparison of the
chosen config(s). Ranking design choices on test would implicitly select on it and make
the final test numbers optimistic/invalid. Use --split test ONLY for that final report.

Usage:
    python ablations/compare_ablations.py                  # val, group=ALL  (ablation/selection)
    python ablations/compare_ablations.py --split test     # FINAL report only (chosen config)
    python ablations/compare_ablations.py --group extrapolation
    python ablations/compare_ablations.py --metrics DICE PSNR SSIM
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
INDEX_PATH = os.path.join(HERE, "runs", "index.json")
MANIFEST_PATH = os.path.join(HERE, "manifest.yaml")


def _manifest():
    """name -> (group, rationale) for ordering/labels; tolerate a missing manifest."""
    if not os.path.exists(MANIFEST_PATH):
        return {}
    import yaml
    with open(MANIFEST_PATH) as f:
        return {m["name"]: (m.get("group", ""), m.get("rationale", "")) for m in yaml.safe_load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["test", "val"],
                    help="VAL for ablations/model selection (default); TEST only for the final report.")
    ap.add_argument("--group", default="ALL",
                    help="held-out group: ALL | interpolation | extrapolation")
    ap.add_argument("--metrics", nargs="+",
                    default=["DICE", "Precision", "Recall", "PSNR", "SSIM", "LPIPS", "LOSS"])
    ap.add_argument("--out", default=os.path.join(HERE, "comparison.csv"))
    ap.add_argument("--runs-dir", default="runs",
                    help="run-dir root under ablations/ holding index.json (default 'runs'; "
                         "use 'runs_r3' for round 3).")
    ap.add_argument("--baseline", default=None,
                    help="ablation to diff against (e.g. r3_base): prints Δ-vs-base per metric + a "
                         "HELPS/HURTS verdict, so each one-knob round-3 ablation reads as a yes/no answer.")
    ap.add_argument("--summary_name", default="leave_one_out_summary.csv",
                    help="Per-run summary file to read. Use leave_one_out_summary_best.csv to compare "
                         "BEST-checkpoint metrics (from summarize_eval.py --best_checkpoint) instead of "
                         "the pooled training-trajectory average.")
    args = ap.parse_args()

    index_path = os.path.join(HERE, args.runs_dir, "index.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"No index at {index_path}. Run ablations/backfill_index.py --runs-dir {args.runs_dir} first.")
    with open(index_path) as f:
        index = json.load(f)
    man = _manifest()

    records = []
    for name, info in index.items():
        run_dir = info.get("run_dir")
        summ = os.path.join(run_dir, args.summary_name) if run_dir else None
        group, rationale = man.get(name, ("", ""))
        if not summ or not os.path.exists(summ):
            records.append({"ablation": name, "group": group, "status": "no summary",
                            "run_dir": run_dir})
            continue
        df = pd.read_csv(summ)
        sel = df[(df["split"] == args.split) & (df["group"] == args.group)]
        if sel.empty:
            records.append({"ablation": name, "group": group, "status": f"no {args.split}/{args.group} row",
                            "run_dir": run_dir})
            continue
        r = sel.iloc[0]
        rec = {"ablation": name, "group": group, "status": "ok",
               "n": int(r.get("n_heldout_visits", 0))}
        for m in args.metrics:
            rec[m] = r.get(f"{m}_mean", np.nan)
            rec[f"{m}_se"] = r.get(f"{m}_se", np.nan)
        if "area_MAE_mm2" in r and not pd.isna(r.get("area_MAE_mm2")):
            rec["areaMAE_mm2"] = float(r["area_MAE_mm2"])
        if "mono_decrease_mm2" in r and not pd.isna(r.get("mono_decrease_mm2")):
            rec["monoDec_mm2"] = float(r["mono_decrease_mm2"])
        ckpt = os.path.join(run_dir, "checkpoint_best.pth")
        rec["best_ckpt"] = ckpt if os.path.exists(ckpt) else "(missing)"
        rec["run_dir"] = run_dir
        records.append(rec)

    out = pd.DataFrame(records)
    # order by ablation group then name; baseline first
    out["_grp"] = out["ablation"].map(lambda n: man.get(n, ("zz", ""))[0])
    out = out.sort_values(by=["_grp", "ablation"],
                          key=lambda s: s.map(lambda x: "" if x == "baseline" else x)).drop(columns="_grp")
    out.to_csv(args.out, index=False)

    # pretty print
    print(f"\n==== GAP-INR ablation comparison  (split={args.split}, held-out={args.group}) ====")
    hdr = f"{'ablation':<16}{'grp':<12}" + "".join(f"{m:<16}" for m in args.metrics)
    has_area = "areaMAE_mm2" in out.columns
    has_mono = "monoDec_mm2" in out.columns
    if has_area:
        hdr += f"{'areaMAE':<10}"
    if has_mono:
        hdr += f"{'monoDec':<10}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in out.iterrows():
        if r.get("status") != "ok":
            print(f"{r['ablation']:<16}{str(r.get('group','')):<12}{r.get('status','?')}")
            continue
        line = f"{r['ablation']:<16}{str(r.get('group','')):<12}"
        for m in args.metrics:
            mu, se = r.get(m, np.nan), r.get(f"{m}_se", np.nan)
            line += f"{(f'{mu:.3f}±{se:.3f}' if pd.notna(mu) else '-'):<16}"
        if has_area:
            am = r.get("areaMAE_mm2", np.nan)
            line += f"{(f'{am:.3f}' if pd.notna(am) else '-'):<10}"
        if has_mono:
            md = r.get("monoDec_mm2", np.nan)
            line += f"{(f'{md:.3f}' if pd.notna(md) else '-'):<10}"
        print(line)
    print(f"\nSaved: {args.out}")

    # ---- Δ-vs-baseline verdicts: each one-knob round-3 ablation reads as a yes/no answer ----
    if args.baseline:
        HIGHER = {"DICE", "Precision", "Recall", "PSNR", "SSIM", "IoU"}   # everything else is lower-is-better
        bl = out[(out["ablation"] == args.baseline) & (out["status"] == "ok")]
        if bl.empty:
            print(f"\n[Δ] baseline '{args.baseline}' has no {args.split}/{args.group} data -- skipping verdicts.")
        else:
            b = bl.iloc[0]
            dmetrics = list(args.metrics) + (["areaMAE_mm2"] if has_area else []) + (["monoDec_mm2"] if has_mono else [])
            key = [m for m in ("DICE", "areaMAE_mm2") if m in dmetrics]   # verdict = seg quality + lesion size
            print(f"\n==== Δ vs baseline '{args.baseline}'  (split={args.split}, held-out={args.group}) "
                  f"-- one-knob architectural verdicts ====")
            hdr = f"{'ablation':<18}" + "".join(f"{m:<13}" for m in dmetrics) + "verdict"
            print(hdr); print("-" * len(hdr))
            for _, r in out.iterrows():
                if r.get("status") != "ok" or r["ablation"] == args.baseline:
                    continue
                line = f"{r['ablation']:<18}"; score = 0
                for m in dmetrics:
                    mu, bmu = r.get(m, np.nan), b.get(m, np.nan)
                    if pd.isna(mu) or pd.isna(bmu):
                        line += f"{'-':<13}"; continue
                    d = float(mu) - float(bmu); better = (d > 0) if m in HIGHER else (d < 0)
                    se = float(np.hypot(r.get(f"{m}_se", 0) or 0, b.get(f"{m}_se", 0) or 0))
                    mark = ("+" if better else "-") if abs(d) > se else "~"     # ~ = within 1 SE
                    line += f"{f'{d:+.3f}{mark}':<13}"
                    if m in key and abs(d) > se:
                        score += 1 if better else -1
                print(line + ("HELPS" if score > 0 else "HURTS" if score < 0 else "~ neutral"))
            print("  (+ better / - worse beyond 1 SE, ~ within 1 SE; verdict from DICE + lesion-area MAE.)")

    # LaTeX
    tex = os.path.join(HERE, "comparison.tex")
    okrows = out[out["status"] == "ok"] if "status" in out.columns else out
    with open(tex, "w") as f:
        cols = "l" + "r" * len(args.metrics) + ("r" if has_area else "")
        f.write("\\begin{tabular}{" + cols + "}\n\\toprule\n")
        f.write("Ablation & " + " & ".join(args.metrics) + (" & area MAE" if has_area else "") + " \\\\\n\\midrule\n")
        for _, r in okrows.iterrows():
            cells = [r["ablation"].replace("_", "\\_")]
            for m in args.metrics:
                mu, se = r.get(m, np.nan), r.get(f"{m}_se", np.nan)
                cells.append(f"{mu:.3f}$\\pm${se:.3f}" if pd.notna(mu) else "--")
            if has_area:
                am = r.get("areaMAE_mm2", np.nan)
                cells.append(f"{am:.3f}" if pd.notna(am) else "--")
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"LaTeX:  {tex}")


if __name__ == "__main__":
    main()
