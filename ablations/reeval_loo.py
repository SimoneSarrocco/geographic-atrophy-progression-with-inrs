#!/usr/bin/env python
"""Re-evaluate each registered ablation run under LEAVE-ONE-VISIT-OUT (interpolation +
extrapolation) WITHOUT retraining, then refresh its leave_one_out_summary.csv so
compare_ablations.py shows interp + extrap instead of extrapolation-only.

Each run's existing checkpoint_best.pth is re-evaluated with --holdout_strategy leave_one_out
into <run_dir>/loo_eval, summarized there, and the resulting leave_one_out_summary.csv is copied
over <run_dir>/leave_one_out_summary.csv (the old extrapolation-only one is backed up once to
leave_one_out_summary_extrap.csv). VAL only (--test off) so the held-out TEST eyes are not touched.

Caveat: checkpoint_best was SELECTED on extrapolation-val-DICE; re-scoring it under LOO is fine and
defensible for ablation ranking. If you want selection itself on LOO, retrain (regenerate configs
with the updated make_configs.py, which now defaults to leave_one_out).

Run from the GAP-INR repo on the server that holds the runs:
    python ablations/reeval_loo.py                              # all runs in index.json
    python ablations/reeval_loo.py baseline a1_timeinput_freq6  # a subset
    CUDA_VISIBLE_DEVICES=0 python ablations/reeval_loo.py       # pin a GPU
"""
import os
import sys
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
INDEX = os.path.join(HERE, "runs", "index.json")
PY = sys.executable


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="ablation names (default: all in index.json)")
    ap.add_argument("--holdout", choices=["last", "leave_one_out"], default="leave_one_out",
                    help="holdout strategy for the re-eval. 'last' = fast extrapolation-only "
                         "(1 fold/eye); 'leave_one_out' = interp+extrap (~4x slower). Both now also "
                         "compute LPIPS + full-trajectory monotonicity from checkpoint_best.pth.")
    ap.add_argument("--runs-dir", default="runs",
                    help="run-dir root under ablations/ holding index.json (default 'runs'; use 'runs_r3').")
    args = ap.parse_args()

    INDEX = os.path.join(HERE, args.runs_dir, "index.json")
    if not os.path.exists(INDEX):
        sys.exit(f"No index at {INDEX}. Run ablations/backfill_index.py --runs-dir {args.runs_dir} first.")
    index = json.load(open(INDEX))
    targets = args.names or list(index)
    # Non-destructive: write to a dedicated summary so the existing best-checkpoint summaries are kept.
    summ_name = f"leave_one_out_summary_reeval_{args.holdout}.csv"

    done, failed, lpips_warn = [], [], []
    for name in targets:
        info = index.get(name)
        if not info or not info.get("run_dir"):
            print(f"[{name}] not in index / no run_dir -- skip"); failed.append(name); continue
        run_dir = info["run_dir"]
        if not os.path.exists(os.path.join(run_dir, "checkpoint_best.pth")):
            print(f"[{name}] no checkpoint_best.pth in {run_dir} -- skip"); failed.append(name); continue

        eval_dir = os.path.join(run_dir, f"reeval_{args.holdout}")
        print(f"\n=== [{name}] {args.holdout} re-eval (best ckpt, +LPIPS) -> {eval_dir} ===")
        rc = subprocess.run(
            [PY, "evaluate.py",
             "--checkpoint", run_dir, "--select_by", "dice",
             "--holdout_strategy", args.holdout,
             "--test", "off",
             "--output_dir", eval_dir],
            cwd=REPO_ROOT).returncode
        if rc != 0:
            print(f"[{name}] evaluate.py FAILED (rc {rc}) -- skip"); failed.append(name); continue

        # Single validation pass at the loaded checkpoint -> its summary IS the best-checkpoint result.
        subprocess.run([PY, "summarize_eval.py", "--eval_dir", eval_dir], cwd=REPO_ROOT, check=False)
        src = os.path.join(eval_dir, "leave_one_out_summary.csv")
        if not os.path.exists(src):
            print(f"[{name}] no summary produced under {eval_dir} -- skip"); failed.append(name); continue

        shutil.copy2(src, os.path.join(run_dir, summ_name))
        print(f"[{name}] wrote {os.path.join(run_dir, summ_name)}  (DICE/PSNR/SSIM/LPIPS/areaMAE/monoDec)")
        done.append(name)

        # Guard: the whole point of this pass is LPIPS. If the lpips package didn't import on this
        # server, every eye's LPIPS is empty -> LPIPS_mean is all-NaN and the comparison is useless.
        # Flag it loudly NOW rather than after the full sweep.
        try:
            import csv as _csv
            with open(src, newline="") as _fh:
                _rows = list(_csv.DictReader(_fh))
            _lp = [r.get("LPIPS_mean", "") for r in _rows]
            _has = any(v not in ("", "nan", "NaN") and v == v for v in
                       [float(x) if x not in ("", "nan", "NaN") else float("nan") for x in _lp])
            if not _has:
                lpips_warn.append(name)
                print(f"  !! [{name}] LPIPS_mean is ALL-NaN -- lpips likely not importable here. "
                      f"Check: python -c 'import lpips' in this env.")
        except Exception as _e:
            print(f"  (LPIPS-presence check skipped: {type(_e).__name__}: {_e})")

    if lpips_warn:
        print("\n" + "=" * 70)
        print(f"WARNING: LPIPS was EMPTY (all-NaN) for {len(lpips_warn)} run(s): {lpips_warn}")
        print("The `lpips` package did not import in this env, so the perceptual metric was NOT")
        print("computed. Fix with `pip install lpips` (weights load offline), verify with")
        print("  python -c \"import lpips, torch; lpips.LPIPS(net='alex',verbose=False); print('ok')\"")
        print("then re-run this pass. (Other metrics are valid.)")
        print("=" * 70)

    print(f"\nDone. refreshed={done}  failed={failed}")
    grp = "extrapolation" if args.holdout == "last" else "ALL"
    print(f"\nNow build the comparison WITH LPIPS + monotonicity:")
    print(f"  python ablations/compare_ablations.py --split val --group {grp} \\")
    print(f"      --metrics DICE PSNR SSIM LPIPS LOSS --summary_name {summ_name} \\")
    print(f"      --out ablations/comparison_reeval_{args.holdout}.csv")


if __name__ == "__main__":
    main()
