#!/usr/bin/env python
"""Back-fill ablations/runs/index.json from run dirs that ALREADY trained but were never
registered (so compare_ablations.py shows them). For each <name> under ablations/runs/:
  - locate its actual output dir (the parent of tb_logs; newest if several),
  - run summarize_eval.py on it if the target summary is missing,
  - add {name: {config, run_dir}} to the index.
No retraining. Idempotent: already-indexed runs with a valid run_dir keep their index entry.

Run from the GAP-INR repo on whichever server holds the runs:
    python ablations/backfill_index.py                  # pooled training-average summary
    python ablations/backfill_index.py --best_checkpoint # ALSO build the best-checkpoint summary
    python ablations/backfill_index.py --best_checkpoint --force   # regenerate even if present

--best_checkpoint runs summarize_eval.py --best_checkpoint, which writes
leave_one_out_summary_best.csv (the selected checkpoint's metrics, not the training average) WITHOUT
touching the pooled leave_one_out_summary.csv. It is also generated for already-indexed runs, so
`compare_ablations.py --summary_name leave_one_out_summary_best.csv` works after one backfill pass.
"""
import os
import sys
import json
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
INDEX = os.path.join(RUNS, "index.json")


def _ensure_summary(run_dir, summary_name, best_checkpoint, force):
    """Run summarize_eval.py for `run_dir` if `summary_name` is missing (or --force)."""
    summ = os.path.join(run_dir, summary_name)
    if os.path.exists(summ) and not force:
        return summ
    cmd = [sys.executable, "summarize_eval.py", "--eval_dir", run_dir]
    if best_checkpoint:
        cmd.append("--best_checkpoint")
    print(f"  summarizing{' (best_checkpoint)' if best_checkpoint else ''} {run_dir}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return summ


def _locate_run_dir(nd):
    """run_dir = the dir that contains tb_logs (run.py's 'Output directory'). Newest wins."""
    cands = glob.glob(os.path.join(nd, "*", "tb_logs")) + glob.glob(os.path.join(nd, "tb_logs"))
    if cands:
        return os.path.dirname(sorted(cands, key=os.path.getmtime)[-1])
    subs = [os.path.join(nd, d) for d in os.listdir(nd) if os.path.isdir(os.path.join(nd, d))]
    return sorted(subs, key=os.path.getmtime)[-1] if subs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best_checkpoint", action="store_true",
                    help="Also build leave_one_out_summary_best.csv (best-checkpoint metrics) for "
                         "every run, including already-indexed ones.")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate the summary even if it already exists.")
    ap.add_argument("--runs-dir", default="runs",
                    help="run-dir root under ablations/ (default 'runs'; use 'runs_r3' for round 3). "
                         "The index.json is written INSIDE this dir, keeping rounds isolated.")
    args = ap.parse_args()
    global RUNS, INDEX
    RUNS = os.path.join(HERE, args.runs_dir)
    INDEX = os.path.join(RUNS, "index.json")

    summary_name = "leave_one_out_summary_best.csv" if args.best_checkpoint else "leave_one_out_summary.csv"
    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}

    for name in sorted(os.listdir(RUNS)):
        nd = os.path.join(RUNS, name)
        if not os.path.isdir(nd):
            continue  # skip index.json and any stray files

        already = name in index and index[name].get("run_dir") and os.path.isdir(index[name]["run_dir"])
        if already:
            # Keep the existing index entry, but still (re)build the requested summary for it -- this is
            # what lets --best_checkpoint backfill the best-checkpoint CSV for previously-indexed runs.
            run_dir = index[name]["run_dir"]
            if args.best_checkpoint or args.force:
                _ensure_summary(run_dir, summary_name, args.best_checkpoint, args.force)
            print(f"[{name}] already indexed -> {run_dir}")
            continue

        run_dir = _locate_run_dir(nd)
        if not run_dir:
            print(f"[{name}] could not locate a run dir (no tb_logs / subdir) -- skipping")
            continue

        summ = _ensure_summary(run_dir, summary_name, args.best_checkpoint, args.force)
        if not os.path.exists(summ):
            print(f"[{name}] WARNING: summarize produced no {summary_name} "
                  f"(no eval JSONs under {run_dir}?) -- indexing anyway, compare will show 'no summary'")

        index[name] = {"config": os.path.join("ablations", "configs", f"{name}.yaml"),
                       "run_dir": run_dir}
        print(f"[{name}] indexed -> {run_dir}")

    json.dump(index, open(INDEX, "w"), indent=2)
    print("\nindex now contains:", sorted(index))


if __name__ == "__main__":
    main()
