#!/usr/bin/env python
"""Train + evaluate every GAP-INR ablation on the SAME 620 split, one command.

For each ablation config it:
  1. runs `run.py --config_atlas <config>` (trains epochs.train, validates every epoch
     with leave-one-out, and runs test() at the end loading checkpoint_best.pth),
  2. captures the run's output directory from stdout,
  3. runs `summarize_eval.py` on it -> leave_one_out_summary.csv (interp/extrap/areaMAE),
  4. records {name -> run_dir, best metrics} in ablations/runs/index.json.

Each config is fully self-contained, so you can ALSO run any single ablation by hand:
    python run.py --config_atlas ablations/configs/a3_latent128.yaml
    python summarize_eval.py --eval_dir <printed output dir>

Usage:
    python ablations/run_ablations.py                 # all configs in ablations/configs/
    python ablations/run_ablations.py a3_latent128 a4_sr0   # only these
    python ablations/run_ablations.py --dry_run       # print commands only
    python ablations/run_ablations.py --skip_existing # skip names already in index.json

After it finishes (or any time):  python ablations/compare_ablations.py
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CONFIG_DIR = os.path.join(HERE, "configs")
INDEX_PATH = os.path.join(HERE, "runs", "index.json")


def _load_index():
    # index.json is a regenerable run registry (gitignored, per-server). Tolerate a missing,
    # empty, or partially-written/corrupt file -> start fresh instead of crashing the launcher.
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH) as f:
                data = f.read().strip()
            return json.loads(data) if data else {}
        except (json.JSONDecodeError, ValueError):
            print(f"[run_ablations] WARNING: {INDEX_PATH} is empty/corrupt -- starting a fresh index.")
            return {}
    return {}


def _save_index(idx):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(idx, f, indent=2)


def _configs(names):
    if names:
        return [os.path.join(CONFIG_DIR, f"{n}.yaml") for n in names]
    return sorted(
        os.path.join(CONFIG_DIR, f) for f in os.listdir(CONFIG_DIR) if f.endswith(".yaml")
    )


def run_one(cfg_path, python, dry_run, gpu=None, log_path=None):
    """Train one ablation (+ summarize). If `gpu` is set, pin it via CUDA_VISIBLE_DEVICES; if
    `log_path` is set (parallel mode), stream output to that file instead of the console."""
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    rel_cfg = os.path.relpath(cfg_path, REPO_ROOT)
    train_cmd = [python, "run.py", "--config_atlas", rel_cfg]
    tag = f"[{name}]" + (f"[gpu {gpu}]" if gpu is not None else "")
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"=== {tag} {' '.join(train_cmd)} (cwd={REPO_ROOT})"
          f"{' -> ' + log_path if log_path else ''} ===")
    if dry_run:
        return name, None

    # Stream + capture stdout so we can recover the run's output directory.
    run_dir = None
    logf = open(log_path, "w") if log_path else None
    try:
        proc = subprocess.Popen(train_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in proc.stdout:
            (logf.write if logf else sys.stdout.write)(line)
            if logf:
                logf.flush()
            m = re.search(r"Output directory:\s*(\S+)", line)
            if m:
                run_dir = m.group(1).strip()
        proc.wait()
    finally:
        if logf:
            logf.close()
    if proc.returncode != 0:
        print(f"{tag} FAILED (exit {proc.returncode})" + (f" -- see {log_path}" if log_path else ""))
        return name, None
    if run_dir and not os.path.isabs(run_dir):
        run_dir = os.path.normpath(os.path.join(REPO_ROOT, run_dir))
    if not run_dir or not os.path.isdir(run_dir):
        print(f"{tag} could not locate run dir (got {run_dir!r})")
        return name, None

    # Summarize (interp/extrap/areaMAE on val + test) from the metric JSONs.
    summ_cmd = [python, "summarize_eval.py", "--eval_dir", run_dir]
    print(f"=== {tag} summarize -> {run_dir} ===")
    subprocess.run(summ_cmd, cwd=REPO_ROOT, check=False, env=env)
    print(f"{tag} DONE -> {run_dir}")
    return name, run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="ablation names (default: all)")
    ap.add_argument("--python", default=sys.executable, help="python interpreter for run.py")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip names already present in runs/index.json")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids to run ablations IN PARALLEL (one run per GPU at a "
                         "time, the rest queued as GPUs free up); e.g. --gpus 0,1,2,3. Each run logs "
                         "to ablations/runs/<name>/train.log. Omit to run sequentially.")
    args = ap.parse_args()

    if not os.path.isdir(CONFIG_DIR) or not os.listdir(CONFIG_DIR):
        sys.exit("No configs found. Run: python ablations/make_configs.py first.")

    index = _load_index()
    todo = []
    for cfg in _configs(args.names):
        name = os.path.splitext(os.path.basename(cfg))[0]
        if args.skip_existing and name in index and index[name].get("run_dir"):
            print(f"[{name}] skip (already in index)")
            continue
        todo.append((name, cfg))

    if args.gpus and not args.dry_run:
        # Parallel: a worker per GPU pulls from a shared queue, so with N GPUs up to N ablations
        # train at once and any extras start as GPUs free up.
        import threading, queue
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
        q = queue.Queue()
        for item in todo:
            q.put(item)
        results, lock = [], threading.Lock()

        def worker(gpu):
            while True:
                try:
                    name, cfg = q.get_nowait()
                except queue.Empty:
                    return
                log_path = os.path.join(HERE, "runs", name, "train.log")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                nm, run_dir = run_one(cfg, args.python, False, gpu=gpu, log_path=log_path)
                if run_dir:
                    with lock:
                        results.append((nm, cfg, run_dir))

        print(f"Launching {len(todo)} ablation(s) across GPUs {gpus} "
              f"(live logs: tail -f ablations/runs/<name>/train.log)")
        threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for nm, cfg, run_dir in results:
            index[nm] = {"config": os.path.relpath(cfg, REPO_ROOT), "run_dir": run_dir}
        _save_index(index)
    else:
        for name, cfg in todo:
            nm, run_dir = run_one(cfg, args.python, args.dry_run)
            if run_dir:
                index[nm] = {"config": os.path.relpath(cfg, REPO_ROOT), "run_dir": run_dir}
                _save_index(index)

    if not args.dry_run:
        print(f"\nAll done. Index -> {INDEX_PATH}")
        print("Build the comparison table:  python ablations/compare_ablations.py")


if __name__ == "__main__":
    main()
