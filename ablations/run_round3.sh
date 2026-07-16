#!/usr/bin/env bash
# Round-3 ablations WITHOUT retraining the baseline.
#
# r3_base (the round-3 reference) is BYTE-IDENTICAL to the already-trained round-2 winner
# r2_a9_chan256 -- same _base_620 + A9 raw-scalar FiLM + latent [256,32,32] + seed 1927; only
# output_dir differs (verify: diff <(grep -v '^#' configs/r3_base.yaml     | grep -v output_dir)
#                                <(grep -v '^#' configs/r2_a9_chan256.yaml | grep -v output_dir)).
# So we REUSE that run as the reference and train only the 7 architectural ablations.
#
# Usage:  bash ablations/run_round3.sh 0,1,2,3      # comma-separated GPU ids
set -euo pipefail
GPUS="${1:?usage: run_round3.sh <comma-separated GPU ids, e.g. 0,1,2,3>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
# The trained round-2 winner lives under runs_r2/ (override with RUNS_DIR=...); the NEW round-3 runs
# + index.json + comparison live under runs/ (OUTRUNS), where run_ablations.py writes them.
RUNS="${RUNS_DIR:-$HERE/runs_r2}"
OUTRUNS="$HERE/runs"
BASE_RUN="$RUNS/r2_a9_chan256"       # the trained round-2 winner (== r3_base)

# 1) Reuse the round-2 reference as the round-3 reference (NO retrain).
if [ ! -d "$BASE_RUN" ] && [ ! -L "$BASE_RUN" ]; then
  echo "ERROR: $BASE_RUN not found. Round-3 reuse needs the trained round-2 winner r2_a9_chan256." >&2
  echo "       set RUNS_DIR=<dir holding r2_a9_chan256> if it lives elsewhere." >&2
  exit 1
fi
mkdir -p "$OUTRUNS"
# symlink into runs/ (next to the new r3_* runs) so name-based lookups resolve to the trained winner
if [ ! -e "$OUTRUNS/r3_base" ]; then
  ln -s "$(cd "$(dirname "$BASE_RUN")" && pwd)/$(basename "$BASE_RUN")" "$OUTRUNS/r3_base"
  echo "linked runs/r3_base -> $BASE_RUN"
fi
# register r3_base in the run index so compare_ablations treats it as the reference
python - "$REPO_ROOT" "$BASE_RUN" <<'PY'
import json, os, sys
gap_inr, base_run = sys.argv[1], sys.argv[2]
idx_path = os.path.join(gap_inr, "ablations", "runs", "index.json")
idx = {}
if os.path.exists(idx_path):
    try: idx = json.load(open(idx_path)) or {}
    except Exception: idx = {}
idx["r3_base"] = {"config": "ablations/configs/r3_base.yaml", "run_dir": os.path.abspath(base_run)}
json.dump(idx, open(idx_path, "w"), indent=2)
print("registered r3_base ->", os.path.abspath(base_run))
PY

# 2) Train ONLY the 7 architectural ablations (r3_base excluded).
python "$REPO_ROOT/ablations/run_ablations.py" --gpus "$GPUS" \
  r3_recon_only_tto r3_sr0_segonly r3_sr1_lightrecon \
  r3_shared_output r3_temporal_dice r3_mono_penalty

echo
echo "Round 3 done. Build the comparison table:  python ablations/compare_ablations.py"
