#!/usr/bin/env bash
# =============================================================================
# GAP-INR end-to-end pipeline: TRAIN -> EVAL (val+test LOO) -> DIAGNOSTICS.
#
# One command runs the whole thing on an automatically-picked free GPU and
# writes every log under the run directory so you can monitor it in one place.
#
#   ./run_pipeline.sh                         # full pipeline, auto GPU, default config
#   ./run_pipeline.sh --gpu 3                 # force a specific GPU
#   ./run_pipeline.sh --config_model configs/config_model.yaml
#   ./run_pipeline.sh --skip-train --run tmp/omega_20260624_204828_loc   # eval+diag only
#   ./run_pipeline.sh --stages train,eval     # subset of stages (train,eval,tsens,traj)
#
# Stages:
#   train  -> run.py                       (validate_splits guard runs first)
#   eval   -> evaluate.py (val LOO + test LOO) -> summarize_eval (auto)
#   tsens  -> temporal_sensitivity.py      (static-collapse check; train, no TTA)
#   traj   -> plot_trajectories.py         (multi-patient trajectory figure, test)
# =============================================================================
set -euo pipefail

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPU=""
RUN_DIR=""
CONFIG_MODEL="configs/config_model.yaml"
STAGES="train,eval,tsens,traj"
SKIP_TRAIN=0
EPOCHS_VAL=""          # optional override for eval/tsens TTA epochs

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --run) RUN_DIR="$2"; shift 2;;
    --config_model) CONFIG_MODEL="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --skip-train) SKIP_TRAIN=1; STAGES="${STAGES/train,/}"; shift;;
    --epochs_val) EPOCHS_VAL="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

# ---- pick the freest GPU (most free memory) if not forced ----
if [[ -z "$GPU" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
export CUDA_VISIBLE_DEVICES="$GPU"
echo "==> Using GPU $GPU"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader -i "$GPU" || true

# =============================================================================
# 1) TRAIN
# =============================================================================
if has_stage train && [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo; echo "================ STAGE: TRAIN ================"
  # run.py timestamps its own output dir under output_dir/. Capture it from the log.
  TRAIN_LOG="$(mktemp)"
  $PY run.py --config_model "$CONFIG_MODEL" 2>&1 | tee "$TRAIN_LOG"
  RUN_DIR=$(grep -oE "Output directory: .*" "$TRAIN_LOG" | tail -1 | sed 's/Output directory: //')
  echo "==> Train run dir: $RUN_DIR"
fi

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: no run dir. Pass --run <dir> (or include the train stage)."; exit 1
fi
RUN_DIR="${RUN_DIR%/}"

# pick the checkpoint: prefer best (best val-eval DICE), else the latest epoch ckpt.
pick_ckpt() {
  if [[ -f "$RUN_DIR/checkpoint_best.pth" ]]; then
    echo "$RUN_DIR/checkpoint_best.pth"
  else
    ls -1v "$RUN_DIR"/checkpoint_epoch_*.pth 2>/dev/null | tail -1
  fi
}
CKPT="$(pick_ckpt)"
echo "==> Checkpoint: ${CKPT:-<none>}"

EV_ARGS=()
[[ -n "$EPOCHS_VAL" ]] && EV_ARGS+=(--epochs_val "$EPOCHS_VAL")

# =============================================================================
# 2) EVAL  (val LOO for model selection + test LOO for the paper number)
#    evaluate.py auto-runs summarize_eval.py at the end (interp vs extrap table).
# =============================================================================
if has_stage eval; then
  echo; echo "================ STAGE: EVAL ================"
  [[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint in $RUN_DIR"; exit 1; }
  $PY evaluate.py --checkpoint "$CKPT" --holdout_strategy leave_one_out --test on \
      "${EV_ARGS[@]}" 2>&1 | tee "$RUN_DIR/eval.log"
fi

# =============================================================================
# 3) TSENS  (static-collapse diagnostic on TRAIN eyes, no TTA -> fast)
# =============================================================================
if has_stage tsens; then
  echo; echo "================ STAGE: TEMPORAL SENSITIVITY ================"
  [[ -z "$CKPT" ]] && { echo "ERROR: no checkpoint in $RUN_DIR"; exit 1; }
  $PY temporal_sensitivity.py --checkpoint "$CKPT" --split train \
      2>&1 | tee "$RUN_DIR/tsens.log"
fi

# =============================================================================
# 4) TRAJ  (multi-patient predicted-trajectory figure from the test lesion CSV)
# =============================================================================
if has_stage traj; then
  echo; echo "================ STAGE: TRAJECTORIES ================"
  CSV=$(ls -1v "$RUN_DIR"/evaluation_*/lesion_analysis/lesion_areas_test_epoch_*.csv 2>/dev/null | tail -1 || true)
  if [[ -n "$CSV" ]]; then
    $PY plot_trajectories.py --csv "$CSV" --split test 2>&1 | tee "$RUN_DIR/traj.log"
  else
    echo "  (no test lesion_areas CSV found yet; run the eval stage first)"
  fi
fi

echo
echo "================ PIPELINE DONE ================"
echo "Run dir : $RUN_DIR"
echo "Monitor : tensorboard --logdir $RUN_DIR/tb_logs"
echo "Key outputs:"
echo "  - eval summary : $RUN_DIR/evaluation_*/leave_one_out_summary.csv  (interp vs extrap DICE/IoU/areaMAE)"
echo "  - collapse chk : $RUN_DIR/temporal_sensitivity_*/*summary.json     (COHORT VERDICT)"
echo "  - trajectories : $RUN_DIR/evaluation_*/lesion_analysis/*overlay.png + *grid.png"
