# Training and evaluation pipeline

The path from a config to the reported numbers, and what to watch while it runs. The stages are:
train, evaluate (val and test leave-one-out), diagnose (collapse check), compare configs.

Run the whole sequence with one command:

```bash
./run_pipeline.sh                 # picks a free GPU, then trains, evaluates, diagnoses
./run_pipeline.sh --gpu 3         # force a GPU
./run_pipeline.sh --skip-train --run runs/faf_ga_YYYYMMDD_HHMMSS_loc   # eval + diagnose an existing run
./run_pipeline.sh --stages eval,tsens --run <dir>                    # only some stages
```

Everything lands under the run dir (`runs/<run_name>/`): checkpoints, `tb_logs/`, `evaluation_*/`,
`temporal_sensitivity_*/`, and a per-stage `*.log`.

---

## Loss is not the selection metric

- Train loss is uninformative: the per-eye latents memorize the training eyes, so it always goes low.
- Val loss is the reconstruction loss of a recon-only-fit latent. It measures pixel fit, not whether
  the predicted future GA is right; reconstruction and segmentation come apart.
- Both are dominated by the roughly 95% background. GA is about 5% of pixels, so the lesion barely
  moves the loss.

Select and compare configs on the task: held-out GA DICE and lesion-area MAE, reported separately for
interpolation (a middle visit held out) and extrapolation (the last visit held out). Use the
validation rows for selection, and touch the test set once for the final number (comparing configs on
the test set leaks).

---

## Stage 1 — train (`run.py`)

```bash
python run.py --config_model configs/config_model.yaml
```

- `validate_splits()` runs first and aborts if train/val/test overlap or the split is inconsistent.
- Checkpoints: `checkpoint_best.pth` (best held-out validation DICE) plus `checkpoint_epoch_*.pth`
  every `validate_every` epochs.
- With `test.activate: true`, a final test leave-one-out pass runs at the end of training.

What to watch in TensorBoard (`tensorboard --logdir runs/<run>/tb_logs`), in order of usefulness:

1. `val_eval_*/dice` (held-out GA DICE) and the interpolation-vs-extrapolation split. This is the
   signal to select on.
2. `diagnostics/time_sensitivity_mean_diff`, a quick scalar for whether time changes the output.
3. The per-term losses (reconstruction vs segmentation), to check they stay balanced, not to select
   on. Both should keep decreasing.
4. The reconstruction figures (GT, prediction, and difference across the visit sequence).

Decide training length from the held-out-DICE curve rather than the loss. If the collapse check
(Stage 3) reports STATIC, more epochs will not help; change the design instead.

---

## Stage 2 — evaluate (`evaluate.py`)

```bash
# val (selection) + test (final), full leave-one-out, auto-summarized:
python evaluate.py --checkpoint runs/<run>/checkpoint_best.pth \
    --holdout_strategy leave_one_out --test on
```

- Re-fits the val/test latents (TTA), runs leave-one-out over every visit position, and writes metric
  JSONs and figures.
- Calls `summarize_eval.py`, which writes `evaluation_*/leave_one_out_summary.csv`: held-out
  DICE/PSNR/SSIM/IoU (mean ± SE) and lesion-area MAE, grouped into interpolation and extrapolation.
- Flags: `--skip_val` (test only), `--support_k K` (fit on the first K visits, predict the rest),
  `--epochs_val N` (TTA budget), `--test on|off`.

This is the table to report, and the validation rows are the ones to select on.

---

## Stage 3 — diagnose the static-collapse check (`temporal_sensitivity.py`)

This check holds each eye's latent fixed and sweeps only the week, which isolates the time pathway. On
the train split it needs no TTA, since the latents come from the checkpoint.

```bash
python temporal_sensitivity.py --checkpoint runs/<run>/checkpoint_best.pth --split train
# val/test (fits latents on all visits first):
python temporal_sensitivity.py --checkpoint runs/<run>/checkpoint_best.pth --split test
```

Outputs (`temporal_sensitivity_*/`):

- `*_summary.json`: cohort verdict (RESPONSIVE or COLLAPSED), the fraction of static eyes, the median
  absolute area slope, and the median relative range.
- `*.csv`: per eye, `area_slope_mm2_per_wk`, `area_range_mm2`, `area_rel_range`, `seg_change_frac`,
  and the verdict.
- `overlay_*.png` (all eyes, colored by growth rate) and `per_eye_*.png` (area-vs-week small
  multiples).

How to read it:

- COLLAPSED means the model predicts the same image and segmentation over time. A loss or DICE that
  looks fine but stays flat across epochs is often this. More epochs will not help; change the design.
- RESPONSIVE means the time pathway works, and quality can now be tuned with the Stage 2 metrics.

If it collapses, the levers are: use FiLM rather than time-as-input, turn on `seg_branch`, shrink
`latent_dim`, raise the `time`/`cond` Fourier frequencies, or add a temporal-monotonicity penalty on
predicted GA area.

---

## Stage 4 — trajectory figure (`plot_trajectories.py`)

A multi-patient predicted-trajectory figure, showing how the trajectories track different progression
rates.

```bash
python plot_trajectories.py \
    --csv runs/<run>/evaluation_*/lesion_analysis/lesion_areas_test_epoch_*.csv --split test
```

It writes `*_overlay.png` (all eyes, colored by ground-truth slope, held-out visit marked) and
`*_grid.png` (fast to slow).

---

## Comparing several configs

1. Train each config to a fixed budget; the best-validation-DICE checkpoint is selected automatically.
2. Evaluate each and read `leave_one_out_summary.csv` (validation rows): extrapolation area-MAE and
   held-out DICE.
3. Run `temporal_sensitivity.py --split train` on each and drop any COLLAPSED config.
4. Rank the survivors on the primary task metric (extrapolation area-MAE and held-out DICE), with FAF
   PSNR as a guardrail that reconstruction has not collapsed.
5. Run the single winner on the test set, once, for the paper.

Rather than gridding latent resolution against loss weight, fix the resolution at the smallest value
that gives acceptable FAF, then sweep `sr_weight` on its own, selecting on the segmentation metric.

---

## Files

| File | Role |
|------|------|
| `run_pipeline.sh` | One-command orchestrator (GPU pick and all stages). |
| `run.py` | Training (splits guarded by `validate_splits`). |
| `evaluate.py` | Val + test leave-one-out eval; `--test on/off`, `--support_k`, `--skip_val`; auto-summarize. |
| `summarize_eval.py` | Interpolation-vs-extrapolation held-out summary table. |
| `temporal_sensitivity.py` | Static-collapse diagnostic (RESPONSIVE/COLLAPSED). |
| `plot_trajectories.py` | Multi-patient predicted-trajectory figure. |
| `seg_growth.py` | Per-eye GA growth-ring / onset / area-vs-week figure. |
| `verify_run_config.py` | Recover the config a checkpoint was trained with (one latent per eye? weeks as FiLM?). |
