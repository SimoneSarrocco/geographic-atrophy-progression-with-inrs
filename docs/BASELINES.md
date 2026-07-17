# Comparison baselines

The paper compares GAP-INR against the **ImageFlowNet** model family. The code is included, in
[`baselines/imageflownet/`](../baselines/imageflownet/), together with the run commands and the
protocol: see its [README](../baselines/imageflownet/README.md) to reproduce the baseline numbers,
and its [ATTRIBUTION.md](../baselines/imageflownet/ATTRIBUTION.md) for provenance and licensing.

> That directory is a derivative of [ImageFlowNet](https://github.com/ChenLiu-1996/ImageFlowNet) and
> is **not** covered by this repository's Apache-2.0 licence — it is governed by the Yale
> Non-Commercial licence shipped alongside it. See the "Scope of this license" note at the end of the
> top-level [`LICENSE`](../LICENSE).

## Methods

| Paper name | `--model` flag | What it is |
|---|---|---|
| **ImageFlowNet (ODE)** | `ImageFlowNetODE` | neural-ODE latent forecaster |
| **Time-conditioned U-Net** (T-UNet) | `T_UNet` | guided-diffusion U-Net, sinusoidal time embedding + FiLM |
| **Time-aware diffusion** (T-I2SBUNet) | `I2SBUNet` | Image-to-Image Schrödinger Bridge diffusion |

All three share one harness. T-UNet and T-I2SBUNet are the two comparison methods used by the
ImageFlowNet paper itself.

## What makes the comparison fair

Every method trains and evaluates on the **same eyes**, the **same visits** and is scored on the
**same 512 grid**, enforced in code rather than by convention. Note that the scoring grid and a
method's working resolution are different things: the ImageFlowNet models run at `--target-dim
(256,256)`, and their predictions and the ground truth are both resized to the 512 scoring grid
(`METRIC_DIM`) before any metric is computed, exactly as for GAP-INR.

- **Split.** Both the baselines and GAP-INR read the `split` column of the same clinical CSV
  (see [`DATA.md`](DATA.md)). GAP-INR checks the resolved eyes against
  [`configs/expected_split.yaml`](../configs/expected_split.yaml); the baselines check theirs against
  `baselines/imageflownet/eval_spec.py`, which hard-asserts that the scored test eyes are exactly
  GAP-INR's test eyes.
- **Preprocessing.** `baselines/imageflownet/common_preproc.py` is the same per-visit min-max
  normalisation GAP-INR applies, and the same 620-crop-then-resize geometry.
- **Protocol.** Leave-one-visit-out folds, bucketed into interpolation (a middle visit is held out)
  and extrapolation (the last visit is held out).

The two split contracts count eyes differently on purpose, and both are correct. GAP-INR's
`expected_split.yaml` lists the eyes with a modality path in the CSV (25 train); `eval_spec.py` lists
the *longitudinal* eyes — at least two visits whose FAF **and** GA-mask files are both on disk (23
train). Val (5) and test (6) are identical under both, because every val/test visit is complete. So
the **scored test set is the same for every method** regardless of internal filtering.

## Protocol

The ImageFlowNet family outputs **images only**, so geographic-atrophy DICE is computed by passing
the predicted FAF through a **single frozen segmentor trained once and reused for all three
forecasters** (so DICE differences reflect the forecaster, not the segmentor). GA is scored against
the **real GA masks**, not against the segmentor's reading of the ground-truth image.

```bash
cd baselines/imageflownet/src
SEG='$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty'

# 0. shared segmentor (train once)
python train_segmentor.py --dataset-name retina_faf_ga --target-dim '(256,256)' --segmentor-ckpt "$SEG"

# 1-3. train each forecaster
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model ImageFlowNetODE --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model T_UNet        --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model I2SBUNet --diffusion-interval 100 --segmentor-ckpt "$SEG"

# 4. evaluation (paper numbers): PSNR/SSIM + DICE(segmentor(pred FAF), real GA mask), bucketed into
# interpolation (target is a middle visit) vs extrapolation (target is the last visit).
python eval_faf_ga.py --model ImageFlowNetODE --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model T_UNet          --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model I2SBUNet        --target-dim '(256,256)' --diffusion-interval 100 --segmentor-ckpt "$SEG"
```

### Paper configuration

- **Resolution 256×256, seed 1** — this is the configuration reported in the paper.
- `--diffusion-interval` (I2SBUNet only, default 100) is the number of diffusion steps and **must
  match** between training and evaluation. T-UNet / ImageFlowNetODE ignore it.
- **Scenario 1** = single-pair forecast (older visit → newer visit). **Scenario 2** = full patient
  history via test-time adaptation. T-I2SBUNet has no test-time-adaptation mechanism, so **Scenario 2
  is not applicable to it**.

Each run writes a `leave_one_out_summary_test_<best_type>.csv` (`leave_one_out_summary_test_seg_dice.csv`
with the default `--best-type seg_dice`) in the same format as GAP-INR's
[`summarize_eval.py`](../summarize_eval.py), so the final comparison table reads identical fields
(DICE / PSNR / SSIM / Hausdorff distance / lesion-area MAE, interpolation vs extrapolation) across
every method. A copy-forward reference (predict the source visit unchanged) is scored alongside as
the floor a forecaster has to beat.

Classical non-learned floors (linear / cubic-spline interpolation, growth-rate and linear-regression
area extrapolation, copy-forward) live in `baselines/imageflownet/comparison/interpolation/`.
