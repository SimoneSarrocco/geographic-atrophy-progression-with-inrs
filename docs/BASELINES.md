# Comparison baselines

The paper compares GAP-INR against the **ImageFlowNet** model family. All baselines are trained and
evaluated on the **same official split** and the **same test eyes** as GAP-INR (train/val/test are
defined by the `split` column of the clinical CSV; see [`DATA.md`](DATA.md) and
`configs/expected_split.yaml`), so every reported number is directly comparable.

The baseline code is **not** part of this repository — it is a separate, lightly adapted copy of the
public [ImageFlowNet](https://github.com/ChenLiu-1996/ImageFlowNet) repository, pointed at the same
data loader / split. This document records the exact protocol so the comparison is reproducible.

## Methods

| Paper name | `--model` flag | What it is |
|---|---|---|
| **ImageFlowNet (ODE)** | `ImageFlowNetODE` | neural-ODE latent forecaster |
| **Time-conditioned U-Net** (T-UNet) | `T_UNet` | guided-diffusion U-Net, sinusoidal time embedding + FiLM |
| **Time-aware diffusion** (T-I2SBUNet) | `I2SBUNet` | Image-to-Image Schrödinger Bridge diffusion |

All three share one harness. T-UNet and T-I2SBUNet are the two comparison methods used by the
ImageFlowNet paper itself.

## Protocol

The ImageFlowNet family outputs **images only**, so geographic-atrophy DICE is computed by passing
the predicted FAF through a **single frozen segmentor trained once and reused for all three
forecasters** (so DICE differences reflect the forecaster, not the segmentor).

```bash
# 0. shared segmentor (train once)
python train_segmentor.py   --dataset-name retina_faf_ga --target-dim '(256,256)' --segmentor-ckpt "$SEG"

# 1-3. train each forecaster
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model ImageFlowNetODE --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model T_UNet        --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model I2SBUNet --diffusion-interval 100 --segmentor-ckpt "$SEG"

# evaluation (paper numbers): PSNR/SSIM + DICE(segmentor(pred FAF), real GA mask), bucketed into
# interpolation (target is a middle visit) vs extrapolation (target is the last visit).
python eval_omega.py --model ImageFlowNetODE --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_omega.py --model T_UNet          --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_omega.py --model I2SBUNet        --target-dim '(256,256)' --diffusion-interval 100 --segmentor-ckpt "$SEG"
```

### Paper configuration

- **Resolution 256×256, seed 1** — this is the configuration reported in the paper.
- `--diffusion-interval` (I2SBUNet only, default 100) is the number of diffusion steps and **must
  match** between training and evaluation. T-UNet / ImageFlowNetODE ignore it.
- **Scenario 1** = single-pair forecast (older visit → newer visit). **Scenario 2** = full patient
  history via test-time adaptation. T-I2SBUNet has no test-time-adaptation mechanism, so **Scenario 2
  is not applicable to it**.

Each run writes a `leave_one_out_summary_test.csv` in the same format as GAP-INR's
[`summarize_eval.py`](../summarize_eval.py), so the final comparison table reads identical fields
(DICE / PSNR / SSIM / Hausdorff distance / lesion-area MAE, interpolation vs extrapolation) across
every method.
