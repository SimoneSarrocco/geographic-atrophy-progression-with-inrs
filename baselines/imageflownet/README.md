# ImageFlowNet baselines

The comparison methods reported in the paper, adapted from [ImageFlowNet](https://github.com/ChenLiu-1996/ImageFlowNet)
to the same cohort, the same split as GAP-INR. Provenance, the list of
changes made for this paper, and the licensing of this directory are in [ATTRIBUTION.md](ATTRIBUTION.md).

> **Licence.** This directory is **not** covered by the repository's Apache-2.0 licence. It is a
> derivative of ImageFlowNet and is governed by the Yale Non-Commercial licence in
> [LICENSE.md](LICENSE.md); the vendored `external_src/I2SB/` carries NVIDIA's own non-commercial
> licence. Research and evaluation use only.

## Methods

| Paper name | `--model` | What it is |
|---|---|---|
| ImageFlowNet (ODE) | `ImageFlowNetODE` | neural-ODE latent forecaster |
| Time-conditioned U-Net (T-UNet) | `T_UNet` | guided-diffusion U-Net, sinusoidal time embedding + FiLM |
| Time-aware diffusion (T-I2SBUNet) | `I2SBUNet` | Image-to-Image Schrödinger Bridge diffusion |

All three share one harness. T-UNet and T-I2SBUNet are the two comparison methods used by the
ImageFlowNet paper itself.

## Why the numbers are comparable to GAP-INR

These models output **images only**, so GA segmentation comes from a single frozen segmentor trained
once and reused for all three forecasters. Everything else is shared with GAP-INR by construction rather than by convention:

- `eval_spec.py` makes sure that we use exactly GAP-INR's patient-wise data split. A split drift fails instead of
  silently producing an incomparable table.
- `common_preproc.py` is the same intensity normalisation GAP-INR uses.
- The loader center-crops the native 768 to 620 (a direct 512 crop clips GA in ~5/133 visits) and
  resizes to 512; predictions and ground truth are built identically and scored on that grid.

Both files read the repo's `data/clinical_metadata.csv` by default (`GAPINR_CSV` overrides the path),
so the baselines and GAP-INR read one table. See [`../../docs/DATA.md`](../../docs/DATA.md).

## Install

```bash
pip install -r ../../requirements.txt -r requirements.txt
```

## Run

All commands are run from `src/`. `$ROOT` expands to this directory.

```bash
cd baselines/imageflownet/src
SEG='$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty'

# 0. shared segmentor (train once, reused by all three forecasters)
python train_segmentor.py --dataset-name retina_faf_ga --target-dim '(256,256)' --segmentor-ckpt "$SEG"

# 1-3. train each forecaster
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model ImageFlowNetODE --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model T_UNet         --segmentor-ckpt "$SEG"
python train_2pt_all.py --dataset-name retina_faf_ga --target-dim '(256,256)' --model I2SBUNet --diffusion-interval 100 --segmentor-ckpt "$SEG"

# 4. evaluate (the paper numbers)
python eval_faf_ga.py --model ImageFlowNetODE --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model T_UNet          --target-dim '(256,256)' --segmentor-ckpt "$SEG"
python eval_faf_ga.py --model I2SBUNet        --target-dim '(256,256)' --diffusion-interval 100 --segmentor-ckpt "$SEG"
```

Each evaluation writes `leave_one_out_summary_test_<best_type>.csv` into the run directory
(`..._seg_dice.csv` with the default `--best-type`), in the same format as
GAP-INR's [`summarise_eval.py`](../../summarise_eval.py), so the comparison table reads identical
fields across every method: DICE, PSNR, SSIM, LPIPS, Hausdorff distance and lesion-area MAE, split into
interpolation and extrapolation. A copy-forward reference (predict the source or last available GT visit unchanged) is
scored alongside, as the floor any model has to beat.

### Paper configuration

- **Resolution 256×256, seed 1.**
- `--diffusion-interval` applies to `I2SBUNet` only and **must match** between training and
  evaluation; the other two models ignore it.
- **Scenario 1** is a single-pair forecast (older visit → newer visit); **scenario 2** uses the full
  patient history via test-time adaptation. T-I2SBUNet has no test-time-adaptation mechanism, so
  scenario 2 does not apply to it.

## Classical floors

`comparison/interpolation/` scores the references: linear
and cubic-spline pixel interpolation, and
copy-forward. `run_seg_interp.py` needs only numpy/scipy/PIL and the shared spec: no GPU, no
segmentor, because it interpolates the GA masks directly.

## Layout

```
eval_spec.py        canonical data split, shared with GAP-INR 
common_preproc.py   canonical intensity normalisation, shared with GAP-INR
monitor_panel.py    shared TensorBoard validation panel
src/
  train_segmentor.py    trains the shared frozen GA segmentor
  train_2pt_all.py      trains a model (--model selects which)
  eval_faf_ga.py        leave-one-visit-out evaluation against the real GA masks
  datasets/             the cohort loader
  nn/                   model implementations
  utils/, data_utils/   upstream helpers
external_src/I2SB/  I2SB + guided-diffusion (see ATTRIBUTION.md)
comparison/         classical interpolation/extrapolation floors
```
