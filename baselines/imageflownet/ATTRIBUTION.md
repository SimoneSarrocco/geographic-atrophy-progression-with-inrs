# Attribution

This directory is a derivative of **ImageFlowNet**, adapted to run the paper's comparison baselines
on the longitudinal FAF / geographic-atrophy cohort. The model implementations, the training harness
(`train_2pt_all.py`), the segmentor trainer (`train_segmentor.py`) and the surrounding utilities are
upstream work; see "What was changed" below for the parts added for this paper.

ImageFlowNet: Forecasting Multiscale Image-Level Trajectories of Disease Progression with
Irregularly-Sampled Longitudinal Medical Images — Chen Liu, Ke Xu, Liangbo L. Shen, Guillaume
Huguet, Zilong Wang, Alexander Tong, Danilo Bzdok, Jay Stewart, Jay C. Wang, Lucian V. Del Priore,
Smita Krishnaswamy. https://github.com/ChenLiu-1996/ImageFlowNet

## Licensing

| Path | License | Terms |
|---|---|---|
| this directory | Non-Commercial License, Yale University © 2024 — [`LICENSE.md`](LICENSE.md) | non-commercial use only |
| `external_src/I2SB/` | NVIDIA Source Code License for I2SB — [`external_src/I2SB/LICENSE`](external_src/I2SB/LICENSE) | non-commercial use only |
| `external_src/I2SB/guided_diffusion/` | MIT © 2021 OpenAI — [`LICENSE_GUIDED_DIFFUSION`](external_src/I2SB/guided_diffusion/LICENSE_GUIDED_DIFFUSION) | permissive |

The rest of the GAP-INR repository is Apache-2.0. That license does **not** extend to this directory;
see the "Scope of this license" note at the end of the repository's top-level `LICENSE`. Both the
Yale and NVIDIA licenses restrict use to non-commercial research or evaluation.

`external_src/I2SB/LICENSE` was retrieved from the upstream I2SB project
(https://github.com/NVlabs/I2SB); the vendored copy of I2SB carried the source headers referencing
it but not the file itself.

The vendored I2SB code (NVIDIA) and guided-diffusion code (OpenAI) reach this directory through
ImageFlowNet, which vendored them; they are reproduced unmodified.

## What was changed

Added for this paper:

- `src/datasets/retina_faf_ga.py` — cohort loader: reads the clinical CSV directly, one subject per
  patient-eye, weeks-from-baseline as the time axis, 620 crop then resize, and the official
  eye-level split exposed as `predefined_split`.
- `src/eval_faf_ga.py` — unified evaluation for all three forecasters: leave-one-visit-out folds
  bucketed into interpolation vs extrapolation, GA scored by a shared frozen segmentor, and
  test-time adaptation (scenario 2).
- `eval_spec.py`, `common_preproc.py` — the split contract and the intensity normalisation, shared
  verbatim with GAP-INR so every method sees identical data.
- `monitor_panel.py` — the shared TensorBoard validation panel, so runs from either method are
  comparable by eye.
- `comparison/interpolation/` — the classical interpolation/extrapolation floors.

Modified upstream files: `src/train_2pt_all.py` and `src/train_segmentor.py` (cohort support, the
`--crop-size` option, segmentor-based GA metrics), `src/data_utils/prepare_dataset.py` (honour the
predefined split), and small fixes in `src/nn/imageflownet_ode.py`, `src/nn/unet_i2sb.py` and
`src/nn/unet_t_emb.py`.

Removed from the upstream tree: the other cohorts (AREDS, UCSF, brain MS/GBM, synthetic) and their
preprocessing/registration scripts, the `external_src/SuperRetina` registration model, the
neural-process training entry points, and upstream's pretrained checkpoints and result archives —
none of which the paper's comparison uses.
