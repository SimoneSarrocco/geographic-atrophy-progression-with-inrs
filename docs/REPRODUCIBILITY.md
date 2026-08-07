# Reproducing the paper

This is the end-to-end recipe to reproduce the paper's numbers. It assumes the environment is set up
(`conda env create -f environment.yml && conda activate gap-inr`) and the data is in place per
[`DATA.md`](DATA.md).

## 0. Split and determinism

- The train/val/test partition (25 / 5 / 6 eyes) is fixed by the CSV `split` column and guarded by
  `configs/expected_split.yaml`; every run aborts on a mismatch.
- Seeds are set from the config (`seed`); the best model uses `seed: 1927`. GPU nondeterminism can
  cause small run-to-run variation; report over the fixed split.
- **Selection discipline:** choose models on the **validation** leave-one-out metric; touch the
  **test** set once for the final number. Comparing configs on test is leakage.

## 1. Best GAP-INR model

[`configs/config_model.yaml`](../configs/config_model.yaml) ships the paper's configuration, so it
reproduces the main result unchanged. Key hyperparameters:

| Group | Setting |
|---|---|
| Decoder | SIREN, `hidden_size: 384`, `num_hidden_layers: 7` (⇒ 8 sine layers), all FiLM-modulated (`modulated_layers: [0..7]`) |
| SIREN frequency | `omega_0 = omega_start = omega_end = 30`, `schedule_type: constant` |
| Latent | `latent_dim: [256, 32, 32]` (256 channels on a 32×32 grid), one per eye |
| Conditioning | FiLM vector of two scalars: weeks-from-baseline (the temporal axis) and age at visit (`faf_ga_twovar_wktemporal_512`; `cond_dims` is derived from the enabled conditions at runtime) |
| Heads / loss | FAF reconstruction (MSE) + GA segmentation (CE + Dice); `sr_weight: 10`, `seg_weight: 1` |
| Optimisation | `lr_inr: 1.0e-4`, `lr_latent: 5.0e-3`, `n_samples: 10000` |
| Schedule | `epochs.train: 50`, `epochs.val: 25` (TTA budget) |
| Data | 620 centre-crop → 512 grid (`faf_ga_twovar_wktemporal_512`) |

Train and evaluate it:

```bash
python run.py
python evaluate.py --checkpoint runs/faf_ga/<run>/checkpoint_best.pth \
    --holdout_strategy leave_one_out --test on
```

`<run>` is the timestamped directory `run.py` prints as `Output directory:`. It is nested under the
config's `output_dir`, so the full path is `runs/faf_ga/faf_ga_twovar_wktemporal_512_<timestamp>_loc/`.

`evaluate.py` writes `evaluation_*/leave_one_out_summary.csv`, holding held-out DICE / PSNR / SSIM and
lesion-area MAE, grouped **interpolation vs extrapolation**. That is the GAP-INR row of the results
table.

### Forecast scenarios

The forecasting scenarios and the commands that run them are in the README, under
[The workflow](../README.md#the-workflow-train-validate-test-adapt).

## 2. Changing the model

Every architectural knob is documented in `configs/config_model.yaml` and in the README's model
architecture options. Change one at a time, in a copy of the config:

```bash
cp configs/config_model.yaml configs/my_config.yaml   # edit it, then:
python run.py --config_model configs/my_config.yaml
```

A few knobs are also exposed on the command line. These are the only ones `run.py` accepts; anything
else must go in the config. List-valued flags take space-separated integers, not a bracketed string:

```bash
python run.py --inr_decoder__latent_dim 64 32 32   # smaller latent grid
python run.py --inr_decoder__hidden_size 256
python run.py --seed 1927
```

`python run.py --help` lists the full set (`--config_data`, `--config_model`, `--seed`, the
`--inr_decoder__*` shape flags, `--model_gen__cond_scale`, `--n_subjects__*` and the `--overfit*`
flags).

Select on the validation leave-one-out metric, not on test.

## 3. Baselines

The comparison methods are the **ImageFlowNet family** (ImageFlowNet ODE, Time-conditioned U-Net,
Time-aware diffusion), trained and evaluated on the **same split and test eyes**, at **256×256, seed
1** (the paper configuration). The protocol is in [`BASELINES.md`](BASELINES.md); the code is in
[`../baselines/imageflownet/`](../baselines/imageflownet/), under a non-commercial licence.

## 4. Figures

```bash
# Per-eye GA-area progression panel (the paper figure)
python plot_lesion_size_trajectories.py \
    --csv runs/faf_ga/<run>/evaluation_*/lesion_analysis/lesion_areas_test_epoch_*.csv --split test \
    --holdout-dir runs/faf_ga/<run>/evaluation_*/holdout_timeline_arrays
```

