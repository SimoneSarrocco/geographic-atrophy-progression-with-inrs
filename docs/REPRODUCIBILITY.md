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
| Conditioning | weeks-from-baseline as FiLM vector |
| Heads / loss | FAF reconstruction (MSE) + GA segmentation (CE + Dice); `sr_weight: 10`, `seg_weight: 1` |
| Optimization | `lr_inr: 1.0e-4`, `lr_latent: 5.0e-3`, `n_samples: 10000` |
| Schedule | `epochs.train: 50`, `epochs.val: 25` (TTA budget) |
| Data | 620 center-crop → 512 grid (`faf_ga_twovar_wktemporal_512`) |

Train and evaluate it:

```bash
python run.py
python evaluate.py --checkpoint runs/<run>/checkpoint_best.pth \
    --holdout_strategy leave_one_out --test on
```

`evaluate.py` writes `evaluation_*/leave_one_out_summary.csv` — held-out DICE / PSNR / SSIM and
lesion-area MAE, grouped **interpolation vs extrapolation**. That is the GAP-INR row of the results
table.

### Forecast scenarios

- **Scenario 1 (single pair):** forecast a target visit from one earlier visit.
- **Scenario 2 (full history):** adapt the eye's latent on all available past visits (TTA), then
  forecast. Use `evaluate.py --support_k K` to fit on the first `K` visits and predict the rest.

## 2. Changing the model

Every architectural knob is documented in `configs/config_model.yaml` and in the README's model
architecture options. Change one at a time, either in a copy of the config or on the command line:

```bash
python run.py --inr_decoder__latent_dim "[64, 32, 32]"   # smaller latent grid
python run.py --inr_decoder__shared_output_layer true    # one shared output head
python run.py --config_model path/to/your_config.yaml
```

Select on the validation leave-one-out metric, not on test.

## 3. Baselines

The comparison methods are the **ImageFlowNet family** (ImageFlowNet ODE, Time-conditioned U-Net,
Time-aware diffusion), trained and evaluated on the **same split and test eyes**, at **256×256, seed
1** (the paper configuration). The protocol is in [`BASELINES.md`](BASELINES.md). The baseline code is
a separate adapted copy of the public ImageFlowNet repository and is not bundled here.

## 4. Diagnostics and figures

```bash
# Is the temporal pathway responsive (not collapsed)?
python temporal_sensitivity.py --checkpoint runs/<run>/checkpoint_best.pth --split test

# Multi-eye predicted-trajectory figure (faithful across progression rates)
python plot_trajectories.py --csv runs/<run>/evaluation_*/lesion_analysis/lesion_areas_test_epoch_*.csv --split test
```

See [`PIPELINE.md`](PIPELINE.md) for the full train → eval → diagnose workflow and what to monitor.
