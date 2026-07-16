# GAP-INR ablation suite

> **Comparison baselines** (the ImageFlowNet family: ImageFlowNet ODE, Time-conditioned U-Net,
> Time-aware diffusion) — see [`../docs/BASELINES.md`](../docs/BASELINES.md) for how to train/evaluate
> each on the same CSV split. This file covers GAP-INR's own ablations only.

One command trains + evaluates every ablation on the **same official 620-crop split**,
records best checkpoints/metrics, and produces a single test-set comparison table.
Every ablation is the **same short-training 620 reference config with exactly one knob
changed**, so differences are attributable to that knob alone.

## The short-run base (the "is 25 epochs enough?" question)

The base is `ablations/_base_620.yaml` (the frozen ablation reference, moved out of `configs/`
so the main run uses only `config_atlas.yaml` + `config_data.yaml`) with:

| override | value | why |
|---|---|---|
| `epochs.train` | 50 | the train cosine `T_max == epochs.train`, so 50 epochs **anneal fully** (5e-3→eta_min). |
| `validate_every` | 2 | run validation every 2 epochs |
| `validation.holdout_strategy` | `last` | hold out the LAST visit at validation → the metric that selects `checkpoint_best.pth` is the **forecast-the-future-visit** task (extrapolation), matching the ImageFlowNet comparison |
| `train_eval_every` | 1 | train-set metrics every epoch |

`a0_constant25` is the annealing **control** (constant LR at the base epoch count). NB with the base
now at 50 epochs, `a0_cosine50` duplicates `baseline`; the A0-*length* ablations keep their own
`epochs.train` (50/100) via per-ablation override.

## Ablations (`make_configs.py` is the source of truth)

| group | configs | tests |
|---|---|---|
| A0 train/anneal | `baseline`, `a0_constant25`, `a0_cosine50`, `a0_cosine100` | is short training + annealing enough? |
| A1 temporal **(headline)** | `baseline` (FiLM) vs `a1_timeinput_freq6` | FiLM modulation vs time-as-coordinate, **fair** (time Fourier-encoded, 6 bands — not a bare scalar) |
| A2 representation | `baseline` (per-eye+FiLM) vs `a2_pervisit` | continuous time-conditioned field vs one latent per visit |
| A3 latent resolution | `a3_latent16`, `baseline`(=32), `a3_latent64`, `a3_latent128` | capacity / boundary sharpness |
| A4 recon aux | `a4_sr0`, `a4_sr1`, `baseline`(=sr10) | does predicting FAF help segmentation? |
| A5 SIREN freq | `baseline`(=ω30) vs `a5_omega100` | high-frequency boundary detail |
| A6 seg loss | `a6_ce` vs `baseline`(=CE+Dice) | Dice term value under GA imbalance |
| A7 augmentation | `baseline`(off) vs `a7_aug_on` | pseudo-eye aug on 37 eyes |
| A8 cond embedding | `baseline` (raw scalar) vs `a8_condmlp` | FiLM condition as raw scalar vs MLP-embedded (dim 16) |

Cells equal to the reference (latent 32, sr 10, ω 30, CE+Dice, FiLM, per-eye, aug off,
cosine) are **not** re-emitted — they **are** the `baseline` run, which every group
compares against. To add/change an ablation, edit the `ABLATIONS` list in
`make_configs.py` and re-run it.

### A8 — TTO budget (post-hoc, no retraining)
How many test-time latent-fit steps a new eye needs. Reuse the baseline checkpoint:
```bash
for k in 5 10 25 50 100; do
  python evaluate.py --checkpoint ablations/runs/baseline/checkpoint_best.pth --epochs_val $k \
    --output_dir ablations/runs/baseline/tto_$k
done
```

### Clinical forecast (support_k) — the money plot
```bash
python evaluate.py --checkpoint ablations/runs/baseline/checkpoint_best.pth --support_k 1 --skip_val
```
Fit the latent on the baseline visit only, forecast visits 2..N (GT-scored).

## Staged run (recommended): temporal mechanism first, then everything else

Pick the best **temporal mechanism** on the standard config (changing ONLY that), then build every
other ablation on the winner. This avoids ablating each knob against a temporal mechanism that may
not be the best one.

```bash
cd GAP-INR
# --- STAGE 1: the three temporal mechanisms (differ ONLY in how time enters) ---
python ablations/make_configs.py                       # scalar base (default)
# run the 3 IN PARALLEL, one per GPU (e.g. the 4x12GB server). Omit --gpus to run sequentially.
python ablations/run_ablations.py baseline a8_condmlp a1_timeinput_freq6 --gpus 0,1,2
#   live logs: tail -f ablations/runs/<name>/train.log
python ablations/compare_ablations.py                  # winner = best held-out DICE
#   baseline           = FiLM modulation, RAW SCALAR weeks
#   a8_condmlp         = FiLM modulation, MLP-embedded weeks (dim 16)
#   a1_timeinput_freq6 = time as INPUT coordinate, Fourier-encoded (6 bands)

# --- STAGE 2: bake the winner into the base, run the remaining ablations on top ---
mv ablations/runs ablations/runs_stage1_temporal       # archive Stage-1 (avoid run-dir collision)
python ablations/make_configs.py --temporal <winner>   # winner = scalar | mlp | timeinput
python ablations/run_ablations.py baseline a0_constant25 a0_cosine50 a0_cosine100 \
    a2_pervisit a3_latent16 a3_latent64 a3_latent128 a4_sr0 a4_sr1 a5_omega100 a6_ce a7_aug_on
python ablations/compare_ablations.py
```
- If the winner is **scalar**, skip the `--temporal`/archive step — the base is already scalar; just run the remaining ablations.
- In Stage 2 the regenerated `baseline` **encodes the winning mechanism**, so it's the reference every Stage-2 ablation compares against. Don't re-run `a8_condmlp` / `a1_timeinput_freq6` — their mechanism is now the base, and their Stage-1 numbers stand.
- `--temporal` is recorded in each config header + `manifest.yaml` (`temporal_base`) for traceability.

## Run it

```bash
# 1. (re)generate configs
python ablations/make_configs.py

# 2. train + eval everything (or pass names for a subset)
python ablations/run_ablations.py                      # all
python ablations/run_ablations.py baseline a1_timeinput_freq6 # subset
python ablations/run_ablations.py --skip_existing       # resume
python ablations/run_ablations.py --dry_run             # print commands only

# 3. one comparison table on the TEST set (held-out = ALL by default)
python ablations/compare_ablations.py
python ablations/compare_ablations.py --group extrapolation   # forecast-only
python ablations/compare_ablations.py --split val
```

## Run any single ablation by hand
Each config is standalone:
```bash
python run.py --config_atlas ablations/configs/a3_latent128.yaml
python summarize_eval.py --eval_dir <the "Output directory:" path printed above>
```

## Outputs
- `ablations/configs/*.yaml` — the generated configs
- `ablations/manifest.yaml` — name → group/rationale/config
- `ablations/runs/index.json` — name → run_dir (written by `run_ablations.py`)
- `ablations/runs/<name>/` — each run dir: `checkpoint_best.pth`, metric JSONs,
  `leave_one_out_summary.csv`, `tb_logs/`
- `ablations/comparison.csv` + `comparison.tex` — the final table (best-checkpoint
  test metrics: DICE/Precision/Recall + PSNR/SSIM + GA-area MAE (mm²), interpolation
  vs extrapolation). DICE is the **GA-foreground** Dice (background excluded);
  Precision/Recall expose the predict-everywhere / predict-nothing failure modes Dice hides.

All runs share the official split (subject_ids + `split` column), so every number is
comparable across ablations and against the ImageFlowNet baselines, which are aligned to the
same 620 crop + leave-one-out + seg-supervised test-time fitting.
```
