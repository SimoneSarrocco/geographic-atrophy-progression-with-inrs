# Configuration

Runs are driven by a **two-file** YAML system merged into one dict at startup (`run.py`).

| File | Role |
|---|---|
| `config_model.yaml` | **Model / optimizer / training** config: the INR decoder (architecture, `latent_dim`, `hidden_size`, SIREN `omega_*` schedule, `out_dim = [#intensity, #seg]`, seg head), the optimizer (`lr_inr`, `lr_latent`, loss weights, test-time-adaptation early-stopping), `epochs`, `validate_every`, `n_samples`, and `model_gen`. It carries a top-level `config_data:` key naming which dataset section to use. |
| `config_data.yaml` | **Dataset** config: a base section `faf_ga` (a YAML anchor `&faf_ga`) plus variants that merge it (`<<: *faf_ga`) and change only the resolution / conditioning / sampling (e.g. `faf_ga_512`, `faf_ga_620`, `faf_ga_twovar_wktemporal_512`, `faf_ga_timeinput`). Because the model is a coordinate INR, resolution variants differ only in the sampling grid — a 512-trained checkpoint can be evaluated at another resolution with no retraining. |
| `subject_ids.yaml` | Optional explicit train/val/test ID lists. The `faf_ga` lists are empty on purpose so the loader uses the CSV `split` column instead. |
| `expected_split.yaml` | The canonical paper split as explicit eye-ID lists (25 / 5 / 6). `validate_splits()` aborts if a run resolves a different partition. |
| `lakefs_cfg.example.yaml` | Template for the **optional** remote object-store credentials. Copy to `lakefs_cfg.yaml` (git-ignored) and fill in, or ignore entirely to read local files. |

Select a dataset section on the command line (`python run.py --config_data faf_ga_620`) or via the
`config_data:` field inside the chosen `--config_model` file. Any nested field can be overridden from
the CLI with `--section__key value` (see `run.py`). Data paths are relative to the repo root by
default; edit `tsv_file` in `config_data.yaml` to point at your CSV (see [`../docs/DATA.md`](../docs/DATA.md)).
