# Models

The neural components of the conditional INR. The decoder is trained in the main loop and **frozen**
at test time (only the per-eye latent is adapted).

| File | Contents |
|---|---|
| `inr_decoder.py` | `INR_Decoder`. Ties everything together: bilinearly samples the per-eye latent grid at a query coordinate, encodes the temporal condition, forms the FiLM modulation vector, runs the SIREN trunk, and emits FAF intensity + GA logits. Also the FiLM `Modulator`. `forward(...)` for training, `inference(...)` for dense grid decoding. |
| `siren.py` | `SineLayer` and `Siren`. The SIREN MLP trunk with per-feature FiLM (scale/shift inside the sine) and the reconstruction / segmentation heads. |
| `omega_scheduler.py` | `get_omega_schedule`. Per-layer SIREN activation frequency (ω) schedule: `constant`, `linear`, or `exponential`. **ω is the SIREN frequency hyperparameter**, unrelated to any dataset name. |
| `encodings.py` | Coordinate / condition / time encoders: hash-grid, Fourier, Gaussian, identity, and MLP condition encodings, with `get_encoding` / `get_condition_encoding` factories. |
