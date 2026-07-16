# Individual GA Progression as a Latent Trajectory over an Implicit Atlas

A method for **forecasting geographic-atrophy (GA) progression for an individual
patient-eye** from a few longitudinal FAF + segmentation observations, and for
rendering a **dense, continuous "movie" of the GA segmentation over time** that can
be queried at any timepoint `t`.

---

## 1. The problem

For each patient-eye we observe a short, irregularly-spaced longitudinal series of
**FAF images** with their **GA segmentation masks** (typically ~4 visits over ~1 year).
We want to:

1. **Represent** each eye's observed visits faithfully (image + lesion).
2. **Forecast** the lesion at an **unobserved** visit (a held-out future or in-between
   timepoint) — *honest missing-visit prediction*.
3. Produce a **continuous disease trajectory** — a dense segmentation field `S(x,y,t)`
   that yields the lesion "snapshot" at any `t`, plus the lesion-area growth curve.

The hard constraints are **tiny data** (~26 train eyes × ~4 visits) and the requirement
that evaluation be **fair** (the model must never see the visit it is asked to predict).

---

## 2. Why the obvious designs fall short (what we learned)

**(a) One latent per eye + time as a FiLM scalar.** A single per-eye code, moved
through time by a `weeks` conditioning scalar. The decoder *can* render different
visits, and observed-visit reconstruction is sharp (PSNR ~31). **But forecasting is
weak**: the held-out visit relies on a single scalar driving a shared decoder, and the
latent has to "average" all visits. The temporal signal is too thin to extrapolate
reliably.

**(b) One latent per visit (independent visits), no time model.** Each visit gets its
own latent → **excellent per-visit reconstruction** (this is the config whose images
look great). **But there is no way to predict an unobserved visit**: the held-out
visit's latent is never informed by the observed ones, and interpolation/extrapolation
is undefined. So per-visit latents are a great *representation* but, alone, **not a
forecasting model** — the held-out "prediction" degenerates to a population default.

**Conclusion.** Representation and dynamics are two different jobs. Trying to do both
with one mechanism either starves the dynamics (a) or omits it entirely (b).

---

## 3. The approach: decouple representation from dynamics

We split the problem into two stages, which is exactly what makes both tractable on
small data.

### Stage 1 — Per-visit implicit representation
Train a shared **SIREN implicit decoder** with **one latent per visit** (auto-decoder).
Each visit `v` of eye `e` is encoded by a compact latent `z_{e,v}` such that the decoder
reconstructs that visit's FAF and GA mask from its coordinates. This yields, for every
observed visit, a point `z_{e,v}` in a shared latent space — a faithful, low-dimensional
*fingerprint* of that eye at that time.

> Key property we require and verify: the latent space must be **smooth and
> time-ordered** — within an eye, latents should move coherently with time
> (`check_latent_smoothness.py`). If they don't, Stage 2 cannot work, and we add a
> latent-regularisation / smoothness prior.

### Stage 2 — Latent dynamics
Learn a **shared temporal operator** over the per-visit latents that maps
`(z(t_i), Δt) → z(t_j)` for visits of the same eye (all ordered pairs are training
examples; `models/latent_transition.py`). This is the "physics" of progression,
**shared across all eyes** so it can be learned from limited data, while a small per-eye
component captures individual rate. Two interchangeable forms:

- a **time-shift operator** `g(z, Δt)` (residual; simplest), or
- a **latent ODE** `dz/dt = f(z)` (continuous; handles irregular visit spacing natively).

### Inference = forecasting
Given a new eye's observed visits: obtain their latents (test-time fit on the observed
masks), then **predict** the held-out latent `ẑ(t*)` with the Stage-2 operator and
**decode** it → predicted FAF/mask at `t*`. Querying a dense grid of `t*` produces the
**continuous segmentation movie**; the lesion area of each frame gives the **growth
trajectory**.

This replaces the degenerate "random held-out latent" of the per-visit-only model with
a *predicted* one — turning a representation into a forecaster.

---

## 4. Output focus: the segmentation trajectory `S(x,y,t)`

The clinically meaningful quantity in GA is the **lesion** (location, shape, area, growth
rate) — not FAF texture. So the **primary output is the segmentation field** `S(x,y,t)`:
a dense, time-aware mask volume, sliced at any `t`. We deliberately do **not** chase
high-fidelity FAF *intensity* prediction (the hardest, least clinically relevant part);
predicted masks are **overlaid on the real FAF** for visualization.

### The role of FAF (and why it earns its place)
The longitudinal *signal* lives in the **mask sequence**; the registered FAFs are nearly
identical across visits. FAF's genuine, untapped value is **prognostic**: the
**perilesional / junctional hyperautofluorescence** at the lesion border predicts *where
and how fast* GA will expand — information a binary mask discards. We exploit it by
feeding the **last-observed FAF (and baseline mask) as a per-coordinate input** that
grounds anatomy and conditions growth direction (gated, `faf_as_input`). Because FAF is
the *input anchor*, optionally predicting an FAF channel becomes the well-posed task of
"**evolving the given image**" rather than generating a retina from scratch.

FAF's contribution is **measured, not assumed**: we ablate with-FAF vs. without-FAF.

---

## 5. Evaluation (fair by construction)

- **Leave-one-visit-out.** Hold out one real visit per eye; fit/observe the rest; predict
  the held-out one; score against its ground truth. Holding out the **last** visit tests
  **extrapolation**; a **middle** visit tests **interpolation**. The model never sees the
  held-out visit (no leakage).
- **Metrics:** held-out **DICE** (report **GA-class**, not background-inflated mean) and
  **lesion-area trajectory error** (you have GT area at every real visit → the predicted
  `area(t)` curve is quantitatively scored).
- **Mandatory baselines:** copy-last-mask, isotropic dilation by Δt, linear-area
  extrapolation. The contribution is *beating* these — i.e., predicting **directional,
  eye-specific** growth, not generic expansion.
- **The dense movie between visits has no ground truth** → it is a *visualization*; the
  measured claims are held-out DICE + area-trajectory error.

---

## 6. Honest limitations

- Data is small; the *individual* dynamics the Stage-2 operator can learn beyond the
  shared population trend are limited — the win may be modest but it is honest.
- A model conditioned on the baseline mask can score deceptively well by "just dilating"
  → beating the dilation baseline is the entire bar.
- Stage 2 requires a smooth Stage-1 latent space; we verify this before trusting it.

---

## 7. Status / where things live

- **Stage 1 (per-visit latents):** set `config_data: faf_ga_indep` in `config_atlas.yaml`
  (`independent_visits: true`). (The old standalone `configs/config_stage1.yaml` was removed —
  edit the two main configs instead.)
- **Smoothness check:** `check_latent_smoothness.py` (run on the Stage-1 checkpoint).
- **Stage 2 (latent dynamics):** `models/latent_transition.py`, `train_latent_transition.py`,
  `predict_future.py` (grouping fixed to per-`Eye_ID`).
- **Segmentation-trajectory model (one latent/eye + time-as-input):** `S(x,y,t)`, seg-primary,
  FAF as a light anchor. (The old standalone `configs/config_seg_movie.yaml` was removed — set
  `time_as_input: true` + `config_data: faf_ga_timeinput` in the two main configs instead.)
- **FAF-as-input (v2):** gated `faf_as_input` flag in `models/inr_decoder.py`
  (decoder side done; `anchor_grids` population in `build_atlas` is the remaining step).

The paper's spine: **per-visit implicit representation (Stage 1) + shared latent dynamics
(Stage 2) → individual GA segmentation trajectory + honest missing-visit forecasting,
with FAF's perilesional signal as a measured contributor.**
