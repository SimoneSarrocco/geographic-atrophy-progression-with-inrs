# Attribution

This repository adapts the open-source CINeMA framework (Conditional Implicit Neural Multi-Modal
Atlas), originally developed for spatio-temporal atlases of the perinatal brain. The SIREN-based
conditional INR decoder, the FiLM latent/condition modulation, the auto-decoder training scheme, and
the YAML configuration system come from that project and are reused here under its Apache-2.0 license.

The contribution of the accompanying paper is the adaptation of the framework to longitudinal
geographic-atrophy forecasting from fundus autofluorescence imaging: the 2-D retinal data pipeline
and preprocessing, the joint FAF-reconstruction and GA-segmentation heads, the temporal
(weeks-from-baseline) conditioning and test-time latent
adaptation for future-visit prediction, the leave-one-out interpolation/extrapolation evaluation, and
the comparison protocol.

## Upstream framework

CINeMA: Conditional Implicit Neural Multi-Modal Atlas for a Spatio-Temporal Representation of the
Perinatal Brain — Dannecker, Sideri-Lampretsa, Starck, Mihailov, Milh, Girard, Auzias, Rueckert.
*IEEE Transactions on Medical Imaging*, 2025. doi:10.1109/TMI.2025.3605194.
Source: https://github.com/m-dannecker/CINeMA (Apache-2.0).

```bibtex
@article{dannecker2025cinema,
  author  = {Dannecker, Maik and Sideri-Lampretsa, Vasiliki and Starck, Sophie and Mihailov, Angeline and Milh, Mathieu and Girard, Nadine and Auzias, Guillaume and Rueckert, Daniel},
  journal = {IEEE Transactions on Medical Imaging},
  title   = {CINeMA: Conditional Implicit Neural Multi-Modal Atlas for a Spatio-Temporal Representation of the Perinatal Brain},
  year    = {2025},
  doi     = {10.1109/TMI.2025.3605194}
}
```

## Third-party components

- **SIREN** — sinusoidal representation networks (Sitzmann et al., 2020).
- **tiny-cuda-nn** — the spatial hash in `models/encodings.py` follows the reference implementation
  (NVIDIA, BSD-3-Clause); see the citation in that file.
- **Comparison baselines** — the ImageFlowNet family, in [`baselines/imageflownet/`](baselines/imageflownet/);
  see [`docs/BASELINES.md`](docs/BASELINES.md). That directory is a lightly adapted copy of the public
  ImageFlowNet repository and is **not** covered by this repository's Apache-2.0 license: it is
  governed by the Yale Non-Commercial license, and the I2SB code it vendors by NVIDIA's own
  non-commercial license. See [`baselines/imageflownet/ATTRIBUTION.md`](baselines/imageflownet/ATTRIBUTION.md).
