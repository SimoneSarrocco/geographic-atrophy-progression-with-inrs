# Attribution

This repository adapts the open-source **CINeMA** framework (Conditional Implicit Neural Multi-Modal
Atlas) **by Dannecker et al., 2026**, originally developed for spatio-temporal atlases of the perinatal brain. The SIREN-based
conditional INR decoder, the FiLM latent/condition modulation, the auto-decoder training scheme, and
the YAML configuration system come from that project and are reused here under its Apache-2.0 licence.

The contribution of our paper is the adaptation of the framework to longitudinal individual
geographic atrophy prediction from fundus autofluorescence imaging: the 2D retinal data pipeline
and preprocessing, the FAF-reconstruction and GA-segmentation heads, the eye-specific latent vectors shared across all 
visits of a single eye, the temporal (weeks-from-baseline) conditioning, the interpolation/extrapolation evaluation on different scenarios.

## Upstream framework

CINeMA: Conditional Implicit Neural Multi-Modal Atlas for a Spatio-Temporal Representation of the
Perinatal Brain. Dannecker, Sideri-Lampretsa, Starck, Mihailov, Milh, Girard, Auzias, Rueckert.
*IEEE Transactions on Medical Imaging*, 2025. doi:10.1109/TMI.2025.3605194.
Source: https://github.com/m-dannecker/CINeMA (Apache-2.0).

```bibtex
@ARTICLE{11150663,
  author={Dannecker, Maik and Sideri-Lampretsa, Vasiliki and Starck, Sophie and Mihailov, Angeline and Milh, Mathieu and Girard, Nadine and Auzias, Guillaume and Rueckert, Daniel},
  journal={IEEE Transactions on Medical Imaging}, 
  title={CINeMA: Conditional Implicit Neural Multi-Modal Atlas for a Spatio-Temporal Representation of the Perinatal Brain}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2025.3605194}}
```

## Comparison methods

**Comparison baselines**: the **ImageFlowNet** family, in [`baselines/imageflownet/`](baselines/imageflownet/);
see [`docs/BASELINES.md`](docs/BASELINES.md). That directory is a lightly adapted copy of the public
**ImageFlowNet repository by Liu et al., 2025** (https://github.com/KrishnaswamyLab/ImageFlowNet/tree/main), which is
governed by the Yale Non-Commercial licence, and the I2SB code it vendors by NVIDIA's own
non-commercial licence. See [`baselines/imageflownet/ATTRIBUTION.md`](baselines/imageflownet/ATTRIBUTION.md).

```bibtex
@inproceedings{liu2025imageflownet,
  title={ImageFlowNet: Forecasting Multiscale Image-Level Trajectories of Disease Progression with Irregularly-Sampled Longitudinal Medical Images},
  author={Liu, Chen and Xu, Ke and Shen, Liangbo L and Huguet, Guillaume and Wang, Zilong and Tong, Alexander and Bzdok, Danilo and Stewart, Jay and Wang, Jay C and Del Priore, Lucian V and Krishnaswamy, Smita},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE}
}
```
