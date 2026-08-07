# Data loading

| File | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dataset.py` | `Data(Dataset)`: the whole data path: resolves splits, loads FAF images + GA masks (grader consensus), applies the crop → resize → normalisation chain, samples coordinates, and attaches the temporal condition. `__getitem__` returns coordinate/value/condition tensors (not full images). Also `validate_splits` (the leakage guard), `EyeBatchSampler` (one batch per eye), `get_longitudinal_indices` (opt vs held-out visit split for TTA), and `sample_subject_ids` (CSV-split branch). |
| `lakefsloader.py` | `LakeFSLoader`: optional backend that fetches images from an S3-compatible object store (lakeFS) into a local cache on first use. |
| `lakefs_config.py` | Resolves the optional lakeFS connection from `configs/lakefs_cfg.yaml` or `GAPINR_LAKEFS_*` environment variables, and is the single source of truth shared by `dataset.py` and `download_lakefs_data.py`. Returns `None` when lakeFS is not configured, so images are read from local disk, and raises a message naming the missing setting when it is only half-configured. `python download_lakefs_data.py --check` verifies credentials without starting a run. |

Images are loaded as **2-D grayscale** (PNG/BMP); the pipeline is 2-D only.

The exact data format, on-disk layout, preprocessing steps, and split mechanism are documented in
[`../docs/DATA.md`](../docs/DATA.md).
