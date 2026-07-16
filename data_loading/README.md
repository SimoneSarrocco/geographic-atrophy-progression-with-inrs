# Data loading

| File | Contents |
|---|---|
| `dataset.py` | `Data(Dataset)` — the whole data path: resolves splits, loads FAF images + GA masks (grader consensus), applies the crop → resize → normalization chain, samples coordinates, and attaches the temporal condition. `__getitem__` returns coordinate/value/condition tensors (not full images). Also `validate_splits` (the leakage guard), `EyeBatchSampler` (one batch per eye), `get_longitudinal_indices` (opt vs held-out visit split for TTA), and `sample_subject_ids` (CSV-split branch). |
| `lakefsloader.py` | `LakeFSLoader` — optional backend that fetches images from an S3-compatible object store (lakeFS) into a local cache on first use. Only used if a `lakefs:` section and a credentials file are present; otherwise images are read from local disk. |

Images are loaded as **2-D grayscale** (PNG/BMP); the pipeline is 2-D only.

The exact data format, on-disk layout, preprocessing steps, and split mechanism are documented in
[`../docs/DATA.md`](../docs/DATA.md).
