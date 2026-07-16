# Preprocessing

Scripts that assemble the clinical metadata table and the train/val/test split the training code
reads. They are cohort-specific and are provided to **document** how the paper's CSV was built —
adapt the column names and paths (`./data/...`) to your own data before running.

| Script | What it does |
|---|---|
| `enrich_clinical_data.py` | Merges per-scan physical-scale metadata (SLO scan-info) into the longitudinal clinical CSV. |
| `physical_enrichment.py` | Adds physical pixel-scale / bounding-box columns (`ScaleXSlo`, `ScaleYSlo`, …) used to convert GA pixel counts to mm². |
| `add_split_column.py` | Adds a patient-wise `split` column (train/val/test) so no eye leaks across splits. |

Run order (adjust paths first):

```bash
python preprocessing/enrich_clinical_data.py
python preprocessing/physical_enrichment.py
python preprocessing/add_split_column.py
```

The resulting CSV is what `configs/config_data.yaml → tsv_file` points at. See
[`../docs/DATA.md`](../docs/DATA.md) for the required columns.
