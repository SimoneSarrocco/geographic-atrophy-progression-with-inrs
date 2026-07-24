# Preprocessing

Scripts that assemble the clinical metadata table and the train/val/test split the training code
reads. They are cohort-specific and are provided to **document** how the paper's CSV was built. Please
adapt the column names and paths (`./data/...`) to your own data before running.

| Script | What it does |
|---|---|
| `enrich_clinical_data.py` | Merges per-scan physical-scale metadata (SLO scan-info) into the longitudinal clinical CSV. |
| `physical_enrichment.py` | Adds physical pixel-scale / bounding-box columns (`ScaleXSlo`, `ScaleYSlo`, …) used to convert GA pixel counts to mm². |
| `add_split_column.py` | Adds a patient-wise `split` column (train/val/test) so no eye leaks across splits. |

These are **not** a pipeline to run against a CSV you already have, and they are destructive:

- `enrich_clinical_data.py` and `physical_enrichment.py` each read `./data/clinical_metadata_raw.csv`
  (a raw export, not shipped) and each write `./data/clinical_metadata.csv`. They are two independent
  enrichments of the raw table, not consecutive stages: running the second after the first
  overwrites the first's output rather than adding to it.
- `add_split_column.py` rewrites `./data/clinical_metadata.csv` **in place** with the official split. 
- It has no `__main__` guard, so merely importing it does this. Running it on a CSV
  that already carries the paper's `split` column will replace that column, after which
  `validate_splits` aborts every run against `configs/expected_split.yaml`.

So: read them to see how the shipped CSV was produced, and adapt them to your own cohort. Do not run
them against the CSV from [`../docs/DATA.md`](../docs/DATA.md), that CSV is already the finished
product `configs/config_data.yaml → tsv_file` points at. See that document for the required columns.
