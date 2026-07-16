"""Assign a patient-wise train / val / test split to the FAF-GA metadata CSV.

Design (see also _run_validation_round in build_atlas.py):
  * train     -> fit the INR decoder + training latents.
  * val        -> further split at optimisation time into:
                    - val-opt:  visits used to fit each val patient-eye's latent,
                    - val-eval: held-out later visit(s), used to pick the best
                                training checkpoint (longitudinal generalisation).
  * test       -> evaluated once with the chosen checkpoint (unbiased performance).

Constraints enforced here:
  * PATIENT-WISE split: both eyes of the same patient (`patient_num`) always go to
    the SAME split, so there is no cross-eye leakage between splits.
  * NO single-visit patient-eye in val: val patient-eyes need >= MIN_VAL_VISITS visits
    so they can be split into val-opt + val-eval. Because the split is patient-wise, a
    patient is val-eligible only if ALL of its eyes have >= MIN_VAL_VISITS visits.
    Single-visit eyes are allowed in train and in test (test mirrors the clinical case:
    a new patient, one acquired visit, optimise a latent, then predict future states).
"""

import pandas as pd
import numpy as np

# ---------------------------- Configuration ----------------------------
# This MUST be the CSV that training actually reads (configs/config_data.yaml -> faf_ga.tsv_file),
# and the column names below must match it (id_column = Eye_ID, patient = Patient_ID), otherwise the
# 'split' column is written to the wrong file / wrong keys and the split silently never changes.
CSV_PATH = './data/clinical_metadata.csv'
PATIENT_COL = 'Patient_ID'    # patient identifier (both eyes share it)
EYE_COL = 'Eye_ID'            # patient-eye identifier (one latent per value)
SPLIT_COL = 'split'           # output column (read by Data._filter_data via dataset.split_column)

TRAIN_FRAC = 0.70             # fractions are by PATIENT count (patient-wise split)
VAL_FRAC = 0.15
TEST_FRAC = 0.15              # train fraction is the remainder
MIN_VAL_VISITS = 2            # min visits per val patient-eye (to allow the opt + eval hold-out split)
SEED = 42
# NOTE for this dataset: 30 patients / 37 eyes / 148 visits, every eye has 4 visits (so the
# MIN_VAL_VISITS constraint is a no-op here, but is enforced generally). 70/15/15 by patient gives
# roughly 21/4-5/4-5 patients — val/test are small, so treat their metrics as noisy.
# ------------------------------------------------------------------------

assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-6, "Fractions must sum to 1."

print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Original shape: {df.shape}")

for col in (PATIENT_COL, EYE_COL):
    if col not in df.columns:
        raise KeyError(
            f"Required column '{col}' not found in CSV. Available columns: {df.columns.tolist()}"
        )

# --- Per-eye visit counts and per-patient eligibility for the val split ---
visits_per_eye = df.groupby(EYE_COL).size()                       # eye_id -> #visits
patient_of_eye = df.drop_duplicates(EYE_COL).set_index(EYE_COL)[PATIENT_COL]

# A patient is val-eligible iff every one of its eyes has >= MIN_VAL_VISITS visits.
eye_eligible = visits_per_eye >= MIN_VAL_VISITS
patient_min_visits = visits_per_eye.groupby(patient_of_eye).min()  # patient -> min visits over its eyes
val_eligible_patients = set(patient_min_visits[patient_min_visits >= MIN_VAL_VISITS].index)

unique_patients = df[PATIENT_COL].unique()
n_patients = len(unique_patients)
print(f"\nUnique patients: {n_patients}")
print(f"Unique patient-eyes: {df[EYE_COL].nunique()}")
print(f"Patient-eyes with >= {MIN_VAL_VISITS} visits: {int(eye_eligible.sum())} / {len(eye_eligible)}")
print(f"Val-eligible patients (all eyes have >= {MIN_VAL_VISITS} visits): {len(val_eligible_patients)}")

# --- Patient-wise assignment ---
# 1) test gets any patient (single-visit eyes are fine here),
# 2) val draws ONLY from val-eligible patients,
# 3) train gets the rest.
rng = np.random.default_rng(SEED)
shuffled = rng.permutation(unique_patients)

n_test = int(round(TEST_FRAC * n_patients))
n_val = int(round(VAL_FRAC * n_patients))

split_of_patient = {}

# Test: first n_test patients regardless of eligibility.
test_patients = list(shuffled[:n_test])
for p in test_patients:
    split_of_patient[p] = 'test'

# Val: walk the remaining patients and take eligible ones until the target is met.
remaining = [p for p in shuffled[n_test:]]
val_patients = []
leftover = []
for p in remaining:
    if len(val_patients) < n_val and p in val_eligible_patients:
        val_patients.append(p)
        split_of_patient[p] = 'val'
    else:
        leftover.append(p)

if len(val_patients) < n_val:
    print(f"\n[WARN] Only {len(val_patients)} val-eligible patients available "
          f"for a target of {n_val}; val split is smaller than requested.")

# Train: everything not assigned to test or val.
for p in leftover:
    split_of_patient[p] = 'train'

df[SPLIT_COL] = df[PATIENT_COL].map(split_of_patient)

# ------------------------------ Validation ------------------------------
# No patient may appear in more than one split (patient-wise integrity).
per_patient_splits = df.groupby(PATIENT_COL)[SPLIT_COL].nunique()
assert (per_patient_splits == 1).all(), "Some patients span multiple splits!"

# Both eyes of a patient share the split (implied by the above, checked explicitly).
per_eye_splits = df.groupby(EYE_COL)[SPLIT_COL].nunique()
assert (per_eye_splits == 1).all(), "Some patient-eyes span multiple splits!"

# No val patient-eye may have fewer than MIN_VAL_VISITS visits.
val_eye_visits = df[df[SPLIT_COL] == 'val'].groupby(EYE_COL).size()
if len(val_eye_visits):
    bad = val_eye_visits[val_eye_visits < MIN_VAL_VISITS]
    assert bad.empty, f"Val patient-eyes with < {MIN_VAL_VISITS} visits: {bad.to_dict()}"

# ------------------------------- Report --------------------------------
def report(name):
    sub = df[df[SPLIT_COL] == name]
    n_pat = sub[PATIENT_COL].nunique()
    n_eye = sub[EYE_COL].nunique()
    n_vis = len(sub)
    print(f"  {name:5s}: {n_pat:4d} patients | {n_eye:4d} patient-eyes | {n_vis:5d} visits "
          f"({n_pat / n_patients * 100:4.1f}% of patients)")

print("\nSplit summary:")
for name in ('train', 'val', 'test'):
    report(name)

df.to_csv(CSV_PATH, index=False)
print(f"\n[OK] Saved updated CSV with patient-wise '{SPLIT_COL}' column to: {CSV_PATH}")

print("\nSample rows:")
print(df[[PATIENT_COL, EYE_COL, SPLIT_COL]].drop_duplicates().head(10).to_string(index=False))
