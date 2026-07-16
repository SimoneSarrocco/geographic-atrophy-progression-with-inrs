"""SHARED FAF/GA evaluation spec — the SINGLE SOURCE OF TRUTH for the longitudinal
FAF/GA comparison across GAP-INR + all baselines (NISF, MetaSeg, ImageFlowNet, gliomagrowth).

Every method MUST train/evaluate on the SAME eyes, the SAME visits, the SAME 512 grid, and the
SAME leave-one-visit-out protocol, or the comparison table is apples-to-oranges. Import this module
instead of re-deriving the split locally.

WHY THIS EXISTS (audit 2026-06-28): the CSV lists 37 eyes x 4 visits = 148 rows, but only 133 rows
have BOTH the FAF and the GA-mask FILE on disk. The 15 missing files are ALL in train. Methods that
filtered missing-modality rows differently ended up training on different eye sets (26 vs 25 vs 23),
which is a fairness bug. This module pins the canonical sets.

CANONICAL DEFINITIONS
- split            : taken from the CSV `split` column (patient-wise; both eyes of a patient share a
                    split -> no cross-eye leakage). NOT recomputed here.
- usable (eye,vis) : a visit is usable iff BOTH faf_path AND ga_mask_path exist on disk.
- longitudinal eye : an eye with >= MIN_USABLE_VISITS (=2) usable visits, so it can form at least one
                    context->target pair / one leave-one-visit-out fold.
- TEST/VAL are CLEAN: every val (5) and test (6) eye has all 4 visits usable, all with non-empty GA.
  Only TRAIN drifts (26 CSV eyes -> 23 longitudinal eyes). So the scored TEST set is identical for
  every method regardless of its internal filtering.

Counts: train 23 / val 5 / test 6 eyes; 87/20/24 usable visit-rows; and 0 of
those usable visits has an empty GA mask (so foreground Dice never hits the empty-empty=1.0 case --
no empty-GT special-casing changes the score, though we keep the convention documented below).
"""

import os
from typing import List, Dict, Tuple

# Clinical CSV. Defaults to the GAP-INR repo's data/clinical_metadata.csv (this file lives at
# <repo>/baselines/imageflownet/), so the baselines read exactly the table GAP-INR trains on.
# Override with the GAPINR_CSV environment variable to point at a CSV elsewhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# preprocessing / scoring grid (shared with dump_io.EVAL_DIM)
CSV_PATH = os.environ.get('GAPINR_CSV', os.path.join(_REPO_ROOT, 'data', 'clinical_metadata.csv'))
CROP_SIZE = 620          # native ~768 -> CENTER-CROP 620 (preserves all GA; a direct 512 crop clips GA)
EVAL_DIM = 512           # then RESIZE to 512 (FAF bilinear/cubic, mask NEAREST). All methods score here.
MIN_USABLE_VISITS = 2    # longitudinal eligibility (>=1 context + 1 target)
AREA_PITCH_RF = (CROP_SIZE / EVAL_DIM) ** 2   # mm^2 area scaling 512-grid -> 620-grid pixel pitch

# Empty-GT convention: in this cohort every usable visit HAS GA, so this never fires on real data, but for
# parity across methods fix it explicitly: a held-out visit with an all-zero GT mask is EXCLUDED from
# the Dice mean (not scored as 1.0). Document and apply identically everywhere.
EXCLUDE_EMPTY_GT_FROM_DICE = True

# ---------------------------------------------------------------------------------------------------
# CANONICAL eye lists (longitudinal, >= MIN_USABLE_VISITS usable visits). Hard-coded as the contract;
# `recompute_canonical_eyes()` re-derives them from disk and `verify()` asserts they still match, so a
# data change can't silently drift the split.
CANONICAL_EYES: Dict[str, List[str]] = {
    "train": [
        "EYE01_OD", "EYE02_OD", "EYE04_OS", "EYE05_OS", "EYE06_OD", "EYE06_OS",
        "EYE08_OD", "EYE10_OD", "EYE12_OD", "EYE13_OS", "EYE14_OD", "EYE15_OS",
        "EYE16_OD", "EYE17_OS", "EYE19_OS", "EYE20_OD", "EYE21_OD", "EYE21_OS",
        "EYE22_OS", "EYE25_OS", "EYE29_OD", "EYE29_OS", "EYE30_OD",
    ],
    "val":  ["EYE07_OD", "EYE23_OD", "EYE23_OS", "EYE24_OS", "EYE26_OD"],
    "test": ["EYE09_OS", "EYE18_OD", "EYE18_OS", "EYE27_OD", "EYE27_OS", "EYE31_OD"],
}


def usable_table(csv_path: str = CSV_PATH):
    """Return the CSV as a DataFrame with a boolean `usable` column (both FAF + GA mask exist on disk)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    has_faf = df["faf_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    has_mask = df["ga_mask_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))
    df["usable"] = has_faf & has_mask
    return df


def recompute_canonical_eyes(csv_path: str = CSV_PATH,
                             min_usable_visits: int = MIN_USABLE_VISITS) -> Dict[str, List[str]]:
    """Re-derive the longitudinal eye lists per split from on-disk file availability."""
    df = usable_table(csv_path)
    u = df[df["usable"]].groupby("Eye_ID").size()
    out: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        eyes = df[df["split"] == split]["Eye_ID"].unique()
        out[split] = sorted(e for e in eyes if int(u.get(e, 0)) >= min_usable_visits)
    return out


def eyes(split: str) -> List[str]:
    """Canonical longitudinal eye list for a split ('train'|'val'|'test')."""
    return list(CANONICAL_EYES[split])


def assert_split_parity(eye_ids, split: str, *, source: str = "") -> None:
    """Hard-assert that `eye_ids` is EXACTLY the canonical eye set for `split`.

    Every baseline/GAP-INR evaluator should call this right after it materializes the eyes it will
    actually score, so that a silent split drift -- a regenerated CSV, a moved/missing file, an
    ad-hoc per-loader filter -- becomes an immediate, loud failure instead of a comparison computed
    on the wrong patients (which no reviewer or reader could detect after the fact). Eye IDs are
    compared as strings in the canonical 'EYE09_OS' format; order/duplicates are ignored.
    Raises AssertionError on mismatch, listing exactly what is missing vs extra."""
    got = sorted({str(e) for e in eye_ids})
    want = sorted(CANONICAL_EYES[split])
    if got != want:
        tag = f" [{source}]" if source else ""
        raise AssertionError(
            f"eval_spec split-parity FAILED for {split!r}{tag}:\n"
            f"  got  ({len(got)}): {got}\n"
            f"  want ({len(want)}): {want}\n"
            f"  missing (in canonical, absent here): {sorted(set(want) - set(got))}\n"
            f"  extra   (here, not canonical):       {sorted(set(got) - set(want))}")


def usable_visit_rows(split: str, csv_path: str = CSV_PATH):
    """DataFrame of usable (eye, visit) rows for the canonical eyes of `split`, sorted by eye+visit.
    Columns include Eye_ID, Visit_Number, faf_path, ga_mask_path. This is what loaders should iterate."""
    df = usable_table(csv_path)
    keep = df["usable"] & df["Eye_ID"].isin(set(CANONICAL_EYES[split]))
    return df[keep].sort_values(["Eye_ID", "Visit_Number"]).reset_index(drop=True)


def loo_folds(split: str, csv_path: str = CSV_PATH) -> List[Tuple[str, int, str]]:
    """Leave-one-visit-out folds for evaluation. For each canonical eye, hold out each usable visit
    position in turn; condition/fit on the OTHER usable visits. Returns (eye_id, holdout_visit_number,
    kind) with kind in {'interpolation','extrapolation'} ('extrapolation' = the latest usable visit)."""
    rows = usable_visit_rows(split, csv_path)
    folds: List[Tuple[str, int, str]] = []
    for eye, sub in rows.groupby("Eye_ID"):
        visits = sorted(int(v) for v in sub["Visit_Number"])
        last = visits[-1]
        for v in visits:
            folds.append((eye, v, "extrapolation" if v == last else "interpolation"))
    return folds


def verify(csv_path: str = CSV_PATH) -> None:
    """Assert the hard-coded canonical lists still match what's on disk; print a summary."""
    rederived = recompute_canonical_eyes(csv_path)
    for split in ("train", "val", "test"):
        assert rederived[split] == CANONICAL_EYES[split], (
            f"split '{split}' drifted!\n  on-disk: {rederived[split]}\n  pinned : {CANONICAL_EYES[split]}")
    df = usable_table(csv_path)
    print("FAF/GA shared eval spec — OK")
    for split in ("train", "val", "test"):
        n_eyes = len(CANONICAL_EYES[split])
        n_vis = int((df["usable"] & df["Eye_ID"].isin(set(CANONICAL_EYES[split]))).sum())
        print(f"  {split:5s}: {n_eyes} eyes, {n_vis} usable visit-rows")
    print(f"  grid: crop{CROP_SIZE} -> resize{EVAL_DIM}; min usable visits={MIN_USABLE_VISITS}")


if __name__ == "__main__":
    verify()
