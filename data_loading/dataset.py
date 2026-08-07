import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import *
from .lakefs_config import make_loader, resolve_cache_path
from PIL import Image
import yaml
import copy


def _seg_path_is_reconstructed(args):
    """True when the segmentation mask path is rebuilt from Patient_ID/Eye/Visit_ID rather than read
    from its CSV column -- the condition Data.resolve_path uses to pick the grader-mask branch."""
    ds = args['dataset']
    if ds.get('dataset_name') != 'faf_ga':
        return False
    if not ds.get('modalities'):
        return False
    out_dim = args.get('inr_decoder', {}).get('out_dim') or []
    if not (out_dim and out_dim[-1] > 0):
        return False
    return ds.get('mask_grader_mode', 'single') in ('majority', 'soft', 'augment')


def resolve_split_eyes(args, split, df=None):
    """Resolve the set of Eye_IDs for `split` using the SAME precedence as
    Data.sample_subject_ids: the CSV `split_column` wins when it is set and present in the
    dataframe; otherwise the subject_ids[split] list. A row counts only if the modality columns the
    loader actually reads are populated (see _seg_path_is_reconstructed). Lightweight (CSV only, no
    LakeFS). Returns (set_of_eye_ids, source_str)."""
    ds = args['dataset']
    idc = ds.get('id_column', 'Eye_ID')
    if df is None:
        df = pd.read_csv(ds['tsv_file'])
    modcols = [m for m in ds.get('modalities', []) if m in df.columns]
    # Mirror Data.resolve_path: for faf_ga with a grader mode other than 'single', the segmentation
    # modality (the LAST entry in `modalities`) is reconstructed from Patient_ID/Eye/Visit_ID and its
    # CSV column is never read. Requiring that column here would drop eyes the loader trains on --
    # e.g. an eye graded by 02/03 but not 01 has an empty ga_mask_path yet a valid majority mask.
    if _seg_path_is_reconstructed(args) and modcols:
        modcols = modcols[:-1]
    if modcols:
        df = df.dropna(subset=modcols)
    split_col = ds.get('split_column')
    if split_col and split_col in df.columns:
        vals = ['val', 'validation'] if split == 'val' else [split]
        sub = df[df[split_col].astype(str).str.lower().isin(vals)]
        return set(sub[idc].unique()), 'split_column'
    sids = (ds.get('subject_ids') or {}).get(split) or []
    if sids:
        return set(df[df[idc].isin(sids)][idc].unique()), 'subject_ids'
    return set(), 'none'


def validate_splits(args, verbose=True):
    """Guarantee the train/val/test split is well-formed and leakage-free, using the SAME
    resolution the loader uses (so it covers split_column AND subject_ids). Specifically:
      1. asserts the three eye-sets are PAIRWISE DISJOINT (raises ValueError on ANY overlap),
      2. asserts each eye has a single CSV `split_column` value (no ambiguous assignment),
      3. warns if a non-empty subject_ids list disagrees with the CSV split_column (CSV wins),
      4. prints the resolved eye lists.
    Call at the START of run.py (training) AND evaluate.py (eval) so every run is checked."""
    ds = args['dataset']
    idc = ds.get('id_column', 'Eye_ID')
    df = pd.read_csv(ds['tsv_file'])
    sets, sources = {}, {}
    for sp in ('train', 'val', 'test'):
        sets[sp], sources[sp] = resolve_split_eyes(args, sp, df=df)
    if verbose:
        print(f"\n=== SPLIT VALIDATION (source: {sources.get('train')}) ===")
        for sp in ('train', 'val', 'test'):
            print(f"  {sp:5s} ({len(sets[sp])} eyes, via {sources[sp]}): {sorted(sets[sp])}")

    # 1. pairwise disjoint -> the hard guarantee against train/val/test leakage.
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        ov = sets[a] & sets[b]
        if ov:
            raise ValueError(
                f"SPLIT LEAKAGE: {len(ov)} eye(s) in BOTH '{a}' and '{b}': {sorted(ov)}. "
                f"Fix the split_column/subject_ids config so the splits are disjoint.")

    # 2. each eye must have a single CSV split value.
    split_col = ds.get('split_column')
    if split_col and split_col in df.columns:
        multi = df.groupby(idc)[split_col].nunique()
        bad = list(multi[multi > 1].index)
        if bad:
            raise ValueError(f"Eyes with >1 distinct '{split_col}' value (ambiguous split): {bad}")

    # 3. subject_ids vs CSV consistency (CSV wins; warn so a stale subject_ids list is visible).
    sids = ds.get('subject_ids') or {}
    if split_col and split_col in df.columns and any((sids.get(s) or []) for s in ('train', 'val', 'test')):
        csv_of = {e: sp for sp in ('train', 'val', 'test') for e in sets[sp]}
        conflicts = [(e, sp, csv_of[e]) for sp in ('train', 'val', 'test')
                     for e in (sids.get(sp) or []) if e in csv_of and csv_of[e] != sp]
        if conflicts and verbose:
            print(f"  WARNING: subject_ids and CSV '{split_col}' DISAGREE for {len(conflicts)} "
                  f"eye(s) (CSV is authoritative). e.g. {conflicts[:5]}")

    # 5. Canonical-split guard. Disjointness (step 1) does NOT catch a wrong-but-disjoint
    # partition (e.g. a stale CSV with 30 train eyes and no test split, which silently
    # contaminated a whole sweep). If configs/expected_split.yaml exists, the resolved eyes
    # MUST match it exactly, or we abort before any training/eval starts.
    exp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'configs', 'expected_split.yaml')
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            exp = (yaml.safe_load(f) or {}).get('splits', {})
        diffs = []
        for sp in ('train', 'val', 'test'):
            want = set(exp.get(sp, []))
            if not want:
                continue
            missing, extra = want - sets[sp], sets[sp] - want
            if missing or extra:
                diffs.append(f"  {sp}: resolved {len(sets[sp])} eyes vs expected {len(want)}; "
                             f"missing={sorted(missing)} extra={sorted(extra)}")
        if diffs:
            raise ValueError(
                "SPLIT MISMATCH vs configs/expected_split.yaml (the canonical paper split).\n"
                + "\n".join(diffs)
                + f"\nThe CSV in use is '{ds['tsv_file']}'. Put the correct enriched CSV in place "
                  "(or update/delete expected_split.yaml if the split intentionally changed).")
        if verbose:
            print("  OK: resolved split matches configs/expected_split.yaml exactly.")
    return sets


class EyeBatchSampler:
    """Yields one batch per patient-eye: ALL of that eye's visit rows. Reshuffles the eye order (and
    the per-eye visit order) every epoch. This is what enables the temporal cross-visit losses --
    with one eye's whole visit sequence in a batch, the pooled soft-Dice becomes the stacked/temporal
    Dice (Lachinov et al. Eq. 6), and the monotonicity penalty sees all the eye's visit times."""

    def __init__(self, groups, shuffle=True):
        self.groups = [list(g) for g in groups]   # list of [positional row indices] per eye
        self.shuffle = shuffle
        self._epoch = 0

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            rng = np.random.RandomState(self._epoch)
            rng.shuffle(order)
        for gi in order:
            idxs = list(self.groups[gi])
            if self.shuffle:
                np.random.RandomState(1000 * self._epoch + gi + 1).shuffle(idxs)
            yield idxs
        self._epoch += 1

    def __len__(self):
        return len(self.groups)


class Data(Dataset):
    def __init__(self, args, tsv_file, split, df_loaded=None):
        self.args = args
        self.split = split
        self.modality_keys = self.args['dataset']['modalities']
        self.id_column = self.args['dataset'].get('id_column', 'subject_id')
        self.world_bbox = np.array(self.args['dataset']['world_bbox'])
        self.tsv_file = tsv_file
        self.epoch = 0
        self.lakefs_loader = None
        self._init_lakefs()

        if df_loaded is not None:
            df = df_loaded
        elif isinstance(tsv_file, str):
            # Automatic separator detection
            sep = '\t' if tsv_file.endswith('.tsv') else ','
            print(f"Loading dataframe from {tsv_file} (sep='{sep}')")
            df = pd.read_csv(tsv_file)
        else:
            df = tsv_file

        df = self._add_weeks_from_baseline_col(df)
        self.df = df if df_loaded is not None else self.filter_dataframe(df)

        # Validation/test do leave-one-visit-out, which needs >=2 valid (file-present) visits per eye.
        # At this point self.df holds only visits with all modality files present (remove_missing_
        # modalities), so a per-eye row count IS the valid-visit count. Drop eyes below the threshold
        # from the eval splits so they don't enter the val/test pool with too few visits to hold out.
        if self.split in ('val', 'test'):
            self._exclude_eyes_with_insufficient_visits()

        if self.args['dataset'].get('dataset_name') == 'faf_ga':
            self._apply_mask_grader_mode()

        self._map_subject_ids()
        self._init_data_augmentation()
        self._compute_patient_stats()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _init_lakefs(self):
        # Optional remote store. make_loader returns None when lakeFS is not configured,
        # in which case images are read from local disk, and raises with an actionable
        # message when it is only half-configured. See data_loading/lakefs_config.py.
        self.lakefs_loader = make_loader(self.args['dataset'], verbose=(self.split == 'train'))


    def _add_weeks_from_baseline_col(self, df):
        """
        Computes the weeks from baseline for each visit.
        Checks for visit_date or Visit_date columns (case-insensitive) first.
        If dates are numeric (e.g. Excel serial dates), computes difference directly.
        Otherwise parses as datetime.
        Falls back to visit_week_map if no date columns are found.
        """
        df = df.copy()
        computed_from_dates = False

        # Check for date column (case-insensitive)
        date_col = None
        for col in ['visit_date', 'Visit_date']:
            if col in df.columns:
                date_col = col
                break

        if date_col is not None:
            # Check if dates are already numeric (e.g., Excel serial numbers)
            if pd.api.types.is_numeric_dtype(df[date_col]):
                # Excel serial dates are in days. Compute min per subject.
                baseline_dates = df.groupby(self.id_column)[date_col].transform('min')
                df['weeks_from_baseline'] = (df[date_col] - baseline_dates) / 7.0
                print(f"Dynamically computed weeks_from_baseline from numeric '{date_col}'.")
            else:
                # Parse as datetime
                parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
                baseline_dates = parsed_dates.groupby(df[self.id_column]).transform('min')
                df['weeks_from_baseline'] = (parsed_dates - baseline_dates).dt.days / 7.0
                print(f"Dynamically computed weeks_from_baseline from datetime '{date_col}'.")
            computed_from_dates = True
        elif 'diff' in df.columns:
            # 'diff' is elapsed days from baseline in the CSV
            df['weeks_from_baseline'] = df['diff'].astype(float) / 7.0
            print("Dynamically computed weeks_from_baseline using 'diff' column.")
            computed_from_dates = True
        else:
            # Fallback to FAF-GA visit_week_map
            visit_week_map = self.args['dataset'].get('visit_week_map')
            if visit_week_map is not None:
                weeks_list = []
                for _, row in df.iterrows():
                    visit_num = row.get('Visit_Number')
                    if visit_num is None:
                        visit_id = row.get('Visit_ID')
                        if visit_id is not None:
                            vis_str = str(visit_id).strip()
                            if vis_str.startswith('V'):
                                vis_str = vis_str[1:]
                            if vis_str.isdigit():
                                visit_num = int(vis_str)
                    
                    weeks = None
                    if visit_num is not None:
                        weeks = visit_week_map.get(int(visit_num))
                        if weeks is None:
                            weeks = visit_week_map.get(str(visit_num))
                    if weeks is None:
                        weeks = 0.0
                    weeks_list.append(float(weeks))
                df['weeks_from_baseline'] = weeks_list
            else:
                df['weeks_from_baseline'] = 0.0

        # A few visits have no recorded visit_date (NaN weeks). weeks_from_baseline is a CONSTRAINED
        # column, so leaving NaN would make check_constraints DROP those visits (e.g. EYE28_OS V3/V4),
        # collapsing eyes below the 2-visit minimum. Fill ONLY the undated rows from the nominal
        # protocol schedule (visit_week_map keyed by Visit_Number); dated rows keep their real
        # date-derived value. This preserves all visits while staying date-based where dates exist.
        if computed_from_dates and df['weeks_from_baseline'].isna().any():
            visit_week_map = self.args['dataset'].get('visit_week_map')
            n_missing = int(df['weeks_from_baseline'].isna().sum())
            if visit_week_map is not None:
                def _nominal_week(vn):
                    if vn is None or (isinstance(vn, float) and np.isnan(vn)):
                        return np.nan
                    return visit_week_map.get(int(vn), visit_week_map.get(str(int(vn)), np.nan))
                fill = df['Visit_Number'].map(_nominal_week)
                df['weeks_from_baseline'] = df['weeks_from_baseline'].fillna(fill)
            # Any still-NaN (no map / unknown visit) -> 0.0 baseline so the visit is not dropped.
            still = int(df['weeks_from_baseline'].isna().sum())
            df['weeks_from_baseline'] = df['weeks_from_baseline'].fillna(0.0)
            if self.split == 'train':
                print(f"weeks_from_baseline: filled {n_missing - still} undated visits from "
                      f"visit_week_map, {still} from baseline (0.0).")

        # Set the weeks_from_baseline normalisation bounds from the ACTUAL date-derived values rather
        # than a hardcoded range. This runs on the FULL dataframe (before the train/val/test split, see
        # __init__), so every split's Dataset reads the same CSV and derives the SAME global [min,max],         # keeping condition normalisation consistent across splits. Only overrides when weeks came from
        # real dates (not the visit_week_map fallback) and the auto-range is enabled (default on).
        if computed_from_dates and self.args['dataset'].get('weeks_constraint_from_dates', True):
            w = pd.to_numeric(df['weeks_from_baseline'], errors='coerce')
            w_min, w_max = float(np.nanmin(w)), float(np.nanmax(w))
            if np.isfinite(w_min) and np.isfinite(w_max) and w_max > w_min:
                constraints = self.args['dataset'].setdefault('constraints', {})
                wc = constraints.setdefault('weeks_from_baseline', {'type': 'numeric'})
                wc['min'], wc['max'] = w_min, w_max
                if self.split == 'train':
                    print(f"weeks_from_baseline constraint set from dates: [{w_min:.3f}, {w_max:.3f}]")
        return df

    def _exclude_eyes_with_insufficient_visits(self):
        """Drop eyes with fewer than `min_valid_visits_eval` valid visits from val/test.

        Leave-one-visit-out validation requires at least 2 visits per eye (one held out, >=1 to fit
        the latent). Eyes with too few VALID (file-present) visits, e.g. EYE28_OS, which has only 1
        FAF+mask visit, cannot be held out and would otherwise pollute the eval pool. self.df already
        contains only file-present visits, so the per-eye row count is the valid-visit count.
        """
        min_visits = int(self.args['dataset'].get('min_valid_visits_eval', 2))
        if min_visits <= 1 or self.df.empty:
            return
        counts = self.df.groupby(self.id_column)[self.id_column].transform('size')
        keep = counts >= min_visits
        dropped_eyes = sorted(set(self.df.loc[~keep, self.id_column]))
        if dropped_eyes:
            per_eye = self.df[~keep].groupby(self.id_column).size().to_dict()
            print(f"[{self.split}] Excluding {len(dropped_eyes)} eye(s) with <{min_visits} valid "
                  f"visits from evaluation: " +
                  ", ".join(f"{e}({per_eye[e]})" for e in dropped_eyes))
            self.df = self.df[keep].reset_index(drop=True)

    def _apply_mask_grader_mode(self):
        mode = self.args['dataset'].get('mask_grader_mode', 'single')
        # Row-replication grader augmentation is a TRAINING-only device: replicating val/test visits
        # would create multiple "visits" at the same timepoint and corrupt the longitudinal opt/eval
        # split and the metrics. On val/test fall through to the single/majority/soft path instead.
        if mode == 'augment' and self.split != 'train':
            return
        if mode == 'augment':
            new_rows = []
            for _, row in self.df.iterrows():
                row_dict = row.to_dict()
                available_graders = []
                for m in ['mask01', 'mask02', 'mask03']:
                    path = self._resolve_grader_mask_path(row_dict, m, download=True)
                    if path and os.path.exists(path):
                        available_graders.append(m)
                
                if not available_graders:
                    # Fallback to mask01 if none found
                    row_dict['selected_mask'] = 'mask01'
                    new_rows.append(row_dict)
                else:
                    for g in available_graders:
                        augmented_row = copy.deepcopy(row_dict)
                        augmented_row['selected_mask'] = g
                        new_rows.append(augmented_row)
            self.df = pd.DataFrame(new_rows)
            print(f"[{self.split}] Replicated rows for 'augment' mode. New total rows: {len(self.df)}")
        else:
            pass

    def _resolve_grader_mask_path(self, row_dict, mask_suffix, download=True):
        try:
            pat = str(row_dict['Patient_ID']).strip()
            eye = str(row_dict['Eye']).strip()
            vis = str(row_dict['Visit_ID']).strip()
            if vis.isdigit():
                vis = f"V{int(vis):02d}"
            elif vis.startswith('V') and len(vis) == 2 and vis[1].isdigit():
                vis = f"V0{vis[1]}"
            
            filename = f"{pat}_{eye}_{vis}_{mask_suffix}.png"
            path = f"data/{pat}/{eye}/{vis}/Spectralis_faf/{filename}"
        except KeyError:
            path = str(row_dict.get('ga_mask_path', ""))
            if pd.isna(row_dict.get('ga_mask_path')) or path.strip() == "" or path.strip().lower() == "nan":
                return None
            path = path.strip()

        if self.lakefs_loader:
            local_path, obj_key = self.lakefs_loader.get_local_and_obj_names(path)
            if download and not os.path.exists(local_path):
                try:
                    self.lakefs_loader.check_file(obj_key)
                except Exception as e:
                    return None
            return local_path
        else:
            # No lakeFS: read straight from the cache directory. Same layout the loader
            # downloads into (<cache_path>/<branch>/data/...), so a cache filled by a
            # previous lakeFS run is still found here.
            lakefs_config = self.args['dataset'].get('lakefs', {})
            branch = lakefs_config.get('branch', 'main')
            local_path = os.path.join(resolve_cache_path(lakefs_config), branch, path)
            return local_path

    def _load_majority_vote_mask(self, row_dict):
        base_path = self._resolve_grader_mask_path(row_dict, 'mask01', download=False)
        if not base_path:
            return None
        
        majority_path = base_path.replace('mask01.png', 'majority.png')
        if os.path.exists(majority_path):
            return majority_path
            
        masks_data = []
        for m in ['mask01', 'mask02', 'mask03']:
            path = self._resolve_grader_mask_path(row_dict, m, download=True)
            if path and os.path.exists(path):
                img = Image.open(path).convert('L')
                img_np = np.array(img)
                img_binary = (img_np > 127).astype(np.uint8)
                masks_data.append(img_binary)
                
        if not masks_data:
            raise ValueError(f"No grader masks available for subject {row_dict.get('Patient_ID')}")
            
        masks_sum = np.sum(masks_data, axis=0)
        threshold = (len(masks_data) + 1) // 2
        majority_vote = (masks_sum >= threshold).astype(np.uint8) * 255
        
        os.makedirs(os.path.dirname(majority_path), exist_ok=True)
        majority_img = Image.fromarray(majority_vote)
        majority_img.save(majority_path)
        print(f"Created majority vote mask at {majority_path} using {len(masks_data)} graders.")

        return majority_path

    def _load_soft_consensus_mask(self, row_dict):
        """Soft segmentation target (option B): the per-pixel MEAN of the available grader masks,
        a float in [0, 1] (e.g. {0, 1/3, 2/3, 1} for three graders). Captures inter-rater
        uncertainty at lesion boundaries instead of discarding it via majority vote. Returns a
        float32 (H, W) array, or None if no grader masks are available (caller falls back)."""
        masks_data = []
        for m in ['mask01', 'mask02', 'mask03']:
            path = self._resolve_grader_mask_path(row_dict, m, download=True)
            if path and os.path.exists(path):
                img_np = np.array(Image.open(path).convert('L'))
                masks_data.append((img_np > 127).astype(np.float32))
        if not masks_data:
            return None
        return np.mean(masks_data, axis=0).astype(np.float32)   # (H, W) in [0, 1]

    def _compute_patient_stats(self):
        """
        Pre-computes per-patient-eye intensity statistics used by the patient-level
        normalisation modes:

          * 'minmax_patient' : shared (min, max) across all of an eye's visits. NOTE this is
            DEGENERATE on FAF-GA FAF (min=0 from the black registration frame and max=255 from
            saturated pixels in EVERY visit), so it collapses to a constant /255 and does NOT
            harmonize inter-visit brightness. Kept for backward compatibility.
          * 'ref_match'      : reference-visit stats for robust linear matching (option 2). Each
            eye's BASELINE visit (earliest temporal_condition) is the reference; we store its
            FOREGROUND (non-zero, i.e. excluding the frame) median, IQR and p1/p99 per modality.
            Other visits are mapped onto this reference's centre+spread at load time, then squashed
            to [0,1] by the reference p1/p99, this actually equalises brightness across visits.

        'minmax_patient_robust' (option 1) is purely per-visit and needs NO precomputation here.
        """
        norm_type = self.args['dataset'].get('normalize_values')
        if norm_type not in ('minmax_patient', 'ref_match'):
            return

        print(f"[{self.split}] Pre-computing patient-level statistics across visits "
              f"(normalize_values={norm_type})...")
        self.patient_stats = {}
        unique_subs = sorted(self.df['sub_id_int'].unique())
        has_seg = self.args['inr_decoder']['out_dim'][-1] > 0
        n_mod = len(self.modality_keys) - 1 if has_seg else len(self.modality_keys)
        tkey = self._get_temporal_key()

        from tqdm import tqdm
        for sub_id in tqdm(unique_subs, desc=f"Computing stats ({self.split})"):
            sub_df = self.df[self.df['sub_id_int'] == sub_id]

            if norm_type == 'minmax_patient':
                # Shared min/max across this eye's visits (full image, incl. background).
                sub_mins = np.full(n_mod, np.inf)
                sub_maxs = np.full(n_mod, -np.inf)
                for _, row in sub_df.iterrows():
                    row_dict = row.to_dict()
                    try:
                        modalities = self.load_modalities(row_dict)
                        for i in range(n_mod):
                            data = modalities[self.modality_keys[i]].get_fdata()
                            sub_mins[i] = min(sub_mins[i], data.min())
                            sub_maxs[i] = max(sub_maxs[i], data.max())
                    except Exception as e:
                        print(f"Warning: Could not load visit for sub_id {sub_id}: {e}")
                sub_mins[np.isinf(sub_mins)] = 0.0
                sub_maxs[np.isinf(sub_maxs)] = 1.0
                self.patient_stats[sub_id] = {'min': sub_mins, 'max': sub_maxs}

            else:  # ref_match: foreground stats of the BASELINE (earliest) visit
                if tkey in sub_df.columns and sub_df[tkey].notna().any():
                    ref_row = sub_df.loc[pd.to_numeric(sub_df[tkey], errors='coerce').idxmin()]
                else:
                    ref_row = sub_df.iloc[0]
                med = np.zeros(n_mod, np.float32)
                iqr = np.ones(n_mod, np.float32)
                p1 = np.zeros(n_mod, np.float32)
                p99 = np.ones(n_mod, np.float32)
                try:
                    modalities = self.load_modalities(ref_row.to_dict())
                    for i in range(n_mod):
                        data = modalities[self.modality_keys[i]].get_fdata()
                        fg = data[data > 0]                # exclude the black registration frame
                        if fg.size == 0:
                            fg = data.ravel()
                        med[i] = np.median(fg)
                        q25, q75 = np.percentile(fg, [25, 75])
                        iqr[i] = max(float(q75 - q25), 1e-6)
                        p1[i], p99[i] = np.percentile(fg, [1, 99])
                        if p99[i] - p1[i] < 1e-6:
                            p99[i] = p1[i] + 1.0
                except Exception as e:
                    print(f"Warning: Could not load reference visit for sub_id {sub_id}: {e}")
                self.patient_stats[sub_id] = {
                    'ref_med': med, 'ref_iqr': iqr, 'ref_p1': p1, 'ref_p99': p99,
                }

    def __len__(self):
        """Returns total number of visits (rows) in the dataset."""
        return len(self.df)
    
    @property
    def n_unique_subjects(self):
        """Returns number of unique patient-eyes (distinct sub_id_int values).
        Use this for sizing latent vectors and transformations (one per patient-eye)."""
        if self.args['dataset'].get('independent_visits', False):
            return len(self.df)
        if 'sub_id_int' in self.df.columns:
            return self.df['sub_id_int'].nunique()
        return len(self.df)

    def eye_groups(self):
        """Positional row indices grouped by patient-eye (sub_id_int) -> [[visit rows of eye 1], ...].
        Used by EyeBatchSampler so one batch = one eye's full visit sequence (enables the temporal
        cross-visit losses: stacked Dice + monotonicity)."""
        groups = {}
        sub_ids = self.df['sub_id_int'].to_numpy() if 'sub_id_int' in self.df.columns \
            else np.arange(len(self.df))
        for pos, s in enumerate(sub_ids):
            groups.setdefault(int(s), []).append(pos)
        return list(groups.values())

    def __getitem__(self, idx):
        try:
            row_dict = self.df.iloc[idx].to_dict()  # this is one row/visit of a patient-eye from the CSV file
            modalities = self.load_modalities(row_dict)  # image paths of that specific row/visit of a patient-eye
            coords, values = self.load_coords_and_values(modalities, row_dict)  # coordinates and corresponding pixel intensity values of that specific row/visit of a patient-eye
            coords = torch.tensor(coords, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            conditions = self.load_conditions(row_dict)[None, :].expand(coords.shape[0], -1)
            
            # sub_id_int: patient-eye index (shared across visits), for latent and transformation lookup
            if self.args['dataset'].get('independent_visits', False):
                if hasattr(self, 'parent_indices'):
                    sub_idx = self.parent_indices[idx]
                else:
                    sub_idx = idx
            else:
                sub_idx = row_dict.get('sub_id_int', idx)
            idx_df = torch.tensor(sub_idx, dtype=torch.int32).unsqueeze(0).expand(coords.shape[0], -1)
            
            # Load time coordinate
            time_val = self.load_time(row_dict)  # normalised temporal condition (e.g. AgeatVisit) in [-1,1], the time-input coordinate
            time_vals = time_val.expand(coords.shape[0], -1)
            
            return coords, values, conditions, idx_df, time_vals
        except Exception as e:
            # Fail loudly: a data error (missing/corrupt image or mask, bad path, shape mismatch)
            # must surface, not be silently masked by substituting a different visit, which would
            # corrupt training/metrics and duplicate samples. Re-raise with full context.
            sub_id = row_dict.get(self.id_column, 'unknown') if 'row_dict' in locals() else 'unknown'
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"[{self.split}] Failed to load dataset index {idx} (Subject {sub_id}): {e}"
            ) from e

    def collate_fn(self, batch, shuffle=True):
        coords = torch.concat([b[0] for b in batch], dim=0)
        values = torch.concat([b[1] for b in batch], dim=0)
        conditions = torch.concat([b[2] for b in batch], dim=0)
        idx_df = torch.concat([b[3] for b in batch], dim=0)
        time_vals = torch.concat([b[4] for b in batch], dim=0)
        if shuffle:
            perm = torch.randperm(coords.shape[0])
            coords = coords[perm]
            values = values[perm]
            conditions = conditions[perm]
            idx_df = idx_df[perm]
            time_vals = time_vals[perm]
        return coords, values, conditions, idx_df, time_vals

    def resolve_path(self, row_dict, mod_key, download=True):
        """
        Resolves a modality key to a local absolute path, handling LakeFS reconstruction.
        """
        is_faf_ga = self.args['dataset'].get('dataset_name') == 'faf_ga'
        has_seg = self.args['inr_decoder']['out_dim'][-1] > 0
        is_seg = has_seg and (mod_key == self.modality_keys[-1])

        if is_faf_ga and is_seg:
            mode = self.args['dataset'].get('mask_grader_mode', 'single')
            if mode == 'augment':
                selected = row_dict.get('selected_mask', 'mask01')
                return self._resolve_grader_mask_path(row_dict, selected, download=download)
            elif mode in ('majority', 'soft'):
                # 'soft' builds a float consensus target directly in load_modalities for TRAINING; for
                # path-based loading (evaluation GT, metrics) it uses the hard majority vote as reference.
                if not download:
                    majority_path = self._resolve_grader_mask_path(row_dict, 'mask01', download=False)
                    if majority_path:
                        maj_file = majority_path.replace('mask01.png', 'majority.png')
                        if os.path.exists(maj_file):
                            return maj_file
                    for m in ['mask01', 'mask02', 'mask03']:
                        path = self._resolve_grader_mask_path(row_dict, m, download=False)
                        if path and os.path.exists(path):
                            return path
                    return None
                else:
                    return self._load_majority_vote_mask(row_dict)
            else:
                return self._resolve_grader_mask_path(row_dict, 'mask01', download=download)

        if mod_key not in row_dict:
            raise ValueError(f"Modality {mod_key} not found in row_dict")
        path = str(row_dict.get(mod_key, ""))
        if pd.isna(row_dict.get(mod_key)) or path.strip() == "" or path.strip().lower() == "nan":
            return None
        
        path = str(path).strip()
            
        # If the path is absolute and already exists, return immediately
        if os.path.isabs(path) and os.path.exists(path):
            return path
        
        # If the path is absolute and within the LakeFS cache but doesn't exist yet,
        # let it fall through to the LakeFS download logic below
        # (no early return, the LakeFSLoader will handle the download)

        # FAF-GA Path Reconstruction Logic
        if self.lakefs_loader and self.args['dataset']['dataset_name'] == 'faf_ga':
            try:
                pat = str(row_dict['Patient_ID']).strip()
                eye = str(row_dict['Eye']).strip()
                vis = str(row_dict['Visit_ID']).strip()
                # Ensure V0x format if simple integer or V1/V2
                if vis.isdigit(): # e.g. "1" -> "V01"
                    vis = f"V{int(vis):02d}"
                elif vis.startswith('V') and len(vis) == 2 and vis[1].isdigit(): # e.g. "V1" -> "V01"
                    vis = f"V0{vis[1]}"
                filename = os.path.basename(path)
                
                if 'FAF' in filename or 'faf_path' in mod_key:
                    mod_folder = 'Spectralis_faf'
                elif 'SLO' in filename or 'slo_path' in mod_key:
                    mod_folder = 'Spectralis_slo'
                elif 'mask' in filename or 'ga_mask_path' in mod_key:
                    mod_folder = 'Spectralis_faf'
                elif 'OCT' in filename or 'oct' in mod_key.lower():
                    mod_folder = 'Spectralis_oct'
                else:
                    mod_folder = 'Spectralis_slo'
                    
                path = f"data/{pat}/{eye}/{vis}/{mod_folder}/{filename}"
            except KeyError:
                pass 

        if self.lakefs_loader:
            # Check and download if needed
            local_path, obj_key = self.lakefs_loader.get_local_and_obj_names(path)
            if download and not os.path.exists(local_path):
                try:
                    self.lakefs_loader.check_file(obj_key)
                except Exception as e:
                    print(f"Failed to download {obj_key} from LakeFS: {e}")
            return local_path

        return path

    def _is_left_eye(self, row_dict):
        """True if this row's eye is a LEFT (OS) eye, from the Eye_ID suffix (…_OS) or a
        Laterality/Eye column. Used by dataset.canonicalize_laterality to mirror OS eyes to OD."""
        eid = str(row_dict.get(self.id_column, '') or row_dict.get('Eye_ID', '')).upper()
        lat = str(row_dict.get('Laterality', row_dict.get('Eye', '')) or '').upper()
        return eid.endswith('_OS') or '_OS_' in eid or lat in ('OS', 'LEFT', 'L')

    def load_modalities(self, row_dict):
        modalities = {}
        has_seg_mod = self.args['inr_decoder']['out_dim'][-1] > 0
        is_faf_ga = self.args['dataset'].get('dataset_name') == 'faf_ga'
        soft_mode = self.args['dataset'].get('mask_grader_mode') == 'soft'
        for i, mod_key in enumerate(self.modality_keys):
            # Soft consensus target (option B): for the segmentation modality during TRAINING, build a
            # per-pixel mean-of-graders float mask directly (kept in [0,1], never binarised). Val/test
            # don't train segmentation (seg_loss_val: false), so they use the hard majority reference.
            if (soft_mode and is_faf_ga and has_seg_mod and mod_key == self.modality_keys[-1]
                    and self.split == 'train'):
                soft = self._load_soft_consensus_mask(row_dict)
                if soft is not None:
                    modalities[mod_key] = Simple2DImage(soft.astype(np.float32))
                    continue
                # else: fall through to the normal (majority) path below
            path = self.resolve_path(row_dict, mod_key)
            if path is None:
                raise ValueError(f"Modality {mod_key} is empty or could not be resolved")

            # Load the 2D grayscale image (PNG/BMP)
            img = Image.open(path).convert('L') # Grayscale
            
            # FAF-GA: Resize to world_bbox ONLY if no sampling_bbox is specified.
            # When sampling_bbox is set, keep original resolution so that
            # load_coords_and_values can centre-crop exact pixels via
            # coordinate filtering (no interpolation artifacts).
            if self.args['dataset'].get('dataset_name') == 'faf_ga':
                # Mask modality (last key) must resample with NEAREST to stay binary; the FAF
                # image uses BILINEAR.
                is_seg_mod = has_seg_mod and (i == len(self.modality_keys) - 1)
                resample = Image.Resampling.NEAREST if is_seg_mod else Image.Resampling.BILINEAR
                crop_pre = self.args['dataset'].get('crop_before_resize')
                if crop_pre is not None:
                    # Centre-crop native (768) -> crop_pre (e.g. 620) to remove the per-visit black
                    # registration frame while keeping ALL GA (max GA half-extent 302px -> >=604 crop),
                    # THEN resize the crop DOWN to world_bbox (e.g. 512). Lets the whole baseline
                    # comparison run at 512 with no GA clipping and no upsampling (a direct 512 crop
                    # would clip GA in 5/133 visits). NEAREST on the mask keeps it binary.
                    cw, ch = int(crop_pre[0]), int(crop_pre[1])
                    W, H = img.size  # PIL: (width, height)
                    left, top = (W - cw) // 2, (H - ch) // 2
                    img = img.crop((left, top, left + cw, top + ch))
                    img = img.resize(tuple(self.args['dataset']['world_bbox'][:2]), resample)
                elif self.args['dataset'].get('sampling_bbox') is None:
                    # Ensure we only use 2D dimensions for PIL resize (width, height)
                    target_size = tuple(self.args['dataset']['world_bbox'][:2])
                    img = img.resize(target_size, resample)
            
            # Canonicalize laterality (DETERMINISTIC, not augmentation): mirror LEFT (OS) eyes to
            # RIGHT (OD) orientation so the WHOLE dataset shares one orientation. Applied to every
            # modality (FAF + mask) with the SAME horizontal flip -> they stay aligned; a horizontal
            # flip commutes with the centre-crop/resize above and leaves ScaleXSlo/area unchanged.
            if self.args['dataset'].get('canonicalize_laterality', False) and self._is_left_eye(row_dict):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            img_np = np.array(img).astype(np.float32)
            
            # Construct physical 2D affine from enriched CSV info
            # ScaleXSlo/ScaleYSlo are mm/pixel. 
            # If columns are missing (e.g. before enrichment), default to 1.0 (pixels)
            scale_x = float(row_dict.get('ScaleXSlo', 1.0))
            scale_y = float(row_dict.get('ScaleYSlo', 1.0))

            modalities[mod_key] = Simple2DImage(img_np)
        # check that all modalities have the same shape
        shapes = [modalities[mod_key].shape for mod_key in modalities]
        if len(set(shapes)) != 1:
            raise ValueError(f"Modalities have different shapes: {shapes}")

        # Ensure mask values are binary indices [0, 1] for FAF-GA masks (only if seg exists)
        has_seg = self.args['inr_decoder']['out_dim'][-1] > 0  # check if we have segmentation
        if has_seg:
            mask_key = self.modality_keys[-1]  # segmentation
            mask_data = modalities[mask_key].get_fdata()
            if mask_data.max() > len(self.args['dataset'].get('label_names', ['BG'])) - 1:
                # print(f"Normalizing mask values from {np.unique(mask_data)} to binary [0, 1]")
                mask_data = (mask_data > 0).astype(np.float32)
                modalities[mask_key] = Simple2DImage(mask_data)
        return modalities

    def load_coords_and_values(self, modalities, row_dict, normalize=True):
        modalities_data = self.augment_modalities(modalities, row_dict)
        # Use last modality for shape reference (works for both seg and no-seg cases)
        ref_key = self.modality_keys[-1] if self.args['inr_decoder']['out_dim'][-1] > 0 else self.modality_keys[0]
        # ref_key = self.modality_keys[0]
        last_mod = modalities_data[ref_key]
        # affine = modalities[ref_key].affine

        if self.args['dataset'].get('sampling_strategy', 'mask') == 'all':
            # Sample all pixels in the image volume
            c_nz = np.argwhere(np.ones_like(last_mod) > 0)  # this means we sample from all pixels in the image
        else:
            # Traditional GAP-INR restricted sampling
            c_nz = np.argwhere(last_mod > 0)
            if len(c_nz) == 0:
                print(f"Warning: Empty mask for subject. Fallback to full image sampling.")
                c_nz = np.argwhere(np.ones_like(last_mod) > 0)

        # Apply sampling_bbox filter if specified
        sampling_bbox = self.args['dataset'].get('sampling_bbox')
        ndim = len(last_mod.shape)
        is_native_2d = (ndim == 2)  # H, W
        is_compat_2d = (ndim == 3 and last_mod.shape[2] == 1)  # H, W, 1

        x_min, y_min, z_min = 0, 0, 0
        w_box, h_box, d_box = None, None, None

        if sampling_bbox is not None:  # if sampling_bbox is specified, we centre-crop the original images
            if is_native_2d or is_compat_2d:  # if we have 2d images
                h, w = last_mod.shape[:2]
                if len(sampling_bbox) == 2:
                    w_box, h_box = sampling_bbox
                    x_min = (w - w_box) // 2
                    y_min = (h - h_box) // 2
                    x_max = x_min + w_box - 1
                    y_max = y_min + h_box - 1
                elif len(sampling_bbox) == 4:
                    x_min, y_min, x_max, y_max = sampling_bbox
                    w_box = x_max - x_min + 1
                    h_box = y_max - y_min + 1
                else:
                    raise ValueError(f"sampling_bbox must have 2 or 4 elements for 2D, got {len(sampling_bbox)}")
                
                # 2D coordinates in c_nz are (row, col) i.e. (y, x)
                bbox_mask = (
                    (c_nz[:, 1] >= x_min) & (c_nz[:, 1] <= x_max) &
                    (c_nz[:, 0] >= y_min) & (c_nz[:, 0] <= y_max)
                )
                c_nz = c_nz[bbox_mask]
            else:
                d, h, w = last_mod.shape
                if len(sampling_bbox) == 3:
                    d_box, h_box, w_box = sampling_bbox
                    z_min = (d - d_box) // 2
                    y_min = (h - h_box) // 2
                    x_min = (w - w_box) // 2
                    z_max = z_min + d_box - 1
                    y_max = y_min + h_box - 1
                    x_max = x_min + w_box - 1
                elif len(sampling_bbox) == 6:
                    x_min, y_min, z_min, x_max, y_max, z_max = sampling_bbox
                    w_box = x_max - x_min + 1
                    h_box = y_max - y_min + 1
                    d_box = z_max - z_min + 1
                else:
                    raise ValueError(f"sampling_bbox must have 3 or 6 elements for 3D, got {len(sampling_bbox)}")
                
                # 3D coordinates in c_nz are (depth, height, width) i.e. (z, y, x)
                bbox_mask = (
                    (c_nz[:, 2] >= x_min) & (c_nz[:, 2] <= x_max) &
                    (c_nz[:, 1] >= y_min) & (c_nz[:, 1] <= y_max) &
                    (c_nz[:, 0] >= z_min) & (c_nz[:, 0] <= z_max)
                )
                c_nz = c_nz[bbox_mask]
            
            # Check if filtered coordinates are empty
            if len(c_nz) == 0:
                print("Warning: sampling_bbox filtered out all pixels. Fallback to unfiltered sampling.")
                if self.args['dataset'].get('sampling_strategy', 'mask') == 'all':
                    c_nz = np.argwhere(np.ones_like(last_mod) > 0)
                else:
                    c_nz = np.argwhere(last_mod > 0)
                    if len(c_nz) == 0:
                        c_nz = np.argwhere(np.ones_like(last_mod) > 0)

        # Extract values using appropriate indexing
        values_list = []
        for mod in self.modality_keys:
            mod_data = modalities_data[mod]
            if is_native_2d:
                # If the modality itself has extra channel dimension
                if mod_data.ndim > 2:
                    v = mod_data[c_nz[:, 0], c_nz[:, 1], :] # shape (N, C)
                else:
                    v = mod_data[c_nz[:, 0], c_nz[:, 1]][:, np.newaxis] # shape (N, 1)
            else:
                # 3D
                if mod_data.ndim > 3:
                    v = mod_data[c_nz[:, 0], c_nz[:, 1], c_nz[:, 2], :] # shape (N, C)
                else:
                    v = mod_data[c_nz[:, 0], c_nz[:, 1], c_nz[:, 2]][:, np.newaxis] # shape (N, 1)
            values_list.append(v)
        values = np.concatenate(values_list, axis=-1)
        
        # Since data is pre-registered (intra-visit and across-visits), we bypass 
        # the transformation into world coordinates and use local pixel coordinates 
        # normalised to [-1, 1].
        if is_native_2d or is_compat_2d:
            c_nz_xy = c_nz[:, :2] if is_compat_2d else c_nz  # drop the singleton depth index
            coords = c_nz_xy[:, ::-1].astype(np.float32)
            h, w = last_mod.shape[:2]
            if normalize:
                if w_box is None or h_box is None:
                    w_box, h_box = w, h
                    x_min, y_min = 0, 0
                coords[:, 0] = 2.0 * (coords[:, 0] - x_min) / w_box - 1.0
                coords[:, 1] = 2.0 * (coords[:, 1] - y_min) / h_box - 1.0
        else:
            # 3D: (d, h, w) -> (x, y, z) where x=w, y=h, z=d
            coords = c_nz[:, ::-1].astype(np.float32)
            d, h, w = last_mod.shape
            if normalize:
                if w_box is None or h_box is None or d_box is None:
                    w_box, h_box, d_box = w, h, d
                    x_min, y_min, z_min = 0, 0, 0
                coords[:, 0] = 2.0 * (coords[:, 0] - x_min) / w_box - 1.0
                coords[:, 1] = 2.0 * (coords[:, 1] - y_min) / h_box - 1.0
                coords[:, 2] = 2.0 * (coords[:, 2] - z_min) / d_box - 1.0

        if normalize:
            coords = np.clip(coords, -1.0, 1.0)
            assert_correct_coord_normalization(coords)
            
            # Apply Normalisation
            norm_type = self.args['dataset']['normalize_values']
            sub_id = row_dict['sub_id_int']
            has_seg = self.args['inr_decoder']['out_dim'][-1] > 0
            
            if norm_type == 'minmax_patient' and hasattr(self, 'patient_stats'):
                # Patient-level normalisation (across visits)
                stats = self.patient_stats[sub_id]
                if has_seg:
                    values_mod = values[..., :-1]
                else:
                    values_mod = values
                v_min = stats['min'].astype(np.float32)
                v_max = stats['max'].astype(np.float32)

                # Broad-castable subtraction: (N, C) - (C,)
                denom = v_max - v_min
                denom = np.where(denom == 0, 1.0, denom).astype(np.float32)

                values_mod = (values_mod - v_min) / denom
                if has_seg:
                    values[..., :-1] = values_mod
                else:
                    values = values_mod
            elif norm_type == 'minmax_patient_robust':
                # OPTION 1, per-VISIT robust percentile scaling on the foreground modality
                # channels: map this visit's [p1, p99] -> [0, 1] and clip. Excluding the
                # extremes (the zero frame already isn't sampled; p99 ignores saturated pixels)
                # removes per-visit exposure/gain mismatch that plain min-max leaves untouched.
                # Purely per-visit -> needs no patient_stats.
                values_mod = values[..., :-1] if has_seg else values
                p1 = np.percentile(values_mod, 1, axis=0).astype(np.float32)
                p99 = np.percentile(values_mod, 99, axis=0).astype(np.float32)
                denom = np.where((p99 - p1) <= 0, 1.0, (p99 - p1)).astype(np.float32)
                values_mod = np.clip((values_mod - p1) / denom, 0.0, 1.0)
                if has_seg:
                    values[..., :-1] = values_mod
                else:
                    values = values_mod
            elif norm_type == 'ref_match' and hasattr(self, 'patient_stats'):
                # OPTION 2, reference-based robust LINEAR matching. Align this visit's foreground
                # centre+spread (median, IQR) to the eye's BASELINE visit, then squash to [0,1] by
                # the reference p1/p99. Because every visit is mapped onto the same reference
                # distribution, inter-visit brightness is genuinely equalized (unlike min-max).
                stats = self.patient_stats[sub_id]
                values_mod = values[..., :-1] if has_seg else values
                m_v = np.median(values_mod, axis=0).astype(np.float32)
                q25, q75 = np.percentile(values_mod, [25, 75], axis=0)
                s_v = np.maximum(q75 - q25, 1e-6).astype(np.float32)
                matched = (values_mod - m_v) / s_v * stats['ref_iqr'] + stats['ref_med']
                denom = (stats['ref_p99'] - stats['ref_p1'])
                denom = np.where(denom <= 0, 1.0, denom).astype(np.float32)
                values_mod = np.clip((matched - stats['ref_p1']) / denom, 0.0, 1.0)
                if has_seg:
                    values[..., :-1] = values_mod
                else:
                    values = values_mod
            else:
                # Default: visit-level normalisation (individual)
                values = normalize_intensities(values, norm_type, has_seg=has_seg)
        return coords, values
    
    def augment_modalities(self, modalities, row_dict):
        if self.data_augmentation:
            sub_id = row_dict.get('sub_id_int', 0)
            epoch = getattr(self, 'epoch', 0)
            seed = int((sub_id * 100000) + epoch)
            rng = np.random.RandomState(seed)
            
            args_aug = self.args['data_augmentation']
            
            
            # Check translation probability
            trans_p = args_aug.get('augment_translation', {}).get('p', 0.0)
            max_shift = args_aug.get('augment_translation', {}).get('max_shift', 0)
            apply_trans = rng.rand() < trans_p
            if apply_trans and max_shift > 0:
                tx = rng.randint(-max_shift, max_shift + 1)
                ty = rng.randint(-max_shift, max_shift + 1)
            else:
                tx, ty = 0, 0
                
            # Check noise probability
            noise_p = args_aug.get('augment_noise', {}).get('p', 0.0)
            apply_noise = rng.rand() < noise_p
            noise_mean = args_aug.get('augment_noise', {}).get('mean', 0.0)
            noise_std = args_aug.get('augment_noise', {}).get('std', 0.0)
            
            augmented_modalities = {}
            for mod_key in modalities:
                data = modalities[mod_key].get_fdata().copy()
                

                # 2. Translate (with zero boundary padding)
                if tx != 0 or ty != 0:
                    shifted = np.roll(data, shift=(ty, tx), axis=(0, 1))
                    if ty > 0:
                        shifted[:ty, :] = 0.0
                    elif ty < 0:
                        shifted[ty:, :] = 0.0
                    if tx > 0:
                        shifted[:, :tx] = 0.0
                    elif tx < 0:
                        shifted[:, tx:] = 0.0
                    data = shifted
                    
                # 3. Noise (only for intensity modalities, not the segmentation mask)
                has_seg = self.args['inr_decoder']['out_dim'][-1] > 0
                is_seg = has_seg and (mod_key == self.modality_keys[-1])
                if apply_noise and not is_seg:
                    noise = rng.normal(noise_mean, noise_std, size=data.shape).astype(np.float32)
                    data = data + noise
                    
                augmented_modalities[mod_key] = data
            return augmented_modalities
        else:
            modalities_data = {mod_key: modalities[mod_key].get_fdata() for mod_key in modalities}
        return modalities_data

    def _get_temporal_key(self):
        temporal_key = self.args['dataset'].get('temporal_condition')
        if temporal_key is None:
            # Fallback: first enabled condition
            for key, enabled in self.args['dataset']['conditions'].items():
                if enabled:
                    temporal_key = key
                    break
        return temporal_key

    def load_time(self, row_dict, normalize=True, allow_extrapolation=False):
        """Extract and normalise the temporal condition (the `temporal_condition` variable, e.g.
        AgeatVisit) to [-1, 1] as the time-input scalar.

        allow_extrapolation=True bypasses the in-range clamp so novel-visit predictions can advance
        PAST the observed training horizon (the normalised value then exceeds |1|). Used by the
        future/extrapolation generation paths; real-visit reconstruction keeps the clamp.

        Applies model_gen.cond_scale so the time input stays consistent with the render-generation
        path (utils.normalize_condition), which also multiplies by cond_scale. With cond_scale=1.0
        this is a no-op; the multiplication only matters if cond_scale is ever changed, in which case
        training and render generation would otherwise silently diverge.
        """
        temporal_key = self._get_temporal_key()
        val = float(row_dict[temporal_key])
        if normalize:
            c_min = self.args['dataset']['constraints'][temporal_key]['min']
            c_max = self.args['dataset']['constraints'][temporal_key]['max']
            cond_scale = self.args['model_gen']['cond_scale']
            # Clamp out-of-range time (e.g. extrapolated visits past the last real one) to the actual
            # observed range so the SIREN sees an in-distribution input (boundary appearance) instead
            # of saturating to black/white. No-op for all in-range training/interpolation values
            # (real visits are already filtered to the range by check_constraints). Skipped when
            # allow_extrapolation=True so future predictions progress past the horizon.
            if not allow_extrapolation:
                val = min(max(val, c_min), c_max)
            val = (2.0 * (val - c_min) / (c_max - c_min) - 1.0) * cond_scale
        return torch.tensor([val], dtype=torch.float32)

    def load_conditions(self, row_dict, normalize=True, allow_extrapolation=False):
        conditions = []
        time_as_input = self.args['inr_decoder'].get('time_as_input', False)
        temporal_key = self._get_temporal_key()
        # has_weeks_map = (self.args['dataset'].get('visit_week_map') is not None)
        
        for key in self.args['dataset']['conditions']:
            if self.args['dataset']['conditions'][key]: # if condition is enabled
                if time_as_input and key == temporal_key:
                    continue  # Skip temporal condition, it goes through input path --> to avoid using the same variable as both input and conditioning vector
                val = row_dict[key]
                if normalize:
                    c_min, c_max = self.args['dataset']['constraints'][key]['min'], self.args['dataset']['constraints'][key]['max']
                    # Clamp out-of-range conditions (e.g. extrapolated weeks_from_baseline past the
                    # last real visit) to the observed range so FiLM stays in-distribution instead of
                    # driving the SIREN to saturation. No-op for in-range train/interpolation values
                    # (real visits are already filtered to the range by check_constraints). Skipped
                    # when allow_extrapolation=True so future predictions progress past the horizon.
                    if not allow_extrapolation:
                        val = min(max(float(val), c_min), c_max)
                    val = (((val - c_min) / (c_max - c_min)) * 2 - 1) * self.args['model_gen']['cond_scale']  # we normalise between -1 and 1
                conditions.append(val)
        
        return torch.tensor(conditions, dtype=torch.float32)

    def filter_dataframe(self, df):
        '''
        Filter the dataframe based on the constraints in the args.
        Also check for missing modalities and remove subjects with missing modalities.
        Returns a dataframe with the filtered data, i.e. the final subjects
        that will be used for training, validation, or testing. 
        '''
        print("--------------------------------")
        print("Sampling data for split ", self.split, "\n")
        df = self.sample_subject_ids(df)
        df = self.remove_missing_modalities(df)
        df = self.check_constraints(df)
        df = self.sample_subjects(df)
        print("Sampled ", len(df), " subjects for split ", self.split, "\n")
        print("--------------------------------")
        return df

    def sample_subject_ids(self, df, verbose=True):
        '''
        Sample subjects based on split column or subject_ids list.
        Also filters by site if configured.
        '''
        # Filter by specific overfit subject ID if specified
        overfit_sid = self.args.get('overfit_subject_id')
        if overfit_sid is not None:
            if verbose: print(f"Overfitting on specific subject ID: {overfit_sid}")
            sid = str(overfit_sid)
            idcol = df[self.id_column].astype(str)
            # Accept either the full id_column value (e.g. Eye_ID 'EYE01_OD') OR a patient-level id
            # that is a prefix of it (e.g. 'EYE01' -> 'EYE01_OD'/'EYE01_OS'); the eye filter below
            # disambiguates. The bare `== sid` check matched nothing when sid was a patient id but
            # id_column is the per-eye Eye_ID, yielding an empty dataset (num_samples=0).
            mask = (idcol == sid) | idcol.str.startswith(sid + "_")
            if 'Patient_ID' in df.columns:
                mask = mask | (df['Patient_ID'].astype(str) == sid)
            df = df[mask].reset_index(drop=True)

            # Filter by eye laterality if specified
            overfit_eye = self.args.get('overfit_eye_laterality')
            if overfit_eye is not None:
                if verbose: print(f"Filtering overfit subject by eye: {overfit_eye}")
                # Support both column names: 'Eye_Laterality' or 'Eye'
                eye_col = 'Eye_Laterality' if 'Eye_Laterality' in df.columns else ('Eye' if 'Eye' in df.columns else None)
                if eye_col:
                    df = df[df[eye_col].astype(str).str.upper() == str(overfit_eye).upper()].reset_index(drop=True)
                else:
                    print("Warning: Eye laterality column not found in dataframe.")
            
            if len(df) == 0:
                print(f"Warning: No rows found for overfit_subject_id {overfit_sid} (eye: {overfit_eye})")
            return df

        # 1. Filter by Site
        site_col = self.args['dataset'].get('site_column')
        sites = self.args['dataset'].get('sites')
        if site_col and sites:
            if verbose: print(f"Filtering by sites: {sites}")
            df = df[df[site_col].isin(sites)]

        # 2. Filter by Split
        split_col = self.args['dataset'].get('split_column')
        if split_col and split_col in df.columns:
            if verbose: print(f"Filtering by split column '{split_col}' for split '{self.split}'")
            # Map self.split to column values
            # GAP-INR splits: 'train', 'val', 'test'
            # CSV values might be 'train', 'test', 'val', 'validation'
            target_split = self.split.lower()
            if target_split == 'val':
                possible_vals = ['val', 'validation']
            else:
                possible_vals = [target_split]
            
            df = df[df[split_col].astype(str).str.lower().isin(possible_vals)]
            
            if verbose: print(f"Number of subjects after split filtering: {len(df)}")
            
        # 3. Fallback/Additional Filter by subject_ids.yaml only if subject_ids list is NOT empty
        # This allows using ONLY the CSV split if subject_ids.yaml has empty lists (as we set them)
        elif self.args['dataset']['subject_ids'][self.split]:
            if verbose: 
                print(f"Sampling subjects from subject_ids list. Number of subjects in dataframe: {len(df)}, "
                      f"Number of subjects in subject_ids list: {len(self.args['dataset']['subject_ids'][self.split])}")
            df = df[df[self.id_column].isin(self.args['dataset']['subject_ids'][self.split])]
            if verbose: print(f"Number of subjects in dataframe after sampling: {len(df)} \n")
            
        return df

    def _map_subject_ids(self):
        """
        Maps the id_column (e.g. eye_id, patient_eye_id) to a unique integer 0..N-1.
        This ensures that multiple visits of the same patient-eye get the SAME latent code.
        """
        # Now map the ID column to integers
        if self.id_column in self.df.columns:
            unique_ids = sorted(self.df[self.id_column].unique())  # we extract the unique Eye_IDs and sort them
            id_mapping = {uid: i for i, uid in enumerate(unique_ids)}
            self.df['sub_id_int'] = self.df[self.id_column].map(id_mapping)
            print(f"Mapped {len(unique_ids)} unique subjects (patient-eyes) from {len(self.df)} total visits.")
        else:
            print(f"Warning: id_column '{self.id_column}' not found. Using row index as subject index (unique per visit).")
            self.df['sub_id_int'] = np.arange(len(self.df))

        if self.args['dataset'].get('independent_visits', False):
            self.sub_id_map = np.arange(len(self.df)).astype(int)
        else:
            self.sub_id_map = self.df['sub_id_int'].values.astype(int)
        
        # Save dataframe with sub_id_int for verification
        out_path = os.path.join(self.args.get('output_dir', '.'), f'mapped_df_{self.split}.csv')
        self.df.to_csv(out_path, index=False)
        # print(f"Saved mapped dataframe to {out_path}")

    def _temporal_sort_col(self):
        """Column used to order an eye's visits chronologically (first enabled condition,
        else Visit_Number)."""
        for key, enabled in self.args['dataset'].get('conditions', {}).items():
            if enabled:
                return key
        return 'Visit_Number' if 'Visit_Number' in self.df.columns else self.df.columns[0]

    def get_longitudinal_indices(self, holdout_position=None, single_visit_to='eval', support_k=None,
                                 pair_source=None, pair_target=None):
        """
        Groups visits by subject and splits them into optimisation and evaluation indices.

        Args:
            holdout_position: 1-indexed chronological position of the visit to hold out.
                              None = hold out the last visit (default behaviour).
                              e.g., 2 = hold out the 2nd chronological visit.
            pair_source/pair_target: if BOTH set (1-indexed positions, source < target), run the
                              per-PAIR forecast: opt = ONLY the visit at `pair_source`, eval = ONLY
                              the visit at `pair_target`, per eye. Eyes lacking either position are
                              skipped. Matches ImageFlowNet's pairwise eval. Overrides holdout/support_k.
            single_visit_to:  where to route patient-eyes with a single visit (no visit to
                              hold out). 'eval' (default, used for validation) keeps the lone
                              visit as evaluation-only; 'opt' (used for the test set) fits the
                              latent on the lone acquired visit instead, mirroring the clinical
                              case where a new patient is imaged once and the latent is optimised
                              on that image to then predict future states.
            support_k:        if set, OVERRIDES holdout: fit the latent on the FIRST `support_k`
                              chronological visits (opt) and evaluate on ALL remaining (later)
                              visits (eval), i.e. "given the first k scans, forecast the rest".
                              support_k=1 = the one-visit clinical forecast (fit on baseline,
                              predict every future visit, scored against GT).

        Returns:
            opt_pos_idcs:  list of positional indices for latent optimisation.
            eval_pos_idcs: list of positional indices for evaluation (held-out visits).
        """
        if holdout_position == 'none':
            all_indices = list(range(len(self.df)))
            return all_indices, []

        # pairwise mode: opt = {visit at pair_source}, eval = {visit at pair_target}, per eye.
        # 1-indexed; only eyes that HAVE both positions (with source < target) participate.
        if pair_source is not None and pair_target is not None:
            s0, t0 = int(pair_source) - 1, int(pair_target) - 1
            temporal_col = self._temporal_sort_col()
            opt_pos_idcs, eval_pos_idcs = [], []
            for sub_id in self.df['sub_id_int'].unique():
                sub_sorted = self.df[self.df['sub_id_int'] == sub_id].sort_values(temporal_col)
                indices = [self.df.index.get_loc(idx) for idx in sub_sorted.index]
                if 0 <= s0 < t0 < len(indices):
                    opt_pos_idcs.append(indices[s0])
                    eval_pos_idcs.append(indices[t0])
            return opt_pos_idcs, eval_pos_idcs

        # support_k mode: opt = first k visits, eval = all later visits (forecast targets).
        if support_k is not None:
            k = max(1, int(support_k))
            temporal_col = self._temporal_sort_col()
            opt_pos_idcs, eval_pos_idcs = [], []
            for sub_id in self.df['sub_id_int'].unique():
                sub_sorted = self.df[self.df['sub_id_int'] == sub_id].sort_values(temporal_col)
                indices = [self.df.index.get_loc(idx) for idx in sub_sorted.index]
                opt_pos_idcs.extend(indices[:k])
                eval_pos_idcs.extend(indices[k:])   # later visits = forecast targets (empty if <=k visits)
            return opt_pos_idcs, eval_pos_idcs

        opt_pos_idcs = []
        eval_pos_idcs = []
        
        # Resolve temporal column for chronological sorting
        temp_key = self.args['dataset'].get('conditions', {})
        temporal_col = None
        for key, enabled in temp_key.items():
            if enabled:
                temporal_col = key
                break
        
        if temporal_col is None:
            temporal_col = 'Visit_Number' if 'Visit_Number' in self.df.columns else self.df.columns[0]

        # Group by sub_id_int
        for sub_id in self.df['sub_id_int'].unique():
            sub_group = self.df[self.df['sub_id_int'] == sub_id]
            sub_group_sorted = sub_group.sort_values(temporal_col)
            indices = [self.df.index.get_loc(idx) for idx in sub_group_sorted.index]
            
            if len(indices) <= 1:
                # Single visit: no visit to hold out. For validation we keep it as
                # evaluation-only; for the test set we instead fit the latent on it.
                if single_visit_to == 'opt':
                    opt_pos_idcs.append(indices[0])
                else:
                    eval_pos_idcs.append(indices[0])
                continue

            # Determine which index to hold out (0-indexed internally)
            if holdout_position is None:
                ho_idx = len(indices) - 1  # last visit
            else:
                ho_idx = holdout_position - 1  # convert 1-indexed to 0-indexed
                if ho_idx < 0 or ho_idx >= len(indices):
                    # Fallback to last visit if position exceeds available visits
                    ho_idx = len(indices) - 1

            eval_pos_idcs.append(indices[ho_idx])
            opt_pos_idcs.extend(idx for j, idx in enumerate(indices) if j != ho_idx)
                
        return opt_pos_idcs, eval_pos_idcs

    def update_with_indices(self, pos_indices):
        """Creates a copy of self with only the specified positional indices."""
        new_data = copy.copy(self)
        new_data.df = self.df.iloc[pos_indices].reset_index(drop=True)
        
        # Track parent indices to preserve mapping to original validation latent slots
        if hasattr(self, 'parent_indices'):
            new_data.parent_indices = [self.parent_indices[i] for i in pos_indices]
        else:
            new_data.parent_indices = list(pos_indices)
            
        return new_data

    def remove_missing_modalities(self, df, verbose=True):
        """
        Removes rows if any required modality file is missing on the filesystem.
        """
        if verbose:
            print("Initial number of subjects:", len(df))

        modalities = self.args['dataset']['modalities']
        keep_mask = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            keep_row = True
            for modality in modalities:
                try:
                    # Resolve path without triggering download if LakeFS is active
                    path = self.resolve_path(row_dict, modality, download=False)
                    if path is None or path == "":
                        keep_row = False
                        break
                    # If not using LakeFS, we must verify that the file actually exists locally
                    if self.lakefs_loader is None and not os.path.exists(path):
                        keep_row = False
                        break
                except Exception:
                    keep_row = False
                    break
            keep_mask.append(keep_row)

        keep_mask = np.array(keep_mask, dtype=bool)
        dropped_count = (~keep_mask).sum()
        if dropped_count > 0 and verbose:
            print(f"Dropping {dropped_count} visits/subjects due to missing files on disk.")

        df = df[keep_mask].reset_index(drop=True)
        if verbose:
            print("Number of subjects after removing missing modalities:", len(df))

        return df

    def check_constraints(self, df, verbose=True):
        """
        Drops rows if any numeric constraint is outside [min, max].
        """
        if verbose:
            print("Initial number of subjects:", len(df))

        constraints_dict = self.args['dataset'].get('constraints', {})

        # Build a mask to keep all that pass constraints
        keep_mask = np.ones(len(df), dtype=bool)

        for ckey, cinfo in constraints_dict.items():
            if cinfo.get('type') == 'numeric':
                cmin = cinfo.get('min', None)
                cmax = cinfo.get('max', None)
                if cmin is not None and cmax is not None and ckey in df.columns:
                    # Convert to numeric, coercing errors (like 'na') to NaN
                    vals = pd.to_numeric(df[ckey], errors='coerce').values
                    # Comparisons with NaN result in False, so rows with 'na' will be dropped
                    this_mask = (vals >= cmin) & (vals <= cmax)
                    keep_mask &= this_mask
            elif cinfo.get('type') == 'categoric':
                if ckey in df.columns:
                    this_mask = df[ckey].isin(cinfo.get('values'))
                    keep_mask &= this_mask
                else:
                    raise ValueError(f"Constraint column {ckey} not found in dataframe")
            else:
                raise ValueError(f"Constraint type {cinfo.get('type')} not supported")

        dropped_count = np.count_nonzero(~keep_mask)
        if dropped_count > 0 and verbose:
            print(f"Dropping {dropped_count} subjects outside constraint ranges.")
        df = df[keep_mask].reset_index(drop=True)

        if verbose:
            print("Number of subjects after constraints check:", len(df))

        return df
    

    def sample_subjects(self, df, verbose=True):
            """
            Nested sampling across constraints that have a 'distribution' with 'priority'.
            """
            n_unique = df[self.id_column].nunique()
            # Default to using all available subjects if n_subjects is not configured for this
            # split (e.g. 'test', which is evaluated in full).
            if self.split not in self.args['n_subjects']:
                self.args['n_subjects'][self.split] = n_unique
            if self.args['n_subjects'][self.split] > n_unique:
                print(f"Warning: n_subjects ({self.args['n_subjects'][self.split]}) > unique patients available ({n_unique}). "
                      f"Using all {n_unique} patients.")
                self.args['n_subjects'][self.split] = n_unique

            max_num_subjects = self.args['n_subjects'][self.split]
            # Gather constraints that define a distribution + priority
            constraints_with_prio = []
            for cname, cinfo in self.args['dataset'].get('constraints', {}).items():
                dist_info = cinfo.get('distribution', {})
                if 'priority' in dist_info:  # we only consider constraints that define a priority
                    constraints_with_prio.append((cname, cinfo))

            # Sort by ascending priority (higher = first).
            constraints_with_prio.sort(key=lambda x: x[1]['distribution']['priority'], reverse=True)

            if verbose:
                print(f"[sample_subjects] # subjects before sampling: {len(df)}")
                print("[sample_subjects] Constraints in priority order:",
                    [c[0] for c in constraints_with_prio])

            # Convert to list for recursion 
            constraints_list = [(cname, cinfo) for (cname, cinfo) in constraints_with_prio]
            
            # If no constraints, return all subjects (skip sampling)
            if len(constraints_list) == 0:
                unique_ids = df[self.id_column].unique()
                if verbose:
                    print(f"[sample_subjects] No constraints - using unique {self.id_column} sampling")
                
                if len(unique_ids) > max_num_subjects:
                    sampled_ids = np.random.choice(unique_ids, size=max_num_subjects, replace=False)
                    df = df[df[self.id_column].isin(sampled_ids)]
                    if verbose:
                        print(f"[sample_subjects] Truncated to max_num_subjects={max_num_subjects} patients")
                return df
            
            # Sampling currently only uniformly over the highest priority constraint, random over the rest
            df_sampled = self._shallow_sampling(df, constraints_list, max_num_subjects, verbose)


            # Global cap if still too large (ensure we cap by unique subjects)
            current_n_unique = df_sampled[self.id_column].nunique()
            if current_n_unique > max_num_subjects:
                unique_ids = df_sampled[self.id_column].unique()
                sampled_ids = np.random.choice(unique_ids, size=max_num_subjects, replace=False)
                df_sampled = df_sampled[df_sampled[self.id_column].isin(sampled_ids)].reset_index(drop=True)
                if verbose:
                    print(f"[sample_subjects] Truncated final set to {max_num_subjects} unique patients")

            if verbose:
                print(f"[sample_subjects] # subjects after nested sampling: {len(df_sampled)}")

            # Print & Save histogram if verbose
            if verbose:
                for constraint in constraints_list:
                    self._print_and_save_hist(df_sampled, constraint)

            return df_sampled

    def _shallow_sampling(self, df, constraints_list, max_num_subjects, verbose=True):
        """
        Sample subjects (keeping all visits) uniformly over the highest priority constraint.
        """
        current_constraint = constraints_list[0]
        current_constraint_name = current_constraint[0]
        current_constraint_info = current_constraint[1]
        c_min = current_constraint_info.get('min', None)
        c_max = current_constraint_info.get('max', None)
        bins = current_constraint_info.get('distribution').get('bins')
        
        # Prepare subject-level representative values (e.g., from the first visit)
        # to ensure the same subject isn't split across bins or sampled multiple times
        sub_df = df.sort_values([self.id_column, current_constraint_name]).groupby(self.id_column).first().reset_index()
        
        if bins is None:
            bins = int(c_max - c_min) if (c_max is not None and c_min is not None) else 10
        
        # create bins
        edges = np.linspace(c_min if c_min is not None else sub_df[current_constraint_name].min(), 
                            c_max if c_max is not None else sub_df[current_constraint_name].max(), 
                            bins+1)
        edges[-1] += 1e-6
        
        values = sub_df[current_constraint_name].values
        bin_idx = np.digitize(values, edges) - 1
        bin_sizes = np.bincount(bin_idx, minlength=bins)
        
        if verbose:
            print(f"[Longitudinal Sampling] Unique subjects available: {len(sub_df)}")
            print(f"Sampling {max_num_subjects} patients from bins...")
        
        # sample IDs from each bin
        samples_drawn_per_bin = [0] * bins
        drained_bins = []
        while np.sum(samples_drawn_per_bin) < max_num_subjects and len(drained_bins) < bins:
            num_remaining_bins = bins - len(np.unique(drained_bins))
            num_remaining_subjects = max_num_subjects - np.sum(samples_drawn_per_bin)
            required_samples_per_bin = math.ceil(num_remaining_subjects / num_remaining_bins)
            
            for i in range(bins):
                if i in drained_bins: continue
                available_in_bin = bin_sizes[i] - samples_drawn_per_bin[i]
                n_samples = min(required_samples_per_bin, available_in_bin)
                samples_drawn_per_bin[i] += n_samples
                if samples_drawn_per_bin[i] == bin_sizes[i]:
                    drained_bins.append(i)
        
        sampled_ids = []
        for i, n_samples in enumerate(samples_drawn_per_bin):
            if n_samples > 0:
                bin_ids = sub_df[bin_idx == i][self.id_column]
                sampled_ids.extend(bin_ids.sample(n=n_samples, random_state=42).tolist())
        
        # Return all visits for the sampled IDs
        df_sampled = df[df[self.id_column].isin(sampled_ids)].reset_index(drop=True)
        return df_sampled


    def _print_and_save_hist(self, df, constraint):
        """
        Print the histogram data (counts per bin) for `column_name` in `df`,
        and save a bar plot as an image. Uses 30 bins by default, or fewer if df is small.
        """
        indent = "  "
        current_cname, current_cinfo = constraint
        # Drop NaNs if they exist
        col_data = df[current_cname].dropna().values
        if len(col_data) == 0:
            print(f"{indent}[Histogram] No data for column '{current_cname}'")
            return

        dist_info = current_cinfo.get('distribution', {})
        dist_type = dist_info.get('type')
        c_min = current_cinfo.get('min', None)
        c_max = current_cinfo.get('max', None)
        if c_min is None:
            c_min = col_data.min()
        if c_max is None:
            c_max = col_data.max()
            
        bins = dist_info.get('bins')
        if bins is None:
            # Simple heuristic for bin count if not specified
            if np.issubdtype(col_data.dtype, np.number):
                bins = min(20, len(np.unique(col_data)))
            else:
                bins = 10 

        if np.issubdtype(col_data.dtype, np.number):
            edges = np.linspace(c_min, c_max, bins+1)
            edges[-1] += 1e-9  # ensures inclusive of c_max
        else:
            # Categorical fallback (e.g. if strings slipped through)
            print(f"{indent}[Histogram] Categorical/Non-numeric data in '{current_cname}', skipping numeric histogram.")
            return

        counts, bin_edges = np.histogram(col_data, bins=edges)
        print(f"{indent}[Histogram] {current_cname} counts: {counts.tolist()}")
        print(f"{indent}[Histogram] {current_cname} bin_edges: {bin_edges.tolist()}")

        # Save a bar plot
        plt.figure()
        # Midpoints for bar chart
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # plot ticks for every bin centre, with slight rotation of the labels
        plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9)
        plt.xticks(bin_centers, rotation=45)
        # plot every y-tick 
        plt.yticks(range(0, max(counts)+1))
        plt.title(f"Histogram of {current_cname} with {dist_type} distribution")
        plt.xlabel(current_cname)
        plt.ylabel("Count")

        out_path = os.path.join(self.args['output_dir'], f"hist_{current_cname}_dist_{dist_type}_{self.split}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()  # free memory
        print(f"{indent}Saved histogram to {out_path}")

    def _init_data_augmentation(self):
        """
        Initialise data augmentation.
        """
        aug_cfg = self.args['data_augmentation']
        mode = aug_cfg.get('mode', 'pseudo_eye')
        # Legacy STOCHASTIC per-load augmentation (translation/noise via augment_modalities) is only
        # used in 'stochastic' mode. 'pseudo_eye' applies deterministic transforms in load_modalities.
        self.data_augmentation = bool(aug_cfg.get('activate', False)) and (self.split == 'train') and (mode == 'stochastic')
