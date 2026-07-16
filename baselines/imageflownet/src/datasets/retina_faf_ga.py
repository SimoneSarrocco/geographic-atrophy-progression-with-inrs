"""Longitudinal FAF + GA dataset for ImageFlowNet.

Drop-in analogue of ``retina_ucsf.py`` for this cohort, so the ImageFlowNet
harness (``train_2pt_all.py``) and the segmentor trainer (``train_segmentor.py``)
run on it UNCHANGED. Unlike the UCSF loader (which reads pre-exported PNG
folders and parses time from the filename), this reads the cohort directly from the
clinical CSV + cached FAF/GA paths, so it stays consistent with the other GAP-INR
baselines (NISF / MetaSeg / amd-inr-progression):

  * one SUBJECT = one patient-eye (``Eye_ID``); visits sorted by ``Visit_Number``.
  * time = weeks-from-baseline, computed per eye from ``visit_date`` (NOT a CSV
    column), global range ~[0, 54] weeks. Returned RAW (weeks); the harness scales
    it via ``t_multiplier = ode_max_t / max_t``.
  * geometry = centered 620 crop of the native 768 (removes the per-visit black
    registration frame without clipping GA), then resize to ``target_dim``.
  * intensity = ImageFlowNet's own normalization (percentile-clip + z-score ->
    [-1, 1]) so the models + shared segmentor see inputs in their expected range.
  * SPLIT = the official ``split`` column (train/val/test) at the eye level,
    exposed via ``predefined_split`` (prepare_dataset uses it instead of a random
    fraction split) so the comparison matches GAP-INR's split.

The forecasting dataset returns ``(images, timestamps)`` exactly like the UCSF
one. GA mask paths are retained per record (``records_by_patient``) so a later
unified shared-segmentor evaluation can score predicted FAF against the REAL GA
masks; ``__getitem__`` itself returns only images+timestamps (harness contract).
"""

import itertools
import os
import sys
from glob import glob
from typing import List, Literal, Tuple

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])   # .../baselines/imageflownet

# Shared canonical preprocessing (baselines/imageflownet/common_preproc.py), imported by GAP-INR and
# every baseline so all methods normalize identically.
sys.path.insert(0, root_dir)
import common_preproc as _common_preproc        # noqa: E402

# Clinical CSV: the GAP-INR repo's data/clinical_metadata.csv by default (see eval_spec.CSV_PATH);
# override with the GAPINR_CSV environment variable.
from eval_spec import CSV_PATH as DEFAULT_CSV   # noqa: E402
CROP_SIZE = 620   # registration-frame crop shared with GAP-INR faf_ga_620 (native 768 -> 620)
MIN_VISITS = 2    # need >= 2 real visits for a progression sequence


# --------------------------------------------------------------------------- #
# image / time helpers
# --------------------------------------------------------------------------- #
def normalize_image(image: np.array) -> np.array:
    '''CANONICAL plain per-visit min-max (models/common_preproc.normalize): (img-min)/(max-min) ->
    [0, 1], then map to [-1, 1] (ImageFlowNet's input range, kept so the rest of the pipeline /
    segmentor are unchanged). Shared by every method for a fair comparison. NB: changing the canonical
    normalization REQUIRES retraining the segmentor and the forecasters.'''
    return _common_preproc.normalize_pm1(image)


def _center_crop_np(arr: np.array, size: int) -> np.array:
    '''Center-crop a (H, W) array to (size, size). No-op if already <= size or size is None.'''
    if size is None:
        return arr
    h, w = arr.shape[:2]
    if h <= size and w <= size:
        return arr
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return arr[top:top + size, left:left + size]


def _load_raw(path: str, target_dim: Tuple[int], crop_size: int, is_mask: bool) -> np.array:
    '''Load grayscale, centered 620-crop, then resize to target_dim. Returns a raw
    float32 (H, W) array (NOT yet intensity-normalized / thresholded).'''
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    img = _center_crop_np(img, crop_size)
    if target_dim is not None:
        dsize = (int(target_dim[0]), int(target_dim[1]))
        if (img.shape[1], img.shape[0]) != dsize:
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
            img = cv2.resize(img, dsize, interpolation=interp)
    return img.astype(np.float32)


def add_channel_dim(array: np.array) -> np.array:
    assert len(array.shape) == 2
    return array[None, :, :]


def load_image(path: str, target_dim: Tuple[int] = None, crop_size: int = CROP_SIZE,
               normalize: bool = True) -> np.array:
    '''FAF loader (crop -> resize -> optional z-score normalize).'''
    image = _load_raw(path, target_dim, crop_size, is_mask=False)
    if normalize:
        image = normalize_image(image)
    return image


# --------------------------------------------------------------------------- #
# CSV -> per-eye records
# --------------------------------------------------------------------------- #
def _build_records(csv: str, num_visits_cap: int = None, min_visits: int = MIN_VISITS):
    '''Group the clinical CSV into per-eye records. Returns (records_by_patient,
    split_by_patient, max_t_weeks). Each record is a dict
    {faf, mask, t, eye_id, visit, sx, sy} where sx/sy are the RAW ScaleXSlo /
    ScaleYSlo for that visit (mm/pixel) used for GAP-INR-compatible lesion areas.'''
    df = pd.read_csv(csv)
    df = df.dropna(subset=['faf_path', 'ga_mask_path']).copy()

    # weeks-from-baseline per eye (computed from visit_date, like the other baselines)
    df['visit_date'] = pd.to_numeric(df['visit_date'], errors='coerce')
    df['weeks'] = df.groupby('Eye_ID')['visit_date'].transform(lambda s: (s - s.min()) / 7.0)
    df = df.dropna(subset=['weeks'])

    records_by_patient, split_by_patient = [], []
    max_t = 0.0
    for eye, g in df.groupby('Eye_ID'):
        g = g.sort_values('Visit_Number')
        if len(g) < min_visits:
            continue
        if num_visits_cap is not None:
            g = g.iloc[:num_visits_cap]
        recs = [{'faf': r['faf_path'], 'mask': r['ga_mask_path'], 't': float(r['weeks']),
                 'eye_id': str(eye), 'visit': int(r['Visit_Number']),
                 'sx': float(r.get('ScaleXSlo', 1.0)), 'sy': float(r.get('ScaleYSlo', 1.0))}
                for _, r in g.iterrows()]
        records_by_patient.append(recs)
        split_by_patient.append(str(g.iloc[0]['split']).lower())
        max_t = max(max_t, max(r['t'] for r in recs))

    if not records_by_patient:
        raise RuntimeError(f'No eyes (>= {min_visits} visits) loaded from {csv}')
    return records_by_patient, split_by_patient, max_t


# --------------------------------------------------------------------------- #
# forecasting dataset
# --------------------------------------------------------------------------- #
class RetinaFafGaDataset(Dataset):

    def __init__(self,
                 csv: str = DEFAULT_CSV,
                 target_dim: Tuple[int] = (620, 620),
                 crop_size: int = CROP_SIZE,
                 num_visits_cap: int = None):
        super().__init__()
        self.target_dim = target_dim
        self.crop_size = crop_size

        self.records_by_patient, self.split_by_patient, self.max_t = _build_records(
            csv, num_visits_cap=num_visits_cap)
        # `image_by_patient` kept as an alias so split-by-index logic mirrors UCSF.
        self.image_by_patient = self.records_by_patient

        self.predefined_split = {
            'train': [i for i, s in enumerate(self.split_by_patient) if s == 'train'],
            'val':   [i for i, s in enumerate(self.split_by_patient) if s == 'val'],
            'test':  [i for i, s in enumerate(self.split_by_patient) if s == 'test'],
        }

    def __len__(self) -> int:
        return len(self.records_by_patient)

    def num_image_channel(self) -> int:
        return 1

    def return_statistics(self) -> None:
        print('max time (weeks):', self.max_t)
        print('Number of eyes:', len(self.records_by_patient))
        for split, idx in self.predefined_split.items():
            print(f'  {split}: {len(idx)} eyes')
        num_visit_map = {}
        for recs in self.records_by_patient:
            num_visit_map[len(recs)] = num_visit_map.get(len(recs), 0) + 1
        for k, v in sorted(num_visit_map.items()):
            print('%d visits: %d eyes.' % (k, v))


class RetinaFafGaSubset(RetinaFafGaDataset):

    def __init__(self,
                 main_dataset: RetinaFafGaDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'all_subsequences',
                                              'all_subarrays', 'full_sequence'],
                 transforms=None,
                 transforms_aug=None):
        # NOTE: do not call RetinaFafGaDataset.__init__ (no re-read of CSV).
        Dataset.__init__(self)
        self.target_dim = main_dataset.target_dim
        self.crop_size = main_dataset.crop_size
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug

        self.records_by_patient = [main_dataset.records_by_patient[i] for i in subset_indices]

        self.all_pairs, self.all_subsequences, self.all_subarrays = [], [], []
        for recs in self.records_by_patient:
            for (i, j) in itertools.combinations(np.arange(len(recs)), r=2):
                self.all_pairs.append([recs[i], recs[j]])
                self.all_subarrays.append(recs[i:j + 1])
            for n in range(2, len(recs) + 1):
                for combo in itertools.combinations(np.arange(len(recs)), r=n):
                    self.all_subsequences.append([recs[k] for k in combo])

    def __len__(self) -> int:
        return {
            'one_pair': len(self.records_by_patient),
            'all_pairs': len(self.all_pairs),
            'all_subsequences': len(self.all_subsequences),
            'all_subarrays': len(self.all_subarrays),
            'full_sequence': len(self.records_by_patient),
        }[self.return_format]

    def _load_seq(self, recs) -> Tuple[np.array, np.array]:
        images = np.array([_load_raw(r['faf'], self.target_dim, self.crop_size, is_mask=False)
                           for r in recs])
        timestamps = np.array([r['t'] for r in recs], dtype=np.float32)
        return images, timestamps

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            recs = self.records_by_patient[idx]
            pair_idx = list(itertools.combinations(np.arange(len(recs)), r=2))
            recs = [recs[i] for i in pair_idx[np.random.choice(len(pair_idx))]]
            images, timestamps = self._load_seq(recs)
        elif self.return_format == 'all_pairs':
            images, timestamps = self._load_seq(self.all_pairs[idx])
        elif self.return_format == 'all_subsequences':
            images, timestamps = self._load_seq(self.all_subsequences[idx])
        elif self.return_format == 'all_subarrays':
            images, timestamps = self._load_seq(self.all_subarrays[idx])
        elif self.return_format == 'full_sequence':
            images, timestamps = self._load_seq(self.records_by_patient[idx])

        if self.return_format in ['one_pair', 'all_pairs']:
            assert len(images) == 2
            image1, image2 = images[0], images[1]

            if self.transforms is not None:
                transformed = self.transforms(image=image1, image_other=image2)
                image1, image2 = transformed['image'], transformed['image_other']

            if self.transforms_aug is not None:
                transformed_aug = self.transforms_aug(image=image1, image_other=image1)
                image1_aug = add_channel_dim(normalize_image(transformed_aug['image']))

            image1 = add_channel_dim(normalize_image(image1))
            image2 = add_channel_dim(normalize_image(image2))

            if self.transforms_aug is not None:
                images = np.vstack((image1[None, ...], image2[None, ...], image1_aug[None, ...]))
            else:
                images = np.vstack((image1[None, ...], image2[None, ...]))

        else:  # all_subsequences / all_subarrays / full_sequence
            num_images = len(images)
            assert 2 <= num_images < 10
            image_list = np.rollaxis(images, axis=0)
            data_dict = {'image': image_list[0]}
            for i in range(num_images - 1):
                data_dict['image_other%d' % (i + 1)] = image_list[i + 1]
            if self.transforms is not None:
                data_dict = self.transforms(**data_dict)
            images = normalize_image(add_channel_dim(data_dict['image']))[None, ...]
            for i in range(num_images - 1):
                images = np.vstack((images,
                                    normalize_image(add_channel_dim(
                                        data_dict['image_other%d' % (i + 1)]))[None, ...]))

        return images, timestamps


# --------------------------------------------------------------------------- #
# segmentation dataset (for train_segmentor.py: image -> GA mask)
# --------------------------------------------------------------------------- #
class RetinaFafGaSegDataset(Dataset):

    def __init__(self,
                 csv: str = DEFAULT_CSV,
                 target_dim: Tuple[int] = (620, 620),
                 crop_size: int = CROP_SIZE):
        super().__init__()
        self.target_dim = target_dim
        self.crop_size = crop_size
        # Use ALL eyes (min_visits=1) for segmentation training; one sample per visit.
        self.records_by_patient, self.split_by_patient, _ = _build_records(csv, min_visits=1)
        self.image_by_patient = [[r['faf'] for r in recs] for recs in self.records_by_patient]
        self.mask_by_patient = [[r['mask'] for r in recs] for recs in self.records_by_patient]
        self.predefined_split = {
            'train': [i for i, s in enumerate(self.split_by_patient) if s == 'train'],
            'val':   [i for i, s in enumerate(self.split_by_patient) if s == 'val'],
            'test':  [i for i, s in enumerate(self.split_by_patient) if s == 'test'],
        }

    def __len__(self) -> int:
        return len(self.records_by_patient)

    def num_image_channel(self) -> int:
        return 1


class RetinaFafGaSegSubset(RetinaFafGaSegDataset):

    def __init__(self,
                 main_dataset: RetinaFafGaSegDataset = None,
                 subset_indices: List[int] = None,
                 transforms=None):
        Dataset.__init__(self)
        self.target_dim = main_dataset.target_dim
        self.crop_size = main_dataset.crop_size
        self.transforms = transforms

        image_by_patient = [main_dataset.image_by_patient[i] for i in subset_indices]
        mask_by_patient = [main_dataset.mask_by_patient[i] for i in subset_indices]
        self.image_list = [im for folder in image_by_patient for im in folder]
        self.mask_list = [m for folder in mask_by_patient for m in folder]
        assert len(self.image_list) == len(self.mask_list)

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = _load_raw(self.image_list[idx], self.target_dim, self.crop_size, is_mask=False)
        mask = _load_raw(self.mask_list[idx], self.target_dim, self.crop_size, is_mask=True)
        if self.transforms is not None:
            # Run albumentations on UINT8 [0,255] (its intended range). On our float32 [0,255] FAF,
            # RandomBrightnessContrast assumes float images are in [0,1] and CLIPS to it -> the image
            # collapses to a constant (~all 1.0) -> completely BLACK after min-max normalize. uint8
            # makes brightness/contrast operate in [0,255] correctly; geometry (ShiftScaleRotate) is
            # unaffected. Same augmentations as the original repo, just the correct dtype.
            transformed = self.transforms(image=np.clip(image, 0, 255).astype(np.uint8),
                                          mask=np.clip(mask, 0, 255).astype(np.uint8))
            image = transformed['image'].astype(np.float32)
            mask = transformed['mask'].astype(np.float32)
        image = add_channel_dim(normalize_image(image))
        mask = add_channel_dim(mask > 128)
        return image, mask


if __name__ == '__main__':
    ds = RetinaFafGaDataset(target_dim=(620, 620))
    ds.return_statistics()
