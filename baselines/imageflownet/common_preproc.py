"""Canonical preprocessing shared by EVERY method (GAP-INR, ImageFlowNet family, NISF, MetaSeg)
so the comparison is fair: identical resolution and identical intensity normalization.

THE PROCEDURE (agreed for the paper):
  1. resolution : center-crop the native 768 to 620 (preserves all GA; a direct 512 crop clips GA in
                  ~5/133 visits), then resize 620 -> 512 (FAF bicubic, mask nearest). All scoring on 512.
  2. normalize  : PLAIN PER-VISIT MIN-MAX (canonical, user decision 2026-06-27). For each image
                  independently, (img - min)/(max - min) on the full image -> [0, 1]. Every repo
                  imports the canonical `normalize` / `normalize_pm1` below, so this is the single
                  flip-point. (normalize_robust = the alternative foreground-p1/p99 variant.)

This matches GAP-INR's per-image min-max (normalize_values NOT in {minmax_patient, *_robust, ref_match}
-> per-visit min/max); the other repos call these functions so they are byte-compatible.

Range convention: this returns [0, 1]. A model that wants [-1, 1] (e.g. ImageFlowNet) should map
`x*2 - 1` AFTER calling normalize_robust -- the *normalization* (what information is kept/equalized)
is then identical across methods; only the final affine input range differs, which is invertible and
fairness-neutral (PSNR/SSIM compare pred vs GT in the same range; DICE is range-independent).
"""
import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False

CROP_SIZE = 620   # native 768 -> 620 center crop (loss-free for GA; min square keeping all lesions)
EVAL_DIM = 512    # 620 -> 512 resize; everything is scored on this grid

# foreground robust-percentile bounds (exclude exposure outliers / saturated pixels)
P_LOW = 1.0
P_HIGH = 99.0


def center_crop(img, crop=CROP_SIZE):
    """Center-crop a (H, W[, C]) array to crop x crop (no resize)."""
    h, w = img.shape[:2]
    top, left = (h - crop) // 2, (w - crop) // 2
    return img[top:top + crop, left:left + crop]


def crop_resize(img, crop=CROP_SIZE, out=EVAL_DIM, is_mask=False):
    """Canonical spatial op: center-crop `crop` then resize to `out` (mask nearest, FAF bicubic)."""
    if not _HAVE_CV2:
        raise RuntimeError("crop_resize needs opencv (cv2); install it or use your repo's own resize.")
    img = center_crop(img, crop)
    if (img.shape[1], img.shape[0]) != (int(out), int(out)):
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
        img = cv2.resize(img, (int(out), int(out)), interpolation=interp)
    return img


def normalize_robust(img, low_pct=P_LOW, high_pct=P_HIGH, eps=1e-6):
    """PER-VISIT robust min-max -> [0, 1]. Clip to FOREGROUND (pixels > 0) [p_low, p_high] then scale.

    Each image is normalized independently (no per-eye/cross-visit stats), so inter-visit exposure/gain
    differences are equalized. Background (0) stays 0 (it maps below p_low and is clipped to 0)."""
    img = np.asarray(img, dtype=np.float32)
    fg = img[img > 0]
    if fg.size == 0:
        fg = img.ravel()
    p_lo, p_hi = np.percentile(fg, [low_pct, high_pct])
    denom = (p_hi - p_lo) if (p_hi - p_lo) > eps else 1.0
    return np.clip((img - p_lo) / denom, 0.0, 1.0).astype(np.float32)


def normalize_robust_pm1(img, **kw):
    """Same robust normalization, mapped to [-1, 1] for models that expect that input range."""
    return normalize_robust(img, **kw) * 2.0 - 1.0


def normalize_minmax(img, eps=1e-6):
    """PLAIN PER-VISIT min-max -> [0, 1]: (img - min) / (max - min) on the FULL image (incl. the 0
    registration frame). Each visit scaled by its own raw min/max. This is the canonical paper
    normalization (chosen 2026-06-27). NB on this cohort FAF min=0 (black frame) and max=255 (saturation)
    in essentially every visit, so this is ~ img/255 and does NOT equalize inter-visit brightness
    (use normalize_robust for that); kept because it is the simplest, most standard choice to report."""
    img = np.asarray(img, dtype=np.float32)
    mn, mx = float(img.min()), float(img.max())
    denom = (mx - mn) if (mx - mn) > eps else 1.0
    return np.clip((img - mn) / denom, 0.0, 1.0).astype(np.float32)


def normalize_minmax_pm1(img, **kw):
    """Plain per-visit min-max mapped to [-1, 1] for models that expect that input range."""
    return normalize_minmax(img, **kw) * 2.0 - 1.0


# --- CANONICAL normalization selector (the single flip-point shared by ALL methods) -----------------
# Every repo imports `normalize` / `normalize_pm1` from here, so switching the paper's normalization is
# a ONE-LINE change. Currently: PLAIN per-visit min-max (user decision 2026-06-27). For robust, set
# these to normalize_robust / normalize_robust_pm1.
normalize = normalize_minmax
normalize_pm1 = normalize_minmax_pm1
