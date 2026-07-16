"""Shared classical temporal primitives for the FAF/GA interpolation/extrapolation floors.

Imported by BOTH:
  * run_baseline_interp.py  -- FAF-space (interpolate images, then external segmentor)
  * run_seg_interp.py       -- mask-space, segmentor-free (GAP-INR-comparable floor)

Keeping the linear / cubic-spline / growth-rate maths in ONE place means the two scripts
apply identical methods, so their interpolation/extrapolation numbers differ only by the
scoring path (segmentor-FAF vs direct-mask), never by the interpolation itself.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def interp_pixelwise(method, t_sup, arr_sup, t_h):
    """Reconstruct a per-pixel field at time t_h from support (t_sup, arr_sup), t_sup ASCENDING.
    Works for any per-pixel array -- FAF intensities OR 0/1 masks.
      linear            : line through the two BRACKETING support visits; linear EXTRAPOLATION from
                          the nearest TWO support visits when t_h is outside the support range. This
                          is the faithful ImageFlowNet baseline (their `_linear_interp` hardcodes the
                          last two support visits, image_arr[-2], image_arr[-1]); it does NOT use the
                          whole history.
      linear_regression : per-pixel least-squares line (deg-1 polyfit) through ALL support visits --
                          i.e. it DOES use the WHOLE patient history -- evaluated at t_h. This is the
                          "linear regression (all visits)" floor, distinct from `linear` above.
      cubic_spline      : cubic spline through ALL support visits (whole history), evaluated at t_h.
    """
    t_sup = np.asarray(t_sup, dtype=np.float64)
    if method == "linear":
        j = int(np.searchsorted(t_sup, t_h))
        if j == 0:
            i0, i1 = 0, 1
        elif j >= len(t_sup):
            i0, i1 = len(t_sup) - 2, len(t_sup) - 1
        else:
            i0, i1 = j - 1, j
        w = (t_h - t_sup[i0]) / (t_sup[i1] - t_sup[i0])
        return arr_sup[i0] + (arr_sup[i1] - arr_sup[i0]) * w
    elif method == "linear_regression":
        # per-pixel least-squares line through ALL support visits (whole history). np.polyfit fits one
        # deg-1 polynomial per COLUMN, so flattening pixels into columns vectorises the per-pixel fit.
        if len(t_sup) < 2:
            return np.asarray(arr_sup[0], dtype=np.float32)
        flat = arr_sup.reshape(len(arr_sup), -1).astype(np.float64)   # (n_visits, n_pixels)
        slope, intercept = np.polyfit(t_sup, flat, 1)                 # each (n_pixels,)
        pred = slope * float(t_h) + intercept
        return pred.reshape(arr_sup.shape[1:]).astype(np.float32)
    elif method == "cubic_spline":
        flat = arr_sup.reshape(len(arr_sup), -1)
        return CubicSpline(t_sup, flat)(t_h).reshape(arr_sup.shape[1:]).astype(np.float32)
    raise ValueError(method)


def fit_predict_area(t_sup, a_sup, t_h):
    """GA-area linear GROWTH-RATE: least-squares fit  area ~ a0 + b*t  over the support visits'
    OBSERVED GA areas, then predict at t_h -- interpolation for an interior held-out visit,
    extrapolation for a last-visit hold-out. Clamped to >= 0 (a GA area cannot be negative).
    Falls back to the single support area when < 2 support visits (can't fit a slope)."""
    t = np.asarray(t_sup, dtype=np.float64)
    a = np.asarray(a_sup, dtype=np.float64)
    if len(t) < 2:
        return float(a[0]) if len(a) else 0.0
    b, a0 = np.polyfit(t, a, 1)          # np.polyfit returns [slope, intercept] for deg 1
    return float(max(0.0, a0 + b * t_h))
