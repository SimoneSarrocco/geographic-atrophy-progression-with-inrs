"""Shared TRAINING/VALIDATION monitor panel — the same layout and colormaps for GAP-INR and for the
ImageFlowNet-family baselines, so per-run TensorBoard images are directly comparable by eye.

Canonical 1x4 panel for one (held-out / target) visit:

    [ GT FAF | Pred FAF | |Pred-GT| (magma, 0..1) | Pred-GA vs GT-GA (TP green / FP red / FN blue) ]

Title carries PSNR / SSIM / DICE. Inputs are 2D numpy arrays on the SAME grid (the 512 eval grid is
recommended); FAF in [0,1], masks binary {0,1}. Pure numpy + matplotlib (skimage only for PSNR/SSIM,
and only if metrics aren't passed in). Import by adding this directory to sys.path.

`log_panel(...)` renders + logs to TensorBoard (handles a raw SummaryWriter OR a PyTorch-Lightning
logger) and/or saves to disk, then closes the figure.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _as01(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:                      # (H,W,C) or grayscale-as-RGB -> single channel
        a = a.mean(axis=-1)
    if a.size and (a.min() < -1e-3 or a.max() > 1.0 + 1e-3):
        if a.min() < -1e-3:              # tolerate [-1,1]
            a = (a + 1.0) / 2.0
        elif a.max() > 1.5:              # tolerate [0,255]
            a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _asmask(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    return (a > 0.5).astype(np.uint8)


def _tpfpfn_overlay(faf, pred_mask, gt_mask):
    """RGB overlay of grayscale FAF with TP (green) / FP (red) / FN (blue) of pred vs GT GA."""
    rgb = np.repeat(faf[..., None], 3, axis=2).astype(np.float32)
    p, g = pred_mask > 0, gt_mask > 0
    rgb[p & g] = [0.0, 1.0, 0.0]
    rgb[p & ~g] = [1.0, 0.0, 0.0]
    rgb[~p & g] = [0.0, 0.4, 1.0]
    return np.clip(rgb, 0, 1)


def seg_stats(pred_mask, gt_mask):
    """GA-foreground DICE / precision / recall (matches GAP-INR + the comparison figure)."""
    p, g = pred_mask > 0, gt_mask > 0
    tp = float((p & g).sum()); fp = float((p & ~g).sum()); fn = float((~p & g).sum())
    dice = 1.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
    prec = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    rec = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    return dice, prec, rec


def make_monitor_panel(gt_faf, pred_faf, gt_mask, pred_mask, title="",
                       psnr=None, ssim=None, dice=None, mask_note=""):
    """Return the canonical 1x4 monitor figure. Computes PSNR/SSIM/DICE if not provided."""
    gt_faf, pred_faf = _as01(gt_faf), _as01(pred_faf)
    gt_mask, pred_mask = _asmask(gt_mask), _asmask(pred_mask)
    if dice is None:
        dice, _, _ = seg_stats(pred_mask, gt_mask)
    if (psnr is None or ssim is None):
        try:
            from skimage.metrics import peak_signal_noise_ratio as _psnr
            from skimage.metrics import structural_similarity as _ssim
            if psnr is None:
                psnr = float(_psnr(gt_faf, pred_faf, data_range=1.0))
            if ssim is None:
                ssim = float(_ssim(gt_faf, pred_faf, data_range=1.0))
        except Exception:
            pass
    diff = np.abs(pred_faf - gt_faf)

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(gt_faf, cmap="gray", vmin=0, vmax=1);   ax[0].set_title("GT FAF")
    ax[1].imshow(pred_faf, cmap="gray", vmin=0, vmax=1); ax[1].set_title("Pred FAF")
    ax[2].imshow(diff, cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("|Pred - GT|  MAE %.4f" % float(diff.mean()))
    ax[3].imshow(_tpfpfn_overlay(pred_faf, pred_mask, gt_mask))
    ax[3].set_title("Pred GA vs GT GA" + (("  (%s)" % mask_note) if mask_note else ""))
    ax[3].legend(handles=[Patch(color=[0, 1, 0], label="TP"), Patch(color=[1, 0, 0], label="FP"),
                          Patch(color=[0, 0.4, 1], label="FN")],
                 loc="lower right", fontsize=7, framealpha=0.6)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    _ps = "PSNR %.2f" % psnr if psnr is not None else "PSNR --"
    _ss = "SSIM %.3f" % ssim if ssim is not None else "SSIM --"
    fig.suptitle("%s  |  %s  %s  DICE %.3f" % (title, _ps, _ss, float(dice)), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def log_panel(sink, tag, step, gt_faf, pred_faf, gt_mask, pred_mask,
              title="", save_path=None, **metric_kw):
    """Render the canonical panel and log it to TensorBoard and/or disk, then close.

    `sink` may be a raw torch SummaryWriter (has .add_figure), a PyTorch-Lightning logger (has
    .experiment.add_figure), or None. `save_path` (optional) also writes a PNG to disk."""
    fig = make_monitor_panel(gt_faf, pred_faf, gt_mask, pred_mask, title=title, **metric_kw)
    try:
        if sink is not None:
            if hasattr(sink, "add_figure"):                         # SummaryWriter
                sink.add_figure(tag, fig, global_step=step)
            elif hasattr(sink, "experiment") and hasattr(sink.experiment, "add_figure"):  # PL logger
                sink.experiment.add_figure(tag, fig, global_step=step)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=110)
    finally:
        plt.close(fig)
