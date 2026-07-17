"""Unified evaluation for the ImageFlowNet-family baselines (GAP-INR comparison).

Why this exists: the native harness (train_2pt_all.py) computes segmentation DICE as
``dice(segmentor(pred_image), segmentor(GT_image))`` -- i.e. it segments the GROUND-TRUTH
IMAGE as the reference, not the real GA mask. For the GAP-INR paper we want the fair,
correct protocol the user chose:

    DICE = dice(segmentor(predicted_FAF), REAL GA mask)          [shared frozen segmentor]

computed for every forecasting pair on the TEST split, under TWO stratifications:
  (1) by visit position (GAP-INR framing, matching summarize_eval):
    * INTERPOLATION  -- the predicted (target) visit is NOT the eye's last visit
    * EXTRAPOLATION  -- the predicted (target) visit IS the eye's last visit
  (2) by GA-growth magnitude (ImageFlowNet Table 1 framing), from the GT masks only:
    * MINOR_GROWTH   -- source->target GT-mask change (1 - DSC) < 0.1 (copy-forward nearly solves it)
    * MAJOR_GROWTH   -- (1 - DSC) > 0.1 (real change; where a forecaster earns its keep)
Metrics per bucket: DICE (real-mask), Hausdorff distance HD (EVAL_DIM-grid px), PSNR, SSIM, LPIPS, and the
GAP-INR lesion-area MAE (mm^2). A COPY-FORWARD reference (predict = source) is scored the same way.

It rebuilds the same config/model as train_2pt_all.py (so checkpoints resolve), loads the
chosen best checkpoint, applies the shared segmentor, and writes
``<run>/leave_one_out_summary_test_<best_type>.csv`` plus per-pair rows.

All metrics are scored on the NATIVE 512 grid -- no upsampling of the prediction. The loader
center-crops 620 from the native 768 (which preserves GA; a direct 512 crop would clip GA in
~5/133 visits) then resizes to target_dim=512. The forecaster output is therefore natively 512;
the GT FAF and GT GA mask are built the SAME way (620-crop then resize to 512, FAF bicubic /
mask nearest), and the shared frozen segmentor (trained at 512) is applied at 512. So pred and
GT are both native 512 and DICE / PSNR / SSIM / lesion-area-MAE are all computed on 512.
Lesion area (mm^2) = (#GA px on the 512 grid) * RAW per-visit ScaleXSlo*ScaleYSlo * (620/512)^2:
ScaleXSlo/ScaleYSlo are mm/pixel at NATIVE resolution, so the 620->512 resize is corrected by
(crop_size/EVAL_DIM)^2 to recover physically-correct native-pitch area (same factor for pred+GT).
A trivial COPY-FORWARD reference (predict = the source visit unchanged) is also scored.

Usage (mirror the training args so the run dir / checkpoint resolve):
    python eval_faf_ga.py --dataset-name retina_faf_ga --target-dim '(512,512)' \
        --model ImageFlowNetODE --segmentor-ckpt '$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty' \
        --run-count 1 [--best-type seg_dice|pred_psnr]
"""
import argparse
import ast
import csv
import os
import sys

import numpy as np
import torch

# eval_spec / common_preproc live at the root of this directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
try:
    import dump_io          # optional: cross-model prediction dumps, not part of this release
except Exception:
    dump_io = None
import eval_spec as _spec   # canonical split (single source of truth) for parity assert

# Map the ImageFlowNet-family class names to the canonical comparison method keys.
_DUMP_METHOD = {'ImageFlowNetODE': 'imageflownet_ode', 'ImageFlowNetSDE': 'imageflownet_sde',
                'T_UNet': 't_unet', 'I2SBUNet': 'i2sb'}

from datasets.retina_faf_ga import CROP_SIZE, RetinaFafGaDataset, _load_raw, normalize_image

EVAL_DIM = 512  # GENERATION grid = model's native training res; OVERRIDDEN from config.target_dim[0].
METRIC_DIM = 512  # SCORING grid; set from --metric-resize. Render @EVAL_DIM (512), then resize pred+GT
                  # to METRIC_DIM (e.g. 256) for the metrics ONLY -- so a 512-trained forecaster is
                  # scored at 256 exactly like GAP-INR's metric_resize (generate@512 -> score@256).
                  # Dumps/figures stay native EVAL_DIM. METRIC_DIM==EVAL_DIM -> no resize (default).
_CROP = CROP_SIZE  # crop before resize; OVERRIDDEN at runtime from config.crop_size (768 = no crop).
# ALL metrics scored on the NATIVE model grid: the loader center-crops 620
                # (preserving GA) then resizes to target_dim=512, so the prediction, the GT FAF,
                # and the GT GA mask are all native 512 -- no upsampling of the prediction.
from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import dice_coeff, hausdorff, psnr, ssim
from utils.parse import parse_settings
from utils.seed import seed_everything

# model classes (same registry as the harness)
from nn.imageflownet_ode import ImageFlowNetODE          # noqa: F401
from nn.imageflownet_sde import ImageFlowNetSDE          # noqa: F401
from nn.unet_t_emb import T_UNet                          # noqa: F401
from nn.unet_i2sb import I2SBUNet                         # noqa: F401
from i2sb.diffusion import Diffusion
from i2sb.runner import make_beta_schedule


def _to_tensor(faf_norm, device):
    return torch.from_numpy(faf_norm[None, None, ...]).float().to(device)


def _seg_mask(segmentor, faf_norm, device):
    '''Run the (512-trained) frozen segmentor on a native-512 FAF (in [-1, 1]); return
    a binary (512, 512) uint8 GA mask. Prediction and segmentor are both native 512, so
    there is NO resampling of the prediction anywhere.'''
    x = torch.from_numpy(faf_norm[None, None, ...]).float().to(device)
    m = (segmentor(x) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)
    return m


def _area_mm2(mask, sx, sy, crop_size=None):
    '''Lesion area (mm^2) on the native pitch. ScaleXSlo/ScaleYSlo are mm/pixel at NATIVE resolution;
    the 768->crop_size(620) crop keeps native pitch, but the crop_size->METRIC_DIM resize enlarges each
    pixel footprint, so we rescale the METRIC_DIM-grid GA pixel count back to native-pitch area by
    (crop_size/METRIC_DIM)^2. `mask` MUST already be on the METRIC_DIM scoring grid (see _mg).
    Applied identically to pred and GT.'''
    if crop_size is None:
        crop_size = _CROP
    px_area = float(sx) * float(sy) * (crop_size / METRIC_DIM) ** 2
    return float(np.sum(mask > 0.5) * px_area)


def _mg(a, seg):
    '''Resize a native EVAL_DIM array to the METRIC_DIM scoring grid (no-op if equal). FAF -> bicubic;
    mask -> nearest + re-binarise. Used to score a 512-generated prediction at 256.'''
    if METRIC_DIM == EVAL_DIM:
        return a
    import cv2
    interp = cv2.INTER_NEAREST if seg else cv2.INTER_CUBIC
    r = cv2.resize(a.astype(np.float32), (METRIC_DIM, METRIC_DIM), interpolation=interp)
    return (r > 0.5).astype(np.uint8) if seg else r


def _score(pred_faf01, gt_faf01, pred_mask, gt_mask, sx, sy, device):
    '''All 6 metrics on the METRIC_DIM grid: resize pred+GT (FAF & mask) then score. Returns a dict with
    Dice/HD/PSNR/SSIM/LPIPS + Pred_Area_mm2/GT_Area_mm2. mask inputs are native EVAL_DIM; resized here.'''
    pf, gf = _mg(pred_faf01, False), _mg(gt_faf01, False)
    pm, gm = _mg(pred_mask, True), _mg(gt_mask, True)
    return {'Dice': dice_coeff(label_pred=pm, label_true=gm),
            'HD': hausdorff(label_pred=pm, label_true=gm),
            'PSNR': psnr(pf, gf, max_value=1), 'SSIM': ssim(pf, gf, data_range=1),
            'LPIPS': _lpips(pf, gf, device),
            'Pred_Area_mm2': _area_mm2(pm, sx, sy), 'GT_Area_mm2': _area_mm2(gm, sx, sy)}


def _faf_to01(faf_norm):
    '''Map a FAF from [-1, 1] to [0, 1] and clip (for PSNR/SSIM with data_range=1).'''
    return np.clip((faf_norm + 1.0) / 2.0, 0.0, 1.0)


_LPIPS_NET = None
_LPIPS_DISABLED = False


def _lpips(pred01, gt01, device):
    '''LPIPS (AlexNet) perceptual distance between two HxW images in [0, 1] (lower is better).
    LPIPS expects (N, 3, H, W) in [-1, 1]; the single FAF channel is replicated to 3. The net is
    built once and cached. Same prediction/GT the PSNR/SSIM see (so all three are consistent).
    Returns None if lpips is unavailable (it needs the package and a one-off AlexNet weight
    download), so the rest of the metrics still report.'''
    global _LPIPS_NET, _LPIPS_DISABLED
    if _LPIPS_DISABLED:
        return None
    if _LPIPS_NET is None:
        try:
            import lpips as _lpips_pkg
            _LPIPS_NET = _lpips_pkg.LPIPS(net='alex', verbose=False).to(device).eval()
        except Exception as e:
            print('LPIPS unavailable (%s); reporting None for it.' % e)
            _LPIPS_DISABLED = True
            return None

    def _t(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None].to(device)  # (1,1,H,W) [0,1]
        return t.repeat(1, 3, 1, 1) * 2.0 - 1.0                                        # (1,3,H,W) [-1,1]

    with torch.no_grad():
        return float(_LPIPS_NET(_t(pred01), _t(gt01)).item())


def _predict(model, config, x_start, dt_weeks, max_t, device):
    """Predict the FAF at the target visit, dt_weeks after the source. Returns (1,1,H,W)."""
    if config.model == 'I2SBUNet':
        delta_norm = dt_weeks / max_t
        step_max = max(int(config.diffusion_interval * delta_norm), 2)
        steps = np.int16(np.linspace(0, step_max - 1, step_max)).tolist()
        _, pred = model.ddpm_sampling(x_start=x_start, steps=steps)
        return pred[:, -1, ...].to(device)
    t = torch.tensor([dt_weeks * config.t_multiplier], device=device).float()
    return model(x=x_start, t=t)


# Models whose flow field can be test-time-optimized via the standard model(x, t) forward +
# freeze_time_independent (section 5.5). I2SBUNet forecasts via ddpm_sampling (a different path)
# and is intentionally excluded from TTO.
_TTO_MODELS = ('ImageFlowNetODE', 'ImageFlowNetSDE', 'T_UNet')


def _tto_predict_last(model, config, ckpt, faf_list, t_wk, device):
    """ImageFlowNet section-5.5 TEST-TIME OPTIMIZATION (no retraining of the network).

    Reload the already-trained checkpoint fresh for THIS eye (no cross-patient contamination),
    fine-tune ONLY the flow field f_theta (model.freeze_time_independent() keeps the encoder/decoder
    fixed) on random pairs drawn from the patient HISTORY {x1..x_{n-1}} -- the last visit is NEVER
    used (no GT leakage) -- then forecast the LAST visit from the FIRST with the fine-tuned model.
    Returns the predicted last-visit FAF as a native-512 numpy array in [-1, 1]. Requires n >= 3.
    """
    n = len(faf_list)
    x_list = [_to_tensor(f, device) for f in faf_list]          # each (1,1,H,W) in [-1,1]
    tt = torch.tensor(t_wk, device=device).float()

    model.load_weights(ckpt, device=device)                     # pristine trained weights for this eye
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=config.tto_lr)
    mse = torch.nn.MSELoss()
    with torch.enable_grad():                                   # override evaluate()'s @torch.no_grad
        for _ in range(config.tto_iters):
            # history only: indices in {0..n-2}; the last visit (n-1) is held out (NOTE: no cheating).
            i, j = sorted(np.random.choice(n - 1, size=2, replace=False))
            try:
                model.freeze_time_independent()                 # tune f_theta only
            except AttributeError:
                pass
            pred = model(x=x_list[i], t=((tt[j] - tt[i]).unsqueeze(0) * config.t_multiplier))
            loss = mse(pred, x_list[j])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred_last = model(x=x_list[0], t=((tt[-1] - tt[0]).unsqueeze(0) * config.t_multiplier))
    return pred_last.cpu().numpy()[0, 0]


def _agg(vals):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return (float('nan'), 0.0, 0.0, 0)
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0          # sample standard deviation
    se = float(sd / np.sqrt(a.size)) if a.size > 1 else 0.0   # standard error of the mean
    return (float(a.mean()), sd, se, int(a.size))


@torch.no_grad()
def evaluate(config):
    device = torch.device('cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    # Scoring grid + crop are config-driven so the SAME evaluator serves the 620->512 track and the
    # no-crop 768->256 ImageFlowNet-style track. EVAL_DIM/_CROP feed the area pitch (crop/EVAL_DIM)^2.
    global EVAL_DIM, METRIC_DIM, _CROP
    EVAL_DIM = int(config.target_dim[0])                       # GENERATION grid (model native)
    METRIC_DIM = int(config.get('metric_resize') or EVAL_DIM)  # SCORING grid (render@EVAL_DIM -> score here)
    _CROP = int(config.get('crop_size', CROP_SIZE))
    print('grid: generate@%d -> score@%d (crop %d); area pitch (crop/%d)^2' %
          (EVAL_DIM, METRIC_DIM, _CROP, METRIC_DIM))
    ds = RetinaFafGaDataset(target_dim=config.target_dim, crop_size=_CROP)
    test_idx = ds.predefined_split['test']
    max_t = ds.max_t
    config.t_multiplier = config.ode_max_t / max_t

    # ---- build model (mirror train_2pt_all.test) ----
    kwargs = {}
    if config.model == 'I2SBUNet':
        step_to_t = torch.linspace(1e-4, 1, config.diffusion_interval, device=device) * config.diffusion_interval
        betas = make_beta_schedule(n_timestep=config.diffusion_interval, linear_end=1 / config.diffusion_interval)
        betas = np.concatenate([betas[:config.diffusion_interval // 2], np.flip(betas[:config.diffusion_interval // 2])])
        kwargs = {'step_to_t': step_to_t, 'diffusion': Diffusion(betas, device)}
    model = globals()[config.model](device=device, num_filters=config.num_filters, depth=config.depth,
                                    ode_location=config.ode_location, in_channels=1, out_channels=1,
                                    image_size=config.target_dim[0],
                                    contrastive=config.coeff_contrastive + config.coeff_invariance > 0, **kwargs)
    model.to(device)
    ckpt = config.model_save_path.replace('.pty', '_best_%s.pty' % config.best_type)
    model.load_weights(ckpt, device=device)
    model.eval()
    print('Loaded forecaster:', ckpt)

    # ---- shared frozen segmentor ----
    import monai
    segmentor = torch.nn.Sequential(
        monai.networks.nets.DynUNet(spatial_dims=2, in_channels=1, out_channels=1,
                                    kernel_size=[5, 5, 5, 5], filters=[16, 32, 64, 128],
                                    strides=[1, 1, 1, 1], upsample_kernel_size=[1, 1, 1, 1]),
        torch.nn.Sigmoid()).to(device)
    segmentor.load_state_dict(torch.load(config.segmentor_ckpt, map_location=device))
    segmentor.eval()
    print('Loaded segmentor:', config.segmentor_ckpt)

    # ---- print test split for parity check with GAP-INR ----
    test_eyes = [ds.records_by_patient[i][0]['eye_id'] for i in test_idx]
    print('Split sizes  train=%d  val=%d  test=%d' % (
        len(ds.predefined_split['train']), len(ds.predefined_split['val']), len(test_idx)))
    print('TEST Eye_IDs (%d):' % len(test_eyes), test_eyes)
    # Parity guard: ImageFlowNet filters by CSV split (NOT the spec's os.path.exists), so assert the
    # scored test eyes are EXACTLY GAP-INR's canonical 6 -- a silent drift would corrupt the whole table.
    _spec.assert_split_parity(test_eyes, 'test', source='ImageFlowNet/eval_faf_ga')

    rows = []        # per-pair forecaster records
    cf_rows = []     # per-pair copy-forward (trivial: predict = source) records
    for i in test_idx:
        recs = ds.records_by_patient[i]
        n = len(recs)
        eye_id = recs[0]['eye_id']
        # Model input, GT FAF, and GT GA mask are ALL built the SAME way: 620-crop (preserves
        # GA) then resize to EVAL_DIM=512 (FAF bicubic, mask nearest). Prediction is natively
        # 512, so it is scored against native-512 GT with NO resampling of the prediction.
        assert int(config.target_dim[0]) == EVAL_DIM and ds.crop_size == _CROP, \
            'eval scores at the native model grid (target-dim) with crop_size=%d' % _CROP
        faf_in = [normalize_image(_load_raw(r['faf'], (EVAL_DIM, EVAL_DIM), ds.crop_size, is_mask=False))
                  for r in recs]
        msk_gt = [(_load_raw(r['mask'], (EVAL_DIM, EVAL_DIM), ds.crop_size, is_mask=True) > 128).astype(np.uint8)
                  for r in recs]
        t_wk = [r['t'] for r in recs]

        for a in range(n):
            for b in range(a + 1, n):
                is_extrap = int(b == n - 1)
                weeks = t_wk[b] - t_wk[a]
                sx_b, sy_b = recs[b]['sx'], recs[b]['sy']
                gt_mask = msk_gt[b]                          # native EVAL_DIM (dumps/growth)
                gt01 = _faf_to01(faf_in[b])                  # native EVAL_DIM (dumps)
                # GA "growth" magnitude = dissimilarity between the SOURCE and TARGET GT masks
                # (1 - DSC). ImageFlowNet's Table 1 stratifies forecasting by this: MINOR growth
                # (<0.1, where copy-forward already nearly solves it) vs MAJOR growth (>0.1, where a
                # forecaster must predict real change and the method earns its keep). Computed from
                # GT masks only, so the buckets are identical for every method (incl GAP-INR).
                growth = 1.0 - dice_coeff(label_pred=msk_gt[a], label_true=gt_mask)
                is_major = int(growth > 0.1)

                # ---- forecaster prediction (native 512) -> segment @512 (no resampling) ----
                x0 = _to_tensor(faf_in[a], device)
                pred = _predict(model, config, x0, weeks, max_t, device)
                pred512 = pred.cpu().numpy()[0, 0]
                pred_mask = _seg_mask(segmentor, pred512, device)   # native EVAL_DIM (dumps)
                sc = _score(_faf_to01(pred512), gt01, pred_mask, gt_mask, sx_b, sy_b, device)
                d, hd, p, s, lp = sc['Dice'], sc['HD'], sc['PSNR'], sc['SSIM'], sc['LPIPS']
                pred_area, gt_area = sc['Pred_Area_mm2'], sc['GT_Area_mm2']
                rows.append({'Patient_Eye': eye_id, 'Set': 'test_eval', 'Weeks': weeks,
                             'src_visit': recs[a]['visit'], 'tgt_visit': recs[b]['visit'],
                             'is_extrap': is_extrap, 'growth': growth, 'is_major': is_major,
                             'Dice': d, 'HD': hd, 'PSNR': p, 'SSIM': s, 'LPIPS': lp,
                             'GT_Area_mm2': gt_area, 'Pred_Area_mm2': pred_area,
                             'area_MAE_mm2': abs(pred_area - gt_area)})

                # ---- copy-forward baseline: carry the SOURCE visit's GROUND-TRUTH GA mask forward,
                # scored vs the target GT mask. NOT segmentor(source image) -- copy-forward must be a
                # clean copied GT mask (no segmentor false positives). Image metrics still use src FAF.
                src512 = faf_in[a]
                cf_mask = msk_gt[a]
                scf = _score(_faf_to01(src512), gt01, cf_mask, gt_mask, sx_b, sy_b, device)
                cf_area = scf['Pred_Area_mm2']
                cf_rows.append({'Patient_Eye': eye_id, 'Set': 'copyforward', 'Weeks': weeks,
                                'src_visit': recs[a]['visit'], 'tgt_visit': recs[b]['visit'],
                                'is_extrap': is_extrap, 'growth': growth, 'is_major': is_major,
                                'Dice': scf['Dice'], 'HD': scf['HD'], 'PSNR': scf['PSNR'],
                                'SSIM': scf['SSIM'], 'LPIPS': scf['LPIPS'],
                                'GT_Area_mm2': gt_area, 'Pred_Area_mm2': cf_area,
                                'area_MAE_mm2': abs(cf_area - gt_area)})

                # ---- cross-model comparison dump (one chosen pair per eye/scenario) ----
                # matched = baseline-only (v1 -> last visit), apples-to-apples with GAP-INR --support_k 1.
                # full    = each method at its best; the pairwise forecaster uses its most-recent anchor
                #           (v_{n-1} -> last visit), the shortest/easiest horizon.
                dump_root = getattr(config, 'dump_root', None)
                scen = getattr(config, 'dump_scenario', None)
                want = (dump_io is not None and dump_root and (
                    (scen == 'matched' and a == 0 and b == n - 1) or
                    (scen == 'full' and a == n - 2 and b == n - 1)))
                if want:
                    dump_io.write_case(
                        dump_root, method=_DUMP_METHOD.get(config.model, config.model.lower()),
                        scenario=scen, eye_id=eye_id, tgt_visit='last',
                        pred_faf=_faf_to01(pred512), gt_faf=gt01, pred_mask=pred_mask, gt_mask=gt_mask,
                        src_faf=_faf_to01(faf_in[a]), src_visit=recs[a]['visit'], weeks=weeks,
                        dice=d, hd=hd, psnr=p, ssim=s, gt_area_mm2=gt_area, pred_area_mm2=pred_area,
                        growth=growth, is_major=is_major, is_extrap=is_extrap)
                    # copy-forward is method-independent: dump once (idempotent overwrite across models).
                    dump_io.write_case(
                        dump_root, method='copyforward', scenario=scen, eye_id=eye_id,
                        tgt_visit='last', pred_faf=_faf_to01(src512), gt_faf=gt01,
                        pred_mask=cf_mask, gt_mask=gt_mask, src_faf=_faf_to01(src512),
                        src_visit=recs[a]['visit'], weeks=weeks,
                        dice=scf['Dice'], hd=scf['HD'], psnr=scf['PSNR'], ssim=scf['SSIM'],
                        gt_area_mm2=gt_area, pred_area_mm2=cf_area,
                        growth=growth, is_major=is_major, is_extrap=is_extrap)

    # ---- aggregate interp / extrap / ALL ----
    def summarize(records):
        def bucket(name, sel):
            sub = [r for r in records if sel(r)]
            out = {'Set': name, 'n': len(sub)}
            for col, key in (('DICE_mean', 'Dice'), ('HD_mean', 'HD'), ('PSNR_mean', 'PSNR'),
                             ('SSIM_mean', 'SSIM'), ('LPIPS_mean', 'LPIPS'),
                             ('area_MAE_mm2', 'area_MAE_mm2')):
                mu, sd, se, _ = _agg([r[key] for r in sub])
                base = col[:-5] if col.endswith('_mean') else col   # 'DICE_mean'->'DICE'; 'area_MAE_mm2' stays
                out[col] = mu
                out[base + '_std'] = sd
                out[base + '_se'] = se
            return out
        # Two stratifications: (1) interp/extrap by visit position (GAP-INR framing); (2) minor/major
        # GA growth by GT-mask change (ImageFlowNet's Table 1 framing -- where forecasters matter).
        return [bucket('interpolation', lambda r: not r['is_extrap']),
                bucket('extrapolation', lambda r: r['is_extrap']),
                bucket('minor_growth', lambda r: not r['is_major']),
                bucket('major_growth', lambda r: r['is_major']),
                bucket('ALL', lambda r: True)]

    summary_rows = summarize(rows)
    cf_summary_rows = summarize(cf_rows)

    run_dir = os.path.dirname(config.model_save_path)
    # (a) per-pair lesion CSV (GAP-INR-compatible columns + extra book-keeping cols)
    pairs_csv = os.path.join(run_dir, 'pairs_test_%s.csv' % config.best_type)
    lesion_cols = ['Patient_Eye', 'Set', 'Weeks', 'GT_Area_mm2', 'Pred_Area_mm2', 'Dice', 'HD',
                   'PSNR', 'SSIM', 'LPIPS', 'area_MAE_mm2', 'growth', 'is_major', 'src_visit',
                   'tgt_visit', 'is_extrap']
    with open(pairs_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=lesion_cols)
        w.writeheader(); w.writerows(rows); w.writerows(cf_rows)

    # (b) leave_one_out_summary_test_<best_type>.csv: {interpolation, extrapolation, ALL} for
    # forecaster + copy-forward. The best_type suffix keeps the pred_psnr and seg_dice runs from
    # clobbering each other in the SAME run dir (only pairs_test_<best_type>.csv was suffixed
    # before). collect_loo_tables.py reads the suffixed name (with legacy-name fallback).
    summ_cols = ['Set', 'DICE_mean', 'DICE_std', 'HD_mean', 'HD_std', 'PSNR_mean', 'PSNR_std',
                 'SSIM_mean', 'SSIM_std', 'LPIPS_mean', 'LPIPS_std', 'area_MAE_mm2',
                 'area_MAE_mm2_std', 'n']
    summ_csv = os.path.join(run_dir, 'leave_one_out_summary_test_%s.csv' % config.best_type)
    with open(summ_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction='ignore')
        w.writeheader(); w.writerows(summary_rows)
    cf_summ_csv = os.path.join(run_dir, 'leave_one_out_summary_test_copyforward_%s.csv' % config.best_type)
    with open(cf_summ_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction='ignore')
        w.writeheader(); w.writerows(cf_summary_rows)

    def _show(tag, srows):
        print('\n==== %s (real-mask DICE, %d grid) | model=%s best=%s | seg=%s ====' %
              (tag, EVAL_DIM, config.model, config.best_type, os.path.basename(config.segmentor_ckpt)))
        for r in srows:
            print('  %-14s n=%-3d  DICE %.3f±%.3f  HD %.1f±%.1f  PSNR %.2f±%.2f  SSIM %.3f±%.3f  '
                  'LPIPS %.3f±%.3f  areaMAE %.4f±%.4f mm^2' % (
                r['Set'], r['n'],
                r['DICE_mean'], r['DICE_std'], r['HD_mean'], r['HD_std'],
                r['PSNR_mean'], r['PSNR_std'], r['SSIM_mean'], r['SSIM_std'],
                r['LPIPS_mean'], r['LPIPS_std'], r['area_MAE_mm2'], r['area_MAE_mm2_std']))

    _show('Unified eval', summary_rows)
    _show('COPY-FORWARD reference', cf_summary_rows)
    print('Saved:', summ_csv, '\n      ', cf_summ_csv, '\n      ', pairs_csv)

    # ---- ImageFlowNet section-5.5 TEST-TIME OPTIMIZATION (separate `_tto` row; NO retraining) ----
    # Per test eye with >=3 visits: fine-tune the flow field on the history {x1..x_{n-1}} (no GT
    # leakage) then forecast the LAST visit (extrapolation), scored with the SAME real-mask protocol
    # + shared 512 grid. Directly comparable to the no-TTO single-anchor row AND to GAP-INR-full
    # (both are test-time optimization that trade compute to exploit patient history). ODE/SDE/T_UNet
    # only (I2SB forecasts via ddpm_sampling). On by default (--tto-iters 100 = paper default).
    tto_summary_rows = None
    if config.tto_iters > 0:
        if config.model not in _TTO_MODELS:
            print('[TTO] skipped: %s forecasts via a non-flow-field path (no freeze_time_independent).'
                  % config.model)
        else:
            base_method = _DUMP_METHOD.get(config.model, config.model.lower())
            dump_root = getattr(config, 'dump_root', None)
            scen = getattr(config, 'dump_scenario', None)
            tto_rows, n_skip = [], 0
            for i in test_idx:
                recs = ds.records_by_patient[i]
                n = len(recs)
                eye_id = recs[0]['eye_id']
                if n < 3:                                   # need >=2 history visits to TTO + 1 to predict
                    n_skip += 1
                    continue
                faf_in = [normalize_image(_load_raw(r['faf'], (EVAL_DIM, EVAL_DIM), ds.crop_size, is_mask=False))
                          for r in recs]
                msk_gt = [(_load_raw(r['mask'], (EVAL_DIM, EVAL_DIM), ds.crop_size, is_mask=True) > 128).astype(np.uint8)
                          for r in recs]
                t_wk = [r['t'] for r in recs]
                b = n - 1                                   # forecast the LAST visit (extrapolation)
                weeks = t_wk[b] - t_wk[0]
                gt_mask = msk_gt[b]                          # native EVAL_DIM (dumps/growth)
                sx_b, sy_b = recs[b]['sx'], recs[b]['sy']
                gt01 = _faf_to01(faf_in[b])                  # native EVAL_DIM (dumps)
                growth = 1.0 - dice_coeff(label_pred=msk_gt[0], label_true=gt_mask)
                is_major = int(growth > 0.1)

                pred512 = _tto_predict_last(model, config, ckpt, faf_in, t_wk, device)
                pred_mask = _seg_mask(segmentor, pred512, device)   # native EVAL_DIM (dumps)
                sc = _score(_faf_to01(pred512), gt01, pred_mask, gt_mask, sx_b, sy_b, device)
                d, hd, p, s, lp = sc['Dice'], sc['HD'], sc['PSNR'], sc['SSIM'], sc['LPIPS']
                pred_area, gt_area = sc['Pred_Area_mm2'], sc['GT_Area_mm2']
                tto_rows.append({'Patient_Eye': eye_id, 'Set': 'tto', 'Weeks': weeks,
                                 'src_visit': recs[0]['visit'], 'tgt_visit': recs[b]['visit'],
                                 'is_extrap': 1, 'growth': growth, 'is_major': is_major,
                                 'Dice': d, 'HD': hd, 'PSNR': p, 'SSIM': s, 'LPIPS': lp,
                                 'GT_Area_mm2': gt_area, 'Pred_Area_mm2': pred_area,
                                 'area_MAE_mm2': abs(pred_area - gt_area)})

                if dump_io is not None and dump_root and scen:
                    dump_io.write_case(
                        dump_root, method='%s_tto' % base_method, scenario=scen,
                        eye_id=eye_id, tgt_visit='last',
                        pred_faf=_faf_to01(pred512), gt_faf=gt01, pred_mask=pred_mask, gt_mask=gt_mask,
                        src_faf=_faf_to01(faf_in[0]), src_visit=recs[0]['visit'], weeks=weeks,
                        dice=d, hd=hd, psnr=p, ssim=s, gt_area_mm2=gt_area, pred_area_mm2=pred_area,
                        growth=growth, is_major=is_major, is_extrap=1)

            # TTO left the model fine-tuned on the last eye -> restore the pristine trained weights.
            model.load_weights(ckpt, device=device); model.eval()

            tto_summary_rows = summarize(tto_rows)
            tto_csv = os.path.join(run_dir, 'leave_one_out_summary_test_tto_%s.csv' % config.best_type)
            with open(tto_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=summ_cols, extrasaction='ignore')
                w.writeheader(); w.writerows(tto_summary_rows)
            # Raw per-eye TTO rows (one last-visit forecast per eye) so SD/SE is recoverable for
            # scenario 2, mirroring pairs_test_<best_type>.csv for scenario 1. Without this the
            # only TTO output is the means file above and the spread is lost.
            tto_pairs_csv = os.path.join(run_dir, 'tto_pairs_test_%s.csv' % config.best_type)
            with open(tto_pairs_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=lesion_cols, extrasaction='ignore')
                w.writeheader(); w.writerows(tto_rows)
            _show('TTO (sec 5.5: history->last, %d iters @ lr %.0e)' % (config.tto_iters, config.tto_lr),
                  tto_summary_rows)
            print('[TTO] eyes used=%d  skipped(<3 visits)=%d  ->' % (len(tto_rows), n_skip), tto_csv)

    # ---- eval-time TensorBoard: every summary metric as a scalar (single step) for full provenance ----
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(run_dir, 'tb_eval_%s' % config.best_type)
        writer = SummaryWriter(tb_dir)
        writer.add_text('provenance',
                        'model=%s | best=%s | generate@%d -> score@%d | crop=%d | ckpt-seed=%d | '
                        'eval-seed=%d | seg=%s' % (
                            config.model, config.best_type, EVAL_DIM, METRIC_DIM, _CROP,
                            config.random_seed, config.get('eval_seed') or config.random_seed,
                            os.path.basename(config.segmentor_ckpt)))
        _METR = ['DICE_mean', 'HD_mean', 'PSNR_mean', 'SSIM_mean', 'LPIPS_mean', 'area_MAE_mm2']
        for scen_tag, srows in (('S1_forecaster', summary_rows), ('copyforward', cf_summary_rows),
                                ('S2_tto', tto_summary_rows or [])):
            for r in srows:
                for m in _METR:
                    if r.get(m) is not None:
                        writer.add_scalar('%s/%s/%s' % (scen_tag, r['Set'], m.replace('_mean', '')),
                                          float(r[m]), 0)
        writer.flush(); writer.close()
        print('TensorBoard (eval scalars):', tb_dir)
    except Exception as e:
        print('[warn] eval TensorBoard logging skipped:', e)

    return {'forecaster': summary_rows, 'copyforward': cf_summary_rows, 'tto': tto_summary_rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified eval for ImageFlowNet-family baselines.')
    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('--run-count', default=1, type=int)
    parser.add_argument('--best-type', default='seg_dice', choices=['seg_dice', 'pred_psnr'])
    parser.add_argument('--dataset-name', default='retina_faf_ga', type=str)
    parser.add_argument('--target-dim', default='(512, 512)', type=ast.literal_eval)
    parser.add_argument('--crop-size', default=620, type=int,
                        help="crop of native 768 before resize (must MATCH training): 620 default; "
                             "768 = no crop (256 track). Sets the area pitch (crop/target)^2.")
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retina_faf_ga_512_seed1.pty', type=str)
    parser.add_argument('--model', default='ImageFlowNetODE', type=str)
    parser.add_argument('--random-seed', default=1, type=int,
                        help="Seed that NAMES the checkpoint dir (…_seed_{random_seed}/). Use 1 to load "
                             "the 256-track seed_1 checkpoints.")
    parser.add_argument('--eval-seed', default=None, type=int,
                        help="RNG seed for the evaluation itself (decoupled from --random-seed). Set 1927 "
                             "to fix a shared eval seed across all methods; important for I2SB's sampler.")
    parser.add_argument('--metric-resize', default=None, type=int,
                        help="Score at this grid: RENDER at --target-dim (model native, e.g. 512) then "
                             "resize pred+GT to this before metrics ONLY (dumps stay native). Use 256 to "
                             "score a 512-trained forecaster on the S1/S2 256 grid, matching GAP-INR's "
                             "metric_resize. Omit -> score at the generation grid.")
    parser.add_argument('--ode-max-t', default=5.0, type=float)
    parser.add_argument('--ode-location', default='all_connections', type=str)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--num-filters', default=64, type=int)
    parser.add_argument('--diffusion-interval', default=100, type=int)
    parser.add_argument('--no-l2', action='store_true')
    parser.add_argument('--coeff-smoothness', default=0, type=float)
    parser.add_argument('--coeff-latent', default=0, type=float)
    parser.add_argument('--coeff-contrastive', default=0, type=float)
    parser.add_argument('--coeff-invariance', default=0, type=float)
    parser.add_argument('--dump-root', default=None, type=str,
                        help='if set, write cross-model comparison .npz dumps under this root '
                             '(consumed by models/comparison/make_comparison_figure.py)')
    parser.add_argument('--dump-scenario', default=None, choices=['matched', 'full'],
                        help="matched: v1->last visit (baseline-only, fair vs GAP-INR --support_k 1); "
                             "full: v_{n-1}->last visit (pairwise model's best anchor)")
    parser.add_argument('--tto-iters', default=100, type=int,
                        help="ImageFlowNet section-5.5 test-time optimization: N flow-field fine-tune "
                             "steps on each eye's history {x1..x_{n-1}} (no retraining), then forecast "
                             "the last visit. Default 100 = the original paper's "
                             "test_time_optimization.py --opt-iters. 0 = off. Writes a separate "
                             "`<method>_tto` dump row + leave_one_out_summary_test_tto.csv + raw "
                             "tto_pairs_test_<best_type>.csv. ODE/SDE/T_UNet only.")
    parser.add_argument('--tto-lr', default=1e-4, type=float,
                        help="learning rate for the section-5.5 test-time flow-field fine-tuning")
    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, log_settings=False, run_count=config.run_count)
    # --random-seed names the checkpoint dir (…_seed_{random_seed}/); --eval-seed (if given) seeds the
    # evaluation RNG independently, so a seed_1-trained checkpoint can be evaluated with a FIXED, shared
    # eval seed (e.g. 1927) across every method -- matters for the stochastic I2SB sampler.
    eff_eval_seed = config.eval_seed if config.get('eval_seed') is not None else config.random_seed
    config.eval_seed = eff_eval_seed
    seed_everything(eff_eval_seed)
    print('checkpoint-seed=%d | eval RNG seed=%d' % (config.random_seed, eff_eval_seed))
    evaluate(config)
