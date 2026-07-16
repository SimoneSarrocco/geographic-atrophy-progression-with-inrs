import math
import json
from types import SimpleNamespace
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb as wd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import hausdorff_distance

# ---- LPIPS perceptual metric (lazy, optional) -------------------------------------------------
# LPIPS is a learned perceptual distance (lower = more similar) -- complements PSNR/SSIM, which the
# user notes are sensitive to the inter-visit brightness shifts in FAF-GA FAF. The AlexNet weights ship
# with the lpips package (loads fully offline). Built once on CPU; if lpips is unavailable it disables
# itself and returns None so the metric is simply absent (eval never crashes).
_LPIPS_MODEL = None
_LPIPS_DISABLED = False


def _lpips_score(pred01, ref01):
    """LPIPS(AlexNet) between two 2D [0,1] grayscale arrays (replicated to 3 channels, mapped to
    [-1,1]). Returns a float distance, or None if lpips can't be loaded."""
    global _LPIPS_MODEL, _LPIPS_DISABLED
    if _LPIPS_DISABLED:
        return None
    try:
        if _LPIPS_MODEL is None:
            import lpips as _lpips_pkg
            _LPIPS_MODEL = _lpips_pkg.LPIPS(net='alex', verbose=False).eval()
            for _p in _LPIPS_MODEL.parameters():
                _p.requires_grad_(False)

        def _to_t(a):
            t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).clamp(0.0, 1.0)
            t = t * 2.0 - 1.0                          # [0,1] -> [-1,1]
            return t[None, None].repeat(1, 3, 1, 1)    # (1, 3, H, W)

        with torch.no_grad():
            d = _LPIPS_MODEL(_to_t(pred01), _to_t(ref01))
        return float(d.reshape(-1)[0].item())
    except Exception as e:
        print(f"[LPIPS] disabled ({e}); skipping the perceptual metric.")
        _LPIPS_DISABLED = True
        return None


def _resize_2d(a, size, seg=False):
    """Resize a 2D array to (H, W). Segmentation masks use nearest-neighbour (label-preserving);
    continuous images use area interpolation (correct for downscaling). Used to bring a prediction
    generated at the checkpoint's native grid down to a fixed METRIC grid (e.g. 512 -> 256) so the
    scores are computed on the same resolution as another model (ImageFlowNet: crop620 -> 256)."""
    th, tw = int(size[0]), int(size[1])
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    mode = 'nearest' if seg else 'area'
    out = torch.nn.functional.interpolate(t, size=(th, tw), mode=mode)
    return out[0, 0].numpy()
import scipy.ndimage as ndi
from PIL import Image

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def dict_to_simplenamespace(d):
    """ Recursively converts dictionary to SimpleNamespace. """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_simplenamespace(value)
        return SimpleNamespace(**d)
    else:
        return d




class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.tf_weight = args['optimizer']['tf_weight']
        self.n_classes = args['inr_decoder']['out_dim'][-1]-1  # number of classes for segmentation excluding background
        self.sr_dims = sum(args['inr_decoder']['out_dim'][:-1])
        self.n_seg_channels = args['inr_decoder']['out_dim'][-1]
        self.has_seg = self.n_seg_channels > 0
        # Default reconstruction-loss weight (config-driven). Counterbalances the
        # CrossEntropy segmentation loss, which is otherwise ~10-50x larger than the
        # MSE on [0,1]-normalized intensities and dominates training.
        self.sr_weight = args['optimizer'].get('sr_weight', 1.0)
        # Relative weight of the soft-Dice term inside the segmentation loss (NISF uses Dice + BCE;
        # here it is Dice + weighted CrossEntropy). Dice is robust to the strong class imbalance of
        # small lesions (e.g. GA). 0.0 -> pure CrossEntropy (original behaviour).
        self.seg_dice_weight = args['optimizer'].get('seg_dice_weight', 0.0)
        # Tversky option for the segmentation loss. Tversky index = TP / (TP + alpha*FP + beta*FN);
        # loss = 1 - TI. alpha>beta penalises FALSE POSITIVES more strongly, suppressing spurious GA
        # far from the real lesion. When active it REPLACES the soft-Dice term (Dice is the special
        # case alpha=beta=0.5).
        tv = args['optimizer'].get('seg_tversky', {}) or {}
        self.tversky_active = bool(tv.get('activate', False))
        self.tversky_alpha = float(tv.get('alpha', 0.7))    # FP penalty weight
        self.tversky_beta = float(tv.get('beta', 0.3))      # FN penalty weight
        self.tversky_weight = float(tv.get('weight', 1.0))  # weight of the Tversky term added to CE
        # Soft (option B) consensus targets: when mask_grader_mode == 'soft' the seg target is a
        # per-pixel probability (mean of graders) rather than a hard class index. Only the binary
        # case (n_seg_channels == 2) is supported as soft; otherwise we fall back to hard labels.
        self.soft_seg = (args['dataset'].get('mask_grader_mode') == 'soft') and (self.n_seg_channels == 2)

        self.criterion_sr = nn.MSELoss() if args['optimizer']['loss_metric'] == 'mse' else nn.L1Loss()
        if self.has_seg:
            self.ce_weights = torch.tensor(args['dataset']['class_weights'], dtype=torch.float32, device=args['device']) if args['dataset'].get('class_weights') is not None else None
            self.criterion_seg = nn.CrossEntropyLoss(weight=self.ce_weights)
        else:
            self.ce_weights = None
            self.criterion_seg = None

    def _target_prob(self, target, seg_logits):
        """Build a per-coordinate class-probability target (N, C) from the raw seg target column.
        Hard labels -> one-hot; soft labels (mask_grader_mode 'soft', binary) -> [1-t, t]."""
        if self.soft_seg:
            t = target[..., -1].clamp(0.0, 1.0)                       # (N,) GA probability
            return torch.stack([1.0 - t, t], dim=-1).to(seg_logits.dtype)  # (N, 2)
        target_idx = target[..., -1].to(torch.int64)
        return F.one_hot(target_idx, num_classes=self.n_seg_channels).to(seg_logits.dtype)

    def _soft_dice_loss(self, seg_logits, target_prob):
        """Class-weighted soft Dice over the flat set of sampled coordinates (the batch acts as the
        pixel population). target_prob is a (N, C) probability target (hard one-hot or soft)."""
        probs = torch.softmax(seg_logits, dim=-1)                                  # (N, C)
        inter = (probs * target_prob).sum(dim=0)                                   # (C,)
        denom = probs.sum(dim=0) + target_prob.sum(dim=0)                          # (C,)
        dice = (2.0 * inter + 1.0) / (denom + 1.0)                                 # (C,) Laplace-smoothed
        dice_loss = 1.0 - dice                                                     # (C,)
        if self.ce_weights is not None:
            w = self.ce_weights / self.ce_weights.sum()
            return (dice_loss * w).sum()
        return dice_loss.mean()

    def _tversky_loss(self, seg_logits, target_prob):
        """Soft Tversky loss over the flat set of sampled coordinates (the batch acts as the pixel
        population). Per class: TI = (TP + s) / (TP + alpha*FP + beta*FN + s), loss = 1 - TI, then
        class-weighted (same convention as _soft_dice_loss). alpha > beta makes false positives cost
        more than false negatives -> fewer spurious GA pixels far from the real lesion. target_prob is
        a (N, C) probability target (hard one-hot or soft)."""
        probs = torch.softmax(seg_logits, dim=-1)                                  # (N, C)
        tp = (probs * target_prob).sum(dim=0)                                      # (C,)
        fp = (probs * (1.0 - target_prob)).sum(dim=0)                              # (C,)
        fn = ((1.0 - probs) * target_prob).sum(dim=0)                              # (C,)
        ti = (tp + 1.0) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1.0)  # Laplace-smoothed
        tv_loss = 1.0 - ti                                                         # (C,)
        if self.ce_weights is not None:
            w = self.ce_weights / self.ce_weights.sum()
            return (tv_loss * w).sum()
        return tv_loss.mean()

    def _seg_ce(self, seg_logits, target, target_prob):
        """Cross-entropy term. Hard labels use nn.CrossEntropyLoss; soft labels use a weighted
        soft cross-entropy -sum_c w_c t_c log softmax(logits)_c (mean over coordinates)."""
        if self.soft_seg:
            logp = F.log_softmax(seg_logits, dim=-1)                               # (N, C)
            if self.ce_weights is not None:
                return -(self.ce_weights * target_prob * logp).sum(dim=-1).mean()
            return -(target_prob * logp).sum(dim=-1).mean()
        return self.criterion_seg(seg_logits, target[..., -1].to(torch.int64))

    def forward(self, output, target, sr_weight=None, seg_weight=1.0):
        # Components are stored UNWEIGHTED (raw) for interpretable logging; the
        # sr_weight / seg_weight factors are applied only once, in loss['total'].
        sr_weight = self.sr_weight if sr_weight is None else sr_weight
        loss = {'seg': torch.tensor(0.0),
                'sr': self.criterion_sr(output[..., :self.sr_dims], target[..., :self.sr_dims]),
                'trafo': torch.tensor(0.0),
                'total': 0.0}

        if seg_weight > 0 and self.has_seg:
            seg_logits = output[..., self.sr_dims:self.sr_dims + self.n_seg_channels]
            target_prob = self._target_prob(target, seg_logits)
            seg_loss = self._seg_ce(seg_logits, target, target_prob)
            if self.tversky_active:
                # Tversky replaces the Dice term (Dice == Tversky with alpha=beta=0.5).
                seg_loss = seg_loss + self.tversky_weight * self._tversky_loss(seg_logits, target_prob)
            elif self.seg_dice_weight > 0:
                seg_loss = seg_loss + self.seg_dice_weight * self._soft_dice_loss(seg_logits, target_prob)
            loss['seg'] = seg_loss

        loss['total'] = sr_weight * loss['sr'] + seg_weight * loss['seg']
        return loss


def compute_ncc(prediction, reference):
    mean_pred = np.mean(prediction)
    mean_ref = np.mean(reference)
    numerator = np.sum((prediction - mean_pred) * (reference - mean_ref))
    denominator = np.sqrt(np.sum((prediction - mean_pred) ** 2) * np.sum((reference - mean_ref) ** 2))
    ncc = numerator / denominator
    return ncc.astype(np.float64)


def embed2affine(embed):
    R = euler2rot(embed[..., :3])
    t = embed[..., 3:6]
    if embed.shape[-1] >= 9:  # add 3D scaling
        S = torch.diag_embed(1.0 + embed[..., 6:9])
        R = torch.matmul(R, S)
    if embed.shape[-1] == 12:  # add 3D shear
        S_x = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_y = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_z = torch.diag_embed(torch.ones_like(embed[..., 9:12]))
        S_x[..., 0, 1] = embed[..., 10]  # y
        S_x[..., 0, 2] = embed[..., 11]  # z
        S_y[..., 1, 0] = embed[..., 9]   # x
        S_y[..., 1, 2] = embed[..., 11]  # z
        S_z[..., 2, 0] = embed[..., 9]   # x
        S_z[..., 2, 1] = embed[..., 10]  # y
        R = torch.matmul(R, S_x)
        R = torch.matmul(R, S_y)
        R = torch.matmul(R, S_z)
    return R, t


def embed2affine2d(embed):
    """
    2D version of the embedding to affine transformation.
    embed: (N, 3) --> [theta, tx, ty]
    """
    # 1. Rotation angle (theta)
    theta = embed[..., 0]
    
    # 2. Translation vector (tx, ty)
    t = embed[..., 1:3]
    
    # 3. Construct 2x2 Rotation Matrix
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    # R = [[cos(theta), -sin(theta)],
    #      [sin(theta),  cos(theta)]]
    row1 = torch.stack([cos_t, -sin_t], dim=-1)
    row2 = torch.stack([sin_t,  cos_t], dim=-1)
    R = torch.stack([row1, row2], dim=-2) 
    
    return R, t 


def euler2rot(theta):
    c1 = torch.cos(theta[..., 0])
    s1 = torch.sin(theta[..., 0])
    c2 = torch.cos(theta[..., 1])
    s2 = torch.sin(theta[..., 1])
    c3 = torch.cos(theta[..., 2])
    s3 = torch.sin(theta[..., 2])
    r11 = c1*c3 - c2*s1*s3
    r12 = -c1*s3 - c2*c3*s1
    r13 = s1*s2
    r21 = c3*s1 + c1*c2*s3
    r22 = c1*c2*c3 - s1*s3
    r23 = -c1*s2
    r31 = s2*s3
    r32 = c3*s2
    r33 = c2
    R = torch.stack([r11, r12, r13, r21, r22, r23, r31, r32, r33], dim=-1)
    R = R.view(R.shape[:-1] + (3, 3))
    return R


def harmonize_labels(subject_seg, dataset):
    # 2D FAF / geographic-atrophy data uses simple {0, 1} labels; no label harmonization is needed.
    return subject_seg


def to_device(x, device='cuda'):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(t) for t in x)
    elif isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    else:
        return x  # Leave non-tensor data as is
    

def normalize_condition(args, condition_key, condition_values, cond_scale=None):
    c_scale = args['atlas_gen']['cond_scale'] if cond_scale is None else cond_scale
    c_min = args['dataset']['constraints'][condition_key]['min']
    c_max = args['dataset']['constraints'][condition_key]['max']
    # Clamp out-of-range values to the observed [min,max] so atlas grids stay in-distribution
    # (boundary appearance) instead of saturating the SIREN. No-op when in range. Skipped when
    # extrapolate_beyond_range is set, so atlas/condition grids may progress past the horizon
    # (consistent with the novel-visit extrapolation path).
    if not args['dataset'].get('extrapolate_beyond_range', False):
        condition_values = np.clip(condition_values, c_min, c_max)
    cv = 2 * ((condition_values - c_min) / (c_max - c_min) - 0.5)
    cv = cv * c_scale
    return cv


def generate_combinations(args_data, conditions, keys=None, idx=0, current=None, results=None):
    if conditions is None: 
        return [[]]
    if keys is None:
        keys = list(conditions.keys())
    if current is None:
        current = []
    if results is None:
        results = []

    key = keys[idx]
    values = conditions[key]['values']

    for value in values:
        if not conditions[key]['normed_values']:
            value = normalize_condition(args_data, key, value)
        else:
            value = value * args_data['atlas_gen']['cond_scale']
        next_current = current + [value]
        if idx == len(keys) - 1:
            results.append(next_current)
        else:
            generate_combinations(args_data, conditions, keys, idx + 1, next_current, results)

    return results


def generate_world_grid(args, normed=True, device='cpu'):
    world_bbox = args['dataset']['world_bbox']
    spacing = args['atlas_gen']['spacing']
    sampling_bbox = args['dataset'].get('sampling_bbox')
    
    if len(world_bbox) == 3:
        x_min, y_min, z_min = 0, 0, 0
        w_box, h_box, d_box = world_bbox[0], world_bbox[1], world_bbox[2]
        
        if sampling_bbox is not None:
            if len(sampling_bbox) == 3:
                d_box, h_box, w_box = sampling_bbox
                z_min = (world_bbox[2] - d_box) // 2
                y_min = (world_bbox[1] - h_box) // 2
                x_min = (world_bbox[0] - w_box) // 2
            elif len(sampling_bbox) == 6:
                x_min, y_min, z_min, x_max, y_max, z_max = sampling_bbox
                w_box = x_max - x_min + 1
                h_box = y_max - y_min + 1
                d_box = z_max - z_min + 1
                
        x_max = x_min + w_box - 1
        y_max = y_min + h_box - 1
        z_max = z_min + d_box - 1
        
        x = torch.arange(x_min, x_max + 1, spacing[0], device=device)
        y = torch.arange(y_min, y_max + 1, spacing[1], device=device)
        z = torch.arange(z_min, z_max + 1, spacing[2], device=device)
        
        if normed:
            x = 2.0 * (x - x_min) / w_box - 1.0
            y = 2.0 * (y - y_min) / h_box - 1.0
            z = 2.0 * (z - z_min) / d_box - 1.0
            
        grid = torch.meshgrid(x, y, z, indexing='ij')
        grid_shape = list(grid[0].shape)
        grid_coords = torch.stack(grid, dim=-1).reshape(-1, 3)
        # affine = torch.diag(torch.tensor([spacing[0], spacing[1], spacing[2], 1.0], device=device))
    elif len(world_bbox) == 2:
        # Native 2D support - return (N, 2) coordinates
        x_min, y_min = 0, 0
        w_box, h_box = world_bbox[0], world_bbox[1]
        
        if sampling_bbox is not None:
            if len(sampling_bbox) == 2:
                w_box, h_box = sampling_bbox
                x_min = (world_bbox[0] - w_box) // 2
                y_min = (world_bbox[1] - h_box) // 2
            elif len(sampling_bbox) == 4:
                x_min, y_min, x_max, y_max = sampling_bbox
                w_box = x_max - x_min + 1
                h_box = y_max - y_min + 1
                
        x_max = x_min + w_box - 1
        y_max = y_min + h_box - 1
        
        x = torch.arange(x_min, x_max + 1, spacing[0], device=device)  # columns (width)
        y = torch.arange(y_min, y_max + 1, spacing[1], device=device)  # rows (height)

        if normed:
            x = 2.0 * (x - x_min) / w_box - 1.0
            y = 2.0 * (y - y_min) / h_box - 1.0

        # Build a ROW-MAJOR (row, col) grid: grid_shape = (H, W) so the reconstruction reshapes to
        # (H, W) and lines up with GT arrays indexed [row, col]. Coordinates are emitted in
        # (x=col, y=row) order to match training (load_coords_and_values) and PyTorch grid_sample.
        # This is correct for NON-SQUARE images; the previous (meshgrid(x,y) + stack([y,x])) form
        # only happened to cancel out when H == W.
        Y, X = torch.meshgrid(y, x, indexing='ij')   # each of shape (H, W)
        grid_shape = list(Y.shape)                    # [H, W]
        grid_coords = torch.stack([X, Y], dim=-1).reshape(-1, 2)  # (x, y) per coordinate
    else:
        raise ValueError(f"world_bbox must be 2 or 3 dims, got {len(world_bbox)}")
        
    return grid_coords, grid_shape  # , affine


def save_atlas(args, atlas, affine, temp_steps, condition_vectors, epoch, tb_writer=None):
    # atlas is of shape  (*spatial, num_modalities, n_conds, t)
    # where spatial is (x, y, z) or (x, y)
    shape = atlas.shape
    num_spatial = len(args['dataset']['world_bbox'])
    num_mods = shape[num_spatial]
    n_conds = shape[num_spatial + 1]
    t = shape[num_spatial + 2]
    is_2d = (num_spatial == 2)
    
    if isinstance(atlas, torch.Tensor):
        try:
            atlas = atlas.detach().cpu().numpy()
        except:
            atlas = atlas.numpy()
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    mod_labels = args['dataset']['modalities']
    if args['save_certainty_maps']:
        seg_labels = [f"CertaintyMaps/{label}" for label in args['dataset']['label_names']]
        mod_labels = mod_labels + seg_labels
    for c in range(n_conds):
        for i in range(len(mod_labels)):
            is_certainty = "CertaintyMaps" in mod_labels[i]
            # save each temporal frame individually
            ext = '.nii.gz' if is_certainty else '.bmp'
            for t_idx in range(t):
                filename = f'{mod_labels[i]}_ga={temp_steps[t_idx]}_cond={c}_ep={epoch}{ext}'
                save_img(atlas[..., i, c, t_idx],
                         output_path=args['output_dir'], filename=filename)
    
    # Print intensity ranges for debugging "black image" issues
    for i, mod in enumerate(mod_labels):
        mod_data = atlas[..., i, :, :]
        print(f"[Atlas Range] {mod}: min={mod_data.min():.4f}, max={mod_data.max():.4f}")
    
    print('Atlas saved to {}'.format(args['output_dir']))

    # Log atlas images to TensorBoard
    if tb_writer is not None:
        for c in range(n_conds):
            for i in range(len(mod_labels)):
                for t_idx in range(t):
                    img_data = atlas[..., i, c, t_idx]
                    # For 3D: take middle slice; for 2D: use as-is
                    if img_data.ndim == 3:
                        mid = img_data.shape[2] // 2
                        img_2d = img_data[:, :, mid]
                    else:
                        img_2d = img_data
                    # Normalize to [0, 1] for display
                    img_2d = img_2d.astype(np.float32)
                    # Sanitize NaNs and Infs
                    img_2d = np.nan_to_num(img_2d, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    vmin, vmax = img_2d.min(), img_2d.max()
                    if vmax > vmin:
                        img_2d = (img_2d - vmin) / (vmax - vmin)
                    else:
                        img_2d = np.zeros_like(img_2d)
                    
                    # Fixed tag: atlas/{mod_label}/cond{c}_t{t_val}
                    tag = f'atlas/{mod_labels[i]}/cond{c}_t{temp_steps[t_idx]}'
                    tb_writer.add_image(tag, img_2d, epoch, dataformats='HW')


def typecheck_img(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # if isinstance(affine, torch.Tensor):
    #     affine = affine.detach().cpu().numpy()
    if img.dtype == 'int64' or img.dtype == 'int32':
        img = img.astype(np.int16)
    elif img.dtype == 'float64':
        img = img.astype(np.float32)
    return img  # , affine


def save_subject(args, img, sub_row_df=None, sub_name=None, epoch=0, split='train', visit_suffix=None, tb_writer=None, baseline_volume=None):
    """
    Save a subject to disk and log to Tensorboard, including longitudinal change maps if baseline provided.
    Args:
        baseline_volume: Optional reconstruction of the baseline visit for change map computation.
    """
    id_col = args['dataset'].get('id_column', 'subject_id')
    eye_id = None
    visit_id = None
    if sub_row_df is not None:
        eye_id = str(sub_row_df.get(id_col, 'unknown'))
        # Extract visit identifier (supports FAF-GA Visit_Number/Visit_ID and Visit)
        if visit_suffix is not None:
            visit_id = visit_suffix
        elif 'Visit_Number' in sub_row_df:
            visit_id = f"V{sub_row_df['Visit_Number']}"
        elif 'Visit_ID' in sub_row_df:
            visit_id = f"VID{sub_row_df['Visit_ID']}"
        elif 'Visit' in sub_row_df:
            visit_id = f"V{sub_row_df['Visit']}"
        else:
            visit_id = 'V0'
        sub_name = f"{eye_id}_{visit_id}"
    elif sub_name is None:
        print("No subject name provided, assigning random subject name.")
        eye_id = 'subject_' + str(np.random.randint(100000))
        visit_id = visit_suffix if visit_suffix is not None else 'V0'
        sub_name = eye_id
    elif visit_suffix is not None:
        # Use sub_name as eye_id and visit_suffix as visit_id
        eye_id = sub_name
        visit_id = visit_suffix
        sub_name = f"{eye_id}_{visit_id}"

    img = typecheck_img(img)
    mytx = None
    modalities = args['dataset']['modalities']
    has_seg = args['inr_decoder']['out_dim'][-1] > 0
    for i, mod in enumerate(modalities): # for each modality (last modality is segmentation if has_seg)
        is_seg = has_seg and (i == len(modalities)-1)
        if is_seg:
            # inference output: [imgs(sr_dims), seg_hard(1), seg_soft(n_classes)]
            # Skip seg_hard, argmax only over seg_soft channels
            sr_dims = len(modalities) - 1
            n_seg_classes = img.shape[-1] - sr_dims - 1

            if n_seg_classes > 0:
                seg_soft = img[..., sr_dims + 1:sr_dims + 1 + n_seg_classes]  # skip seg_hard channel, explicit bound
                img_mod = np.argmax(seg_soft, axis=-1).astype(np.int16)
            else:
                img_mod = img[..., i].astype(np.int16)
        else:
            img_mod = img[..., i].astype(np.float32)
        # Structured path: {split}/{eye_id}/{mod}_{visit_id}_ep={epoch}
        filename = f'{split}/{eye_id}/{mod}_{visit_id}_ep={epoch}.nii.gz'
        # save_img(img_mod, affine, args['output_dir'], filename)

        # Log to TensorBoard with fixed tag for slider
        if tb_writer is not None:
            img_2d = img_mod
            if img_2d.ndim == 3:
                mid = img_2d.shape[2] // 2
                img_2d = img_2d[:, :, mid]
            # Clip to [0, 1] (data already in a consistent [0,1] space) rather than per-image
            # min-max, so GT/pred and successive visits share the same scale and off-intensity
            # reconstructions are not visually hidden.
            img_2d = np.clip(img_2d.astype(np.float32), 0.0, 1.0)
            # Structured tag: {split}/{eye_id}/{mod}/{visit_id}
            tag = f'{split}/{eye_id}/{mod}/{visit_id}'
            tb_writer.add_image(tag, img_2d, epoch, dataformats='HW')

            # --- Longitudinal Change/Difference Maps ---
            if baseline_volume is not None:
                sr_dims = len(modalities) - 1
                if is_seg:
                    # GA Change Map: Compare current argmax with baseline argmax
                    n_seg_classes = img.shape[-1] - sr_dims - 1

                    if n_seg_classes > 0:
                        seg_soft = img[..., sr_dims + 1:sr_dims + 1 + n_seg_classes]
                        pred_mask = np.argmax(seg_soft, axis=-1)
                        base_mask = (baseline_volume[..., -1] > 0) # simplified check
                        
                        base_n_seg_classes = baseline_volume.shape[-1] - sr_dims - 1
                        if base_n_seg_classes > 0:
                            base_mask = np.argmax(baseline_volume[..., sr_dims+1:sr_dims+1+base_n_seg_classes], axis=-1)
                    else:
                        pred_mask = (img_mod > 0).astype(np.uint8)
                        base_mask = (baseline_volume[..., i] > 0).astype(np.uint8)

                    # Build RGB Change Map
                    # Green [0,1,0]: Stable, Red [1,0,0]: Growth, Blue [0,0,1]: Resolved
                    change_map = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
                    stable = (pred_mask > 0) & (base_mask > 0)
                    growth = (pred_mask > 0) & (base_mask == 0)
                    lost   = (pred_mask == 0) & (base_mask > 0)
                    change_map[stable] = [0.2, 0.8, 0.4] # Green
                    change_map[growth] = [0.9, 0.1, 0.1] # Red
                    change_map[lost]   = [0.2, 0.4, 0.9] # Blue
                    
                    # Log as RGB image
                    tag_change = f'{split}/{eye_id}/change_map/{mod}/{visit_id}'
                    tb_writer.add_image(tag_change, change_map, epoch, dataformats='HWC')
                else:
                    # Intensity Difference Map
                    base_mod = baseline_volume[..., i]
                    # Normalize both to [0, 1] for fair comparison
                    p_min, p_max = img_mod.min(), img_mod.max()
                    b_min, b_max = base_mod.min(), base_mod.max()
                    p_norm = (img_mod - p_min) / (p_max - p_min + 1e-8)
                    b_norm = (base_mod - b_min) / (b_max - b_min + 1e-8)
                    
                    diff_map = (p_norm - b_norm) # [-1, 1]
                    # Log as heatmap (shifted to [0, 1])
                    diff_norm = (diff_map + 1.0) / 2.0
                    tag_diff = f'{split}/{eye_id}/difference/{mod}/{visit_id}'
                    tb_writer.add_image(tag_diff, diff_norm, epoch, dataformats='HW')


def save_img(img, output_path, filename):
    if img.dtype == 'int64' or img.dtype == 'int32':
        img = img.astype(np.int16)
    elif img.dtype == 'float64':
        img = img.astype(np.float32)
    full_path = os.path.join(output_path, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    # Save 2D images as BMP.
    img_2d = img[:, :, 0] if len(img.shape) == 3 else img
    # No transpose needed - data is already in (H, W) = (row, col) order
    # Normalize to 0-255 for BMP
    vmin, vmax = img_2d.min(), img_2d.max()
    if vmax > vmin:
        img_uint8 = ((img_2d - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        img_uint8 = np.zeros_like(img_2d, dtype=np.uint8)
    bmp_path = full_path.replace('.nii.gz', '.bmp').replace('.nii', '.bmp')
    Image.fromarray(img_uint8).save(bmp_path)
    print(f'Saved {os.path.basename(bmp_path)} to {output_path}')


def assert_correct_coord_normalization(coords, min_val=-1.0, max_val=1.0, tolerance=0.1):
    """
    Args:
        coords: numpy array of shape (n, dim)
        min_val: minimum allowed value of the normalized coordinates
        max_val: maximum allowed value of the normalized coordinates
        tolerance: allowed overshoot beyond min_val/max_val (e.g. 0.1 for 10%)
    """
    # t_min = min_val - tolerance
    # t_max = max_val + tolerance
    t_min = min_val
    t_max = max_val
    
    if coords.shape[1] == 3:
        min_x, min_y, min_z = coords.min(axis=0)
        max_x, max_y, max_z = coords.max(axis=0)
        assert min_x >= t_min, f"min_x = {min_x} is less than {t_min}"
        assert max_x <= t_max, f"max_x = {max_x} is greater than {t_max}"
        assert min_y >= t_min, f"min_y = {min_y} is less than {t_min}"
        assert max_y <= t_max, f"max_y = {max_y} is greater than {t_max}"
        assert min_z >= t_min, f"min_z = {min_z} is less than {t_min}"
        assert max_z <= t_max, f"max_z = {max_z} is greater than {t_max}"
    else:
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        assert min_x >= t_min, f"min_x = {min_x} is less than {t_min}"
        assert max_x <= t_max, f"max_x = {max_x} is greater than {t_max}"
        assert min_y >= t_min, f"min_y = {min_y} is less than {t_min}"
        assert max_y <= t_max, f"max_y = {max_y} is greater than {t_max}"

# Wraps a 2D array with the get_fdata()/affine/shape accessors the loaders call on every modality.
class Simple2DImage:
    def __init__(self, data, affine=None):
        self._data = data
        self.affine = affine

    @property
    def shape(self):
        return self._data.shape
        
    def get_fdata(self):
        return self._data


# add background halo to segmentation to allow masking of the brain in the postprsocessing step
def add_background_halo(label_names, seg_nii, halo_width=1.5, background_label_str='BG'):
    seg = seg_nii.get_fdata()
    bg_label = label_names.index(background_label_str)
    mask_bg = (seg>0).astype(np.float32)
    mask_bg = (ndi.gaussian_filter(mask_bg, sigma=halo_width) > 0.001).astype(np.uint8) * bg_label
    mask_bg[seg > 0] = seg[seg > 0]
    return Simple2DImage(mask_bg, seg_nii.affine)


def mask_nifti(nii, mask):
    data = nii.get_fdata()
    data *= mask
    return Simple2DImage(data, nii.affine)
    



def compute_metrics(
    args,
    pred,
    # affine,
    df_row_dict,
    epoch=0,
    split='train',
    reg_type='Rigid',
    bg_label=None,
    tb_writer=None,
    baseline_volume=None,
    gt_baseline_row=None,       # NEW: row dict of the GT baseline visit
    return_images=False,        # NEW: if True, return image arrays for tiling
    patient_stats=None,         # NEW: dict of patient-level min/max stats
):
    """
    Compute metrics and log images + difference maps to TensorBoard.

    Images logged per modality per visit
    ─────────────────────────────────────
    Always (intra-visit differences):
      {split}/{eye_id}/{mod}/{visit_id}_pred        predicted current visit
      {split}/{eye_id}/{mod}/{visit_id}_ref         GT current visit
      {split}/{eye_id}/{mod}/{visit_id}_diff_pred_gt  pred minus GT (FAF only)
      {split}/{eye_id}/{mod}/{visit_id}_diff_pred_gt_seg  TP/FP/FN map (seg only)

    When baseline_volume is provided (differences wrt predicted baseline, with the aim of understanding whether the model is wrongly predicting always the same images regardless of time or not):
      {split}/{eye_id}/longitudinal_diff/{mod}/{visit_id}         intensity diff
      {split}/{eye_id}/longitudinal_change/{mod}/{visit_id}       RGB change map (seg)

    When gt_baseline_row is provided (GT-level difference maps, new):
      {split}/{eye_id}/gt_longitudinal_diff/{mod}/{visit_id}      pred_current - GT_baseline (FAF)
      {split}/{eye_id}/gt_longitudinal_change/{mod}/{visit_id}    pred_current vs GT_baseline (seg)
      {split}/{eye_id}/gt_gt_change/{mod}/{visit_id}              GT_current vs GT_baseline (seg)

    Args:
        args:             experiment config dict
        pred:             INR output — (H, W, C) for 2D or (H, W, Z, C) for 3D
        affine:           affine matrix (np.ndarray)
        df_row_dict:      current visit row as dict
        epoch:            training epoch (used in filenames and TB step)
        split:            'train' | 'val'
        reg_type:         '2D' registration type (unused when commented out)
        bg_label:         segmentation background label index
        tb_writer:        TensorBoard SummaryWriter or None
        baseline_volume:  model prediction at baseline age — same shape as pred, or None
        gt_baseline_row:  row dict of the chronologically first visit of this patient-eye, or None
    """
    pred = typecheck_img(pred)
    if baseline_volume is not None and hasattr(baseline_volume, 'cpu'):
        baseline_volume = baseline_volume.cpu().numpy()

    # Ensure 4×4 affine
    # if affine.shape != (4, 4):
    #     affine = np.eye(4)

    id_col = args['dataset'].get('id_column', 'subject_id')
    eye_id = str(df_row_dict.get(id_col, 'unknown'))

    if 'Visit_Number' in df_row_dict:
        visit_id = f"V{df_row_dict['Visit_Number']}"
    elif 'Visit_ID' in df_row_dict:
        visit_id = f"VID{df_row_dict['Visit_ID']}"
    elif 'Visit' in df_row_dict:
        visit_id = f"V{df_row_dict['Visit']}"
    else:
        visit_id = 'V0'

    sub_id = f"{eye_id}_{visit_id}"
    metrics = {'Subject': sub_id, 'PSNR': [], 'SSIM': [], 'LPIPS': [], 'DICE': [], 'Precision': [], 'Recall': [], 'IoU': [], 'HD': [], 'LOSS': []}
    modalities = args['dataset']['modalities']

    if bg_label is None and args['dataset'].get('label_names') and 'BG' in args['dataset']['label_names']:
        bg_label = args['dataset']['label_names'].index('BG')
    elif bg_label is None:
        bg_label = 0

    is_2d = (pred.ndim == 3)   # (H, W, C)

    if is_2d:
        # 2D PATH
        has_seg = args['inr_decoder']['out_dim'][-1] > 0
        img_dict = {}

        for i, mod in enumerate(modalities):
            is_seg = has_seg and (i == len(modalities) - 1)
            ref_path = df_row_dict[mod]
            mod_imgs = {}

            # Load GT current visit normalised between [0,1]
            ref_data = load_2d_modality(ref_path, is_seg, patient_stats=patient_stats, mod_index=i, args=args)

            # Crop reference using sampling_bbox if specified
            sampling_bbox = args['dataset'].get('sampling_bbox')
            if sampling_bbox is not None:
                h_ref, w_ref = ref_data.shape[:2]
                if len(sampling_bbox) == 2:
                    w_box, h_box = sampling_bbox
                    x_min = (w_ref - w_box) // 2
                    y_min = (h_ref - h_box) // 2
                    x_max = x_min + w_box - 1
                    y_max = y_min + h_box - 1
                elif len(sampling_bbox) == 4:
                    x_min, y_min, x_max, y_max = sampling_bbox
                ref_data = ref_data[y_min:y_max+1, x_min:x_max+1]

            # Extract prediction channel
            if is_seg:
                sr_dims = len(modalities) - 1
                # n_seg_classes = pred.shape[-1] - sr_dims - 1
                # we extract the soft segmentations
                # seg_soft = pred[..., sr_dims + 1:sr_dims + 1 + n_seg_classes]
                seg_hard = pred[..., sr_dims]
                pred_data = seg_hard.astype(np.float32)
                # pred_data = np.argmax(seg_soft, axis=-1).astype(np.float32)
            else:
                pred_data = pred[..., i].astype(np.float32)
                # pred_data = _minmax(pred_data)

            # Center crop reference to match prediction shape if they differ
            if ref_data.shape != pred_data.shape:
                H_ref, W_ref = ref_data.shape
                H_pred, W_pred = pred_data.shape
                h_start = max(0, (H_ref - H_pred) // 2)
                w_start = max(0, (W_ref - W_pred) // 2)
                ref_data = ref_data[h_start:h_start + H_pred, w_start:w_start + W_pred]
                if ref_data.shape != pred_data.shape:
                    H_ref_new, W_ref_new = ref_data.shape
                    h_start_pred = max(0, (H_pred - H_ref_new) // 2)
                    w_start_pred = max(0, (W_pred - W_ref_new) // 2)
                    pred_data = pred_data[h_start_pred:h_start_pred + H_ref_new, w_start_pred:w_start_pred + W_ref_new]

            # Optional METRIC-GRID resize: generate at the checkpoint's native world_bbox, then score
            # on a fixed grid (dataset.metric_resize, e.g. 512 -> 256) to match another model's metric
            # resolution (ImageFlowNet: crop620 -> 256). Resize COPIES only -- pred_data/ref_data stay
            # native for the figures/saves/dumps below. Both pred and GT go to the same grid.
            _mres = args['dataset'].get('metric_resize')
            if _mres is not None:
                _tgt = (int(_mres[0]), int(_mres[1]))
                pred_m = pred_data if pred_data.shape == _tgt else _resize_2d(pred_data, _tgt, seg=is_seg)
                ref_m = ref_data if ref_data.shape == _tgt else _resize_2d(ref_data, _tgt, seg=is_seg)
            else:
                pred_m, ref_m = pred_data, ref_data

            # Metrics --> always computed between predicted current visit and corresponding GT current visit
            if not is_seg:
                metrics['PSNR'].append(psnr_metric(pred_m, ref_m, data_range=1.0))
                metrics['SSIM'].append(ssim_metric(pred_m, ref_m, data_range=1.0))
                _lp = _lpips_score(pred_m, ref_m)
                if _lp is not None:
                    metrics['LPIPS'].append(_lp)
            else:
                seg_m = compute_seg_overlap_metrics(pred_m, ref_m, bg_label)
                metrics['DICE'].append(seg_m['DICE'])
                metrics['Precision'].append(seg_m['Precision'])  # TP/(TP+FP): drops with false positives
                metrics['Recall'].append(seg_m['Recall'])        # TP/(TP+FN): drops with missed GA
                metrics['IoU'].append(seg_m['IoU'])
                metrics['HD'].append(seg_m['HD'])                # Hausdorff distance (grid px)

            # File saves
            if args.get('save_imgs', {}).get(split, False):
                save_img(pred_data, args['output_dir'],
                         f'{split}/{eye_id}/{mod}_{visit_id}_ep={epoch}.bmp')
                save_img(ref_data, args['output_dir'],
                         f'{split}/{eye_id}/{mod}_{visit_id}_ref.bmp')

            # Image Collection for Tiling
            if return_images:
                # mod_imgs['pred'] = _to_tb(pred_data, is_seg)
                # mod_imgs['ref'] = _to_tb(ref_data, is_seg)
                mod_imgs['pred'] = pred_data  # predicted current visit
                mod_imgs['ref'] = ref_data  # GT current visit
                if not is_seg:
                    # mod_imgs['diff_intra'] = _signed_diff_map(pred_data, ref_data)
                    mod_imgs['diff_intra'] = _signed_diff_map_gray(pred_data, ref_data)  # difference map between predicted and GT current visit
                else:
                    mod_imgs['diff_intra'] = _seg_tpfpfn_map(pred_data, ref_data)  # segmentation change map between predicted and GT current visit

            # Individual TensorBoard Logging (skip if tiling)
            if tb_writer is not None and not return_images:
                tb_writer.add_image(
                    f'{split}/{eye_id}/{mod}/{visit_id}_pred',
                    _to_tb(pred_data, is_seg), epoch, dataformats='HW')
                tb_writer.add_image(
                    f'{split}/{eye_id}/{mod}/{visit_id}_ref',
                    _to_tb(ref_data, is_seg), epoch, dataformats='HW')

                if not is_seg:
                    tb_writer.add_image(
                        f'{split}/{eye_id}/{mod}/{visit_id}_diff_pred_gt',
                        _signed_diff_map_gray(pred_data, ref_data), epoch, dataformats='HWC')
                else:
                    tb_writer.add_image(
                        f'{split}/{eye_id}/{mod}/{visit_id}_diff_pred_gt_seg',
                        _seg_tpfpfn_map(pred_data, ref_data), epoch, dataformats='HWC')

            # Longitudinal Difference Maps --> comparison between model's current predicted visit and the predicted baseline visit --> to see if the model is learning temporal dynamics or just predicting the same image across different timepoints
            if baseline_volume is not None:
                sr_dims = len(modalities) - 1
                base_n_seg_classes = baseline_volume.shape[-1] - sr_dims - 1
                
                if not is_seg:
                    # base_mod = _minmax(baseline_volume[..., i].astype(np.float32))
                    base_mod = baseline_volume[..., i].astype(np.float32)  # predicted baseline visit
                    diff_long = _signed_diff_map_gray(pred_data, base_mod)  # difference map between predicted current visit and predicted baseline visit
                    if return_images: mod_imgs['diff_long'] = diff_long
                    if tb_writer is not None and not return_images:
                        tb_writer.add_image(f'{split}/{eye_id}/longitudinal_diff/{mod}/{visit_id}', diff_long, epoch, dataformats='HW')
                else:
                    if base_n_seg_classes > 0:
                        # base_mask = np.argmax(baseline_volume[..., sr_dims + 1:sr_dims + 1 + base_n_seg_classes], axis=-1).astype(np.float32)
                        base_mask = baseline_volume[..., sr_dims].astype(np.float32)  # predicted baseline segmentation
                    else:
                        base_mask = (baseline_volume[..., i] > 0).astype(np.float32)
                    change_rgb = _seg_change_map(pred_data, base_mask)  # segmentation change map between predicted current visit and predicted baseline visit
                    if return_images: mod_imgs['change_long'] = change_rgb
                    if tb_writer is not None and not return_images:
                        tb_writer.add_image(f'{split}/{eye_id}/longitudinal_change/{mod}/{visit_id}', change_rgb, epoch, dataformats='HWC')

            # GT-level difference maps (new)
            if gt_baseline_row is not None:
                gt_base_path = gt_baseline_row.get(mod, None)
                if gt_base_path is not None:
                    gt_base_data = load_2d_modality(gt_base_path, is_seg, patient_stats=patient_stats, mod_index=i, args=args)  # GT baseline visit

                    # Crop baseline using sampling_bbox if specified
                    if sampling_bbox is not None:
                        h_gt, w_gt = gt_base_data.shape[:2]
                        if len(sampling_bbox) == 2:
                            w_box, h_box = sampling_bbox
                            x_min = (w_gt - w_box) // 2
                            y_min = (h_gt - h_box) // 2
                            x_max = x_min + w_box - 1
                            y_max = y_min + h_box - 1
                        elif len(sampling_bbox) == 4:
                            x_min, y_min, x_max, y_max = sampling_bbox
                        gt_base_data = gt_base_data[y_min:y_max+1, x_min:x_max+1]

                    # Center crop baseline to match prediction shape if they differ
                    if gt_base_data.shape != pred_data.shape:
                        H_gt, W_gt = gt_base_data.shape
                        H_pred, W_pred = pred_data.shape
                        h_start = max(0, (H_gt - H_pred) // 2)
                        w_start = max(0, (W_gt - W_pred) // 2)
                        gt_base_data = gt_base_data[h_start:h_start + H_pred, w_start:w_start + W_pred]
                        if gt_base_data.shape != pred_data.shape:
                            H_gt_new, W_gt_new = gt_base_data.shape
                            h_start_pred = max(0, (H_pred - H_gt_new) // 2)
                            w_start_pred = max(0, (W_pred - W_gt_new) // 2)
                            pred_data = pred_data[h_start_pred:h_start_pred + H_gt_new, w_start_pred:w_start_pred + W_gt_new]
                            # If pred_data was cropped, also crop ref_data to keep them aligned
                            if ref_data.shape != pred_data.shape:
                                ref_data = ref_data[h_start_pred:h_start_pred + H_gt_new, w_start_pred:w_start_pred + W_gt_new]

                    if not is_seg:
                        diff_gt_base = _signed_diff_map_gray(pred_data, gt_base_data)  # difference map between predicted current visit and GT baseline visit
                        diff_gt_gt = _signed_diff_map_gray(ref_data, gt_base_data)  # difference map between GT current visit and GT baseline visit
                        if return_images:
                            mod_imgs['gt_long_diff'] = diff_gt_base  # difference map between predicted current visit and GT baseline visit
                            mod_imgs['gt_gt_diff'] = diff_gt_gt  # difference map between GT current visit and GT baseline visit
                        if tb_writer is not None and not return_images:
                            tb_writer.add_image(f'{split}/{eye_id}/gt_longitudinal_diff/{mod}/{visit_id}', diff_gt_base, epoch, dataformats='HW')
                            tb_writer.add_image(f'{split}/{eye_id}/gt_gt_diff/{mod}/{visit_id}', diff_gt_gt, epoch, dataformats='HW')
                    else:
                        change_gt_base = _seg_change_map(pred_data, gt_base_data)  # segmentation change map between predicted current visit and GT baseline visit
                        if return_images: mod_imgs['gt_long_change'] = change_gt_base
                        if tb_writer is not None and not return_images:
                            tb_writer.add_image(f'{split}/{eye_id}/gt_longitudinal_change/{mod}/{visit_id}', change_gt_base, epoch, dataformats='HWC')

                        gt_gt_change = _seg_change_map(ref_data, gt_base_data)  # segmentation change map between GT current visit and GT baseline visit
                        if return_images: mod_imgs['gt_gt_change'] = gt_gt_change
                        if tb_writer is not None and not return_images:
                            tb_writer.add_image(f'{split}/{eye_id}/gt_gt_change/{mod}/{visit_id}', gt_gt_change, epoch, dataformats='HWC')
            
            img_dict[mod] = mod_imgs

        # Combined held-out LOSS: a SINGLE trade-off number that incorporates BOTH the segmentation
        # and the reconstruction quality, so a checkpoint/config can be selected on it directly (the
        # user-requested "validation loss") in addition to the held-out DICE. Composition mirrors the
        # training loss: seg term (1 - GA Dice) + recon_weight * recon MSE. MSE is recovered from PSNR
        # (data_range=1 -> MSE = 10^(-PSNR/10)). Lower is better.
        if metrics['DICE'] and metrics['PSNR']:
            _srw = float(args['optimizer'].get('outer_sr_weight',
                                               args['optimizer'].get('sr_weight', 1.0)))
            _mse = float(np.mean([10.0 ** (-float(p) / 10.0) for p in metrics['PSNR']]))
            _segloss = 1.0 - float(np.mean(metrics['DICE']))
            metrics['LOSS'] = [float(_segloss + _srw * _mse)]

        if return_images:
            return metrics, img_dict

    if return_images:
        return metrics, img_dict
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — all pure functions, no side effects
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]. Returns zeros if range is zero."""
    mn, mx = x.min(), x.max()
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)


def center_crop_2d(img: np.ndarray, sampling_bbox) -> np.ndarray:
    """Center-crop a 2D (H, W) or 3D (H, W, C) array using sampling_bbox."""
    if sampling_bbox is None:
        return img
    h, w = img.shape[:2]
    if len(sampling_bbox) == 2:
        w_box, h_box = sampling_bbox
        x_min = (w - w_box) // 2
        y_min = (h - h_box) // 2
        x_max = x_min + w_box - 1
        y_max = y_min + h_box - 1
    elif len(sampling_bbox) == 4:
        x_min, y_min, x_max, y_max = sampling_bbox
    else:
        return img
    return img[y_min:y_max+1, x_min:x_max+1]

def _to_tb(img: np.ndarray, is_seg: bool) -> np.ndarray:
    """Prepare a 2D array for TensorBoard add_image (HW, float32 in [0,1]).

    Intensity (recon) and GT are already in a consistent per-patient [0,1] space (the decoder output
    is clamped to [0,1]; GT is minmax_patient-normalised), so CLIP to [0,1] rather than re-applying a
    per-image min-max stretch. Per-image min-max put GT and prediction on different scales and
    visually hid low-contrast / off-intensity reconstructions; clipping keeps them on the same scale
    and consistent with the composite figures (which use fixed vmin=0, vmax=1). `is_seg` is kept for
    signature compatibility — segmentation masks are already {0,1} so clipping is a no-op for them.
    """
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    return img.clip(0, 1)


def _signed_diff_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns (a - b) mapped to an RGB image using a Blue-White-Red (bwr) diverging colormap.
    No change (0.0) is mapped to white, positive changes to red, and negative to blue.
    Both inputs should already be in [0, 1].
    """
    diff = a.astype(np.float32) - b.astype(np.float32)
    diff = np.clip(diff, -1.0, 1.0)
    
    h, w = diff.shape[:2]
    rgb = np.ones((h, w, 3), dtype=np.float32)
    
    # Positive differences (0 to 1) -> transition from white [1, 1, 1] to red [1, 0, 0]
    pos_mask = diff > 0
    if np.any(pos_mask):
        val = diff[pos_mask]
        rgb[pos_mask, 1] = 1.0 - val # G decreases
        rgb[pos_mask, 2] = 1.0 - val # B decreases
        
    # Negative differences (-1 to 0) -> transition from white [1, 1, 1] to blue [0, 0, 1]
    neg_mask = diff < 0
    if np.any(neg_mask):
        val = -diff[neg_mask]
        rgb[neg_mask, 0] = 1.0 - val # R decreases
        rgb[neg_mask, 1] = 1.0 - val # G decreases
        
    return rgb


def _signed_diff_map_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns (a - b) mapped to [0, 1] where 0.5 = no change.
    Both inputs should already be in [0, 1].
    """
    diff = a.astype(np.float32) - b.astype(np.float32)
    return ((diff + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)



def _seg_tpfpfn_map(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    RGB map comparing predicted segmentation to GT for the SAME visit.
      Green  [0.2, 0.8, 0.4]  — True positive  (pred=GA, GT=GA)
      Red    [0.9, 0.1, 0.1]  — False positive  (pred=GA, GT=BG)
      Blue   [0.2, 0.4, 0.9]  — False negative  (pred=BG, GT=GA)
      Black  [0,   0,   0  ]  — True negative   (pred=BG, GT=BG)
    Both pred and ref are expected to be float arrays with 0=BG, >0=GA.
    Returns (H, W, 3) float32.
    """
    p = (pred > 0).astype(bool)
    r = (ref  > 0).astype(bool)
    rgb = np.zeros((*pred.shape[:2], 3), dtype=np.float32)
    rgb[p  & r ] = [0.2, 0.8, 0.4]   # TP
    rgb[p  & ~r] = [0.9, 0.1, 0.1]   # FP
    rgb[~p & r ] = [0.2, 0.4, 0.9]   # FN
    return rgb


def _seg_change_map(current: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    RGB map comparing current segmentation to a baseline segmentation.
      Green  — stable GA (both current and baseline positive)
      Red    — new / grown GA (current positive, baseline negative)
      Blue   — resolved GA (current negative, baseline positive)
      Black  — stable background
    Returns (H, W, 3) float32.
    """
    c = (current  > 0).astype(bool)
    b = (baseline > 0).astype(bool)
    rgb = np.zeros((*current.shape[:2], 3), dtype=np.float32)
    rgb[c  & b ] = [0.2, 0.8, 0.4]   # stable GA
    rgb[c  & ~b] = [0.9, 0.1, 0.1]   # new / grown GA
    rgb[~c & b ] = [0.2, 0.4, 0.9]   # resolved GA
    return rgb


def _match_geometry(data: np.ndarray, is_seg: bool, args: dict) -> np.ndarray:
    """Apply the SAME spatial preprocessing the training dataset applies, so a GT image loaded here
    lands on the EXACT field of view the model (and its predictions) live on.

    The model is trained on coordinates normalised over the image the dataset produces. For FAF-GA the
    dataset (data_loading/dataset.py) center-crops native (768) -> `crop_before_resize` (e.g. 620),
    then resizes that crop to `world_bbox` (e.g. 512), mask NEAREST / FAF BILINEAR. The SIREN's
    [-1, 1] domain therefore spans the 620-crop FOV. If GT is loaded WITHOUT this step (native, then
    naively center-cropped to the prediction's pixel SHAPE) it lands on a DIFFERENT, tighter FOV
    (the central-512 direct crop — no optic disc), so PSNR/SSIM/DICE compare misaligned fields of
    view. Mirroring dataset.py:806-826 here keeps GT and prediction on the same FOV.

    Only the geometric step is replicated (crop_before_resize -> resize, or resize-to-world_bbox when
    no sampling_bbox). The old direct-crop path (sampling_bbox set, no crop_before_resize) keeps native
    resolution here and is handled by the caller's existing sampling_bbox crop, so this is a no-op then.
    """
    ds = args.get('dataset', {}) if args else {}
    if ds.get('dataset_name') != 'faf_ga':
        return data
    wb = ds.get('world_bbox')
    if not wb:
        return data
    wb = (int(wb[0]), int(wb[1]))  # PIL (width, height)
    resample = Image.Resampling.NEAREST if is_seg else Image.Resampling.BILINEAR
    crop_pre = ds.get('crop_before_resize')
    if crop_pre is not None:
        cw, ch = int(crop_pre[0]), int(crop_pre[1])
        H, W = data.shape[:2]
        left, top = (W - cw) // 2, (H - ch) // 2
        data = data[top:top + ch, left:left + cw]
        data = np.asarray(Image.fromarray(data.astype(np.float32)).resize(wb, resample),
                          dtype=np.float32)
    elif ds.get('sampling_bbox') is None:
        data = np.asarray(Image.fromarray(data.astype(np.float32)).resize(wb, resample),
                          dtype=np.float32)
    return data


def load_2d_modality(path: str, is_seg: bool, patient_stats: dict = None, mod_index: int = 0,
                     args: dict = None) -> np.ndarray:
    """
    Load a 2D modality from disk. Returns a float32 (H, W) array.
    For intensity images: min-max normalised to [0, 1] using patient_stats if provided,
    otherwise using individual-visit stats.
    For segmentation:    binarised to {0, 1}.

    patient_stats['min']/['max'] may be per-intensity-modality arrays (shape (n_mod,)) — the
    same layout produced by Data._compute_patient_stats and used in load_coords_and_values.
    `mod_index` selects this modality's entry so the normalisation matches training exactly
    (the previous scalar form only worked for a single intensity modality and threw for >1).

    `args`: pass the run config so the GT undergoes the SAME crop_before_resize -> resize geometry as
    training (see `_match_geometry`). REQUIRED for any GT used in metrics/figures whenever
    `crop_before_resize` is set, else GT and prediction land on different fields of view.
    """
    data = np.array(Image.open(path).convert('L')).astype(np.float32)

    # Match the training/dataset FOV (crop_before_resize -> resize) BEFORE normalise/binarise.
    if args is not None:
        data = _match_geometry(data, is_seg, args)

    # Canonicalize laterality: mirror LEFT (OS) eyes so GT lands in the SAME flipped orientation as the
    # training data (dataset.py). Detected from the path (…/OS/… or …_OS_…). Must match dataset.py's
    # PIL FLIP_LEFT_RIGHT -> horizontal flip = data[:, ::-1].
    if args is not None and args.get('dataset', {}).get('canonicalize_laterality', False):
        pu = path.upper(); base = os.path.basename(pu)
        if '/OS/' in pu or '_OS_' in base or base.rsplit('.', 1)[0].endswith('_OS'):
            data = data[:, ::-1].copy()

    if is_seg:
        return (data > 0).astype(np.float32)
    else:
        data = data.astype(np.float32)
        if patient_stats is not None and 'min' in patient_stats and 'max' in patient_stats:
            mn_arr = np.asarray(patient_stats['min']).reshape(-1)
            mx_arr = np.asarray(patient_stats['max']).reshape(-1)
            i = mod_index if 0 <= mod_index < mn_arr.size else 0
            mn, mx = float(mn_arr[i]), float(mx_arr[i])
            if mx > mn:
                return np.clip((data - mn) / (mx - mn), 0.0, 1.0)
            else:
                return np.zeros_like(data)
        return _minmax(data)




def compute_dice(pred, ref, bg_label):
    """
    Compute the Dice score between a predicted and a reference segmentation.
    Ignores the background labels 0 and bg_label.
    
    Args:
        pred (np.ndarray): predicted segmentation, shape (X,Y,Z)
        ref (np.ndarray): reference segmentation, shape (X,Y,Z)
        bg_label (int): label value to ignore (in addition to 0)
    
    Returns:
        float: Average Dice score across non-background labels.
    
    Raises:
        ValueError: If input shapes of 'pred' and 'ref' do not match.
    """
    # Ensure that the shapes of pred and ref match
    if pred.shape != ref.shape:
        raise ValueError("The shape of pred and ref must be the same.")
    
    # Compute the union of labels from both pred and ref 
    # so that we don't miss any label that appears in one but not in the other.
    labels = np.union1d(np.unique(pred), np.unique(ref))
    
    # Exclude background labels: both 0 and the provided bg_label.
    labels = labels[(labels != 0) & (labels != bg_label)]
    
    # If no labels remain, either everything is background 
    # or there is no relevant information, return 1.0 (perfect match) or raise an error.
    if len(labels) == 0:
        return 1.0
    
    dice_scores = []
    for label in labels:
        # Create boolean masks for the current label
        pred_mask = (pred == label)
        ref_mask = (ref == label)
        
        # Compute intersection and size of each mask using count_nonzero
        intersection = np.count_nonzero(pred_mask & ref_mask)
        pred_sum = np.count_nonzero(pred_mask)
        ref_sum = np.count_nonzero(ref_mask)
        
        # When both masks are empty, consider the score perfect (dice = 1.0).
        if pred_sum + ref_sum == 0:
            dice = 1.0
        else:
            dice = (2.0 * intersection) / (pred_sum + ref_sum)
        dice_scores.append(dice)
    
    # Return the mean dice over all labels.
    return np.mean(dice_scores)


def compute_seg_overlap_metrics(pred, ref, bg_label):
    """Overlap metrics for the foreground (non-background) labels, averaged over labels.

    Returns a dict with:
      DICE      = 2·TP / (2·TP + FP + FN)   — overlap; INSENSITIVE to small FP when the GT is large.
      Precision = TP / (TP + FP)            — drops when there are false positives (e.g. GA predicted
                                              far from the real lesion). Use this to catch the failure
                                              mode Dice hides.
      Recall    = TP / (TP + FN)            — drops when real GA is missed.
      IoU       = TP / (TP + FP + FN)       — Jaccard; stricter than Dice.

    Background labels 0 and `bg_label` are ignored, matching compute_dice. When both pred and ref are
    empty for every foreground label, all metrics are 1.0 (a true empty match). When pred predicts
    nothing for a label that IS present in the GT, precision is defined as 1.0 (no false positives)
    while recall is 0.0 — so a "predict nothing" degenerate solution shows up as low recall, and a
    "predict everywhere" solution shows up as low precision.
    """
    if pred.shape != ref.shape:
        raise ValueError("The shape of pred and ref must be the same.")
    labels = np.union1d(np.unique(pred), np.unique(ref))
    labels = labels[(labels != 0) & (labels != bg_label)]
    if len(labels) == 0:
        return {'DICE': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'IoU': 1.0, 'HD': 0.0}

    dice_s, prec_s, rec_s, iou_s, hd_s = [], [], [], [], []
    for label in labels:
        pm = (pred == label)
        rm = (ref == label)
        tp = np.count_nonzero(pm & rm)
        fp = np.count_nonzero(pm & ~rm)
        fn = np.count_nonzero(~pm & rm)
        dice_s.append(1.0 if (2 * tp + fp + fn) == 0 else (2.0 * tp) / (2 * tp + fp + fn))
        prec_s.append(1.0 if (tp + fp) == 0 else tp / (tp + fp))
        rec_s.append(1.0 if (tp + fn) == 0 else tp / (tp + fn))
        iou_s.append(1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn))
        # Hausdorff distance (grid pixels) with eval_omega's empty-mask convention: both-empty -> 0;
        # one-empty -> the array diagonal (max possible distance); else skimage HD.
        if not pm.any() and not rm.any():
            hd_s.append(0.0)
        elif not pm.any() or not rm.any():
            hd_s.append(float(np.sqrt(sum(float(d) ** 2 for d in pm.shape))))
        else:
            hd_s.append(float(hausdorff_distance(pm, rm)))
    return {'DICE': float(np.mean(dice_s)), 'Precision': float(np.mean(prec_s)),
            'Recall': float(np.mean(rec_s)), 'IoU': float(np.mean(iou_s)), 'HD': float(np.mean(hd_s))}


def log_metrics(args, metrics, epoch, df=None, split='train', tb_writer=None):
    """
    Log metrics to wandb, TensorBoard and save to disk
    Metrics are of the form: [metrics_sub-1, ..., metrics_sub-N] with
    metrics_sub-i = {metric-1: [val-mod1, val-mod2, ...], metric-2: [val-mod1, val-mod2, ...]}
    """
    metrics_keys = ['PSNR', 'SSIM', 'LPIPS', 'DICE', 'Precision', 'Recall', 'IoU', 'HD', 'LOSS']
    mod_keys = args['dataset']['modalities']
    # Map each metric to the modalities it was actually computed on, mirroring compute_metrics'
    # rule (modality i is segmentation iff has_seg and i == last). Intensity metrics (PSNR/SSIM) are
    # computed on the intensity modalities (all but the last when seg is present); segmentation
    # metrics (DICE/Precision/Recall/IoU) are computed on the seg modality (the last one). Without
    # this, the per-subject seg list has one element and was mislabelled under the FIRST modality.
    has_seg = args['inr_decoder']['out_dim'][-1] > 0
    intensity_mods = mod_keys[:-1] if has_seg else list(mod_keys)
    seg_mods = [mod_keys[-1]] if has_seg else []
    seg_metric_keys = {'DICE', 'Precision', 'Recall', 'IoU', 'HD'}
    wandb_payload = {}  # batch all (metric, modality) values into ONE wd.log call (shared step)

    for metric_key in metrics_keys:
        # Modalities this metric belongs to (so the tag uses the correct modality name).
        mods_for_metric = seg_mods if metric_key in seg_metric_keys else intensity_mods

        # Collect all values for this metric across all subjects
        # Each entry in metrics[sub][metric_key] is a list [val_mod1, val_mod2, ...]
        all_vals = []
        for m_sub in metrics:
            if metric_key in m_sub and len(m_sub[metric_key]) > 0:
                all_vals.append(m_sub[metric_key])

        if not all_vals:
            continue

        all_vals = np.array(all_vals) # (N_subs, N_mods)
        means = np.mean(all_vals, axis=0) # (N_mods,)
        stds = np.std(all_vals, axis=0)

        for i, mod in enumerate(mods_for_metric):
            if i >= len(means): break
            m_mean = means[i].item()
            m_std = stds[i].item()
            
            tag = f"{split}/{mod}_{metric_key}"
            print(f"{tag}: {m_mean:.3f} +/- {m_std:.3f}")

            if args['logging']:
                wandb_payload[tag] = m_mean
            if tb_writer is not None:
                tb_writer.add_scalar(tag, m_mean, epoch)

    if args['logging'] and wandb_payload:
        wandb_payload[f"{split}/epoch"] = epoch
        wd.log(wandb_payload)  # one batched call (no explicit step; see log_loss note)
    if tb_writer is not None:
        tb_writer.flush()

    # save to disk as json with proper formatting
    metrics_path = os.path.join(args['output_dir'], f'{split}/{split}_metrics_ep={epoch}.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
    if df is not None:
        df.to_csv(os.path.join(args['output_dir'], f'{split}/{split}_df.csv'), index=False)


def log_loss(loss, epoch, split, log=True, tb_writer=None, global_step=None):
    """Log loss components to wandb and TensorBoard.
    Args:
        global_step: If provided, use this as the TensorBoard step instead of epoch.
                     Useful for the validation inner loop to avoid overwriting steps.
    """
    def _val(v):
        return v.item() if hasattr(v, 'item') else v

    tb_step = global_step if global_step is not None else epoch

    if log:
        # Batch all keys into ONE wd.log call so they share a single W&B step; otherwise each
        # wd.log() auto-increments the global step and scatters the components across steps.
        # We do NOT pass an explicit step: call sites use different step scales (per-chunk
        # global_steps vs per-epoch), and W&B requires a monotonically non-decreasing global
        # step. The {split}/epoch entry can be used as a custom x-axis in the W&B UI instead.
        payload = {f"{split}/epoch": epoch}
        if 'total' in loss: payload[f"{split}/loss"] = _val(loss['total'])
        if 'sr' in loss:    payload[f"{split}/loss_sr"] = _val(loss['sr'])
        if 'seg' in loss:   payload[f"{split}/loss_seg"] = _val(loss['seg'])
        for k in ['tf_rot', 'tf_trans', 'tf_scale', 'tf']:
            if k in loss:
                payload[f"{split}/loss_{k}"] = _val(loss[k])
        wd.log(payload)
    # Log to TensorBoard
    if tb_writer is not None:
        for key in ['total', 'sr', 'seg', 'tf', 'tf_rot', 'tf_trans', 'tf_scale']:
            if key in loss:
                tb_writer.add_scalar(f'{split}/loss_{key}', _val(loss[key]), tb_step)
        tb_writer.flush()


def normalize_intensities(values, norm_type, has_seg=True):
    """
    Normalize values according to norm_type
    Args:
        values: numpy array of shape (n_samples, n_modalities)
        norm_type: str, 'minmax' or 'zscore'
        has_seg: bool, if True the last column is segmentation (excluded from normalization)
    Returns:
        normalized_values: numpy array of shape (n_samples, n_modalities)
    """
    values = np.clip(values, 0, None)
    if has_seg:
        values_mod = values[..., :-1]
    else:
        values_mod = values
    if norm_type == 'minmax':
        v_min, v_max = values_mod.min(axis=0), values_mod.max(axis=0)
        # Avoid division by zero
        denom = v_max - v_min
        denom[denom == 0] = 1.0
        values_mod = (values_mod - v_min) / denom
    elif norm_type == 'zscore':
        v_mean, v_std = values_mod.mean(axis=0), values_mod.std(axis=0)
        values_mod = (values_mod - v_mean) / (v_std + 1e-5)
    if has_seg:
        values[..., :-1] = values_mod
    else:
        values = values_mod
    return values


def denormalize_conditions(args, cond_key, values):
    """
    Denormalize values according to the constraints in the dataset
    Args:
        args: arguments
        cond_key: key of the condition in the dataset
        values: numpy array of shape (n_samples, n_modalities)
    Returns:
        denormalized_values: numpy array of shape (n_samples, n_modalities)
    """
    c_min = args['dataset']['constraints'][cond_key]['min']
    c_max = args['dataset']['constraints'][cond_key]['max']
    c_scale = args['atlas_gen']['cond_scale']
    values = (values / c_scale + 1) / 2 * (c_max - c_min) + c_min
    return values


def fig_to_numpy(fig):
    """Convert a Matplotlib figure to a 3D NumPy array (RGB) for TensorBoard."""
    fig.canvas.draw()
    # Use buffer_rgba for compatibility with Matplotlib 3.8+
    data = np.asarray(fig.canvas.buffer_rgba())
    # Extract RGB channels (discard Alpha)
    rgb_data = data[:, :, :3]
    plt.close(fig)
    return rgb_data


def make_longitudinal_tiled_figure(images_row1, images_row2, labels, row1_name="Prediction", row2_name="Reference", title=None,
                                   row1_sublabels=None, row2_sublabels=None):
    """
    Create a tiled figure with two rows.
    images_row1: list of (H, W) or (H, W, 3) arrays
    images_row2: list of (H, W) or (H, W, 3) arrays
    labels: list of column labels (e.g. visit IDs)
    row1_sublabels / row2_sublabels: optional list of per-column annotation strings
        (e.g. metrics / lesion sizes) drawn under each tile via set_xlabel.
    """
    n_cols = len(images_row1)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 4, 8.6), squeeze=False)

    if title:
        fig.suptitle(title, fontsize=16)

    for c in range(n_cols):
        # Row 1 -- annotation goes into the TITLE (top) so it can never overlap row 2's title.
        img1 = images_row1[c]
        if img1.ndim == 2:
            axes[0, c].imshow(img1, cmap='gray', vmin=0, vmax=1)
        else:
            axes[0, c].imshow(img1)
        t1 = f"{row1_name}\n{labels[c]}"
        if row1_sublabels is not None and c < len(row1_sublabels) and row1_sublabels[c]:
            t1 += f"\n{row1_sublabels[c]}"
        axes[0, c].set_title(t1, fontsize=10)
        axes[0, c].axis('off')

        # Row 2 -- annotation goes UNDER the bottom image (xlabel), no neighbour below it.
        img2 = images_row2[c]
        if img2.ndim == 2:
            axes[1, c].imshow(img2, cmap='gray', vmin=0, vmax=1)
        else:
            axes[1, c].imshow(img2)
        axes[1, c].set_title(f"{row2_name}\n{labels[c]}", fontsize=10)
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        if row2_sublabels is not None and c < len(row2_sublabels) and row2_sublabels[c]:
            axes[1, c].set_xlabel(row2_sublabels[c], fontsize=9)
            for sp in axes[1, c].spines.values():
                sp.set_visible(False)
        else:
            axes[1, c].axis('off')

    plt.tight_layout(h_pad=2.5)
    return fig


def make_longitudinal_singlerow_figure(ground_truth, preds, labels, row_name="Future Predictions",
                                   title=None):
    """
    Create a tiled figure with one row.
    images_row: list of (H, W) or (H, W, 3) arrays
    labels: list of column labels (e.g. visit IDs)
    """

    n_cols = len(preds) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4), squeeze=False)

    if title:
        fig.suptitle(title, fontsize=16)

    axes[0, 0].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f"{row_name}\n{'Last GT Visit'}")
    axes[0, 0].axis('off')
    for c in range(1, n_cols):
        img = preds[c-1]
        if img.ndim == 2:
            axes[0, c].imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            axes[0, c].imshow(img)
        axes[0, c].set_title(f"{row_name}\n{labels[c-1]}")
        axes[0, c].axis('off')

    plt.tight_layout()
    return fig


def make_interleaved_figure(images, labels, is_gt_flags, title=None, sublabels=None):
    """
    Single-row figure with GT columns (green border) and Pred columns (orange border).

    Args:
        images:       list of (H, W) or (H, W, 3) arrays, sorted chronologically
        labels:       list of column labels (e.g. 'GT@0w', 'Pred@6w', ...)
        is_gt_flags:  list of bool — True for GT columns, False for predicted
        title:        optional figure title
        sublabels:    optional list of per-column captions shown under each panel
                      (e.g. predicted lesion sizes 'Pred 6.42 mm²'); None entries are skipped
    Returns:
        matplotlib Figure
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4), squeeze=False)
    if title:
        fig.suptitle(title, fontsize=14)

    for c in range(n):
        ax = axes[0, c]
        img = images[c]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(img)
        ax.set_title(labels[c], fontsize=9)
        if sublabels is not None and c < len(sublabels) and sublabels[c]:
            colour = '#2ecc71' if is_gt_flags[c] else '#e67e22'
            ax.set_xlabel(sublabels[c], fontsize=9, color=colour, fontweight='semibold')

        # Hide ticks and labels but keep spines visible for the border
        ax.set_xticks([])
        ax.set_yticks([])

        # Coloured border: green for GT, orange for predictions
        colour = '#2ecc71' if is_gt_flags[c] else '#e67e22'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)

    plt.tight_layout()
    return fig
