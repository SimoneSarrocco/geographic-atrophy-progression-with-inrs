import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.siren import Siren
from models.encodings import get_encoding, TimeEncoding, ConditionEncoding, IdentityEncoding, get_condition_encoding
from utils import embed2affine
from scipy.ndimage import label
import scipy.ndimage as ndi
import math


class INR_Decoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        args_inr = args['inr_decoder']
        self.device = device
        self.sr_dims = sum(args_inr['out_dim'][:-1])
        self.n_seg_channels = args_inr['out_dim'][-1]
        self.out_dim = sum(args_inr['out_dim'])
        self.mod_dim = args_inr['hidden_size'] * len(args_inr['modulated_layers']) * 2
        self.modulator = Modulator(args_inr['latent_dim'], kernel_size=args_inr['cnn_kernel_size'])
        self.arch = args_inr.get('architecture', 'siren')

        # Initialize Coordinate Encoding
        self.encoding = get_encoding(args_inr, args.get('encoding'))
        siren_in_dim = self.encoding.out_dim
        # siren_in_dim = args_inr['in_dim']

        # Time encoding (if enabled)
        self.time_as_input = args_inr.get('time_as_input', False)
        if self.time_as_input:
            # 'time_encoding' selects the time-input encoder: 'mlp' -> learned MLP embedding;
            # anything else falls through to the Fourier/raw logic driven by time_num_frequencies
            # (so existing 'raw'+freq6 configs are unchanged; only 'mlp' is new behavior).
            self.time_encoding = TimeEncoding(
                num_frequencies=args_inr.get('time_num_frequencies', 6),
                kind=args_inr.get('time_encoding', 'fourier'),
                mlp_hidden=args_inr.get('time_mlp_hidden', 16),
                mlp_out=args_inr.get('time_mlp_out', 16),
                mlp_layers=args_inr.get('time_mlp_layers', 2),
            )
            siren_in_dim += self.time_encoding.out_dim
            # siren_in_dim += 1

        # Initialize Condition Encoding. 'cond_encoding' selects the encoder:
        #   'fourier' (default, num_frequencies bands), 'mlp' (learned smooth embedding), or 'raw'.
        cond_dims = args_inr.get('cond_dims', 0)
        cond_num_freqs = args_inr.get('cond_num_frequencies', 6)
        self.cond_encoding = get_condition_encoding(
            cond_dims,
            kind=args_inr.get('cond_encoding', 'fourier'),
            num_frequencies=cond_num_freqs,
            mlp_hidden=args_inr.get('cond_mlp_hidden', 16),
            mlp_out=args_inr.get('cond_mlp_out', 16),
            mlp_layers=args_inr.get('cond_mlp_layers', 2),
        )

        # v2 (gated): per-coordinate ANCHOR inputs (last-observed FAF + baseline mask) fed as
        # extra SIREN inputs to ground prediction in anatomy / exploit perilesional FAF.
        # When faf_as_input is False, n_anchor=0 and behaviour is IDENTICAL to before.
        # anchor_grids (N, n_anchor, H, W) is populated by build_model after the latent bank
        # is sized; it is sampled per-coordinate in forward() via _interpolate_latents.
        self.faf_as_input = args_inr.get('faf_as_input', False)
        self.n_anchor = 2 if self.faf_as_input else 0
        self.anchor_grids = None

        if 'omega_0' in args_inr:
            omega_0 = args_inr['omega_0']
            omega_start = args_inr['omega_start']
            omega_end = args_inr['omega_end']
        else:
            omega_0 = args_inr['omega'][0]
            omega_start = args_inr['omega'][0]
            omega_end = args_inr['omega'][1]

        # self.sr_net = Siren(siren_in_dim, args_inr['latent_dim'][0] + self.cond_encoding.out_dim, self.out_dim,
        #                   args_inr['hidden_size'],
        #                   args_inr['num_hidden_layers'],
        #                   omega_0=omega_0, omega_start=omega_start, omega_end=omega_end,
        #                   schedule_type=args_inr['schedule_type'], outermost_linear=True,
        #                   modulated_layers=args_inr['modulated_layers'])
        self.sr_net = Siren(siren_in_dim,
                            args_inr['latent_dim'][0] + self.cond_encoding.out_dim + self.n_anchor,
                            self.sr_dims, self.n_seg_channels,
                            args_inr['hidden_size'],
                            args_inr['num_hidden_layers'],
                            omega_0=omega_0, omega_start=omega_start, omega_end=omega_end,
                            schedule_type=args_inr['schedule_type'], outermost_linear=True,
                            modulated_layers=args_inr['modulated_layers'],
                            seg_head_num_layers=args_inr.get('seg_head_num_layers', 0),
                            seg_head_hidden_size=args_inr.get('seg_head_hidden_size', 0),
                            seg_head_use_last_features=args_inr.get('seg_head_use_last_features', False),
                            seg_branch_activate=args_inr.get('seg_branch', {}).get('activate', False),
                            seg_branch_num_layers=args_inr.get('seg_branch', {}).get('num_layers', 2),
                            seg_branch_hidden_size=args_inr.get('seg_branch', {}).get('hidden_size', 256),
                            seg_branch_layer=args_inr.get('seg_branch', {}).get('branch_layer', -1),
                            seg_branch_modulate=args_inr.get('seg_branch', {}).get('modulate', True),
                            shared_output_layer=args_inr.get('shared_output_layer', False))

    def forward(self, coords, latent_vecs, condition_vecs, idcs_df=None, time_vals=None):
        """
        Args:
            coords: (N, 2) for 2D or (N, 3) for 3D
            latent_vecs: (N, l_channels, l_x, l_y) for 2D or (N, l_channels, l_x, l_y, l_z) for 3D 
            condition_vecs: (N, c_channels)
            idcs_df: mapping to batch index
            time_vals: (N, 1) or (N, C_time) temporal coordinate
            flips: boolean tensor indicating if spatial latent coordinates are horizontally flipped
        """
        # Apply Coordinate Encoding (e.g. Fourier, HashGrid) — spatial only
        coords_enc = self.encoding(coords)

        # Compute the modulated latent grids for all subjects (small: N_subjects, C, *spatial).
        # NOTE: we deliberately do NOT gather per-sample grids (i.e. self.modulator(latent_vecs)[idcs_df])
        # here, because that materialises an (n_samples, C, H, W) tensor and blows up memory with
        # higher-resolution latent grids. Instead we interpolate per subject inside _interpolate_latents.
        modulations = self.modulator(latent_vecs)

        # --- Standard: concatenate everything ---
        # Append encoded time if enabled
        if self.time_as_input and time_vals is not None:
            time_enc = self.time_encoding(time_vals)
            coords_enc = torch.cat([coords_enc, time_enc], dim=-1)

        # Apply Fourier encoding to conditioning variables
        condition_vecs_enc = self.cond_encoding(condition_vecs)

        # Sample each coordinate's latent from its subject's grid (memory-efficient), then
        # concatenate the conditioning variables (latents + conditions).
        latents_interp = self._interpolate_latents(coords, modulations, idcs_df)
        modulations_interp = torch.cat((latents_interp, condition_vecs_enc), dim=-1)

        # v2: append per-coordinate anchor features (FAF + baseline mask), sampled from the
        # eye's anchor grid the same way as the latent. anchor_grids is fixed (not learned).
        # The SIREN always expects n_anchor extra inputs when faf_as_input is on, so if the
        # anchor bank isn't populated yet we pad zeros (keeps dims consistent).
        if self.faf_as_input:
            if self.anchor_grids is not None:
                anchor_interp = self._interpolate_latents(coords, self.anchor_grids, idcs_df)
            else:
                anchor_interp = torch.zeros(modulations_interp.shape[0], self.n_anchor,
                                            device=modulations_interp.device,
                                            dtype=modulations_interp.dtype)
            modulations_interp = torch.cat((modulations_interp, anchor_interp), dim=-1)

        output = self.sr_net((coords_enc, modulations_interp))

        return output

    def _interpolate_latents(self, coords, modulations, idcs_df):
        """
        For each coordinate sample, bilinearly/trilinearly sample the latent grid of the
        subject it belongs to (idcs_df), WITHOUT materialising a per-sample grid.

        Memory is O(n_samples * C) instead of O(n_samples * C * H * W), which is what makes
        higher-resolution latent grids feasible. Numerically equivalent to gathering
        modulations[idcs_df] and grid_sampling per sample.

        Args:
            coords:      (n_samples, dim) normalized to [-1, 1]
            modulations: (N_subjects, C, H, W) for 2D or (N_subjects, C, D, H, W) for 3D
            idcs_df:     (n_samples,) subject index per coordinate
        Returns:
            (n_samples, C) interpolated latent vectors.
        """
        spatial_dim = modulations.ndim - 2  # 2 or 3
        n_channels = modulations.shape[1]

        if idcs_df is None:
            idcs_df = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        idcs_df = idcs_df.reshape(-1).long()

        out = torch.empty((coords.shape[0], n_channels), device=modulations.device, dtype=modulations.dtype)

        # Group samples by subject; with batch_size=1 (or inference) this is a single iteration.
        for s in torch.unique(idcs_df):
            mask = (idcs_df == s)
            c = coords[mask]
            grid_s = modulations[s:s + 1]  # (1, C, *spatial)
            if spatial_dim == 2:
                grid = c[None, :, None, :2]  # (1, k, 1, 2)
                sampled = F.grid_sample(grid_s, grid, mode='bilinear',
                                        align_corners=True, padding_mode='border')  # (1, C, k, 1)
                out[mask] = sampled[0, :, :, 0].t()
            elif spatial_dim == 3:
                grid = c[None, :, None, None, :3]  # (1, k, 1, 1, 3)
                sampled = F.grid_sample(grid_s, grid, mode='bilinear',
                                        align_corners=True, padding_mode='border')  # (1, C, k, 1, 1)
                out[mask] = sampled[0, :, :, 0, 0].t()
            else:
                raise ValueError(f"Unsupported latent grid dimensionality: {spatial_dim}D")
        return out

    def inference(self, coords, latent_vec, condition_vec, img_shape, tfs=None, step_size=10000, time_val=None):
        """
        Inference of the INR decoder for volume generation.
        """
        if condition_vec is not None and condition_vec.dim() == 1:
            condition_vec = condition_vec.unsqueeze(0)
        if time_val is not None and time_val.dim() == 1:
            time_val = time_val.unsqueeze(0)

        output = torch.empty((coords.shape[0], self.out_dim)).to(device=self.device)
        
        # Transform coordinates if needed (strictly spatial)
        # if tfs is not None:
        #    tfs_expanded = tfs.expand(coords.shape[0], -1)
        #    coords = self.transform(coords, tfs_expanded)

        for i in range(0, coords.shape[0], step_size):
            c = coords[i:i + step_size]
            idcs_df = torch.zeros(c.shape[0], dtype=torch.long, device=self.device)
            cv = condition_vec.expand(c.shape[0], -1)
            tv = time_val.expand(c.shape[0], -1) if (self.time_as_input and time_val is not None) else None
            output[i:i + step_size] = self.forward(c, latent_vec, cv, idcs_df=idcs_df, time_vals=tv)

        imgs = torch.clamp(output[..., :self.sr_dims], 0, 1)
        # (a) Per-image, per-SR-channel min-max renormalization to [0,1] (the original GAP-INR
        # behaviour, restored). The decoder often outputs a compressed sub-range; without this stretch
        # the reconstructions display/measure washed-out and low-contrast. Min/max are taken over all
        # pixels (dim=0) so each channel uses its full dynamic range; the eps guards flat outputs.
        # Config-gated by inr_decoder.renormalize_output (default True). GT is minmax_patient-normalised
        # to [0,1], so a stretched prediction stays comparable for PSNR/SSIM.
        if self.args['inr_decoder'].get('renormalize_output', True):
            mn = imgs.min(dim=0, keepdim=True)[0]
            mx = imgs.max(dim=0, keepdim=True)[0]
            imgs = (imgs - mn) / (mx - mn).clamp_min(1e-6)
        imgs = imgs.reshape(img_shape + [-1])

        # Handle segmentation
        if self.n_seg_channels > 0:
            seg_logits = output[..., self.sr_dims:]
            seg_soft_flat = torch.nn.functional.softmax(seg_logits, dim=-1)
            # Hard mask. For BINARY seg (BG, GA) a config threshold on P(GA) is supported: argmax
            # is equivalent to P(GA) > 0.5, so a higher threshold (e.g. 0.7) keeps only confident GA
            # pixels and suppresses low-confidence false positives far from the real lesion.
            seg_thr = float(self.args['inr_decoder'].get('seg_threshold', 0.5))
            if self.n_seg_channels == 2 and seg_thr != 0.5:
                seg_hard = (seg_soft_flat[..., -1] > seg_thr).long().reshape(img_shape + [-1])
            else:
                seg_hard = torch.argmax(seg_logits, dim=-1).reshape(img_shape + [-1])
            seg_soft = seg_soft_flat.reshape(img_shape + [-1])
            # Connected-component cleanup (binary GA, 2D): drop tiny/spurious islands far from the main
            # lesion and/or keep only the largest component. GA is anatomically contiguous, so this
            # suppresses unrealistic far-from-lesion false positives. Applied at inference -> propagates
            # to all metrics (DICE/Precision/Recall/IoU) and figures consistently.
            seg_hard = self._postprocess_seg(seg_hard, img_shape)
            modalities_rec = torch.cat([imgs, seg_hard, seg_soft], dim=-1)
        else:
            modalities_rec = imgs
            
        if self.args['mask_reconstruction']:
            return self.mask_reconstruction(modalities_rec, seg_hard if self.n_seg_channels > 0 else None)
        else:
            return modalities_rec

    def _postprocess_seg(self, seg_hard, img_shape):
        """Connected-component cleanup of the binary GA mask (2D only).

        Config (inr_decoder.seg_postprocess): activate, keep_largest (keep only the single biggest GA
        component), min_area_px (remove components below this pixel count). No-op when disabled, for
        non-binary seg, or for non-2D shapes. seg_hard has shape img_shape + [n_seg_mods].
        """
        cfg = self.args['inr_decoder'].get('seg_postprocess', {}) or {}
        if not cfg.get('activate', False) or self.n_seg_channels != 2 or len(img_shape) != 2:
            return seg_hard
        keep_largest = bool(cfg.get('keep_largest', False))
        min_area = int(cfg.get('min_area_px', 0))
        if not keep_largest and min_area <= 0:
            return seg_hard
        m = seg_hard[..., 0].detach().cpu().numpy().astype(np.uint8)  # [H,W] in {0,1}
        if m.sum() == 0:
            return seg_hard
        lbl, n = label(m)
        if n <= 1 and not (min_area > 0):
            return seg_hard
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        keep_ids = set(np.arange(1, n + 1)[sizes >= min_area]) if min_area > 0 else set(np.arange(1, n + 1))
        if keep_largest and len(keep_ids) > 0:
            largest = int(np.argmax(sizes)) + 1
            keep_ids = {largest} if largest in keep_ids else keep_ids
        cleaned = np.isin(lbl, list(keep_ids)).astype(np.int64) if keep_ids else np.zeros_like(lbl)
        out = torch.from_numpy(cleaned).to(seg_hard.device).reshape(img_shape + [-1])
        return out

    @staticmethod
    def transform(coords, tfs, inverse=False):
        """
        Transform coordinates using affine transformations.
        Handles both 2D and 3D coordinates automatically.
        """
        if tfs is None or tfs.numel() == 0 or tfs.shape[-1] == 0:
            return coords

        # Detect if coords are 2D or 3D
        is_2d = (coords.shape[-1] == 2)

        # Native 2D Transformation (if tfs matches 2D params)
        if is_2d and tfs.shape[-1] in [3, 6]:
            if tfs.shape[-1] == 3:
                # Interpret tfs as [theta, tx, ty]
                theta = tfs[..., 0]
                t = tfs[..., 1:]  # (N, 2)

                c = torch.cos(theta)
                s = torch.sin(theta)

                # Construct 2D Rotation Matrix
                # R = [[c, -s], [s, c]]
                row1 = torch.stack([c, -s], dim=-1)
                row2 = torch.stack([s, c], dim=-1)
                R = torch.stack([row1, row2], dim=-2)  # (N, 2, 2)
            else:
                # 6-DOF 2D Affine Transformation: [a, b, c, d, tx, ty]
                # Linear part R = [[1+a, b], [c, 1+d]], Trans part t = [tx, ty]
                # We add 1 to a and d so that tfs=0 corresponds to identity mapping
                linear_params = tfs[..., :4]
                t = tfs[..., 4:]

                row1 = torch.stack([1.0 + linear_params[..., 0], linear_params[..., 1]], dim=-1)
                row2 = torch.stack([linear_params[..., 2], 1.0 + linear_params[..., 3]], dim=-1)
                R = torch.stack([row1, row2], dim=-2)  # (N, 2, 2)

            if inverse:
                # General inverse for 2D matrix
                R = torch.inverse(R)
                t = -torch.einsum('nij,nj->ni', R, t)

            # Apply 2D transformation
            return torch.einsum('nxy,ny->nx', R, coords) + t

        # 3D transformation path
        if is_2d:
            raise ValueError(
                f"2D coords received but tf_dim={tfs.shape[-1]} is not 3 or 6. "
                f"Set inr_decoder.tf_dim to 3 (rigid) or 6 (affine) for native 2D."
            )

        # Apply 3D transformation
        if tfs.shape[-1] < 6:
            # If tfs has < 6 columns (e.g. 3), treat as rotation only
            from utils import euler2rot
            R = euler2rot(tfs[..., :3])
            t = torch.zeros((*tfs.shape[:-1], 3), device=tfs.device)
        else:
            R, t = embed2affine(tfs)
        if inverse:
            R = R.inverse()
            t = -torch.einsum('nij,nj->ni', R, t)
        return torch.einsum('nxy,ny->nx', R, coords) + t

    @staticmethod
    def spatial_interpolation(coords, latents, condition_vecs=None):
        """
        Spatial interpolation of the latent vector.
        Determines sampling dimensionality (2D or 3D) based on the latents tensor shape.
        """
        # latents: (N, C, H, W) or (N, C, D, H, W)
        latent_spatial_dim = latents.ndim - 2
        
        if latent_spatial_dim == 2:
            # Latent grid is 2D. Sample using the first 2 dimensions of coords.
            # coords: (N, dim) -> (N, 1, 1, 2)
            # sample_coords = coords[..., :2]
            # grid_coords = sample_coords[:, None, None, :]
            # sampled = F.grid_sample(latents, grid_coords, mode='bilinear', align_corners=True,
            #                        padding_mode='border')
            # latents_interp = sampled.squeeze(-1).squeeze(-1)
            # If batch size was 1, squeeze() might have removed the batch dim. Ensure (N, C)
            # if latents_interp.ndim == 1:
            #    latents_interp = latents_interp.unsqueeze(0)
            coords = coords[:, None, None, :]  # (N, 1, 1, 2)
            latents = F.grid_sample(latents, coords, mode='bilinear', align_corners=True,
                                    padding_mode='border').squeeze()

        elif latent_spatial_dim == 3:
            # Latent grid is 3D. Sample using the first 3 dimensions of coords.
            # coords: (N, dim) -> (N, 1, 1, 1, 3)
            # sample_coords = coords[..., :3]
            coords = coords[:, None, None, None, :] # (N, 1, 1, 1, 3)
            # grid_coords = sample_coords[:, None, None, None, :]
            # sampled = F.grid_sample(latents, grid_coords, mode='bilinear', align_corners=True,
            #                        padding_mode='border')
            # latents_interp = sampled.squeeze(-1).squeeze(-1).squeeze(-1)
            # if latents_interp.ndim == 1:
            #    latents_interp = latents_interp.unsqueeze(0)
            latents = F.grid_sample(latents, coords, mode='bilinear', align_corners=True,
                                    padding_mode='border').squeeze()
        else:
            raise ValueError(f"Unsupported latent grid dimensionality: {latent_spatial_dim}D (latents shape: {latents.shape})")

        if condition_vecs is not None:
            # condition_vecs is (N, C)
            latents = torch.concat((latents, condition_vecs), dim=-1)
        return latents
    

    def mask_reconstruction(self, recs, seg):
        mask = self.connected_components(seg)
        mask = mask.expand_as(recs)
        return recs * mask

    def connected_components(self, seg, bg_label_str='BG'):
        bg_label = self.args['dataset']['label_names'].index(bg_label_str)
        mask = ((seg > 0) & (seg != bg_label)).detach().cpu().numpy()
        shp = np.array(list(mask.shape[:-1]))

        # Handle both 2D and 3D
        ndim = len(shp)
        ps = np.clip(((shp * 0.1) // 2).astype(int), 1, None)  # patch size at least 1
        ps = np.minimum(ps, shp // 2)  # don't exceed boundaries
        cp = shp // 2

        # get connected components
        labeled_mask, num_labels = label(mask)

        # get majority label of center patch
        if ndim == 2:
            center_label = labeled_mask[cp[0] - ps[0]:cp[0] + ps[0], cp[1] - ps[1]:cp[1] + ps[1]].flatten()
        elif ndim == 3:
            center_label = labeled_mask[
                cp[0] - ps[0]:cp[0] + ps[0], cp[1] - ps[1]:cp[1] + ps[1], cp[2] - ps[2]:cp[2] + ps[2]].flatten()
        else:
            raise ValueError(f"Unsupported mask dimensionality: {ndim}D")

        if len(center_label) == 0 or num_labels == 0:
            majority_label = 0
        else:
            majority_label = np.argmax(np.bincount(center_label))

        # set all other labels to 0
        mask = mask * (labeled_mask == majority_label)
        mask_blur = (ndi.gaussian_filter(mask.astype(np.float32), sigma=1.0) > 0.001).astype(np.uint8)

        return torch.from_numpy(mask_blur).to(self.device, torch.float)


class Modulator(nn.Module):
    """
    Modulator for the latent vector based on CNNs.
    Supports both 2D and 3D latent grids.
    Args:
        latent_dims: (C, H, W) for 2D or (C, H, W, D) for 3D
    """

    def __init__(self, latent_dims, kernel_size=3):
        super().__init__()
        if kernel_size > 0:
            # Determine dimensionality
            ndim = len(latent_dims) - 1  # Subtract channel dim
            if ndim == 2:
                self.conv = nn.Conv2d(latent_dims[0], latent_dims[0], kernel_size, padding='same')
            elif ndim == 3:
                self.conv = nn.Conv3d(latent_dims[0], latent_dims[0], kernel_size, padding='same')
            else:
                raise ValueError(f"Unsupported latent grid dimensionality: {ndim}D")
        else:
            self.conv = nn.Identity()

    def forward(self, latent_vecs):
        return self.conv(latent_vecs)
