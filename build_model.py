import wandb as wd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from models.inr_decoder import INR_Decoder
from data_loading.dataset import Data
from utils import *
from utils import _seg_change_map, _signed_diff_map, _seg_tpfpfn_map


class _GatedImageWriter:
    """Thin proxy over a TensorBoard SummaryWriter that forwards EVERYTHING (add_scalar,
    add_text, add_histogram, flush, close, ...) EXCEPT image/figure logging, which is dropped
    when `log_images` is False. This lets scalar curves -- metrics, losses, the checkpoint-
    selection DICE/LOSS, the sensitivity-probe scalars -- stay dense every epoch while the heavy
    per-eye/per-visit image dumps (which wrote ~900 MB/run and blew the disk quota) are throttled
    to an every-Nth-validation cadence. Wrapping self.args['tb_writer'] once per epoch gates every
    add_image call across validate() + both sensitivity probes at a single point."""

    def __init__(self, writer, log_images=True):
        object.__setattr__(self, '_writer', writer)
        object.__setattr__(self, '_log_images', bool(log_images))

    def add_image(self, *a, **k):
        if self._log_images and self._writer is not None:
            self._writer.add_image(*a, **k)

    def add_images(self, *a, **k):
        if self._log_images and self._writer is not None:
            self._writer.add_images(*a, **k)

    def add_figure(self, *a, **k):
        if self._log_images and self._writer is not None:
            self._writer.add_figure(*a, **k)

    def __getattr__(self, name):
        # Delegate any non-image method/attribute to the real writer.
        return getattr(object.__getattribute__(self, '_writer'), name)


class ModelBuilder:
    """
    Trains the conditional INR on a longitudinal cohort and runs validation, test and test-time adaptation.
    """

    def __init__(self, args):
        self.args = args
        # Global seed FIRST — before any model/latent init or dataloader creation — so SIREN weight
        # init, the initial train latent bank, and the DataLoader shuffle order (which uses the global
        # torch RNG) are all reproducible across runs with the same config. NOTE: this does not remove
        # GPU float-accumulation nondeterminism (e.g. grid_sample backward); for that you'd also need
        # torch.use_deterministic_algorithms / cudnn.deterministic (deliberately not enabled here).
        self._seed()
        self.device = args['device']
        self.loss_criterion = Criterion(args).to(args['device'])
        self.time_as_input = self.args['inr_decoder'].get('time_as_input', False)
        self._temporal_key = self._get_temporal_key()
        self.reconstruction_cache = {}
        self._init_training()
        self.global_steps = {'train': 0, 'val': 0}
        self.global_val_steps_monotonic = 0
        # Anchor tensors for the TTA latent regulariser (set per split in _init_validation):
        # None -> regularise toward 0 (random init); tensor -> regularise toward that prior.
        self._latent_anchor = {}
        # Best-checkpoint selection on the held-out (val-eval) DICE.
        self.best_val_dice = -float('inf')
        self.best_val_epoch = -1
        # Parallel selection on the held-out combined LOSS (recon + seg in one number).
        self.best_val_loss = float('inf')
        self.best_val_loss_epoch = -1
        if self.args['epochs']['train'] > 0:
            self.train_on_data()
        if self.args.get('test', {}).get('activate', False):
            self.test()


    def _get_temporal_key(self):
        """
        Dynamically resolve the temporal condition key from the dataset config.
        Returns the temporal_condition if set, otherwise the first enabled condition key.
        """
        temporal_key = self.args['dataset'].get('temporal_condition')
        conditions = self.args['dataset'].get('conditions', {})
        if temporal_key is not None:
            for idx, key in enumerate(conditions.keys()):
                if key == temporal_key:
                    self._temporal_key_idx = idx
                    return temporal_key

        idx = 0
        for key, enabled in conditions.items():
            if enabled:
                self._temporal_key_idx = idx
                return key
            idx += 1
        raise ValueError("No temporal condition key found in dataset config. "
                         "Please specify 'temporal_condition' or enable at least one condition.")


    # Add this to ModelBuilder in build_model.py

    def create_subset_dataloader(self, split, indices):
        """Creates a DataLoader for a subset of the dataset specified by indices."""
        # update_with_indices returns a copy of the dataset with only these rows
        subset_dataset = self.datasets[split].update_with_indices(indices)
        
        return DataLoader(
            subset_dataset,
            batch_size=self.args['batch_size'],
            shuffle=True,
            collate_fn=subset_dataset.collate_fn,
            num_workers=self.args.get('num_workers', 0),
            pin_memory=self.args.get('pin_memory', False)
        )

    def train_on_data(self):
        if len(self.args['load_model']['path']) > 0: self.validate(epoch_train=0)
        loss_hist_epochs = []
        start_time = time.time()
        for epoch in range(self.args['epochs']['train']):
            epoch_start = time.time()
            if self.args['optimizer']['re_init_latents']: self.re_init_latents()
            loss, _ = self.train_epoch(epoch, split='train')
            loss_hist_epochs.append(loss)
            print(
                f"Training: Epoch: {epoch}, Loss: {loss:.4f}, AvgLoss: {np.mean(loss_hist_epochs):.4f}, "
                f"Epoch Time: {time.time() - epoch_start:.2f}s, Total Time: {time.time() - start_time:.2f}s")
            # Throttle image/figure logging to an every-Nth-validation cadence (disk-quota guard).
            # Scalars still log every epoch (the wrapper only drops add_image/add_figure); the heavy
            # per-eye image dumps fire only on image epochs (+ always the final epoch).
            _real_writer = self.args.get('tb_writer', None)
            _gate_images = self._images_enabled_this_epoch(epoch)
            if _real_writer is not None:
                self.args['tb_writer'] = _GatedImageWriter(_real_writer, _gate_images)
            try:
                self.validate(epoch)
                self.run_time_sensitivity_probe(epoch)
                self.run_condition_sensitivity_probe(epoch)
            finally:
                if _real_writer is not None:
                    self.args['tb_writer'] = _real_writer
            self._update_scheduler(split='train')


        return np.mean(loss_hist_epochs)

    def _images_enabled_this_epoch(self, epoch):
        """Decide whether THIS epoch may write images/figures to TensorBoard. Images log only on
        every-Nth VALIDATION epoch (validation.log_images_every, in units of validations) and always
        on the final training epoch. log_images_every <= 1 keeps the original every-validation
        behaviour. Scalars are unaffected (logged every epoch regardless)."""
        every = int(self.args.get('validation', {}).get('log_images_every', 1) or 1)
        is_final = (epoch + 1) == self.args['epochs']['train']
        if every <= 1 or is_final:
            return True
        validate_every = int(self.args.get('validate_every', 1) or 1)
        is_val_epoch = (epoch + 1) % validate_every == 0
        n_val = (epoch + 1) // validate_every          # 1-based index of this validation
        return is_val_epoch and (n_val % every == 0)

    def _load_best_decoder(self):
        """Load decoder weights from 'checkpoint_best.pth' into inr_decoder['train'] so the test
        pass uses the best (by val-eval DICE) checkpoint. Falls back to the current in-memory
        weights if the file is missing (e.g. save_model: false or no validation ran)."""
        best_path = os.path.join(self.args['output_dir'], 'checkpoint_best.pth')
        if not os.path.exists(best_path):
            print(f"[test] No checkpoint_best.pth at {best_path}; using current decoder weights "
                  f"(epoch {self.best_val_epoch}).")
            return
        chkp = torch.load(best_path, weights_only=False)
        self.inr_decoder['train'].load_state_dict(chkp['inr_decoder'])
        self.inr_decoder['train'].eval()
        print(f"[test] Loaded best decoder weights from epoch {chkp.get('epoch')} ({best_path}).")

    def test(self):
        """Final evaluation on the held-out test set with the best training checkpoint.

        Mirrors a validation round but on the 'test' split: load the best decoder (selected by
        val-eval DICE), optimise a fresh latent per test patient-eye on its acquired (non-held-out)
        visits, evaluate on the held-out visit, then generate future-state predictions. Single-visit
        test eyes are optimised on their lone visit (clinical deployment) and only get predictions.
        """
        print(f"\n{'='*60}\n  TEST: final evaluation on the held-out test set\n{'='*60}")

        # Use the best checkpoint's decoder weights, then build the test dataloader.
        self._load_best_decoder()
        self._init_dataloading(split='test')
        if len(self.datasets['test'].df) == 0:
            print("[test] No test rows found in the split column; skipping test evaluation.")
            return None

        tb_writer = self.args.get('tb_writer', None)
        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)

        test_df = self.datasets['test'].df
        picked_test_subs = sorted(test_df['sub_id_int'].unique())

        # Hold-out choice on the test set. support_k (clinical forecast) overrides everything;
        # otherwise mirror validation: 'leave_one_out' holds out EACH visit in turn so the test
        # table gets BOTH interpolation (non-last positions) and extrapolation (last position);
        # 'last'/'specific'/'none' run a single round. Dir naming is f"{split}_{eval|opt}_{tag}",
        # and tag 'holdout_V{p}' is what summarize_eval parses for the interp/extrap split.
        holdout_cfg = self.args.get('validation', {})
        strategy = holdout_cfg.get('holdout_strategy', 'last')
        test_support_k = self.args.get('test', {}).get('support_k', None)
        epoch_tag = self.best_val_epoch if self.best_val_epoch >= 0 else 0

        # Per-pair forecast (matches ImageFlowNet pairwise exactly): overrides support_k / holdout,
        # writes its own summary, and skips the single-round future-state figures below.
        if self.args.get('test', {}).get('pairwise', False):
            return self._run_pairwise_forecast(epoch_tag, tb_writer, grid_coords, grid_shape,
                                                picked_test_subs, split='test')

        if test_support_k is not None:
            print(f"[test] support_k={int(test_support_k)}: fitting latent on the first "
                  f"{int(test_support_k)} visit(s) per eye, forecasting all later visits.")
            test_dice = self._run_validation_round(
                epoch_tag, tb_writer, grid_coords, grid_shape, picked_test_subs,
                holdout_position=None, tag_suffix=f'test_supportk{int(test_support_k)}',
                split='test', support_k=test_support_k)
        elif strategy == 'leave_one_out':
            max_visits = int(test_df.groupby('sub_id_int').size().max())
            print(f"\n[test] Leave-One-Out over {max_visits} visit position(s) on the test set "
                  f"(interpolation = non-last positions, extrapolation = last position).")
            loo_eval_metrics, dices = [], []
            loo_holdout = {}                         # {sub_id: {pos: recon_dict}} -- HELD-OUT pred per visit
            for ho_pos in range(1, max_visits + 1):
                print(f"\n{'='*60}\n  TEST Leave-One-Out: holding out visit {ho_pos}/{max_visits}\n{'='*60}")
                d = self._run_validation_round(
                    epoch_tag, tb_writer, grid_coords, grid_shape, picked_test_subs,
                    holdout_position=ho_pos, tag_suffix=f"holdout_V{ho_pos}", split='test')
                if d is not None:
                    dices.append(d)
                if getattr(self, '_last_round_eval_metrics', None):
                    loo_eval_metrics.extend(self._last_round_eval_metrics)
                # accumulate THIS round's held-out prediction(s) (the eval visit) per subject/visit,
                # so a combined timeline can always show a genuine hold-out prediction for every visit.
                for _sid, _sd in ((getattr(self, '_eval_sets', {}) or {}).get('test_eval', {}) or {}).items():
                    for _pos, _rd in (_sd.get('reconstructions', {}) or {}).items():
                        loo_holdout.setdefault(_sid, {})[int(_pos)] = {
                            k: _rd.get(k) for k in ('gt_faf', 'pred_faf', 'gt_seg', 'pred_seg',
                                                    'psnr', 'ssim', 'dice', 'gt_area', 'pred_area', 'week')}
            # dump accumulated per-visit hold-out arrays to npz (one file per eye) for the standalone
            # hold-out-timeline figure (make_holdout_timeline.py).
            try:
                _tdf = self.datasets['test'].df
                _eye_of = {int(r['sub_id_int']): str(r[self.args['dataset'].get('id_column', 'Eye_ID')])
                           for _, r in _tdf.drop_duplicates('sub_id_int').iterrows()}
                _hd = os.path.join(self.args['output_dir'], 'holdout_timeline_arrays'); os.makedirs(_hd, exist_ok=True)
                for _sid, _pd in loo_holdout.items():
                    _save = {}
                    for _pos, _rd in _pd.items():
                        for _k, _v in _rd.items():
                            if _v is not None and hasattr(_v, 'shape'):
                                _save[f'v{_pos}_{_k}'] = np.asarray(_v, dtype=np.float32)
                            elif _v is not None:
                                _save[f'v{_pos}_{_k}'] = np.asarray(float(_v), dtype=np.float32)
                    np.savez_compressed(os.path.join(_hd, f'{_eye_of.get(int(_sid), _sid)}.npz'), **_save)
                print(f"[holdout-timeline] dumped per-visit hold-out arrays -> {_hd}")
            except Exception as _e:
                print(f"[holdout-timeline] dump skipped: {_e}")
            if loo_eval_metrics:
                print(f"\n{'='*60}\n  TEST Leave-One-Out AVERAGED held-out metrics "
                      f"({len(loo_eval_metrics)} held-out visits over {max_visits} positions)\n{'='*60}")
                log_metrics(self.args, loo_eval_metrics, epoch_tag,
                            split='test_eval_loo_avg', tb_writer=tb_writer)
            test_dice = float(np.mean(dices)) if dices else None
        else:
            if strategy == 'specific':
                ho_pos = holdout_cfg.get('holdout_visit', None)
            elif strategy == 'none':
                ho_pos = 'none'
            else:  # 'last'
                ho_pos = None
            test_dice = self._run_validation_round(
                epoch_tag, tb_writer, grid_coords, grid_shape, picked_test_subs,
                holdout_position=ho_pos, tag_suffix='test', split='test')

        # Future-state predictions for the test eyes (the clinical use case).
        if not self.args['dataset'].get('independent_visits', False):
            self._generate_novel_visits(epoch=epoch_tag, split='test', subject_ids=picked_test_subs,
                                        grid_coords=grid_coords, grid_shape=grid_shape, future=True)

        # Lesion-size analysis on the test set (reuses the cached test reconstructions from the
        # round above). label='_test' keeps its CSV/figures from overwriting the validation ones;
        # force_final=True enables per-eye progression plots + interpolated/extrapolated trajectories.
        if self.args['dataset'].get('dataset_name') == 'faf_ga':
            self.analyze_and_plot_lesion_sizes(epoch_tag, sets=['test_opt', 'test_eval'],
                                               label='_test', force_final=True)
            self.plot_seg_growth_figures(epoch_tag, 'test', picked_test_subs, label='_test')

        if test_dice is not None:
            print(f"[test] Mean held-out DICE on test set: {test_dice:.4f} "
                  f"(decoder from epoch {self.best_val_epoch}).")
            if tb_writer is not None:
                tb_writer.add_scalar("test/eval_dice_mean", test_dice, epoch_tag)
        return test_dice

    @staticmethod
    def _mask_change(seg_a, seg_b):
        """1 - Dice(seg_a, seg_b) between two binary GA masks (the GT-mask change a->b). Matches
        ImageFlowNet's growth metric (is_major = change > 0.1). Both-empty -> 0 (no change)."""
        if seg_a is None or seg_b is None:
            return None
        a = np.asarray(seg_a) > 0.5
        b = np.asarray(seg_b) > 0.5
        denom = float(a.sum() + b.sum())
        if denom == 0:
            return 0.0
        return float(1.0 - 2.0 * float(np.logical_and(a, b).sum()) / denom)

    def _run_pairwise_forecast(self, epoch_tag, tb_writer, grid_coords, grid_shape, picked_subs, split='test'):
        """Per-PAIR forecast matching ImageFlowNet's pairwise eval EXACTLY: for every visit pair
        (a < b) per eye, fit a FRESH latent on ONLY visit a (the older visit of the pair) and predict
        visit b (the newer). Each (a, b) is one `_run_validation_round` with pair_source=a/pair_target=b
        (single-visit opt set), so the latent sees only visit a -- unlike support_k (always visit 1) or
        leave-one-out (all-but-one). Aggregates per (eye, pair) into interp/extrap (b is/ isn't the eye's
        last visit) + minor/major GA-growth buckets, and writes leave_one_out_summary_pairwise.csv in the
        GAP-INR summary format so comparison/collect_loo_tables.py compares it head-to-head with IFN.
        """
        df = self.datasets[split].df
        n_visits = df.groupby('sub_id_int').size().to_dict()   # #visits per eye -> extrapolation test
        max_visits = int(max(n_visits.values())) if n_visits else 0
        print(f"\n{'='*60}\n  {split.upper()} PAIRWISE forecast: every (a<b) pair, fit-on-a -> predict-b "
              f"(max {max_visits} visits)\n{'='*60}")

        rows = []
        for s in range(1, max_visits):
            for t in range(s + 1, max_visits + 1):
                self._run_validation_round(
                    epoch_tag, tb_writer, grid_coords, grid_shape, picked_subs,
                    tag_suffix=f"pair_V{s}toV{t}", split=split, pair_source=s, pair_target=t)
                eval_data = (getattr(self, '_eval_sets', {}) or {}).get(f'{split}_eval', {}) or {}
                opt_data = (getattr(self, '_eval_sets', {}) or {}).get(f'{split}_opt', {}) or {}
                for sub_id, sd in eval_data.items():
                    recs = list((sd.get('reconstructions', {}) or {}).values())
                    if not recs:
                        continue
                    tgt = recs[0]   # eval set holds exactly the target visit t for this eye
                    src_recs = list((opt_data.get(sub_id, {}) or {}).get('reconstructions', {}).values())
                    src = src_recs[0] if src_recs else {}
                    pa, ga = tgt.get('pred_area'), tgt.get('gt_area')
                    growth = self._mask_change(src.get('gt_seg'), tgt.get('gt_seg'))
                    rows.append(dict(
                        Patient_Eye=sub_id, src_visit=s, tgt_visit=t,
                        Dice=tgt.get('dice'), PSNR=tgt.get('psnr'), SSIM=tgt.get('ssim'),
                        LPIPS=tgt.get('lpips'), HD=tgt.get('hd'),
                        area_MAE_mm2=(abs(pa - ga) if (pa is not None and ga is not None) else float('nan')),
                        growth=growth, is_major=(growth is not None and growth > 0.1),
                        is_extrap=(t == int(n_visits.get(sub_id, t)))))

        if not rows:
            print("[pairwise] no eligible pairs (eyes need >= 2 visits).")
            return None
        self._write_pairwise_summary(rows, split)
        ex = [r['Dice'] for r in rows if r['is_extrap'] and r['Dice'] is not None]
        return float(np.mean(ex)) if ex else None

    def _write_pairwise_summary(self, rows, split):
        """Write the per-pair detail CSV + a bucketed summary (GAP-INR leave_one_out format:
        split/group/<metric>_mean/_se) so collect_loo_tables.py ingests it like any GAP-INR run."""
        out_dir = self.args['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        pairs_csv = os.path.join(out_dir, f'pairs_gap_inr_{split}.csv')
        pd.DataFrame(rows).to_csv(pairs_csv, index=False)

        def agg(metric, sub):
            """Return (mean, std, se) over the finite values of `metric`. std/se use ddof=1
            (sample std), matching how the other models' scores are reported (mean±std)."""
            vals = [r[metric] for r in sub if r.get(metric) is not None and not np.isnan(r[metric])]
            if not vals:
                return float('nan'), float('nan'), float('nan')
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            se = sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            return mu, sd, se

        buckets = [('interpolation', lambda r: not r['is_extrap']),
                   ('extrapolation', lambda r: r['is_extrap']),
                   ('minor_growth', lambda r: not r['is_major']),
                   ('major_growth', lambda r: r['is_major']),
                   ('ALL', lambda r: True)]
        # metric key in the rows -> column prefix in the summary
        metric_cols = [('Dice', 'DICE'), ('PSNR', 'PSNR'), ('SSIM', 'SSIM'),
                       ('LPIPS', 'LPIPS'), ('HD', 'HD'), ('area_MAE_mm2', 'areaMAE')]
        summ = []
        for name, pred in buckets:
            sub = [r for r in rows if pred(r)]
            if not sub:
                continue
            row = {'split': split, 'group': name, 'n_heldout_visits': len(sub)}
            for key, col in metric_cols:
                mu, sd, se = agg(key, sub)
                row[f'{col}_mean'] = mu
                row[f'{col}_std'] = sd
                row[f'{col}_se'] = se
            # keep the legacy flat column too (some readers expect it)
            row['area_MAE_mm2'] = row['areaMAE_mean']
            summ.append(row)
        summ_csv = os.path.join(out_dir, 'leave_one_out_summary_pairwise.csv')
        pd.DataFrame(summ).to_csv(summ_csv, index=False)
        print(f"[pairwise] {len(rows)} (eye,pair) records -> {pairs_csv}")
        print(f"[pairwise] bucketed summary (mean±std) -> {summ_csv}")
        for r in summ:
            print(f"    {r['group']:14s} DICE {r['DICE_mean']:.3f}±{r['DICE_std']:.3f}  "
                  f"PSNR {r['PSNR_mean']:.2f}±{r['PSNR_std']:.2f}  "
                  f"SSIM {r['SSIM_mean']:.3f}±{r['SSIM_std']:.3f}  "
                  f"LPIPS {r['LPIPS_mean']:.3f}±{r['LPIPS_std']:.3f}  "
                  f"HD {r['HD_mean']:.2f}±{r['HD_std']:.2f}  "
                  f"areaMAE {r['areaMAE_mean']:.3f}±{r['areaMAE_std']:.3f}  n={r['n_heldout_visits']}")


    def train_epoch(self, epoch, split, epoch_train=None, sub_writer=None):
        """Run one epoch of training or validation.
        Args:
            global_step: If provided, used as the TensorBoard step for log_loss
                         instead of epoch. Prevents step overwrites for val inner loop.
        """
        self.inr_decoder[split].train() if split == 'train' else self.inr_decoder[split].eval()
        if hasattr(self.datasets[split], 'set_epoch'):
            self.datasets[split].set_epoch(epoch)
        loss_hist_batches = []
        loss_component_accum = {}  # accumulate all loss components across batches to be able to log them separately into tensorboard
        time_data_loader = time.time()
        # Add overarching epoch progress bar using tqdm
        # pbar = tqdm(self.dataloaders[split], desc=f"Epoch {epoch} [{split}]")
        tb_writer = self.args.get('tb_writer', None)

        for batch in self.dataloaders[split]:
            print(f"Split: {split}, Current Epoch: {epoch}, Time Loading Batch: {time.time() - time_data_loader:.2f}s")
            start_time = time.time()
            loss, loss_components = self.train_batch(batch, epoch, split, epoch_train=epoch_train, sub_writer=sub_writer)
            loss_hist_batches.append(loss)  # we store the average loss for each batch
            # Accumulate average loss components for each batch
            for key, val in loss_components.items():
               if key not in loss_component_accum:
                   loss_component_accum[key] = []
               loss_component_accum[key].append(val)
            
            # Update progress bar trailing text instead of printing multiple lines
            # pbar.set_postfix({'loss': f"{np.mean(loss_hist_batches):.4f}", 'time/batch': f"{time.time() - start_time:.2f}s"})
            time_data_loader = time.time() # Reset clock for next batch loading time
            print(
                f"Split: {split}, Current Epoch: {epoch}, Loss Batch: {loss:.4f}, Total Training Time Batch: {time.time() - start_time:.2f}s")

        # Average the components across the entire epoch
        epoch_avg_loss = {key: np.mean(vals) for key, vals in loss_component_accum.items()}
        
        # Log epoch-averaged training loss to TensorBoard
        if split == 'train':
            tb_writer = self.args.get('tb_writer', None)
            if tb_writer is not None:
                log_loss(epoch_avg_loss, epoch, split='train_epoch', log=self.args['logging'], tb_writer=tb_writer, global_step=epoch)

        # Log epoch-averaged validation loss if in validation loop
        if split == 'val' and epoch_train is not None:
            tb_writer = self.args.get('tb_writer', None)
            if sub_writer is not None:
                log_loss(epoch_avg_loss, epoch, split='val_inner', log=self.args['logging'], tb_writer=sub_writer, global_step=epoch)
            elif tb_writer is not None:
                val_epochs = self.args['epochs']['val']
                global_step = epoch_train * val_epochs + epoch
                log_loss(epoch_avg_loss, epoch, split='val_inner', log=self.args['logging'], tb_writer=tb_writer, global_step=global_step)
                
        return np.mean(loss_hist_batches), epoch_avg_loss

    def _monotonicity_penalty(self, coords, time_vals, conditions, idx_df, split, cfg):
        """Soft non-decreasing-GA penalty. For each distinct visit (time) in this eye-batch, predict
        the GA probability at a SHARED set of coordinates, order the visits chronologically, and
        penalise any decrease in predicted GA prob from one visit to the next (mean of ReLU(p_t -
        p_{t+1}) over coords & steps). 0 = perfectly non-decreasing. Returns None if <2 visits or no
        seg head. Varies BOTH the time coordinate and the conditioning per visit, so it is correct
        whether the temporal variable is a FiLM condition or a time-as-input coordinate."""
        t = time_vals.reshape(time_vals.shape[0], -1)[:, 0]
        uniq, inv = torch.unique(t, sorted=True, return_inverse=True)
        V = int(uniq.numel())
        if V < 2:
            return None
        out_dim = self.args['inr_decoder']['out_dim']
        sr_dims, n_seg = sum(out_dim[:-1]), out_dim[-1]
        if n_seg < 2:
            return None
        m = min(coords.shape[0], int(cfg.get('n_coords', 2048)))
        c, idf = coords[:m], idx_df[:m]
        probs = []
        for k in range(V):
            first = int((inv == k).float().argmax().item())
            cond_k = conditions[first:first + 1].expand(m, -1)
            time_k = time_vals[first:first + 1].expand(m, -1)
            out = self.inr_decoder[split](c, self.latents[split], cond_k, idcs_df=idf, time_vals=time_k)
            ga = torch.softmax(out[..., sr_dims:sr_dims + n_seg].float(), dim=-1)[..., -1]   # (m,)
            probs.append(ga)
        P = torch.stack(probs, dim=1)                 # (m, V), columns chronologically ordered
        return torch.relu(P[:, :-1] - P[:, 1:]).mean()

    def train_batch(self, batch, epoch, split='train', epoch_train=None, sub_writer=None):
        tb_writer = self.args.get('tb_writer', None)
        loss_hist_samples = []
        loss_component_accum = {}  # accumulate all loss components across inner iterations
        n_smpls = self.args['n_samples']
        if split in ('val', 'test') and not self.args['optimizer'].get('seg_loss_val', True):
            # Test-time optimisation (val/test) fits the latent with the reconstruction loss only;
            # the seg DICE is the held-out selection/early-stopping signal, not an optimisation target.
            seg_weight = 0.0
        else:
            seg_weight = self.args['optimizer']['seg_weight']
            if split == 'train':
                zero_prob = self.args['optimizer'].get('seg_weight_zero_prob', 0.0)
                if zero_prob > 0.0:
                    import random
                    if random.random() < zero_prob:
                        seg_weight = 0.0
        coords_batch, values_batch, conditions_batch, idx_df_batch, time_batch = to_device(batch, device=self.device)
        sample_iterator = range(0, idx_df_batch.shape[0], n_smpls)  # idx_df_batch.shape[0] is the total number of samples in the batch/image
        start_time = time.time()
        
        for i, smpls in enumerate(sample_iterator):
            self.optimizers[split].zero_grad()
            coords = coords_batch[smpls:smpls + n_smpls]
            values = values_batch[smpls:smpls + n_smpls]
            idx_df = idx_df_batch[smpls:smpls + n_smpls].squeeze()
            # with the following, during validation we let the model predict the conditions:
            # conditions = conditions_batch[smpls:smpls + n_smpls] if split == 'train' else self.conditions_val[idx_df] 
            # with the following, during validation we use the ground truth conditions as we do during training rather than trying to learn them through the val optimisation loop
            conditions = conditions_batch[smpls:smpls + n_smpls]
            time_vals = time_batch[smpls:smpls + n_smpls]

            # DEBUG: Check for NaNs
            if torch.isnan(coords).any(): print(f"\nNaN in coords at batch {i}")
            if torch.isnan(values).any(): print(f"\nNaN in values at batch {i}")
            if torch.isnan(conditions).any(): print(f"\nNaN in conditions at batch {i}")
            if torch.isnan(time_vals).any(): print(f"\nNaN in time_vals at batch {i}")
            if torch.isnan(self.latents[split]).any(): print(f"\nNaN in latents at batch {i}")
            # if torch.isnan(self.transformations[split][idx_df]).any(): print(f"\nNaN in transformations at batch {i}")

            with torch.autocast(device_type=self.device, enabled=self.args['amp']):
                values_p = self.inr_decoder[split](coords, self.latents[split], conditions,
                                                   idcs_df=idx_df,
                                                   time_vals=time_vals)
                loss = self.loss_criterion(values_p, values,
                                           seg_weight=seg_weight)

                # Monotonicity penalty (GA is irreversible): the predicted GA probability at a fixed
                # coordinate must be NON-DECREASING in time -- the soft analogue of Lachinov et al.'s
                # phi_dot >= 0 ODE constraint. GT-free. Needs >= 2 visits in the batch, so it is a
                # no-op unless dataset.batch_by_eye groups an eye's visits into one batch.
                mono_cfg = self.args['optimizer'].get('mono_penalty', {}) or {}
                if split == 'train' and mono_cfg.get('activate', False) and seg_weight > 0:
                    mp = self._monotonicity_penalty(coords, time_vals, conditions, idx_df, split, mono_cfg)
                    if mp is not None:
                        loss['mono'] = mp
                        loss['total'] = loss['total'] + float(mono_cfg.get('weight', 0.1)) * mp

            # Anchored latent regularisation during TTA (val/test). The latent is fit on the
            # reconstruction loss only, which is heavily under-determined; this L2 term keeps it
            # near its initialisation prior so it stays in the seg-valid region of latent space.
            # Anchor = population mean / nearest-train init when those modes are used; = 0 for random
            # init (see _init_validation). No-op for split='train' and when val_latent_reg == 0.
            val_reg = self.args['optimizer'].get('val_latent_reg', 0.0)
            if split in ('val', 'test') and val_reg > 0:
                anchor = getattr(self, '_latent_anchor', {}).get(split)
                diff = self.latents[split] if anchor is None else (self.latents[split] - anchor)
                reg_term = val_reg * diff.pow(2).mean()
                loss['latent_reg'] = reg_term
                loss['total'] = loss['total'] + reg_term

            if self.args['amp']:
                self.grad_scalers[split].scale(loss['total']).backward()
                self.grad_scalers[split].step(self.optimizers[split])
                self.grad_scalers[split].update()
            else:
                loss['total'].backward()
                self.optimizers[split].step()

            loss_hist_samples.append(loss['total'].item())  # we append the loss computed for each specific sample of coordinates of the same batch
            self.global_steps[split] += 1
            if split == 'val':
                self.global_val_steps_monotonic += 1

            # Compute the loss components and accumulate over all chunks of that batch
            for key in loss:
                val = loss[key].item() if hasattr(loss[key], 'item') else loss[key]
                if key not in loss_component_accum:
                    loss_component_accum[key] = []
                loss_component_accum[key].append(val)

            # Log average loss components every 100 samples of 10k coordinates (i.e., 5 times for each 512x512 image)
            if i % 100 == 0 or i == (len(sample_iterator) - 1):
                loss_components_running_avg = {key: np.mean(vals) for key, vals in loss_component_accum.items()}
                if split == 'train':
                    log_loss(loss_components_running_avg, epoch, split='train_batch', log=self.args['logging'], tb_writer=tb_writer, global_step=self.global_steps['train'])
                elif split == 'val':
                    if sub_writer is not None:
                        log_loss(loss_components_running_avg, epoch, split='val_inner_batch', log=self.args['logging'], tb_writer=sub_writer, global_step=self.global_steps['val'])
                    elif tb_writer is not None:
                        log_loss(loss_components_running_avg, epoch, split='val_inner_batch', log=self.args['logging'], tb_writer=tb_writer, global_step=self.global_val_steps_monotonic)
                print(f"Split: {split}, Epoch: {epoch}, "
                      f"Elapsed Training Time Batch: {time.time() - start_time:.2f}s"
                      f"Progress: {i / len(sample_iterator):.2f},"
                      f"Loss: {np.mean(loss_hist_samples):.4f},")
                
        if split == 'val' and sub_writer is not None:
            # 2F.1: Latent norm tracking
            sub_id_val = idx_df_batch[0, 0].item()
            latent_norm = self.latents['val'][sub_id_val].norm(p=2).item()
            sub_writer.add_scalar("val_inner/latent_norm", latent_norm, epoch)
            
            # 2F.3: Latent gradient magnitude tracking
            if self.latents['val'].grad is not None:
                grad_norm = self.latents['val'].grad[sub_id_val].norm(p=2).item()
                sub_writer.add_scalar("val_inner/latent_grad_norm", grad_norm, epoch)

        loss_components_avg = {key: np.mean(vals) for key, vals in loss_component_accum.items()}  # average over the whole batch
        return np.mean(loss_hist_samples), loss_components_avg  # we return the average loss for that batch

    def validate(self, epoch_train):
        """
        Validate the model on the validation set, including:
        - optionally generate conditioned renders and save to disk
        - generate training subjects (top 5) and compute eval metrics
        - generate validation subjects (top 3) and compute eval metrics
        - analyze latent space to predict attributes
        - save model state
        """
        validation_cfg = self.args.get('validation', {})
        train_eval_every = validation_cfg.get('train_eval_every', 1)
        
        is_train_eval_epoch = (epoch_train + 1) % train_eval_every == 0 or (epoch_train + 1) == self.args['epochs']['train']
        is_val_epoch = ((epoch_train + 1) % self.args['validate_every'] == 0 or (epoch_train + 1) == self.args['epochs']['train']) and validation_cfg.get('activate', True)

        if not (is_train_eval_epoch or is_val_epoch):
            return

        # Clear cache for the current evaluation epoch
        self.reconstruction_cache = {}
        # Per-set subject_data (reconstructions + lesion areas + metrics), reused by the
        # lesion-size analysis so it never re-reconstructs existing visits.
        self._eval_sets = {'train': {}, 'val_opt': {}, 'val_eval': {}}

        tb_writer = self.args.get('tb_writer', None)
        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)

        if is_val_epoch and self.args['generate_cond_renders']:
            self.generate_renders(epoch_train, n_max=100)

        if is_train_eval_epoch:
            # Training evaluation: first 3 REAL patient-eyes only (never pseudo-eye augmentations,
            # whose GT-vs-pred comparison is invalid; see _is_augmented_eye).
            train_df = self.datasets['train'].df
            picked_train_subs = self._real_eye_subs(train_df)[:3]
            # POSITIONAL indices (df.index.get_loc) so they match _evaluate_visits' iloc-based
            # reconstruction cache keys regardless of the DataFrame index type.
            train_indices = [train_df.index.get_loc(i)
                             for i in train_df[train_df['sub_id_int'].isin(picked_train_subs)].index]

            print(f"Evaluating reconstruction on {len(train_indices)} visits "
                  f"for {len(picked_train_subs)} training subjects...")
            
            metrics_train, train_data = self._evaluate_visits(
                train_indices, 'train', grid_coords, grid_shape, epoch=epoch_train, tb_writer=tb_writer
            )
            self._eval_sets['train'] = train_data
            log_metrics(self.args, metrics_train, epoch_train,
                        df=train_df, split='train',
                        tb_writer=tb_writer)

            self._log_reconstruction_figures(train_data, 'train', epoch_train, tb_writer, tensorboard_tag='existing_visits')

            if not self.args['dataset'].get('independent_visits', False):
                 # In-between visits predictions on the picked training subjects (interpolation)
                 self._generate_novel_visits(epoch=epoch_train, split='train', subject_ids=picked_train_subs,
                                             grid_coords=grid_coords, grid_shape=grid_shape, future=False)
             
                 # Future visits predictions on the picked training subjects (extrapolation)
                 self._generate_novel_visits(epoch=epoch_train, split='train', subject_ids=picked_train_subs,
                                             grid_coords=grid_coords, grid_shape=grid_shape, future=True)

            # Generate and log summary figure for training subjects
            if self.args['dataset'].get('dataset_name') == 'faf_ga':
                self.log_publication_reconstruction_figure(
                    epoch_train, 'train', picked_train_subs, opt_idcs=train_indices, eval_idcs=[],
                    subject_data=train_data
                )

        if is_val_epoch:
            # Validation
            holdout_cfg = self.args.get('validation', {})
            strategy = holdout_cfg.get('holdout_strategy', 'last')
            val_df = self.datasets['val'].df
            picked_val_subs = sorted(val_df['sub_id_int'].unique())

            # Collect the held-out DICE from each round; its mean is the checkpoint-selection signal.
            val_eval_dices = []
            if strategy == 'leave_one_out':
                # Determine max number of visits across all val patient-eyes
                max_visits = val_df.groupby('sub_id_int').size().max()
                loo_eval_metrics = []  # flat list of per-visit held-out metric dicts across ALL positions
                for ho_pos in range(1, max_visits + 1):
                    print(f"\n{'='*60}")
                    print(f"  Leave-One-Out: holding out visit {ho_pos}/{max_visits}")
                    print(f"{'='*60}")
                    tag_suffix = f"holdout_V{ho_pos}"
                    d = self._run_validation_round(
                        epoch_train, tb_writer, grid_coords, grid_shape,
                        picked_val_subs, holdout_position=ho_pos, tag_suffix=tag_suffix
                    )
                    if d is not None:
                        val_eval_dices.append(d)
                    # Collect this position's per-visit held-out metrics for the averaged table.
                    if getattr(self, '_last_round_eval_metrics', None):
                        loo_eval_metrics.extend(self._last_round_eval_metrics)
                # Averaged-across-positions held-out metrics = mean over every (eye, hold-out position)
                # held-out visit — the proper leave-one-out generalisation summary. Reuses log_metrics
                # (TB scalars + JSON under split 'val_eval_loo_avg') with the flattened per-visit list.
                if loo_eval_metrics:
                    print(f"\n{'='*60}\n  Leave-One-Out AVERAGED held-out metrics "
                          f"({len(loo_eval_metrics)} held-out visits over {max_visits} positions)\n{'='*60}")
                    log_metrics(self.args, loo_eval_metrics, epoch_train,
                                split='val_eval_loo_avg', tb_writer=tb_writer)
            else:
                if strategy == 'specific':
                    ho_pos = holdout_cfg.get('holdout_visit', None)
                    tag_suffix = f"holdout_V{ho_pos}" if ho_pos else "holdout_last"
                elif strategy == 'none':
                    ho_pos = 'none'
                    tag_suffix = "holdout_none"
                else:  # 'last'
                    ho_pos = None
                    tag_suffix = "holdout_last"

                d = self._run_validation_round(
                    epoch_train, tb_writer, grid_coords, grid_shape,
                    picked_val_subs, holdout_position=ho_pos, tag_suffix=tag_suffix
                )
                if d is not None:
                    val_eval_dices.append(d)

            # Mean held-out DICE across rounds — the model-selection metric.
            self._last_val_eval_dice = float(np.mean(val_eval_dices)) if val_eval_dices else None

            # Mean held-out combined LOSS (recon + seg) across the same held-out visits — the
            # alternative single-number selection criterion. Pulled from the per-visit metric dicts.
            _loss_metrics = (loo_eval_metrics if strategy == 'leave_one_out'
                             else (getattr(self, '_last_round_eval_metrics', None) or []))
            _losses = [m['LOSS'][0] for m in _loss_metrics
                       if isinstance(m, dict) and m.get('LOSS')]
            self._last_val_loss = float(np.mean(_losses)) if _losses else None

            if not self.args['dataset'].get('independent_visits', False):
                 # Novel visit generation (only once, independent of holdout strategy)
                 self._generate_novel_visits(epoch=epoch_train, split='val', subject_ids=picked_val_subs,
                                             grid_coords=grid_coords, grid_shape=grid_shape, future=True)
                 self._generate_novel_visits(epoch=epoch_train, split='val', subject_ids=picked_val_subs,
                                             grid_coords=grid_coords, grid_shape=grid_shape, future=False)

        # Analyze lesion sizes progression for FAF-GA dataset after each training and validation round
        if is_val_epoch and self.args['dataset'].get('dataset_name') == 'faf_ga':
            self.analyze_and_plot_lesion_sizes(epoch_train)
            # GA-growth overlay/onset figures (per-eye models only; cheap for the few val eyes).
            self.plot_seg_growth_figures(epoch_train, 'val', picked_val_subs)

        if is_val_epoch:
            self.save_state(epoch_train)

            # Best-checkpoint selection on the held-out (val-eval) DICE: this measures
            # longitudinal generalisation and is independent of the inner-loop early stopping
            # (which only decides when to stop optimising a given latent). The decoder weights
            # live on the 'train' split, so we snapshot that split as 'checkpoint_best.pth'.
            val_eval_dice = getattr(self, '_last_val_eval_dice', None)
            if val_eval_dice is not None:
                if val_eval_dice > self.best_val_dice:
                    self.best_val_dice = val_eval_dice
                    self.best_val_epoch = epoch_train
                    self.save_state(epoch_train, split='train', filename='checkpoint_best.pth')
                    print(f"[checkpoint] New best val-eval DICE {val_eval_dice:.4f} at epoch {epoch_train} "
                          f"-> saved checkpoint_best.pth")
                else:
                    print(f"[checkpoint] val-eval DICE {val_eval_dice:.4f} (best {self.best_val_dice:.4f} "
                          f"@ epoch {self.best_val_epoch}); keeping previous best.")
                if tb_writer is not None:
                    tb_writer.add_scalar("val/eval_dice_mean", val_eval_dice, epoch_train)
                    tb_writer.add_scalar("val/best_eval_dice", self.best_val_dice, epoch_train)

            # Parallel best-checkpoint on the held-out combined LOSS (recon + seg). Saved alongside
            # checkpoint_best.pth so the user can select a config by val-DICE OR val-loss.
            val_eval_loss = getattr(self, '_last_val_loss', None)
            if val_eval_loss is not None:
                if val_eval_loss < self.best_val_loss:
                    self.best_val_loss = val_eval_loss
                    self.best_val_loss_epoch = epoch_train
                    self.save_state(epoch_train, split='train', filename='checkpoint_best_loss.pth')
                    print(f"[checkpoint] New best val-eval LOSS {val_eval_loss:.4f} at epoch {epoch_train} "
                          f"-> saved checkpoint_best_loss.pth")
                else:
                    print(f"[checkpoint] val-eval LOSS {val_eval_loss:.4f} (best {self.best_val_loss:.4f} "
                          f"@ epoch {self.best_val_loss_epoch}); keeping previous best.")
                if tb_writer is not None:
                    tb_writer.add_scalar("val/eval_loss_mean", val_eval_loss, epoch_train)
                    tb_writer.add_scalar("val/best_eval_loss", self.best_val_loss, epoch_train)

    def _lesion_px_area_mm2(self, row_dict):
        """SINGLE SOURCE OF TRUTH for mm^2-per-GA-pixel. = ScaleXSlo*ScaleYSlo (native mm/px) * rf,
        where rf = (crop_before_resize/world_bbox)^2 corrects for the 620->512 downsample: each grid
        pixel on the world_bbox(=512) image covers (crop/world)^2 more physical area than a native
        pixel. EVERY lesion-area computation (existing, interpolated, extrapolated, trajectory,
        figures, dumps) MUST use this so the scaling is identical across visits, configs and models.
        For configs WITHOUT crop_before_resize (sampling_bbox native-pitch crop) rf=1."""
        sx = float(row_dict.get('ScaleXSlo', 1.0))
        sy = float(row_dict.get('ScaleYSlo', 1.0))
        cbr = self.args['dataset'].get('crop_before_resize')
        wb = self.args['dataset'].get('world_bbox')
        rf = (float(cbr[0]) / float(wb[0])) * (float(cbr[1]) / float(wb[1])) if (cbr and wb) else 1.0
        return sx * sy * rf

    def analyze_and_plot_lesion_sizes(self, epoch_train, sets=None, label='', force_final=False):
        """
        Lesion-size analysis (mm^2) reusing the reconstructions already computed during
        evaluation (self._eval_sets) -- it does NOT re-reconstruct existing visits.

        Lesion area = (#GA pixels on the sampling grid) * ScaleXSlo * ScaleYSlo, computed
        identically for prediction and ground truth (both on the same center-cropped grid).

        Args:
            epoch_train: epoch tag used in filenames / TB steps.
            sets:        which cached eval sets to analyse. Defaults to the validation triple
                         ['train', 'val_opt', 'val_eval']. Pass ['test_opt', 'test_eval'] for the
                         final test-set analysis.
            label:       suffix inserted into output filenames (e.g. '_test') so a test run does
                         not overwrite the validation CSV/figure at the same epoch.
            force_final: treat this call as the final epoch (enables per-eye plots + continuous
                         interpolated/extrapolated trajectories). Used by the one-shot test() call.

        Produces:
          - one cohort-average plot (one panel per set), GT vs Pred, points joined by lines;
          - one progression plot per patient-eye (final epoch by default), GT vs Pred across visits,
            plus an interpolated/extrapolated continuous trajectory at the final epoch.
        All figures are explicitly closed to avoid matplotlib memory accumulation.
        """
        la_cfg = self.args.get('lesion_analysis', {})
        if not la_cfg.get('activate', True):
            print("Lesion size analysis is deactivated in the config.")
            return

        set_names = list(sets) if sets is not None else ['train', 'val_opt', 'val_eval']

        eval_sets = getattr(self, '_eval_sets', None)
        if not eval_sets or not any(eval_sets.get(k) for k in set_names):
            print("Lesion analysis: no cached reconstructions available (compute_metrics off?); skipping.")
            return

        from collections import defaultdict
        print(f"\n--- Lesion Size Analysis (reusing cached reconstructions) [{', '.join(set_names)}] ---")

        analysis_dir = os.path.join(self.args['output_dir'], "lesion_analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        is_final_epoch = force_final or ((epoch_train + 1) == self.args['epochs']['train'])
        save_indiv = la_cfg.get('save_individual_plots', False)
        save_indiv_final = la_cfg.get('save_individual_plots_final', True)
        should_plot_indiv = save_indiv or (is_final_epoch and save_indiv_final)
        max_indiv = la_cfg.get('max_individual_plots', 10)

        set_titles = {
            'train': 'Training Set',
            'val_opt': 'Validation (Opt Visits)',
            'val_eval': 'Validation (Hold-out Visit)',
            'test_opt': 'Test (Opt Visits)',
            'test_eval': 'Test (Hold-out Visit)',
        }
        # Map a set name to the dataset split it was reconstructed from.
        def _split_of(s):
            return 'train' if s == 'train' else ('test' if s.startswith('test') else 'val')
        color_gt = '#D32F2F'
        color_pred = '#1976D2'

        # 1. Build per-set per-eye visit tables from the cached reconstructions.
        sets_data = {s: {} for s in set_names}
        for set_name in set_names:
            split = _split_of(set_name)
            for sub_id_int, sdata in (eval_sets.get(set_name) or {}).items():
                eye_id = str(sdata.get('eye_id', 'unknown'))
                visits = []
                for _vidx, rec in sdata.get('reconstructions', {}).items():
                    if rec.get('pred_area') is None and rec.get('gt_area') is None:
                        continue
                    visits.append({
                        'weeks': float(rec.get('weeks', 0.0)),
                        'pred_area': rec.get('pred_area'),
                        'gt_area': rec.get('gt_area'),
                        'dice': rec.get('dice'),
                    })
                if not visits:
                    continue
                visits.sort(key=lambda v: v['weeks'])
                for i, v in enumerate(visits):
                    v['visit_idx'] = i
                sets_data[set_name][eye_id] = {'sub_id': int(sub_id_int), 'split': split, 'visits': visits}

        # 2. CSV backup
        csv_rows = []
        for set_name in set_names:
            for eye_id, info in sets_data[set_name].items():
                for v in info['visits']:
                    csv_rows.append({
                        'Patient_Eye': eye_id, 'Set': set_name, 'Weeks': v['weeks'],
                        'GT_Area_mm2': v['gt_area'], 'Pred_Area_mm2': v['pred_area'], 'Dice': v['dice'],
                    })
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(analysis_dir, f"lesion_areas{label}_epoch_{epoch_train}.csv"), index=False)

        # 2b. Monotonicity check: GA is irreversible, so the predicted lesion area should be
        # non-decreasing across chronological visits. We quantify violations as the total area
        # DECREASE along each eye's predicted sequence (mm^2; 0 = perfectly monotone), and report the
        # fraction of eyes with any decrease, per set, alongside the same stats for GT (sanity ref).
        # Logged to TensorBoard and the CSV so unrealistic shrinking predictions are visible.
        tb_writer = self.args.get('tb_writer', None)

        def _drop(seq):
            d = np.diff(np.asarray(seq, dtype=float)) if len(seq) >= 2 else np.array([0.0])
            return float(-d[d < 0].sum())  # total decrease along the sequence (>=0; 0 = non-decreasing)

        mono_rows = []
        for set_name in set_names:
            for eye_id, info in sets_data[set_name].items():
                areas = [v['pred_area'] for v in info['visits'] if v['pred_area'] is not None]
                gtas = [v['gt_area'] for v in info['visits'] if v['gt_area'] is not None]
                mono_rows.append({'Set': set_name, 'Patient_Eye': eye_id,
                                  'pred_decrease_mm2': _drop(areas), 'gt_decrease_mm2': _drop(gtas)})

        # COMBINED FULL-TRAJECTORY monotonicity ('{split}_full'): merge each eye's opt + held-out
        # reconstructions into one chronologically-sorted sequence -> the model's predicted GA area at
        # ALL of that eye's visit times (the clinically meaningful check; the per-set rows above are
        # trivially monotone when a set holds a single visit, e.g. holdout='last' -> one eval visit).
        combined = {}   # eye_id -> {'split', 'visits':[...]}
        for set_name in set_names:
            for eye_id, info in sets_data[set_name].items():
                c = combined.setdefault(eye_id, {'split': info['split'], 'visits': []})
                c['visits'].extend(info['visits'])
        for eye_id, info in combined.items():
            vs = sorted(info['visits'], key=lambda v: v['weeks'])
            areas = [v['pred_area'] for v in vs if v['pred_area'] is not None]
            gtas = [v['gt_area'] for v in vs if v['gt_area'] is not None]
            mono_rows.append({'Set': f"{info['split']}_full", 'Patient_Eye': eye_id,
                              'pred_decrease_mm2': _drop(areas), 'gt_decrease_mm2': _drop(gtas)})

        if mono_rows:
            mdf = pd.DataFrame(mono_rows)
            mdf.to_csv(os.path.join(analysis_dir, f"lesion_monotonicity{label}_epoch_{epoch_train}.csv"), index=False)
            for set_name in sorted(set(mdf['Set'])):
                sub = mdf[mdf['Set'] == set_name]
                if sub.empty:
                    continue
                viol_frac = float((sub['pred_decrease_mm2'] > 1e-6).mean())
                mean_drop = float(sub['pred_decrease_mm2'].mean())
                print(f"  [monotonicity:{set_name}] eyes with shrinking pred GA: {viol_frac*100:.0f}% "
                      f"| mean total decrease: {mean_drop:.3f} mm^2")
                if tb_writer is not None:
                    tb_writer.add_scalar(f"lesion_monotonicity/{set_name}/pred_violation_fraction", viol_frac, epoch_train)
                    tb_writer.add_scalar(f"lesion_monotonicity/{set_name}/pred_mean_decrease_mm2", mean_drop, epoch_train)

        # 3. Cohort-average plot (one panel per set), GT vs Pred, points joined by lines, +/- std bands.
        fig, axes = plt.subplots(1, len(set_names), figsize=(5 * len(set_names), 5), sharey=True)
        axes = np.atleast_1d(axes)
        try:
            fig.suptitle(f"Cohort Average Lesion Size (mm$^2$) - Epoch {epoch_train}",
                         fontsize=14, fontweight='bold')
            for ax_idx, set_name in enumerate(set_names):
                ax = axes[ax_idx]
                ax.set_title(set_titles[set_name], fontsize=12, fontweight='semibold')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlabel("Visit (chronological)", fontsize=11)
                if ax_idx == 0:
                    ax.set_ylabel("Lesion Area (mm$^2$)", fontsize=11)
                subjects = sets_data[set_name]
                if not subjects:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                            transform=ax.transAxes, color='#757575')
                    continue
                gt_by, pred_by = defaultdict(list), defaultdict(list)
                abs_errs = []  # |pred - gt| over all paired visits -> set-level lesion-area MAE
                for info in subjects.values():
                    for v in info['visits']:
                        if v['gt_area'] is not None:
                            gt_by[v['visit_idx']].append(v['gt_area'])
                        if v['pred_area'] is not None:
                            pred_by[v['visit_idx']].append(v['pred_area'])
                        if v['gt_area'] is not None and v['pred_area'] is not None:
                            abs_errs.append(abs(v['pred_area'] - v['gt_area']))
                idcs = sorted(set(list(gt_by) + list(pred_by)))
                if not idcs:
                    continue
                xs = [i + 1 for i in idcs]

                def _mean_se(by, i):
                    """Mean and STANDARD ERROR (std/sqrt(n)) of the cohort at visit i."""
                    a = np.asarray(by[i], dtype=float)
                    if a.size == 0:
                        return np.nan, 0.0
                    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
                    return float(a.mean()), se

                gt_mean, gt_se = zip(*[_mean_se(gt_by, i) for i in idcs])
                pred_mean, pred_se = zip(*[_mean_se(pred_by, i) for i in idcs])
                ax.errorbar(xs, gt_mean, yerr=gt_se, color=color_gt, lw=2.5, marker='o',
                            ms=6, capsize=3, label='GT (mean ± SE)')
                ax.errorbar(xs, pred_mean, yerr=pred_se, color=color_pred, lw=2.5, ls='--',
                            marker='s', ms=6, capsize=3, label='Pred (mean ± SE)')
                # MAE of predicted vs GT lesion area over all visits in this set (paper metric).
                mae = float(np.mean(abs_errs)) if abs_errs else float('nan')
                # Log to TensorBoard so each run shows a held-out lesion-area MAE curve alongside
                # DICE/PSNR/SSIM. val_eval = hold-out visit (the predicted-lesion-size metric).
                if tb_writer is not None and abs_errs:
                    tb_writer.add_scalar(f"lesion_area_MAE/{set_name}", mae, epoch_train)
                n_eyes = len(subjects)
                ax.set_title(f"{set_titles[set_name]}  (n={n_eyes} eyes)\nlesion-area MAE = {mae:.3f} mm$^2$",
                             fontsize=11, fontweight='semibold')
                ax.set_xticks(xs)
                ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
            plt.tight_layout()
            fig.savefig(os.path.join(analysis_dir, f"average_lesion_sizes{label}_epoch_{epoch_train}.png"),
                        dpi=200, bbox_inches='tight')
        finally:
            plt.close(fig)

        # 4. Per-eye progression plots (final epoch by default). Continuous trajectory only at
        #    the final epoch (the only place we run new reconstructions, and only for the picked eyes).
        if should_plot_indiv:
            indiv_dir = os.path.join(analysis_dir, "individual_subjects")
            need_traj = is_final_epoch and not self.args['dataset'].get('independent_visits', False)
            grid_coords, grid_shape = (generate_world_grid(self.args, device=self.device)
                                       if need_traj else (None, None))
            for set_name in set_names:
                set_dir = os.path.join(indiv_dir, set_name)
                os.makedirs(set_dir, exist_ok=True)
                for count, (eye_id, info) in enumerate(sets_data[set_name].items()):
                    if count >= max_indiv:
                        break
                    visits = info['visits']
                    if not visits:
                        continue
                    # Merge the held-out visit(s) from the paired '_eval' set into this '_opt'
                    # figure, so a single per-patient plot shows ALL prediction types distinctly:
                    # observed-visit preds, the HELD-OUT existing visit pred, and the
                    # interpolated/extrapolated synthetic trajectory. Skip the standalone '_eval'
                    # figure (its held-out point now lives in the merged '_opt' figure).
                    if set_name.endswith('_eval') and set_name[:-5] + '_opt' in set_names:
                        continue
                    heldout_visits = []
                    if set_name.endswith('_opt'):
                        ev_info = sets_data.get(set_name[:-4] + '_eval', {}).get(eye_id)
                        if ev_info:
                            heldout_visits = ev_info.get('visits', [])
                    split = info['split']
                    sub_id = info['sub_id']
                    fig = plt.figure(figsize=(8.5, 5.5))
                    try:
                        weeks = [v['weeks'] for v in visits]
                        gt_areas = [v['gt_area'] for v in visits]
                        pred_areas = [v['pred_area'] for v in visits]
                        # per-eye lesion-area MAE over the visits available in this set
                        pe = [abs(p - g) for p, g in zip(pred_areas, gt_areas)
                              if p is not None and g is not None]
                        eye_mae = float(np.mean(pe)) if pe else float('nan')
                        # GT trajectory over ALL real visits (observed + held-out)
                        all_v = sorted(visits + heldout_visits, key=lambda v: v['weeks'])
                        all_w = [v['weeks'] for v in all_v if v['gt_area'] is not None]
                        all_gt = [v['gt_area'] for v in all_v if v['gt_area'] is not None]
                        plt.plot(all_w, all_gt, color=color_gt, marker='o', ms=8, lw=2.0, label='GT visits')
                        plt.plot(weeks, pred_areas, color=color_pred, marker='s', ms=8, lw=2.0,
                                 ls='--', label='Pred (observed visits)')
                        # held-out EXISTING visit prediction -- distinct (red star)
                        hpw = [(v['weeks'], v['pred_area']) for v in heldout_visits
                               if v['pred_area'] is not None]
                        if hpw:
                            hw, hp = zip(*hpw)
                            plt.scatter(hw, hp, color='#D32F2F', marker='*', s=260,
                                        edgecolors='k', linewidths=0.8, zorder=6,
                                        label='Pred (held-out visit)')

                        if need_traj:
                            sub_df = self.datasets[split].df
                            actual_rows = [r.to_dict() for _, r in
                                           sub_df[sub_df['sub_id_int'] == sub_id]
                                           .sort_values(self._temporal_key).iterrows()]
                            if actual_rows and weeks:
                                w_max = max(weeks)
                                cont_w = np.linspace(0.0, w_max + 48.0, 25)
                                traj_w, traj_a = [], []
                                for w in cont_w:
                                    new_row = self._get_interpolated_row_dict(actual_rows, w)
                                    try:
                                        vol = self._reconstruct_visit(
                                            new_row, int(sub_id), grid_coords, grid_shape, split=split,
                                            allow_extrapolation=self.args['dataset'].get('extrapolate_beyond_range', False))
                                        pred_np = typecheck_img(vol)
                                        dec = self.inr_decoder[split]
                                        if dec.n_seg_channels > 0:
                                            seg = pred_np[..., dec.sr_dims] > 0.5
                                            traj_w.append(float(w))
                                            traj_a.append(float(np.sum(seg) * self._lesion_px_area_mm2(new_row)))
                                    except Exception:
                                        continue
                                if traj_w:
                                    traj_w = np.asarray(traj_w); traj_a = np.asarray(traj_a)
                                    plt.plot(traj_w, traj_a, color='#455A64', ls='-', lw=1.3, alpha=0.6,
                                             zorder=1, label='Pred trajectory (continuous)')
                                    # discrete query points, color-coded interpolation vs extrapolation
                                    interp, extrap = traj_w <= w_max, traj_w > w_max
                                    if interp.any():
                                        plt.scatter(traj_w[interp], traj_a[interp], color='#2E7D32', marker='D',
                                                    s=26, zorder=3, label='Pred (interpolated)')
                                    if extrap.any():
                                        plt.scatter(traj_w[extrap], traj_a[extrap], color='#EF6C00', marker='^',
                                                    s=42, zorder=3, label='Pred (extrapolated)')
                                    plt.axvline(x=w_max, color='#9E9E9E', ls=':', alpha=0.8)

                        plt.title(f"Lesion Trajectory: {eye_id} ({set_titles[set_name]})\n"
                                  f"observed-visit MAE = {eye_mae:.3f} mm$^2$  —  Epoch {epoch_train}",
                                  fontsize=11, fontweight='bold')
                        plt.xlabel("Weeks from Baseline", fontsize=10)
                        plt.ylabel("Lesion Area (mm$^2$)", fontsize=10)
                        plt.grid(True, linestyle=':', alpha=0.5)
                        plt.legend(loc='upper left', frameon=True, fontsize=8)
                        fig.savefig(os.path.join(set_dir, f"subject_{eye_id}_epoch_{epoch_train}.png"),
                                    dpi=200, bbox_inches='tight')
                    finally:
                        plt.close(fig)

        print(f"Saved lesion analysis to {analysis_dir}")

    def plot_seg_growth_figures(self, epoch_train, split, subject_ids, label=''):
        """Per-eye GA-growth visualisation: reconstruct the predicted GA mask at a dense set
        of weeks (observed -> extrapolated) from the eye's latent, then render a boundary
        overlay (color = week) + a per-pixel onset 'volume' map (see seg_growth.py).
        Per-eye models only (needs the FiLM/time conditioning to vary the mask with time)."""
        if self.args['dataset'].get('independent_visits', False):
            return
        dec = self.inr_decoder.get(split)
        if dec is None or getattr(dec, 'n_seg_channels', 0) <= 0:
            return
        try:
            from seg_growth import save_seg_growth_figure
        except Exception as e:
            print(f"[seg_growth] import failed: {e}")
            return
        out_dir = os.path.join(self.args['output_dir'], "reconstructions", "seg_growth")
        os.makedirs(out_dir, exist_ok=True)
        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)
        df = self.datasets[split].df
        id_col = self.args['dataset'].get('id_column', 'subject_id')
        extrap = self.args['dataset'].get('extrapolate_beyond_range', False)
        for sub_id in subject_ids:
            sub_df = df[df['sub_id_int'] == sub_id].sort_values(self._temporal_key)
            if sub_df.empty:
                continue
            eye_id = str(sub_df.iloc[0].get(id_col, 'unknown'))
            actual_rows = [r.to_dict() for _, r in sub_df.iterrows()]
            # mm^2 per grid-pixel (same factor the lesion_areas CSV uses) so the GA-area panel
            # is in mm^2 and consistent with the trajectory figures. Masks below are decoded on
            # the SAME world grid that the CSV areas are computed on.
            area_per_px_mm2 = self._lesion_px_area_mm2(actual_rows[0])
            weeks_obs = [float(r.get(self._temporal_key, 0.0)) for r in actual_rows]
            w_max = max(weeks_obs) if weeks_obs else 48.0
            traj_w, masks, faf_bg = [], [], None
            for w in np.linspace(0.0, w_max + 48.0, 10):
                new_row = self._get_interpolated_row_dict(actual_rows, float(w))
                try:
                    vol = self._reconstruct_visit(new_row, int(sub_id), grid_coords, grid_shape,
                                                  split=split, allow_extrapolation=extrap)
                    pred_np = typecheck_img(vol)
                    masks.append(pred_np[..., dec.sr_dims] > 0.5)
                    traj_w.append(float(w))
                    if faf_bg is None and dec.sr_dims > 0:
                        faf_bg = np.clip(pred_np[..., 0], 0, 1)
                except Exception:
                    continue
            if masks:
                try:
                    save_seg_growth_figure(traj_w, masks,
                                           os.path.join(out_dir, f"{eye_id}{label}_growth.png"),
                                           faf_bg=faf_bg, title=f"{eye_id}: predicted GA growth",
                                           area_per_px_mm2=area_per_px_mm2)
                except Exception as e:
                    print(f"[seg_growth] {eye_id} failed: {e}")
        print(f"Saved GA-growth figures to {out_dir}")

    def log_publication_reconstruction_figure(self, epoch, split, subject_ids, opt_idcs, eval_idcs, subject_data=None, tag_suffix=''):
        """
        Generates two publication-ready figures per patient-eye:
        1. FAF combined timeline:
           - Row 1: FAF GT (blank for interpolated/extrapolated)
           - Row 2: FAF Prediction (with border style highlighting the status and PSNR)
           - Row 3: FAF Intra-visit Signed Difference Map
        2. Segmentation combined timeline:
           - Row 1: Segmentation GT (blank for interpolated/extrapolated)
           - Row 2: Segmentation Prediction (with border style highlighting the status and Dice)
           - Row 3: Segmentation TP/FP/FN Intra-visit Change Map
        """
        print(f"\n--- Generating Publication-Ready Figures for {split} split ---")
        from PIL import Image

        # Hold-out position suffix (e.g. '_holdout_V2'): keeps leave-one-out rounds from colliding
        # at the same TB tag / disk filename. Empty for train and single-round validation.
        sfx = f"_{tag_suffix}" if tag_suffix else ""
        tb_writer = self.args.get('tb_writer', None)
        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)
        df = self.datasets[split].df
        id_col = self.args['dataset'].get('id_column', 'subject_id')
        modalities = self.args['dataset']['modalities']
        
        # Color theme: orange for train, blue for val
        color = '#FFA726' if split == 'train' else '#29B6F6'
        
        bg_label = self.args['dataset'].get('label_names', ['BG']).index('BG') if 'BG' in self.args['dataset'].get('label_names', []) else 0
        
        pub_dir = os.path.join(self.args['output_dir'], "reconstructions", "publication_figures")
        os.makedirs(pub_dir, exist_ok=True)
        
        # Border style utility helper
        def set_axis_border(ax, border_style, border_color):
            for spine in ax.spines.values():
                if border_style == 'none':
                    spine.set_visible(False)
                elif border_style == 'dotted':
                    spine.set_visible(True)
                    spine.set_color('#888888')
                    spine.set_linewidth(2.0)
                    spine.set_linestyle((0, (2, 2)))
                elif border_style == 'solid':
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(2.5)
                    spine.set_linestyle('-')
                elif border_style == 'dashed':
                    spine.set_visible(True)
                    spine.set_color(border_color)
                    spine.set_linewidth(2.5)
                    spine.set_linestyle((0, (5, 5)))

        _newvisit_rows = []                        # new-timepoint predicted GA areas (--new-csv schema)
        # Optional: dump NEW-visit FAF/mask cases (no GT) for make_trajectory.py --new-root.
        _nvd = self.args.get('comparison_dump_newvisits') or {}
        _nv_dump = None
        if _nvd.get('enable') and split == _nvd.get('split', 'test'):
            try:
                import sys as _sys
                _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'comparison'))
                import dump_io as _dump_io
                _nv_dump = (_dump_io, _nvd['root'], _nvd.get('method', 'gap_inr'), _nvd.get('scenario', 'static'))
            except Exception as _e:
                print(f"[newvisits_dump] skipped (import failed): {_e}")
        for sub_id in subject_ids:
            sub_df = df[df['sub_id_int'] == sub_id]
            if sub_df.empty:
                continue
            
            sorted_sub_df = sub_df.sort_values(self._temporal_key)
            eye_id = str(sorted_sub_df.iloc[0].get(id_col, 'unknown'))
            patient_stats = self._get_patient_stats(split, sub_id)
            
            actual_visits = []
            actual_rows = []
            
            # 1. Gather actual GT and predictions
            for idx, row in sorted_sub_df.iterrows():
                row_dict = row.to_dict()
                weeks = float(row_dict.get('weeks_from_baseline', 0.0))
                actual_rows.append(row_dict)

                # POSITIONAL index: opt_idcs / eval_idcs and the reconstruction cache are all keyed
                # positionally (df.iloc), while iterrows() yields the index LABEL. Convert once so the
                # keying is robust to any DataFrame index type (RangeIndex or not).
                pos = df.index.get_loc(idx)

                # Determine status and borders
                is_eval = pos in eval_idcs
                gt_border = 'dotted' if is_eval else 'none'

                if is_eval:
                    opt_weeks = [df.iloc[o_idx]['weeks_from_baseline'] for o_idx in opt_idcs if df.iloc[o_idx]['sub_id_int'] == sub_id]
                    if len(opt_weeks) > 0 and min(opt_weeks) < weeks < max(opt_weeks):
                        recon_border = 'solid'
                    else:
                        recon_border = 'dashed'
                else:
                    recon_border = 'none'

                # Load GT modalities and reconstruct predictions
                gt_loaded_ok = False
                pred_loaded_ok = False

                if subject_data is not None and sub_id in subject_data and pos in subject_data[sub_id]['reconstructions']:
                    recon_dict = subject_data[sub_id]['reconstructions'][pos]
                    gt_faf = recon_dict['gt_faf']
                    pred_faf = recon_dict['pred_faf']
                    gt_seg = recon_dict['gt_seg']
                    pred_seg = recon_dict['pred_seg']
                    gt_loaded_ok = gt_faf is not None
                    pred_loaded_ok = pred_faf is not None
                
                if not (gt_loaded_ok and pred_loaded_ok):
                    # Fallback: Load GT modalities on the fly
                    try:
                        faf_path = self.datasets[split].resolve_path(row_dict, modalities[0])
                        gt_faf = load_2d_modality(faf_path, is_seg=False, patient_stats=patient_stats, args=self.args)

                        seg_path = self.datasets[split].resolve_path(row_dict, modalities[1])
                        gt_seg = load_2d_modality(seg_path, is_seg=True, patient_stats=patient_stats, args=self.args)
                        gt_loaded_ok = True
                    except Exception as e:
                        print(f"Warning: Failed to load GT modalities for {eye_id} at week {weeks}: {e}")
                        continue
                    
                    # Fallback: Reconstruct visit on the fly (independent_visits uses the positional
                    # per-visit latent slot, matching _evaluate_visits).
                    latent_idx = pos if self.args['dataset'].get('independent_visits', False) else int(sub_id)
                    try:
                        volume_inf = self._reconstruct_visit(row_dict, latent_idx, grid_coords, grid_shape, split=split)
                        pred_np = typecheck_img(volume_inf)
                        
                        pred_faf = pred_np[..., 0].astype(np.float32)
                        pred_seg = pred_np[..., 1].astype(np.float32)
                        pred_loaded_ok = True
                    except Exception as e:
                        print(f"Warning: Failed to reconstruct visit for {eye_id} at week {weeks}: {e}")
                        continue
                
                # Crop/align dimensions
                if gt_faf.shape != pred_faf.shape:
                    H_ref, W_ref = gt_faf.shape
                    H_pred, W_pred = pred_faf.shape
                    h_start = max(0, (H_ref - H_pred) // 2)
                    w_start = max(0, (W_ref - W_pred) // 2)
                    gt_faf = gt_faf[h_start:h_start + H_pred, w_start:w_start + W_pred]
                    gt_seg = gt_seg[h_start:h_start + H_pred, w_start:w_start + W_pred]
                
                # Metrics & lesion sizes: reuse the values cached during evaluation if present
                # (single source of truth, no recompute); otherwise compute them here.
                rd = recon_dict if (subject_data is not None and sub_id in subject_data
                                    and pos in subject_data[sub_id]['reconstructions']) else {}
                psnr = rd.get('psnr')
                if psnr is None:
                    psnr = psnr_metric(pred_faf, gt_faf, data_range=1.0)
                ssim = rd.get('ssim')
                if ssim is None:
                    ssim = ssim_metric(pred_faf, gt_faf, data_range=1.0)
                dice = rd.get('dice')
                if dice is None:
                    dice = compute_dice(pred_seg, gt_seg, bg_label)

                # Difference & TP/FP/FN maps
                diff_faf = _signed_diff_map(pred_faf, gt_faf)
                diff_seg = _seg_tpfpfn_map(pred_seg, gt_seg)

                _px_mm2 = self._lesion_px_area_mm2(row_dict)
                gt_area = rd.get('gt_area')
                if gt_area is None:
                    gt_area = np.sum(gt_seg > 0.5) * _px_mm2
                pred_area = rd.get('pred_area')
                if pred_area is None:
                    pred_area = np.sum(pred_seg > 0.5) * _px_mm2

                actual_visits.append({
                    'week': weeks,
                    'gt_faf': gt_faf,
                    'pred_faf': pred_faf,
                    'diff_faf': diff_faf,
                    'gt_seg': gt_seg,
                    'pred_seg': pred_seg,
                    'diff_seg': diff_seg,
                    'psnr': psnr,
                    'ssim': ssim,
                    'dice': dice,
                    'gt_border': gt_border,
                    'recon_border': recon_border,
                    'label': ("Baseline" if weeks == 0 else f"Week {weeks:.1f}") + (" [HOLD-OUT]" if is_eval else ""),
                    'status': 'holdout' if is_eval else 'actual',
                    'gt_area': gt_area,
                    'pred_area': pred_area
                })
                
            # 2. Generate interpolated and extrapolated predictions (if sequential latents active)
            interpolated_visits = []
            extrapolated_visits = []
            
            # 2. Generate interpolated and extrapolated predictions (if not independent_visits)
            interpolated_visits = []
            extrapolated_visits = []
            
            if not self.args['dataset'].get('independent_visits', False) and len(actual_visits) > 0:
                actual_weeks = [v['week'] for v in actual_visits]
                
                # Midpoints for interpolation
                pred_weeks_int = []
                for i in range(len(actual_weeks) - 1):
                    midpoint = (actual_weeks[i] + actual_weeks[i+1]) / 2.0
                    pred_weeks_int.append(midpoint)
                    
                for w in pred_weeks_int:
                    new_row = self._get_interpolated_row_dict(actual_rows, w)
                    # Check cache first to avoid redundant reconstructions
                    cache_key = (split, sub_id, round(float(w), 3))
                    if cache_key in self.reconstruction_cache:
                        pred_imgs = self.reconstruction_cache[cache_key]
                    else:
                        latent_vec = self.latents[split][sub_id:sub_id+1]
                        try:
                            volume_inf = self._reconstruct_visit(new_row, latent_vec, grid_coords, grid_shape, split=split,
                                                                 allow_extrapolation=self.args['dataset'].get('extrapolate_beyond_range', False))
                            pred_imgs = self._extract_modality_images(volume_inf)
                            self.reconstruction_cache[cache_key] = pred_imgs
                        except Exception:
                            continue
                    
                    pred_faf = pred_imgs[modalities[0]]
                    pred_seg = pred_imgs[modalities[1]] if len(modalities) > 1 else None
                        
                    blank_faf = np.zeros_like(pred_faf)
                    blank_seg = np.zeros_like(pred_seg) if pred_seg is not None else np.zeros_like(pred_faf)
                    blank_diff_faf = np.zeros((*pred_faf.shape, 3), dtype=np.float32)
                    blank_diff_seg = np.zeros((*blank_seg.shape, 3), dtype=np.float32)
                    
                    pred_area = (np.sum(pred_seg > 0.5) * self._lesion_px_area_mm2(new_row)
                                 if pred_seg is not None else 0.0)

                    interpolated_visits.append({
                        'week': w,
                        'gt_faf': blank_faf,
                        'pred_faf': pred_faf,
                        'diff_faf': blank_diff_faf,
                        'gt_seg': blank_seg,
                        'pred_seg': pred_seg if pred_seg is not None else blank_seg,
                        'diff_seg': blank_diff_seg,
                        'psnr': None,
                        'ssim': None,
                        'dice': None,
                        'gt_border': 'none',
                        'recon_border': 'solid',
                        'label': f"Week {w:.1f} (Int)",
                        'status': 'interpolated',
                        'gt_area': None,
                        'pred_area': pred_area
                    })
                    
                # Future offsets for extrapolation
                offsets_weeks = self.args['model_gen'].get('future_offsets_weeks', [12, 24, 48])
                last_week = actual_weeks[-1]
                pred_weeks_ext = [last_week + offset for offset in offsets_weeks]
                
                for w in pred_weeks_ext:
                    new_row = self._get_interpolated_row_dict(actual_rows, w)
                    # Check cache first to avoid redundant reconstructions
                    cache_key = (split, sub_id, round(float(w), 3))
                    if cache_key in self.reconstruction_cache:
                        pred_imgs = self.reconstruction_cache[cache_key]
                    else:
                        latent_vec = self.latents[split][sub_id:sub_id+1]
                        try:
                            volume_inf = self._reconstruct_visit(new_row, latent_vec, grid_coords, grid_shape, split=split,
                                                                 allow_extrapolation=self.args['dataset'].get('extrapolate_beyond_range', False))
                            pred_imgs = self._extract_modality_images(volume_inf)
                            self.reconstruction_cache[cache_key] = pred_imgs
                        except Exception:
                            continue
                    
                    pred_faf = pred_imgs[modalities[0]]
                    pred_seg = pred_imgs[modalities[1]] if len(modalities) > 1 else None
                        
                    blank_faf = np.zeros_like(pred_faf)
                    blank_seg = np.zeros_like(pred_seg) if pred_seg is not None else np.zeros_like(pred_faf)
                    blank_diff_faf = np.zeros((*pred_faf.shape, 3), dtype=np.float32)
                    blank_diff_seg = np.zeros((*blank_seg.shape, 3), dtype=np.float32)
                    
                    pred_area = (np.sum(pred_seg > 0.5) * self._lesion_px_area_mm2(new_row)
                                 if pred_seg is not None else 0.0)

                    extrapolated_visits.append({
                        'week': w,
                        'gt_faf': blank_faf,
                        'pred_faf': pred_faf,
                        'diff_faf': blank_diff_faf,
                        'gt_seg': blank_seg,
                        'pred_seg': pred_seg if pred_seg is not None else blank_seg,
                        'diff_seg': blank_diff_seg,
                        'psnr': None,
                        'ssim': None,
                        'dice': None,
                        'gt_border': 'none',
                        'recon_border': 'dashed',
                        'label': f"Week {w:.1f} (Ext)",
                        'status': 'extrapolated',
                        'gt_area': None,
                        'pred_area': pred_area
                    })

            # New-timepoint predicted GA areas (interp midpoints + extrap future) for the lesion-size
            # trajectory figure (plot_lesion_size_trajectories.py --new-csv). Same _lesion_px_area_mm2
            # convention as the observed/held-out lesion CSV -> directly comparable scale.
            for _v in interpolated_visits + extrapolated_visits:
                _kind = 'interp' if _v['status'] == 'interpolated' else 'extrap'
                _newvisit_rows.append({'Patient_Eye': eye_id, 'Weeks': float(_v['week']),
                                       'Pred_Area_mm2': _v['pred_area'], 'Kind': _kind})
                # new-visit FAF/mask image dump (no GT): pred only, gt arrays zeroed
                if _nv_dump is not None and _v.get('pred_seg') is not None:
                    _dio, _root, _meth, _scen = _nv_dump
                    _pf = np.asarray(_v['pred_faf'], dtype=np.float32)
                    _pm = (np.asarray(_v['pred_seg']) > 0.5).astype(np.uint8)
                    _dio.write_case(_root, method=_meth, scenario=_scen, eye_id=eye_id,
                                    tgt_visit=f"{_kind[:3]}{_v['week']:.1f}",
                                    pred_faf=_pf, gt_faf=np.zeros_like(_pf),
                                    pred_mask=_pm, gt_mask=np.zeros_like(_pm),
                                    weeks=float(_v['week']), pred_area_mm2=_v['pred_area'],
                                    is_extrap=(0 if _kind == 'interp' else 1),
                                    mask_source='gap_inr_seg_head')

            combined_visits = actual_visits + interpolated_visits + extrapolated_visits
            combined_visits.sort(key=lambda x: (x['week'], x['status'] != 'actual'))
            
            K = len(combined_visits)
            if K == 0:
                continue

            # Color-code each column by status so existing / hold-out / interpolated /
            # extrapolated visits are instantly distinguishable.
            status_color = {
                'actual': color,            # split color (orange=train, blue=val)
                'holdout': '#c8951a',       # gold — held-out visit (darkened for white background)
                'interpolated': '#26A69A',  # teal — interpolated (between existing visits)
                'extrapolated': '#AB47BC',  # purple — extrapolated (future)
            }
            status_style = {'actual': 'solid', 'holdout': 'solid',
                            'interpolated': 'solid', 'extrapolated': 'dashed'}
            legend_txt = ("Column color: actual = %s   |   hold-out = gold   |   "
                          "interpolated = teal   |   extrapolated = purple"
                          % ('orange' if split == 'train' else 'blue'))

            # --- 1. Plot FAF combined timeline figure ---
            fig_faf, axes_faf = plt.subplots(3, K, figsize=(3.5 * K, 9), squeeze=False)
            fig_faf.patch.set_facecolor('white')
            fig_faf.suptitle(f"FAF Timeline: {eye_id} (Split: {split}) - Epoch {epoch}", color='black', fontsize=16, fontweight='bold', y=0.96)
            
            for col_idx, v in enumerate(combined_visits):
                v_color = status_color.get(v['status'], color)
                # Row 1: GT FAF
                ax = axes_faf[0, col_idx]
                ax.imshow(v['gt_faf'], cmap='gray', vmin=0, vmax=1)
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                set_axis_border(ax, v['gt_border'], v_color)
                ax.set_title(v['label'], color=v_color, fontsize=12, fontweight='semibold')
                # GT lesion size (GA pixels x SLO spacing), same value shown on the seg figure.
                if v['gt_area'] is not None:
                    ax.set_xlabel(f"GT Size: {v['gt_area']:.2f} mm²", color=v_color, fontsize=10, fontweight='semibold')
                else:
                    ax.set_xlabel("GT Size: N/A", color=v_color, fontsize=10, fontweight='semibold')

                # Row 2: Pred FAF
                ax = axes_faf[1, col_idx]
                ax.imshow(v['pred_faf'], cmap='gray', vmin=0, vmax=1)
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                set_axis_border(ax, status_style.get(v['status'], 'solid'), v_color)
                # Predicted lesion size always shown; append PSNR/SSIM when available.
                size_txt = f"\nPred Size: {v['pred_area']:.2f} mm²" if v.get('pred_area') is not None else ""
                if v['psnr'] is not None:
                    ssim_txt = f" | SSIM {v['ssim']:.3f}" if v.get('ssim') is not None else ""
                    ax.set_xlabel(f"PSNR {v['psnr']:.1f} dB{ssim_txt}{size_txt}", color=v_color, fontsize=10, fontweight='semibold')
                else:
                    ax.set_xlabel(f"({v['status'].capitalize()}){size_txt}", color=v_color, fontsize=10, fontweight='semibold')

                # Row 3: Difference Map
                ax = axes_faf[2, col_idx]
                ax.imshow(v['diff_faf'])
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                set_axis_border(ax, 'none', v_color)

                if col_idx == 0:
                    axes_faf[0, col_idx].set_ylabel("FAF GT", color='black', fontsize=12, fontweight='bold')
                    axes_faf[1, col_idx].set_ylabel("FAF Pred", color='black', fontsize=12, fontweight='bold')
                    axes_faf[2, col_idx].set_ylabel("FAF Diff", color='black', fontsize=12, fontweight='bold')

            fig_faf.text(0.5, 0.02, legend_txt, color='black', fontsize=10, ha='center', style='italic')
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
            try:
                filepath_faf = os.path.join(pub_dir, f"subject_{eye_id}_faf{sfx}_epoch_{epoch}.png")
                plt.savefig(filepath_faf, dpi=150, bbox_inches='tight', facecolor='white')
                # Log to TensorBoard
                if tb_writer is not None:
                    fig_faf.canvas.draw()
                    rgba = fig_faf.canvas.buffer_rgba()
                    img_arr = np.asarray(rgba)[..., :3]
                    tb_writer.add_image(f"{split}/publication_figures/{eye_id}/faf{sfx}", img_arr, global_step=epoch, dataformats='HWC')
            finally:
                plt.close(fig_faf)
            
            # --- 2. Plot Segmentation combined timeline figure ---
            fig_seg, axes_seg = plt.subplots(3, K, figsize=(3.5 * K, 9), squeeze=False)
            fig_seg.patch.set_facecolor('white')
            fig_seg.suptitle(f"Segmentation Timeline: {eye_id} (Split: {split}) - Epoch {epoch}", color='black', fontsize=16, fontweight='bold', y=0.96)
            
            for col_idx, v in enumerate(combined_visits):
                v_color = status_color.get(v['status'], color)
                # Row 1: GT Seg
                ax = axes_seg[0, col_idx]
                ax.imshow(v['gt_seg'], cmap='gray', vmin=0, vmax=1)
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                set_axis_border(ax, v['gt_border'], v_color)
                ax.set_title(v['label'], color=v_color, fontsize=12, fontweight='semibold')
                if v['gt_area'] is not None:
                    ax.set_xlabel(f"GT Size: {v['gt_area']:.2f} mm²", color=v_color, fontsize=10, fontweight='semibold')
                else:
                    ax.set_xlabel("GT Size: N/A", color=v_color, fontsize=10, fontweight='semibold')

                # Row 2: Pred Seg
                ax = axes_seg[1, col_idx]
                ax.imshow(v['pred_seg'], cmap='gray', vmin=0, vmax=1)
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                set_axis_border(ax, status_style.get(v['status'], 'solid'), v_color)
                if v['dice'] is not None:
                    ax.set_xlabel(f"Dice: {v['dice']:.2f}\nPred Size: {v['pred_area']:.2f} mm²", color=v_color, fontsize=10, fontweight='semibold')
                else:
                    ax.set_xlabel(f"({v['status'].capitalize()})\nPred Size: {v['pred_area']:.2f} mm²", color=v_color, fontsize=10, fontweight='semibold')

                # Row 3: TP/FP/FN Change Map
                ax = axes_seg[2, col_idx]
                ax.imshow(v['diff_seg'])
                ax.set_facecolor('white')
                ax.set_xticks([])
                ax.set_yticks([])
                set_axis_border(ax, 'none', v_color)

                if col_idx == 0:
                    axes_seg[0, col_idx].set_ylabel("Seg GT", color='black', fontsize=12, fontweight='bold')
                    axes_seg[1, col_idx].set_ylabel("Seg Pred", color='black', fontsize=12, fontweight='bold')
                    axes_seg[2, col_idx].set_ylabel("TP/FP/FN Map", color='black', fontsize=12, fontweight='bold')

            fig_seg.text(0.5, 0.02, legend_txt, color='black', fontsize=10, ha='center', style='italic')
            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
            try:
                filepath_seg = os.path.join(pub_dir, f"subject_{eye_id}_seg{sfx}_epoch_{epoch}.png")
                plt.savefig(filepath_seg, dpi=150, bbox_inches='tight', facecolor='white')
                # Log to TensorBoard
                if tb_writer is not None:
                    fig_seg.canvas.draw()
                    rgba = fig_seg.canvas.buffer_rgba()
                    img_arr = np.asarray(rgba)[..., :3]
                    tb_writer.add_image(f"{split}/publication_figures/{eye_id}/seg{sfx}", img_arr, global_step=epoch, dataformats='HWC')
            finally:
                plt.close(fig_seg)
            
        print(f"Decoupled FAF and Segmentation timeline figures generated for {split} split in reconstructions/publication_figures/")

        # Export the new-timepoint predicted areas in the plot_lesion_size_trajectories.py --new-csv
        # schema (Patient_Eye, Weeks, Pred_Area_mm2, Kind). Empty for the per-visit-latent model
        # (independent_visits), which has no weeks-conditioned new visits.
        if _newvisit_rows:
            _nv_dir = os.path.join(self.args['output_dir'], 'lesion_analysis')
            os.makedirs(_nv_dir, exist_ok=True)
            _nv_path = os.path.join(_nv_dir, f"lesion_areas_newvisits_{split}{tag_suffix}_epoch_{epoch}.csv")
            pd.DataFrame(_newvisit_rows).to_csv(_nv_path, index=False)
            print(f"Wrote {len(_newvisit_rows)} new-visit predicted areas -> {_nv_path}")

    @staticmethod
    def _mean_dice(metrics_list):
        """Mean DICE over a list of per-visit metrics dicts (each {'DICE': [vals]}), or None."""
        vals = [d for m in metrics_list for d in (m.get('DICE') or [])]
        return float(np.mean(vals)) if vals else None

    def _run_validation_round(self, epoch_train, tb_writer, grid_coords, grid_shape,
                              picked_val_subs, holdout_position=None, tag_suffix='holdout_last',
                              split='val', support_k=None, pair_source=None, pair_target=None):
        """
        Run a single test-time-optimisation round on `split` ('val' or 'test'): re-init latents,
        optimize on non-held-out visits, evaluate on both optimization and held-out visits.

        Args:
            holdout_position: 1-indexed visit to hold out (None = last).
            tag_suffix:       string appended to TensorBoard/metric tags (e.g. 'holdout_V2').
            split:            'val' (default) or 'test'. For 'test', single-visit patient-eyes
                              are optimised on (not held out), mirroring clinical deployment.

        Returns:
            Mean DICE over the held-out (eval) visits for this round, or None if there are no
            held-out visits / no segmentation. Used by validate() to pick the best checkpoint.
        """
        # Re-initialize latents from scratch
        self._init_validation(split=split)

        # ---- Frozen test-time latents (reproducibility) --------------------------------------------
        # The latent TTO uses grid_sample, whose backward has no deterministic CUDA kernel, so re-runs
        # drift slightly. To make the reported numbers a FIXED, citable artifact: run the TTO ONCE and
        # save the optimised per-round latents; on any later eval, RELOAD them and skip the TTO (pure
        # rendering is deterministic). Keyed by split + round tag so pairwise/LOO/last never collide.
        _lat_mode = self.args.get('test_latents_mode')          # 'save' | 'load' | None
        _lat_path = self.args.get('test_latents_path')
        if _lat_mode and not hasattr(self, '_frozen_latents'):
            self._frozen_latents = (torch.load(_lat_path, map_location='cpu')
                                    if (_lat_mode == 'load' and _lat_path and os.path.exists(_lat_path))
                                    else {})
        _lat_key = f"{split}:{tag_suffix}"
        _loaded_frozen = False
        if _lat_mode == 'load' and _lat_key in getattr(self, '_frozen_latents', {}):
            self.latents[split].data.copy_(self._frozen_latents[_lat_key].to(self.latents[split].device))
            _loaded_frozen = True
            print(f"[{split} {tag_suffix}] Loaded FROZEN latents -> skipping TTO (deterministic re-eval).")

        # Split visits into optimization and evaluation sets. On the test set, a patient-eye with
        # a single visit is optimised on (clinical case) rather than held out for evaluation.
        single_visit_to = 'opt' if split == 'test' else 'eval'
        opt_idcs, eval_idcs = self.datasets[split].get_longitudinal_indices(
            holdout_position=holdout_position, single_visit_to=single_visit_to, support_k=support_k,
            pair_source=pair_source, pair_target=pair_target)
        self.current_val_opt_idcs = opt_idcs
        self.current_val_eval_idcs = eval_idcs
        # Per-visit held-out metrics of THIS round (set below); consumed by the leave_one_out loop
        # to build the averaged-across-positions table. None if there are no held-out visits.
        self._last_round_eval_metrics = None

        ho_label = "none" if holdout_position == 'none' else (f"visit {holdout_position}" if holdout_position else "last visit")
        print(f"[{split} {tag_suffix}] Optimising on {len(opt_idcs)} visits, evaluating on {len(eval_idcs)} (held-out: {ho_label})")

        # 1. Optimise latents on non-held-out visits
        orig_loader = self.dataloaders[split]
        self.dataloaders[split] = self.create_subset_dataloader(split, opt_idcs)

        sub_writer = None
        if tb_writer is not None:
            from torch.utils.tensorboard import SummaryWriter
            sub_log_dir = os.path.join(self.args['output_dir'], 'tb_logs', f'{split}_inner_ep{epoch_train}')
            sub_writer = SummaryWriter(log_dir=sub_log_dir)

        # ---- Early stopping on the optimisation-visit DICE ----
        # The segmentation DICE on the optimisation visits peaks and then degrades as the latents
        # overfit the image intensities -- most sharply under seg_loss_val: false, where the seg
        # head gets no gradient here at all. We monitor that DICE every inner epoch and keep the
        # latents from the best inner epoch, restoring them before the final evaluation.
        es_cfg = self.args['optimizer'].get('val_early_stopping', {}) or {}
        es_active = es_cfg.get('activate', False)
        es_patience = es_cfg.get('patience', 50)
        es_min_delta = es_cfg.get('min_delta', 0.0)
        best_dice = -float('inf')
        best_epoch = -1
        best_latents = None
        epochs_no_improve = 0

        last_val_loss_components = None
        epoch_val = -1
        for epoch_val in range(0 if _loaded_frozen else self.args['epochs']['val']):
            _, loss_components = self.train_epoch(split=split, epoch=epoch_val, epoch_train=epoch_train, sub_writer=sub_writer)
            self._update_scheduler(split=split)
            opt_dice = self._log_inner_val_convergence(epoch_train, epoch_val, picked_val_subs, opt_idcs, eval_idcs, grid_coords, grid_shape, split=split, tag_suffix=tag_suffix)
            last_val_loss_components = loss_components

            # Track the best (pre-overfit) latents and stop once the DICE has stalled.
            if es_active and opt_dice is not None:
                if opt_dice > best_dice + es_min_delta:
                    best_dice = opt_dice
                    best_epoch = epoch_val
                    best_latents = self.latents[split].detach().clone()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= es_patience:
                        print(f"[{split} {tag_suffix}] Early stopping at inner epoch {epoch_val}: opt DICE "
                              f"has not improved for {es_patience} epochs (best {best_dice:.4f} @ epoch {best_epoch}).")
                        break

        # Restore the latents from the best inner epoch (skip if the last epoch was already best).
        if es_active and best_latents is not None and best_epoch != epoch_val:
            self.latents[split].data.copy_(best_latents)
            print(f"[{split} {tag_suffix}] Restored {split} latents from inner epoch {best_epoch} "
                  f"(opt DICE {best_dice:.4f}).")
        if tb_writer is not None and es_active and best_epoch >= 0:
            tb_writer.add_scalar(f"{split}_inner_opt/{tag_suffix}/best_epoch", best_epoch, epoch_train)
            tb_writer.add_scalar(f"{split}_inner_opt/{tag_suffix}/best_dice", best_dice, epoch_train)

        # ---- Frozen test-time latents: persist THIS round's optimised latents for reproducible re-eval.
        if _lat_mode == 'save' and not _loaded_frozen:
            self._frozen_latents[_lat_key] = self.latents[split].detach().cpu().clone()
            if _lat_path:
                torch.save(self._frozen_latents, _lat_path)   # rewrite the full bank after each round (robust to any exit path)

        self.dataloaders[split] = orig_loader

        if sub_writer is not None:
            sub_writer.close()

        # Log final optimisation loss at training epoch step
        if last_val_loss_components is not None:
            log_loss(last_val_loss_components, epoch_train, split=f'{split}_opt_{tag_suffix}', log=self.args['logging'], tb_writer=tb_writer)

        # Clear the reconstruction cache before final evaluation of this round,
        # so that evaluations and novel visit figures use the newly optimized latents.
        self.reconstruction_cache = {}

        # 2. Evaluate on optimisation visits
        metrics_opt, opt_data = self._evaluate_visits(
            opt_idcs, split, grid_coords, grid_shape, epoch=epoch_train, tb_writer=tb_writer
        )
        if split in ('val', 'test'):
            if getattr(self, '_eval_sets', None) is None:
                self._eval_sets = {}
            self._eval_sets[f'{split}_opt'] = opt_data
        log_metrics(self.args, metrics_opt, epoch_train, split=f'{split}_opt_{tag_suffix}', tb_writer=tb_writer)
        self._log_reconstruction_figures(opt_data, split, epoch_train, tb_writer, tensorboard_tag=f'{split}_opt_{tag_suffix}')
        # Static-segmentation dump: every OBSERVED visit (opt_data) for the Part-2 comparison vs
        # NISF/MetaSeg. No-op unless args['comparison_dump_static']['enable'] (evaluate.py --dump_static_root).
        self._maybe_dump_static(opt_data, split)

        # 3. Evaluate on held-out visits
        eval_data = None
        eval_dice = None
        if eval_idcs:
            metrics_eval, eval_data = self._evaluate_visits(
                eval_idcs, split, grid_coords, grid_shape, epoch=epoch_train, tb_writer=tb_writer
            )
            eval_dice = self._mean_dice(metrics_eval)
            self._last_round_eval_metrics = metrics_eval
            if split in ('val', 'test'):
                if getattr(self, '_eval_sets', None) is None:
                    self._eval_sets = {}
                self._eval_sets[f'{split}_eval'] = eval_data
            log_metrics(self.args, metrics_eval, epoch_train, split=f'{split}_eval_{tag_suffix}', tb_writer=tb_writer)
            self._log_reconstruction_figures(eval_data, split, epoch_train, tb_writer, tensorboard_tag=f'{split}_eval_{tag_suffix}')
            # Cross-model comparison dump (held-out target visit only). No-op unless
            # args['comparison_dump']['enable'] is set (by evaluate.py --dump_root).
            self._maybe_dump_comparison(eval_data, split)

            # Held-out reconstruction loss (generalization): computed with the optimized,
            # frozen latents, same seg_weight as the opt loss so the curves are comparable.
            eval_loss = self._compute_loss_on_visits(eval_idcs, split=split)
            if eval_loss is not None:
                log_loss(eval_loss, epoch_train, split=f'{split}_eval_{tag_suffix}',
                         log=self.args['logging'], tb_writer=tb_writer)

        # Generate and log publication-ready summary figure
        if self.args['dataset'].get('dataset_name') == 'faf_ga':
            val_data_combined = {}
            if opt_data:
                val_data_combined.update(opt_data)
            if eval_data:
                for sub_id, data in eval_data.items():
                    if sub_id not in val_data_combined:
                        val_data_combined[sub_id] = data
                    else:
                        val_data_combined[sub_id]['reconstructions'].update(data['reconstructions'])
            self.log_publication_reconstruction_figure(
                epoch_train, split, picked_val_subs, opt_idcs=opt_idcs, eval_idcs=eval_idcs,
                subject_data=val_data_combined, tag_suffix=tag_suffix
            )

        return eval_dice



    def _maybe_dump_comparison(self, eval_data, split):
        """Write one cross-model comparison .npz per held-out eye (target = its latest held-out
        visit) to the shared dump tree consumed by models/comparison/make_comparison_figure.py.

        No-op unless args['comparison_dump']['enable'] is True and split matches. The scenario +
        method key are set by the caller (evaluate.py): scenario='full'/method='gap_inr' for the
        all-(n-1)-visits LOO-last run, scenario='matched'/method='gap_inr_k1' for the --support_k 1
        (baseline-only) run. Arrays come straight from the reconstruction record GAP-INR already
        built (pred/gt FAF + seg on its sampling grid; faf_ga_512 => native 512, comparable to the
        ImageFlowNet-family dumps). DICE/PSNR/SSIM/areas are reused; HD is computed here."""
        cd = self.args.get('comparison_dump') or {}
        if not cd.get('enable') or not eval_data or split != cd.get('split', 'test'):
            return
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'comparison'))
            import dump_io
            from skimage.metrics import hausdorff_distance
        except Exception as e:
            print(f"[comparison_dump] skipped (import failed): {e}")
            return
        root = cd['root']
        scenario = cd.get('scenario', 'full')
        method = cd.get('method', 'gap_inr')
        n = 0
        for sub_id, data in eval_data.items():
            recs = data.get('reconstructions', {})
            if not recs:
                continue
            eye_id = str(data.get('eye_id', sub_id))
            latest = max(recs.keys(), key=lambda k: recs[k].get('weeks', 0.0) or 0.0)
            # 'interp' -> dump EVERY held-out visit, keyed by its own visit index (interior visits for
            # the interpolation comparison + trajectory figure; keying by visit avoids the 'last'
            # collision when leave_one_out holds out each visit in a separate round). Extrapolation
            # scenarios -> single target = the latest held-out visit (== the eye's last visit).
            vks = list(recs.keys()) if scenario == 'interp' else [latest]
            for vk in vks:
                r = recs[vk]
                if r.get('pred_faf') is None or r.get('pred_seg') is None or r.get('gt_seg') is None:
                    continue
                pm = (np.asarray(r['pred_seg']) > 0.5).astype(np.uint8)
                gm = (np.asarray(r['gt_seg']) > 0.5).astype(np.uint8)
                if pm.sum() == 0 and gm.sum() == 0:
                    hd = 0.0
                elif pm.sum() == 0 or gm.sum() == 0:
                    hd = float(np.hypot(*pm.shape))
                else:
                    hd = float(hausdorff_distance(pm, gm))
                tgt = str(vk) if scenario == 'interp' else 'last'
                dump_io.write_case(
                    root, method=method, scenario=scenario, eye_id=eye_id, tgt_visit=tgt,
                    pred_faf=r['pred_faf'], gt_faf=r['gt_faf'], pred_mask=pm, gt_mask=gm,
                    src_visit=None, weeks=r.get('weeks'), dice=r.get('dice'), hd=hd,
                    psnr=r.get('psnr'), ssim=r.get('ssim'),
                    gt_area_mm2=r.get('gt_area'), pred_area_mm2=r.get('pred_area'),
                    is_extrap=(0 if (scenario == 'interp' and vk != latest) else 1),
                    mask_source='gap_inr_seg_head')
                n += 1
        print(f"[comparison_dump] wrote {n} GAP-INR case(s) -> {os.path.join(root, scenario, method)}")

    def _maybe_dump_static(self, opt_data, split):
        """Static-segmentation dump (Part 2): one case per OBSERVED visit, keyed by Visit_Number, so
        GAP-INR can be compared to the static baselines (NISF/MetaSeg) on the segmentation-of-a-seen-
        visit task. method='gap_inr_pervisit' (independent_visits) or 'gap_inr_perpatient' (one latent
        per eye) -- set by evaluate.py. No-op unless args['comparison_dump_static']['enable']."""
        cd = self.args.get('comparison_dump_static') or {}
        if not cd.get('enable') or not opt_data or split != cd.get('split', 'test'):
            return
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'comparison'))
            import dump_io
            from skimage.metrics import hausdorff_distance
        except Exception as e:
            print(f"[comparison_dump_static] skipped (import failed): {e}")
            return
        root, method = cd['root'], cd.get('method', 'gap_inr_pervisit')
        n = 0
        for sub_id, data in opt_data.items():
            eye_id = str(data.get('eye_id', sub_id))
            for _vk, r in (data.get('reconstructions', {}) or {}).items():
                if r.get('pred_faf') is None or r.get('pred_seg') is None or r.get('gt_seg') is None:
                    continue
                vis = r.get('visit')
                if vis is None:
                    continue
                pm = (np.asarray(r['pred_seg']) > 0.5).astype(np.uint8)
                gm = (np.asarray(r['gt_seg']) > 0.5).astype(np.uint8)
                if pm.sum() == 0 and gm.sum() == 0:
                    hd = 0.0
                elif pm.sum() == 0 or gm.sum() == 0:
                    hd = float(np.hypot(*pm.shape))
                else:
                    hd = float(hausdorff_distance(pm, gm))
                dump_io.write_case(
                    root, method=method, scenario='static', eye_id=eye_id, tgt_visit=str(vis),
                    pred_faf=r['pred_faf'], gt_faf=r['gt_faf'], pred_mask=pm, gt_mask=gm,
                    weeks=r.get('weeks'), dice=r.get('dice'), hd=hd, psnr=r.get('psnr'),
                    ssim=r.get('ssim'), gt_area_mm2=r.get('gt_area'), pred_area_mm2=r.get('pred_area'),
                    mask_source='gap_inr_seg_head')
                n += 1
        print(f"[comparison_dump_static] wrote {n} GAP-INR observed-visit case(s) -> "
              f"{os.path.join(root, 'static', method)}")

    def _compute_loss_on_visits(self, visit_idcs, split='val'):
        """Average reconstruction/segmentation loss over the given visits using the CURRENT
        (optimized, frozen) latents. No gradients. Used to log a held-out validation loss.

        Uses the same seg_weight policy as the optimization loss (seg off during val unless
        seg_loss_val), so val_eval_* and val_opt_* loss curves are directly comparable.
        Returns an averaged loss-components dict {total, sr, seg, trafo} or None.
        """
        if not visit_idcs:
            return None
        if split in ('val', 'test') and not self.args['optimizer'].get('seg_loss_val', True):
            seg_weight = 0.0
        else:
            seg_weight = self.args['optimizer']['seg_weight']

        loader = self.create_subset_dataloader(split, visit_idcs)
        n_smpls = self.args['n_samples']
        accum = {}
        self.inr_decoder[split].eval()
        with torch.no_grad():
            for batch in loader:
                coords_b, values_b, conditions_b, idx_b, time_b = to_device(batch, device=self.device)
                for s in range(0, idx_b.shape[0], n_smpls):
                    coords = coords_b[s:s + n_smpls]
                    values = values_b[s:s + n_smpls]
                    idx_df = idx_b[s:s + n_smpls].squeeze()
                    conditions = conditions_b[s:s + n_smpls]
                    time_vals = time_b[s:s + n_smpls]
                    values_p = self.inr_decoder[split](coords, self.latents[split], conditions,
                                                       idcs_df=idx_df, time_vals=time_vals)
                    loss = self.loss_criterion(values_p, values, seg_weight=seg_weight)
                    for k, v in loss.items():
                        accum.setdefault(k, []).append(v.item() if hasattr(v, 'item') else v)
        return {k: float(np.mean(vals)) for k, vals in accum.items()} if accum else None

    def _is_augmented_eye(self, row_or_id):
        """True for pseudo-eye augmentation rows/ids (train-only).

        These MUST be excluded from all evaluation, metrics, lesion analysis and figures: the
        prediction comes from the pseudo-eye's latent and therefore lives in the augmented
        (warped + intensity-jittered) frame, but the on-disk ground truth is the ORIGINAL,
        un-augmented image/mask. Comparing the two is meaningless (it was producing DICE ~0.5
        and garbage publication figures for `*__aug*` eyes). Pseudo-eyes exist only to regularise
        the decoder during training; we never report their reconstruction quality.
        """
        id_col = self.args['dataset'].get('id_column', 'subject_id')
        if isinstance(row_or_id, dict):
            try:
                if int(row_or_id.get('aug_id', 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                pass
            return '__aug' in str(row_or_id.get(id_col, row_or_id.get('Eye_ID', '')))
        return '__aug' in str(row_or_id)

    def _real_eye_subs(self, df):
        """Sorted unique sub_id_int of the NON-augmented (real) patient-eyes in `df`."""
        mask = pd.Series(True, index=df.index)
        if 'aug_id' in df.columns:
            mask &= df['aug_id'].fillna(0).astype(int) == 0
        id_col = self.args['dataset'].get('id_column', 'subject_id')
        if id_col in df.columns:
            mask &= ~df[id_col].astype(str).str.contains('__aug', na=False)
        return sorted(df.loc[mask, 'sub_id_int'].unique())

    def _evaluate_visits(self, visit_indices, split, grid_coords, grid_shape, epoch=0, tb_writer=None):
        """Reconstruct visits, compute metrics, collect images. Returns (metrics, subject_data)."""
        # self.inr_decoder[split].eval()
        metrics = []
        subject_data = {} # sub_id -> {eye_id: str, mods: {mod -> {visits: [], preds: [], refs: [], diff_intra: [], diff_long: []}}}
        baseline_cache = {}

        for visit_idx in tqdm(visit_indices, desc=f"Eval [{split}]", leave=False):
            # For each individual visit of the same patient-eye
            df_row_dict = self.datasets[split].df.iloc[visit_idx].to_dict()  # we take the corresponding row in the dataset

            # Skip pseudo-eye augmentation rows: their GT on disk is un-augmented while the
            # prediction is in the augmented frame, so any metric/figure would be wrong.
            if self._is_augmented_eye(df_row_dict):
                continue
            for mod in self.args['dataset']['modalities']:
                df_row_dict[mod] = self.datasets[split].resolve_path(df_row_dict, mod)  # we save the paths of all image modalities for that visit

            sub_id_int = int(df_row_dict['sub_id_int'])  # patient-eye identifier (unique for each patient-eye)
            eye_id = str(df_row_dict.get(self.args['dataset'].get('id_column', 'subject_id'), 'unknown'))  # eye identifier (left or right)

            # Reconstruct current follow-up visit
            latent_idx = visit_idx if self.args['dataset'].get('independent_visits', False) else sub_id_int
            volume_inf = self._reconstruct_visit(
                df_row_dict, latent_idx, grid_coords, grid_shape, split=split
            )

            # Retrieve or compute baseline visit reconstruction
            if sub_id_int not in baseline_cache:
                baseline_row = (
                    self.datasets[split].df[self.datasets[split].df['sub_id_int'] == sub_id_int]
                    .sort_values(self._temporal_key)
                    .iloc[0]
                )
                baseline_idx = self.datasets[split].df.index.get_loc(baseline_row.name)  # absolute row number of this baseline visit in the static DataFrame
                baseline_row_dict = baseline_row.to_dict()  # convert baseline visit to dictionary
                # iterate over the modalities to resolve their file paths
                for mod in self.args['dataset']['modalities']:
                    baseline_row_dict[mod] = self.datasets[split].resolve_path(baseline_row_dict, mod)
                
                # reconstruct baseline visit
                baseline_latent_idx = baseline_idx if self.args['dataset'].get('independent_visits', False) else sub_id_int
                baseline_inf = self._reconstruct_visit(
                    baseline_row_dict, baseline_latent_idx, grid_coords, grid_shape, split=split
                )
                baseline_cache[sub_id_int] = (baseline_inf, baseline_row)  # store reconstructed baseline and metadata row
            else:
                # any subsequent visit of this patient-eye processed in the loop will hit the else block and immediately retrieve the baseline reconstruction without running inference again
                baseline_inf, baseline_row = baseline_cache[sub_id_int]  

            patient_stats = self._get_patient_stats(split, sub_id_int)  # get patient-eye statistics to have a min-max normalisation based on pixel intensities of ALL visits of that patient-eye

            if self.args['compute_metrics']:
                res_metrics, res_imgs = compute_metrics(
                    self.args, volume_inf, df_row_dict, epoch, split,  # volume_inf is the predicted current follow-up visit
                    tb_writer=tb_writer, 
                    baseline_volume=baseline_inf, # predicted baseline visit
                    gt_baseline_row=baseline_row,  # ground truth baseline visit
                    return_images=True,
                    patient_stats=patient_stats
                )
                metrics.append(res_metrics)
                
                # Group for tiling
                if sub_id_int not in subject_data:
                    subject_data[sub_id_int] = {'eye_id': eye_id, 'mods': {}, 'reconstructions': {}}
                
                visit_label = f"V{df_row_dict.get('Visit_Number', df_row_dict.get('Visit', '0'))}"

                # --- Per-visit scalars computed ONCE and reused by every figure (single source of truth) ---
                modalities = self.args['dataset']['modalities']
                pred_seg = res_imgs[modalities[1]].get('pred') if len(modalities) > 1 else None
                gt_seg = res_imgs[modalities[1]].get('ref') if len(modalities) > 1 else None

                # Lesion sizes in mm^2 = (#GA pixels on the sampling grid) * ScaleXSlo * ScaleYSlo * rf.
                # ScaleXSlo/ScaleYSlo are mm/px at NATIVE resolution. faf_ga_620 samples the native-pitch
                # 620 grid (sampling_bbox=[620,620], spacing 1.0) -> rf=1. faf_ga_512 center-crops 620
                # (native pitch, no GA clip) then DOWNSAMPLES to 512, so each grid pixel covers
                # (620/512)^2 more physical area -> rf=(crop/world)^2 restores the true mm^2. Every
                # 512-scored baseline (NISF/MetaSeg/ImageFlowNet) applies the SAME factor -> comparable.
                px_area = self._lesion_px_area_mm2(df_row_dict)
                pred_area = float(np.sum(pred_seg > 0.5) * px_area) if pred_seg is not None else None
                gt_area = float(np.sum(gt_seg > 0.5) * px_area) if gt_seg is not None else None

                # Carry the per-visit GA area on the metrics dict (single-element lists, matching the
                # other metric values) so it is written into the per-hold-out-position JSON. This lets
                # summarize_eval bucket area-MAE into interpolation/extrapolation exactly like DICE --
                # the pooled lesion CSV only survives the LAST leave-one-out round, so it cannot.
                res_metrics['GT_Area_mm2'] = [gt_area if gt_area is not None else float('nan')]
                res_metrics['Pred_Area_mm2'] = [pred_area if pred_area is not None else float('nan')]

                def _first(lst):
                    return float(lst[0]) if isinstance(lst, (list, tuple)) and len(lst) > 0 else None
                v_psnr = _first(res_metrics.get('PSNR'))
                v_ssim = _first(res_metrics.get('SSIM'))
                v_dice = _first(res_metrics.get('DICE'))
                v_hd = _first(res_metrics.get('HD'))
                v_lpips = _first(res_metrics.get('LPIPS'))

                # Standardized cross-method monitor panel (identical layout/colormaps to gliomagrowth/
                # IFN/NISF + the Phase-4 comparison figure), logged to TB per held-out eye.
                if tb_writer is not None and pred_seg is not None and len(modalities) > 0:
                    try:
                        import sys as _sys, os as _os
                        _comp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                                              "comparison")
                        if _comp not in _sys.path:
                            _sys.path.insert(0, _comp)
                        import monitor_panel
                        _fp = res_imgs[modalities[0]].get('pred')
                        _fg = res_imgs[modalities[0]].get('ref')
                        if _fp is not None and _fg is not None:
                            monitor_panel.log_panel(
                                tb_writer, f"{split}/monitor/{eye_id}/{visit_label}", epoch,
                                _fg, _fp, gt_seg, pred_seg, title=f"GAP-INR {eye_id} {visit_label}",
                                psnr=v_psnr, ssim=v_ssim, dice=v_dice, mask_note="real GT")
                    except Exception as _e:
                        print(f"[monitor_panel] GAP-INR skip: {_e}")

                for mod, imgs in res_imgs.items():
                    if mod not in subject_data[sub_id_int]['mods']:
                        subject_data[sub_id_int]['mods'][mod] = {
                            'visits': [],
                            'preds': [],
                            'refs': [],
                            'diff_intra': [],  # difference map between predicted follow-up and GT follow-up
                            'diff_long': [],  # difference map between predicted follow-up and predicted baseline
                            'gt_long_diff': [],  # difference map between predicted follow-up and GT baseline
                            'gt_gt_diff': [],   # difference map between GT follow-up and GT baseline
                            # per-visit scalars aligned with the lists above (for figure annotations)
                            'psnr': [], 'ssim': [], 'dice': [], 'pred_area': [], 'gt_area': []
                        }
                    subject_data[sub_id_int]['mods'][mod]['visits'].append(visit_label)
                    subject_data[sub_id_int]['mods'][mod]['preds'].append(imgs.get('pred'))
                    subject_data[sub_id_int]['mods'][mod]['refs'].append(imgs.get('ref'))
                    subject_data[sub_id_int]['mods'][mod]['diff_intra'].append(imgs.get('diff_intra'))
                    subject_data[sub_id_int]['mods'][mod]['psnr'].append(v_psnr)
                    subject_data[sub_id_int]['mods'][mod]['ssim'].append(v_ssim)
                    subject_data[sub_id_int]['mods'][mod]['dice'].append(v_dice)
                    subject_data[sub_id_int]['mods'][mod]['pred_area'].append(pred_area)
                    subject_data[sub_id_int]['mods'][mod]['gt_area'].append(gt_area)

                    diff_long_val = imgs.get('diff_long', imgs.get('change_long', None))
                    if diff_long_val is not None:
                        subject_data[sub_id_int]['mods'][mod]['diff_long'].append(diff_long_val)

                    gt_long_val = imgs.get('gt_long_diff', imgs.get('gt_long_change', None))
                    if gt_long_val is not None:
                        subject_data[sub_id_int]['mods'][mod]['gt_long_diff'].append(gt_long_val)

                    gt_gt_val = imgs.get('gt_gt_diff', imgs.get('gt_gt_change', None))
                    if gt_gt_val is not None:
                        subject_data[sub_id_int]['mods'][mod]['gt_gt_diff'].append(gt_gt_val)

                subject_data[sub_id_int]['reconstructions'][visit_idx] = {
                    'pred_faf': res_imgs[modalities[0]].get('pred'),
                    'gt_faf': res_imgs[modalities[0]].get('ref'),
                    'pred_seg': pred_seg,
                    'gt_seg': gt_seg,
                    'pred_area': pred_area,
                    'gt_area': gt_area,
                    'weeks': float(df_row_dict.get('weeks_from_baseline', 0.0)),
                    'visit': df_row_dict.get('Visit_Number', df_row_dict.get('Visit')),
                    'psnr': v_psnr,
                    'ssim': v_ssim,
                    'dice': v_dice,
                    'hd': v_hd,
                    'lpips': v_lpips,
                }

            elif self.args['save_imgs'][split]:
                save_subject(self.args, volume_inf, df_row_dict, epoch=epoch, split=split,
                             tb_writer=None, baseline_volume=baseline_inf)

        return metrics, subject_data

    def _reconstruct_visit(self, row_dict, sub_id_int, grid_coords, grid_shape, split='train', step_size=None,
                           allow_extrapolation=False):
        """
        Helper to reconstruct a single visit for a given subject.
        Loads conditions, time_val, and retrieves the latent and transformation,
        then runs INR decoder inference.
        Returns volume_inf.

        allow_extrapolation=True lets the time/condition inputs exceed the observed training range
        (used for future/extrapolated novel-visit generation; see Data.load_time/load_conditions).
        """
        with torch.no_grad():
            conditions = self.datasets[split].load_conditions(row_dict, allow_extrapolation=allow_extrapolation).to(self.device)
            time_val = self.datasets[split].load_time(row_dict, allow_extrapolation=allow_extrapolation).to(self.device) if self.args['inr_decoder'].get('time_as_input', False) else None
            
            if isinstance(sub_id_int, torch.Tensor):
                latent_vec = sub_id_int
            else:
                latent_vec = self.latents[split][sub_id_int:sub_id_int + 1]
            

            inf_kwargs = {'time_val': time_val}
            if step_size is not None:
                inf_kwargs['step_size'] = step_size

            volume_inf = self.inr_decoder[split].inference(
                grid_coords,
                latent_vec,
                conditions,
                grid_shape,
                **inf_kwargs
            )
        return volume_inf

    def _get_patient_stats(self, split, sub_id_int):
        """Retrieve patient-level min/max stats for normalisation."""
        dataset = self.datasets[split]
        if hasattr(dataset, 'patient_stats') and sub_id_int in dataset.patient_stats:
            p_stats = dataset.patient_stats[sub_id_int]
            return {
                'min': p_stats['min'],
                'max': p_stats['max']
            }
        return None

    def _log_reconstruction_figures(self, subject_data, split, epoch, tb_writer, tensorboard_tag=''):
        """Logs tiled reconstruction and difference figures to TensorBoard and saves them locally."""
        if not subject_data:
            return

        tiled_dir = os.path.join(self.args['output_dir'], "reconstructions", "tiled_figures")
        os.makedirs(tiled_dir, exist_ok=True)

        modalities = self.args['dataset']['modalities']
        has_seg = self.args['inr_decoder']['out_dim'][-1] > 0

        def _fmt(v, suf=''):
            return f"{v:.2f}{suf}" if v is not None else "n/a"

        for sub_id, data in subject_data.items():
            eye_id = data['eye_id']
            for mod, content in data['mods'].items():
                if not content['preds']:
                    continue
                is_seg_mod = has_seg and (mod == modalities[-1])

                # FAF / Seg Tiled (Pred vs Ref) --> predicted visit vs GT visit
                fig = make_longitudinal_tiled_figure(
                    content['preds'], content['refs'], content['visits'],
                    row1_name="Model Prediction", row2_name="Ground Truth",
                    title=f"{eye_id} - {mod} - Longitudinal Reconstruction"
                )
                try:
                    filepath_long = os.path.join(tiled_dir, f"subject_{eye_id}_{mod}_Longitudinal_epoch_{epoch}.png")
                    fig.savefig(filepath_long, dpi=150, bbox_inches='tight')
                finally:
                    plt.close(fig)

                # Difference Maps Tiled --> diff map between predicted visit and GT visit (intra-visit error)
                if content['diff_long']:
                    fig_diff = make_longitudinal_tiled_figure(
                        content['diff_intra'], content['diff_long'], content['visits'],
                        row1_name="Intra-visit Error", row2_name="Temporal Diff (vs Baseline)",
                        title=f"{eye_id} - {mod} - Longitudinal Dynamics"
                    )
                    try:
                        filepath_diff = os.path.join(tiled_dir, f"subject_{eye_id}_{mod}_Differences_epoch_{epoch}.png")
                        fig_diff.savefig(filepath_diff, dpi=150, bbox_inches='tight')
                    finally:
                        plt.close(fig_diff)

                # GT Baseline Difference Maps Tiled --> row 1: pred follow-up vs GT baseline,
                # row 2: GT follow-up vs GT baseline. Annotated with metrics + lesion sizes.
                if content.get('gt_long_diff') and content.get('gt_gt_diff'):
                    n = min(len(content['gt_long_diff']), len(content['gt_gt_diff']), len(content['visits']))
                    if is_seg_mod:
                        row1_sub = [f"Dice {_fmt(content['dice'][i])} | Pred {_fmt(content['pred_area'][i], ' mm²')}"
                                    for i in range(n)]
                        row2_sub = [f"GT {_fmt(content['gt_area'][i], ' mm²')}" for i in range(n)]
                    else:
                        row1_sub = [f"PSNR {_fmt(content['psnr'][i], ' dB')} | SSIM {_fmt(content['ssim'][i])}"
                                    for i in range(n)]
                        row2_sub = [None] * n
                    fig_gt_diff = make_longitudinal_tiled_figure(
                        content['gt_long_diff'][:n], content['gt_gt_diff'][:n], content['visits'][:n],
                        row1_name="Pred Follow-up vs GT Baseline", row2_name="GT Follow-up vs GT Baseline",
                        title=f"{eye_id} - {mod} - GT Baseline Dynamics",
                        row1_sublabels=row1_sub, row2_sublabels=row2_sub
                    )
                    try:
                        filepath_gt_diff = os.path.join(tiled_dir, f"subject_{eye_id}_{mod}_GT_Baseline_Differences_epoch_{epoch}.png")
                        fig_gt_diff.savefig(filepath_gt_diff, dpi=150, bbox_inches='tight')
                        # Log to TensorBoard: row1 = Pred follow-up vs GT baseline, row2 = GT follow-up
                        # vs GT baseline, per modality (FAF + seg). Replaces the former
                        # Future/Interpolation "Differences" images.
                        if tb_writer is not None:
                            tag_gt_diff = f"{split}/{eye_id}/{mod}/GT_Baseline_Differences_{tensorboard_tag}"
                            tb_writer.add_image(tag_gt_diff, fig_to_numpy(fig_gt_diff), epoch, dataformats='HWC')
                    finally:
                        plt.close(fig_gt_diff)

    def _extract_modality_images(self, volume_inf):
        pred = typecheck_img(volume_inf)
        modalities = self.args['dataset']['modalities']
        has_seg = self.args['inr_decoder']['out_dim'][-1] > 0
        img_dict = {}
        # Get one of the decoders to retrieve sr_dims
        decoder = self.inr_decoder['train'] if 'train' in self.inr_decoder else self.inr_decoder['val']
        for i, mod in enumerate(modalities):
            is_seg = has_seg and (i == len(modalities) - 1)
            if is_seg:
                sr_dims = decoder.sr_dims
                n_seg_classes = pred.shape[-1] - sr_dims - 1
                pred_data = pred[..., sr_dims].astype(np.float32)
            else:
                pred_data = pred[..., i].astype(np.float32)
            img_dict[mod] = pred_data
        return img_dict

    def _compute_diff_map(self, mod_name, current, baseline):
        is_seg = 'seg' in mod_name.lower() or 'mask' in mod_name.lower()
        if is_seg:
            return _seg_change_map(current, baseline)
        else:
            return _signed_diff_map(current, baseline)

    def _log_inner_val_convergence(self, epoch_train, epoch_val, picked_subs, opt_idcs, eval_idcs, grid_coords, grid_shape, split='val', tag_suffix=''):
        """Logs quantitative metrics averaged across the whole `split` set, and convergence images for picked subjects.

        Returns the mean DICE over the optimisation visits (the visits used to fit the
        `split` latents) at the current inner epoch, or None if no DICE is available.
        Used as the early-stopping signal in _run_validation_round: the latents are
        optimised with the reconstruction loss only, so the opt-visit DICE peaks and
        then degrades as the latents overfit the intensities.

        The metrics are always computed (so the early-stopping signal is available even with
        no TensorBoard writer); only the TensorBoard image/scalar logging is gated on a writer.
        """
        tb_writer = self.args.get('tb_writer', None)

        # Hold-out position suffix (e.g. '_holdout_V2'): keeps leave-one-out rounds from colliding
        # at the same convergence-image / inner-average-scalar tag. Empty for single-round validation.
        sfx = f"_{tag_suffix}" if tag_suffix else ""

        split_df = self.datasets[split].df
        id_col = self.args['dataset'].get('id_column', 'subject_id')
        val_epochs = self.args['epochs']['val']
        global_step = epoch_train * val_epochs + epoch_val

        self.inr_decoder[split].eval()

        # Accumulators for validation metrics over the entire set
        metrics_accum = {
            'opt': {mod: {'PSNR': [], 'SSIM': [], 'DICE': []} for mod in self.args['dataset']['modalities']},
            'eval': {mod: {'PSNR': [], 'SSIM': [], 'DICE': []} for mod in self.args['dataset']['modalities']}
        }
        # Flat accumulator for the early-stopping signal: every DICE measured on the
        # optimisation visits at this inner epoch (one entry per seg modality per visit).
        opt_dice_vals = []

        # The 'opt' visits are always processed (they provide the early-stopping signal even
        # with no writer). The 'eval' visits and all images/scalars are TensorBoard-only.
        categories = [('opt', opt_idcs)]
        if tb_writer is not None:
            categories.append(('eval', eval_idcs))

        with torch.no_grad():
            # Run inference on the relevant validation indices to compute average metrics
            for category, idcs in categories:
                for idx in idcs:
                    row = split_df.iloc[idx]
                    sub_id = int(row['sub_id_int'])
                    row_dict = row.to_dict()
                    for mod in self.args['dataset']['modalities']:
                        row_dict[mod] = self.datasets[split].resolve_path(row_dict, mod)

                    latent_idx = idx if self.args['dataset'].get('independent_visits', False) else sub_id
                    volume_inf = self._reconstruct_visit(
                        row_dict, latent_idx, grid_coords, grid_shape, split=split
                    )
                    patient_stats = self._get_patient_stats(split, sub_id)
                    res_metrics, res_imgs = compute_metrics(
                        self.args, volume_inf, row_dict, epoch_val, split,
                        tb_writer=None,
                        return_images=True,
                        patient_stats=patient_stats
                    )

                    if category == 'opt' and res_metrics.get('DICE'):
                        opt_dice_vals.extend(res_metrics['DICE'])

                    modalities = self.args['dataset']['modalities']
                    for m_name in ['PSNR', 'SSIM', 'DICE']:
                        if m_name in res_metrics:
                            m_list = res_metrics[m_name]
                            for i, mod in enumerate(modalities):
                                if i < len(m_list):
                                    metrics_accum[category][mod][m_name].append(m_list[i])

                    # For picked subjects, also log convergence images patient-eye-wise
                    if tb_writer is not None and sub_id in picked_subs:
                        eye_id = str(row_dict.get(id_col, 'unknown'))
                        visit_id = f"V{row_dict.get('Visit_Number', row_dict.get('Visit', '0'))}"
                        for mod, imgs in res_imgs.items():
                            fig = make_longitudinal_tiled_figure(
                                [imgs['pred']], [imgs['ref']], [f"Inner Ep {epoch_val}"],
                                row1_name="Prediction", row2_name="Ground Truth",
                                title=f"{eye_id} - {mod} - Convergence - {visit_id} ({category})"
                            )
                            tag = f"{split}_convergence_{category}/{eye_id}/{visit_id}/{mod}_epTrain{epoch_train}{sfx}"
                            tb_writer.add_image(tag, fig_to_numpy(fig), epoch_val, dataformats='HWC')
                            plt.close(fig)

            # Log the average metrics on TensorBoard
            if tb_writer is not None:
                for category in ['opt', 'eval']:
                    for mod in self.args['dataset']['modalities']:
                        for m_name in ['PSNR', 'SSIM', 'DICE']:
                            vals = metrics_accum[category][mod][m_name]
                            if vals:
                                avg_val = np.mean(vals)
                                tb_writer.add_scalar(f"{split}_inner_{category}/average{sfx}/{mod}_{m_name}", avg_val, global_step)

        # Early-stopping signal: mean DICE over the optimisation visits at this inner epoch.
        return float(np.mean(opt_dice_vals)) if opt_dice_vals else None

    def _row_weeks(self, d):
        """Weeks-from-baseline for a row dict — the canonical interpolation axis for novel visits.

        IMPORTANT: novel-visit targets (`w`) are always expressed in WEEKS (midpoints / future
        offsets). The interpolation axis must therefore be weeks_from_baseline, NOT the
        temporal_condition (which may be AgeatVisit, in years). Using the wrong axis was bracketing
        week targets against age values, and clobbering AgeatVisit with week numbers — feeding wildly
        out-of-distribution values to load_time and breaking interpolation/extrapolation.

        Prefers the derived 'weeks_from_baseline' column; falls back to visit_week_map[Visit_Number],
        then the temporal_key value. Works for both time-as-input choices (AgeatVisit or weeks).
        """
        if 'weeks_from_baseline' in d and d['weeks_from_baseline'] is not None:
            try:
                return float(d['weeks_from_baseline'])
            except (TypeError, ValueError):
                pass
        vwm = self.args['dataset'].get('visit_week_map')
        if vwm is not None and 'Visit_Number' in d:
            try:
                vn = int(d['Visit_Number'])
                wk = vwm.get(vn, vwm.get(str(vn)))
                if wk is not None:
                    return float(wk)
            except (TypeError, ValueError):
                pass
        return float(d.get(self._temporal_key, 0.0))

    def _get_interpolated_row_dict(self, actual_rows, w):
        # Bracket the target week `w` against each row's weeks_from_baseline (see _row_weeks).
        weeks = [self._row_weeks(d) for d in actual_rows]

        if len(actual_rows) == 1:
            new_row = copy.deepcopy(actual_rows[0])
            # Convert the week delta to an age delta (age advances linearly with time).
            age_diff = (w - weeks[0]) / 52.0
            for age_key in ('AgeatVisit', 'age_at_visit'):
                if age_key in new_row and new_row[age_key] is not None:
                    new_row[age_key] = float(new_row[age_key]) + age_diff
            new_row['weeks_from_baseline'] = w
            if self._temporal_key == 'weeks_from_baseline':
                new_row[self._temporal_key] = w
            new_row['Visit_Number'] = f"{w}w"
            return new_row

        if w <= weeks[0]:
            w0, w1 = weeks[0], weeks[1]
            r0, r1 = actual_rows[0], actual_rows[1]
            factor = (w - w0) / (w1 - w0) if w1 != w0 else 0.0
            return self._interpolate_rows(r0, r1, factor, w)
        elif w >= weeks[-1]:
            w0, w1 = weeks[-2], weeks[-1]
            r0, r1 = actual_rows[-2], actual_rows[-1]
            factor = 1.0 + (w - w1) / (w1 - w0) if w1 != w0 else 1.0
            return self._interpolate_rows(r0, r1, factor, w)
        else:
            for i in range(len(weeks) - 1):
                if weeks[i] <= w <= weeks[i+1]:
                    w0, w1 = weeks[i], weeks[i+1]
                    r0, r1 = actual_rows[i], actual_rows[i+1]
                    factor = (w - w0) / (w1 - w0) if w1 != w0 else 0.0
                    return self._interpolate_rows(r0, r1, factor, w)

    def _interpolate_rows(self, r0, r1, factor, w):
        new_row = copy.deepcopy(r1)

        # Linearly inter/extrapolate ALL numeric columns by `factor`. Because age and
        # weeks_from_baseline both advance linearly with time, this yields the correct AgeatVisit
        # (in years) AND weeks_from_baseline at the target — each in its own units.
        for key in r0.keys():
            if (key in r0 and key in r1 and
                isinstance(r0[key], (int, float, np.integer, np.floating)) and
                isinstance(r1[key], (int, float, np.integer, np.floating)) and
                key not in ['sub_id_int', 'Study_Subject_ID', 'Eye_ID', 'Patient_ID', 'Site_ID']):

                v0 = float(r0[key])
                v1 = float(r1[key])
                new_row[key] = v0 + factor * (v1 - v0)

        # Pin the canonical time axis to the exact target week. Do NOT clobber the temporal_key
        # (e.g. AgeatVisit) with `w` — its correct value comes from the interpolation above; the old
        # `new_row[temporal_key] = w` is what fed week numbers into the AgeatVisit time input.
        new_row['weeks_from_baseline'] = w
        if self._temporal_key == 'weeks_from_baseline':
            new_row[self._temporal_key] = w
        new_row['Visit_Number'] = f"{w}w"
        return new_row

    def _generate_novel_visits(self, epoch, split, subject_ids, grid_coords=None, grid_shape=None, future=True):
        """
        Generate novel (interpolated or extrapolated) predictions for specified subjects,
        and log interleaved GT+Pred figures and difference map figures.
        """
        tb_writer = self.args.get('tb_writer', None)
        if tb_writer is None:
            return

        if grid_coords is None or grid_shape is None:
            grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)

        # self.inr_decoder[split].eval()
        df = self.datasets[split].df
        id_col = self.args['dataset'].get('id_column', 'subject_id')  
        sampling_bbox = self.args['dataset'].get('sampling_bbox')
        visit_week_map = self.args['dataset'].get('visit_week_map')

        if visit_week_map is not None:
            def get_weeks_for_visit(v_num):
                if v_num is None:
                    return 0.0
                try:
                    w = visit_week_map.get(int(v_num))
                    if w is None:
                        w = visit_week_map.get(str(v_num))
                    return float(w) if w is not None else 0.0
                except Exception:
                    return 0.0
        else:
            def get_weeks_for_visit(v_num):
                return 0.0

        # Retrieve offset weeks configurations
        if future:
            offsets_weeks = self.args['model_gen'].get('future_offsets_weeks', [26, 52, 78])

        for sub_id in subject_ids:
            sub_df = df[df['sub_id_int'] == sub_id]
            if sub_df.empty:
                continue

            sorted_sub_df = sub_df.sort_values(self._temporal_key)  # sort the visits by the temporal condition column
            eye_id = str(sorted_sub_df.iloc[0].get(id_col, 'unknown'))  # extract Eye_ID
            patient_stats = self._get_patient_stats(split, sub_id)  # get patient stats for normalization

            # 1. Load actual GT visits and images
            gt_visits = []
            for idx, row in sorted_sub_df.iterrows():
                row_dict = row.to_dict()
                if 'weeks_from_baseline' in row_dict:
                    week = float(row_dict['weeks_from_baseline'])
                elif visit_week_map is not None:
                    visit_num = row_dict.get('Visit_Number')
                    week = get_weeks_for_visit(visit_num)  # we extract the corresponding number of weeks from baseline
                else:
                    week = float(row_dict.get(self._temporal_key, 0.0))

                gt_imgs = {}
                for mod_i, mod in enumerate(self.args['dataset']['modalities']):
                    mod_path = self.datasets[split].resolve_path(row_dict, mod)
                    is_seg = 'seg' in mod.lower() or 'mask' in mod.lower()
                    img = load_2d_modality(mod_path, is_seg, patient_stats=patient_stats, mod_index=mod_i, args=self.args)
                    img = center_crop_2d(img, sampling_bbox)
                    gt_imgs[mod] = img

                # Keep row_dict's real values (AgeatVisit in years, weeks_from_baseline in weeks).
                # Do NOT overwrite the temporal_key with `week` — that corrupted the AgeatVisit time
                # input during novel-visit reconstruction. The plotting 'week' is tracked separately.
                gt_visits.append({
                    'week': week,
                    'label': f"GT@{int(week)}w",
                    'images': gt_imgs,
                    'row_dict': row_dict,
                    'is_gt': True
                })

            # Sort GT visits chronologically
            gt_visits.sort(key=lambda x: x['week'])

            # 2. Determine target weeks for predictions
            gt_weeks = [v['week'] for v in gt_visits]
            if not future:
                # Interpolation: intermediate weeks are midpoints of consecutive GT visits
                pred_weeks = []
                for i in range(len(gt_weeks) - 1):
                    midpoint = (gt_weeks[i] + gt_weeks[i+1]) / 2.0
                    pred_weeks.append(midpoint)
            else:
                # Extrapolation: future weeks are relative to the last GT visit
                last_week = gt_weeks[-1]
                pred_weeks = [last_week + offset for offset in offsets_weeks]

            # 3. Generate predictions for each target week
            pred_visits = []
            actual_rows = [v['row_dict'] for v in gt_visits]  # ground truth visits
            for w in pred_weeks:
                # Interpolate/extrapolate the row dictionary inputssub_id
                new_row = self._get_interpolated_row_dict(actual_rows, w)
                
                # Check cache first to avoid redundant reconstructions
                cache_key = (split, sub_id, round(float(w), 3))
                if cache_key in self.reconstruction_cache:
                    pred_imgs = self.reconstruction_cache[cache_key]
                else:
                    # Reconstruct predicted visit
                    latent_idx = sub_id
                    if self.args['dataset'].get('independent_visits', False):
                        # Fallback to the first visit index for this patient-eye
                        sub_rows = df[df['sub_id_int'] == sub_id]
                        if not sub_rows.empty:
                            latent_idx = df.index.get_loc(sub_rows.index[0])
                    try:
                        volume_inf = self._reconstruct_visit(
                            new_row, latent_idx, grid_coords, grid_shape, split=split,
                            allow_extrapolation=self.args['dataset'].get('extrapolate_beyond_range', False)
                        )
                        pred_imgs = self._extract_modality_images(volume_inf)
                        self.reconstruction_cache[cache_key] = pred_imgs
                    except Exception as e:
                        print(f"Warning: Failed to reconstruct novel visit for {eye_id} at week {w}: {e}")
                        continue
                
                pred_visits.append({
                    'week': w,
                    'label': f"Pred@{int(w)}w",
                    'images': pred_imgs,
                    'is_gt': False
                })

            # Per-eye pixel->mm^2 scale (≈constant across visits) for the lesion-size captions below.
            # Single source of truth (_lesion_px_area_mm2) -> captions agree with the CSV/metrics/figures.
            has_seg = self.args['inr_decoder']['out_dim'][-1] > 0
            seg_key = self.args['dataset']['modalities'][-1] if has_seg else None
            px_area = self._lesion_px_area_mm2(gt_visits[0]['row_dict']) if gt_visits else 1.0

            # 4. For each modality, build and log figures
            for mod in self.args['dataset']['modalities']:
                # Collect and sort all visits chronologically
                visits_to_plot = []
                for g_v in gt_visits:
                    visits_to_plot.append({
                        'week': g_v['week'],
                        'label': g_v['label'],
                        'image': g_v['images'][mod],
                        'is_gt': True
                    })
                for p_v in pred_visits:
                    visits_to_plot.append({
                        'week': p_v['week'],
                        'label': p_v['label'],
                        'image': p_v['images'][mod],
                        'is_gt': False
                    })
                visits_to_plot.sort(key=lambda x: (x['week'], x['is_gt']))
                
                # Align images to minimum common dimensions
                min_H = min(v['image'].shape[0] for v in visits_to_plot)
                min_W = min(v['image'].shape[1] for v in visits_to_plot)
                for v in visits_to_plot:
                    img = v['image']
                    H, W = img.shape[:2]
                    if H != min_H or W != min_W:
                        h_start = (H - min_H) // 2
                        w_start = (W - min_W) // 2
                        v['image'] = img[h_start:h_start + min_H, w_start:w_start + min_W]

                plot_images = [v['image'] for v in visits_to_plot]
                plot_labels = [v['label'] for v in visits_to_plot]
                plot_is_gts = [v['is_gt'] for v in visits_to_plot]

                # For the segmentation modality, caption each column with the lesion size MEASURED
                # FROM THAT COLUMN'S MASK (predicted mask for Pred/novel columns, GT mask for GT
                # columns) — never an interpolated/artificial size. mm^2 = (#GA px > 0.5) * px_area.
                sublabels = None
                if has_seg and mod == seg_key:
                    sublabels = []
                    for v in visits_to_plot:
                        area = float(np.sum(v['image'] > 0.5) * px_area)
                        sublabels.append(f"{'GT' if v['is_gt'] else 'Pred'} {area:.2f} mm²")

                fig_interleaved = make_interleaved_figure(
                    plot_images, plot_labels, plot_is_gts,
                    title=f"{eye_id} - {mod} - {'Future Extrapolation' if future else 'Interpolation'}",
                    sublabels=sublabels
                )
                
                tag = f"{split}/{eye_id}/{mod}_{'Future' if future else 'Interpolation'}"
                tb_writer.add_image(tag, fig_to_numpy(fig_interleaved), epoch, dataformats='HWC')
                plt.close(fig_interleaved)

                # NOTE: the former Future/Interpolation "Differences" TB images (sequential-diff and
                # baseline-diff of the synthetic interpolated/extrapolated visits) are no longer logged.
                # These novel visits have no ground truth, so a GT-vs-prediction difference is not
                # meaningful here. The GT-baseline difference comparison (Pred follow-up vs GT baseline
                # and GT follow-up vs GT baseline, FAF + seg) is logged instead from
                # _log_reconstruction_figures, where real GT visits are available.



    def generate_renders(self, epoch=0, n_max=100):
        """
        Generate a render for each condition combination in self.args['model_gen']['conditions'].
        """
        print(f"Generating renders (depending on resolution and count this may take some time) ...\n")
        self.inr_decoder['train'].eval()
        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)
        temp_steps = self.args['model_gen']['temporal_values']
        sr_dims = sum(self.args['inr_decoder']['out_dim'][:-1])
        n_seg_channels = self.args['inr_decoder']['out_dim'][-1]
        has_seg = n_seg_channels > 0
        # Rendering expects num_modalities channels = [intensity..., seg_argmax]
        # But inference() now returns [imgs(sr_dims), seg_hard(1), seg_soft(n_seg_channels)]
        # We need to extract [imgs, seg_hard] = sr_dims + 1 channels
        render_list = []
        with torch.no_grad():
            temporal_key = self._temporal_key
            for temp_step in temp_steps:
                temp_step_normed = normalize_condition(self.args, temporal_key, temp_step)
                mean_latent = self.get_mean_latent(temporal_key, temp_step_normed, n_max=n_max)
                condition_vectors = generate_combinations(self.args, self.args['model_gen']['conditions'])
                cond_list = []
                for c_v in condition_vectors:
                    time_as_input = self.args['inr_decoder'].get('time_as_input', False)
                    if time_as_input:
                        keys = list(self.args['model_gen']['conditions'].keys())
                        if temporal_key in keys:
                            temp_idx = keys.index(temporal_key)
                            t_val = c_v[temp_idx]
                            c_v_filtered = [val for idx, val in enumerate(c_v) if idx != temp_idx]
                        else:
                            t_val = temp_step_normed
                            c_v_filtered = c_v
                        
                        c_v_tensor = torch.tensor(c_v_filtered, dtype=torch.float32).to(self.device)
                        t_val_tensor = torch.tensor([t_val], dtype=torch.float32).to(self.device)
                    else:
                        c_v_tensor = torch.tensor(c_v, dtype=torch.float32).to(self.device)
                        t_val_tensor = None
                    
                    values_p = self.inr_decoder['train'].inference(grid_coords, mean_latent, c_v_tensor,
                                                                   grid_shape, None, time_val=t_val_tensor)
                    # Extract only [imgs, seg_hard] for rendering (discard seg_soft channels)
                    if has_seg:
                        imgs = values_p[..., :sr_dims]
                        seg_hard = values_p[..., sr_dims:sr_dims + 1]  # argmax channel
                        values_p = torch.cat((imgs, seg_hard), dim=-1)
                    else:
                        values_p = values_p[..., :sr_dims]
                    cond_list.append(values_p.detach().cpu())
                    torch.cuda.empty_cache()

                render_list.append(torch.stack(cond_list, dim=-1))
        render_list = torch.stack(render_list, dim=-1)  # [*spatial, num_modalities, num_conditions, t]
        save_renders(self.args, render_list, temp_steps, condition_vectors, epoch=epoch,
                   tb_writer=self.args.get('tb_writer', None))
        return render_list

    def get_mean_latent(self, condition_key, condition_mean, n_max=100, split='train'):
        """
        Regress gaussian weighted latent code from subjects weighted by distance to condition mean
        of the condition with condition_key. Weights are clipped to the closest n_max subjects.
        sigma is the standard deviation of the gaussian distribution used to weight the latents
        emperically we want +/- 2 stds (covering 95% of the weights) to span +/- "gaussian_span" weeks of scan age, e.g. 0.75 weeks.
        Therefore:
        - Full range of condition values is [-1, 1], i.e. 2.
        - Full range of scan age is c_max - c_min = c_range, e.g. 46 - 37 = 9 for term neonates.
        - The ratio of condition values to weeks is 2 / c_range = c_ratio, e.g. 2 / 9 = 0.222 units per week.
        ==> 2 std = 0.75 weeks = 0.75 * c_ratio e.g. = 0.165 units.
        ==> sigma = 1 std = 0.5 * 0.75 weeks * c_ratio, e.g. = 0.0825 units for term neonates.
        # Finally, we scale the sigma by the condition scale factor in the config, as scan age is actually normalized to [-cond_scale, cond_scale]
        """
        c_ratio = 2 / (self.args['dataset']['constraints'][condition_key]['max'] -
                       self.args['dataset']['constraints'][condition_key]['min'])
        span_weeks = self.args['model_gen']['gaussian_span']
        sigma = 0.5 * span_weeks * c_ratio
        sigma = sigma * self.args['model_gen']['cond_scale']

        latents = self.latents[split]
        # Expand latents to visit-level: each visit gets its patient-eye's latent
        sub_id_map = self.datasets[split].sub_id_map
        expanded_latents = latents[sub_id_map]  # (N_visits, ...)
        condition_values, df_idcs = self.datasets[split].get_condition_values(condition_key, normed=True,
                                                                              device=self.device)
        assert len(condition_values) == len(expanded_latents), f"Condition values ({len(condition_values)}) \
                                                        and expanded latents ({len(expanded_latents)}) must have the same length!"
        weights = torch.exp(-(condition_values - condition_mean) ** 2 / (2 * (sigma ** 2)))
        n_max = min(n_max, len(weights))
        weights[torch.argsort(weights, descending=True)[n_max:]] = 0
        weights = weights / torch.sum(weights)
        # Dynamically reshape weights to match latents dimensionality
        view_shape = [-1] + [1] * (expanded_latents.ndim - 1)
        weights = weights.view(*view_shape)
        mean_latent = torch.sum(expanded_latents * weights, dim=0, keepdim=True)
        return mean_latent

    def save_state(self, epoch, split='train', filename=None):
        """Save a checkpoint. By default writes 'checkpoint_epoch_{epoch}.pth'; pass `filename`
        (e.g. 'checkpoint_best.pth') to write to a fixed name for best-checkpoint selection."""
        if self.args['save_model']:
            log_dir = self.args['output_dir']
            tb_writer = self.args.pop('tb_writer', None)
            state_dict = {
                'epoch': epoch,
                'latents': self.latents[split].cpu(),
                # 'transformations': self.transformations[split].cpu(),
                'inr_decoder': self.inr_decoder[split].state_dict(),
                'tsv_file': self.datasets[split].tsv_file,
                'dataset_df': self.datasets[split].df,
                'args': self.args
            }

            fname = filename if filename is not None else f'checkpoint_epoch_{epoch}.pth'
            torch.save(state_dict, os.path.join(log_dir, fname))
            if tb_writer is not None:
                self.args['tb_writer'] = tb_writer
            print(f'Saved model state to {os.path.join(log_dir, fname)}')
        else:
            print(f'Not saving model state as save_model is set to False')

    def load_checkpoint(self, chkp_path=None, epoch=None):
        chkp_path = os.path.join(chkp_path, f'checkpoint_epoch_{epoch}.pth')
        if not os.path.exists(chkp_path):
            raise FileNotFoundError(f'State file {chkp_path} not found!')
        chkp = torch.load(chkp_path, weights_only=False)
        # self.args = chkp['args']
        self._init_dataloading(chkp['tsv_file'], chkp['dataset_df'])
        self._init_inr(chkp['inr_decoder'], split='train')
        # self._init_transformations(chkp['transformations'])
        self._init_latents(chkp['latents'])

        print(f'Loaded state from {chkp_path}')

    def _init_training(self):
        self.datasets, self.dataloaders = {}, {}
        self.inr_decoder, self.latents, self.transformations = {}, {}, {}
        self.optimizers, self.grad_scalers = {}, {}
        self.schedulers = {}
        chkp_path = self.args['load_model']['path']
        if len(chkp_path) > 0:
            self.load_checkpoint(chkp_path, self.args['load_model']['epoch'])
        else:
            self._init_dataloading(split='train')
            self._init_inr(split='train')
            # self._init_transformations(split='train')
            self._init_latents(split='train')
        self._init_optimizer(split='train')  # optimizer is not loaded from checkpoint
        if self.args.get('overfit', False):
            print("--- OVERFIT MODE: Reusing training subjects for validation ---")
            self._init_dataloading(df_loaded=self.datasets['train'].df, split='val')
        else:
            self._init_dataloading(split='val')

    def _init_validation(self, split='val'):
        """Re-initialise the latents/optimiser/decoder for a test-time-optimisation split
        ('val' or 'test'): fresh latents, a frozen copy of the trained decoder, and an
        optimiser over the latents only."""
        self._seed()
        self._init_latents(split=split)
        self.global_steps[split] = 0

        # Optionally initialise latents from nearest training latents / population mean
        init_mode = self.args['optimizer'].get('val_latent_init', 'random')
        if init_mode == 'nearest_train':
            self._init_val_latents_from_nearest_train(split=split)
        elif init_mode == 'population_mean':
            self._init_val_latents_from_population_mean(split=split)

        # Snapshot the initialisation as the anchor for the TTA latent regulariser:
        # random -> pull toward 0 (anchor None); population_mean / nearest_train -> pull toward
        # that prior (the just-initialised latent values).
        if init_mode == 'random':
            self._latent_anchor[split] = None
        else:
            self._latent_anchor[split] = self.latents[split].detach().clone()

        # self._init_transformations(split=split)
        self._init_optimizer(split=split)
        # Temporarily remove tb_writer (contains unpicklable thread locks) before deepcopy
        tb_writer = self.args.pop('tb_writer', None)
        self.inr_decoder[split] = copy.deepcopy(self.inr_decoder['train'])
        if tb_writer is not None:
            self.args['tb_writer'] = tb_writer
        self.inr_decoder[split].eval()
        # Freeze INR decoder weights — only latents should be optimised at test time.
        for p in self.inr_decoder[split].parameters():
            p.requires_grad_(False)

    def _init_dataloading(self, tsv_file=None, df_loaded=None, split='train'):
        shuffle = True if split == 'train' else False
        tsv_file = self.args['dataset']['tsv_file'] if tsv_file is None else tsv_file
        self.datasets[split] = Data(self.args, tsv_file, split=split, df_loaded=df_loaded)
        # batch_by_eye (train only): one batch = one eye's full visit sequence, so the pooled soft-Dice
        # becomes the TEMPORAL (stacked) Dice and the monotonicity penalty sees all the eye's visits.
        if self.args['optimizer'].get('batch_by_eye', False) and split == 'train':
            from data_loading.dataset import EyeBatchSampler
            sampler = EyeBatchSampler(self.datasets[split].eye_groups(), shuffle=True)
            self.dataloaders[split] = DataLoader(self.datasets[split], batch_sampler=sampler,
                                                 num_workers=self.args['num_workers'],
                                                 collate_fn=self.datasets[split].collate_fn, pin_memory=True)
            print(f"[batch_by_eye] train batches = {len(sampler)} eyes (temporal cross-visit losses enabled)")
        else:
            self.dataloaders[split] = DataLoader(self.datasets[split], batch_size=self.args['batch_size'],
                                                 num_workers=self.args['num_workers'], shuffle=shuffle,
                                                 collate_fn=self.datasets[split].collate_fn, pin_memory=True)

        print(f"Initialized dataloader for {split} with {len(self.datasets[split])} visits "
              f"({self.datasets[split].n_unique_subjects} unique subjects)")

    def _init_inr(self, state_dict=None, split='train'):
        # get the number of active conditions
        time_as_input = self.args['inr_decoder'].get('time_as_input', False)
        temporal_key = self.args['dataset'].get('temporal_condition')
        if temporal_key is None:
            # Fallback to first enabled condition
            for key, enabled in self.args['dataset']['conditions'].items():
                if enabled:
                    temporal_key = key
                    break

        cond_dims = 0
        for c, enabled in self.args['dataset']['conditions'].items():
            if enabled:
                if time_as_input and c == temporal_key:
                    continue
                cond_dims += 1  # we count only the conditions that are NOT the variable used as time-input 
        
        self.args['inr_decoder']['cond_dims'] = cond_dims
        self.inr_decoder[split] = INR_Decoder(self.args, self.device).to(self.device)
        if state_dict is not None:
            self.inr_decoder[split].load_state_dict(state_dict)

    def _init_transformations(self, tfs=None, split='train'):
        n_subjects = self.datasets[split].n_unique_subjects  # one transformation per patient-eye
        shape = (n_subjects, max(self.args['inr_decoder']['tf_dim'],
                                 0))  # TODO: change to 6 for 3D # at least 6 for rigid, 9 for rigid+scale
        tfs = torch.zeros(shape).to(self.device) if tfs is None else tfs.to(self.device)
        self.transformations[split] = nn.Parameter(tfs) if self.args['inr_decoder'][
                                                               'tf_dim'] > 0 else tfs  # if tf_dim=0, set trafos to 0 and fix

    def _init_latents(self, lats=None, split='train'):
        n_subjects = self.datasets[split].n_unique_subjects
        shape = (n_subjects, *self.args['inr_decoder']['latent_dim'])  # (N, C, X_1, X_2)
        lats = torch.normal(0, 0.01, size=shape).to(self.device) if lats is None else lats.to(self.device)
        self.latents[split] = nn.Parameter(lats)
        # if split == 'val' and self.args['inr_decoder']['cond_dims'] > 0:
        #    shape_cond_val = (n_subjects, self.args['inr_decoder']['cond_dims'])
        #    self.conditions_val = nn.Parameter(torch.normal(0, 0.01, size=shape_cond_val).to(self.device))

    def _init_val_latents_from_nearest_train(self, split='val'):
        """
        Initialise each test-time latent (val or test) from the nearest training latent,
        where 'nearest' is defined by the temporal condition (e.g. AgeatVisit).
        For each patient-eye we take the mean condition value across its
        visits and find the training patient-eye whose mean condition value is
        closest.  The training latent is then copied as the starting point.
        """
        temporal_key = self._temporal_key
        train_df = self.datasets['train'].df
        split_df = self.datasets[split].df

        # Compute mean temporal value per training patient-eye
        train_means = train_df.groupby('sub_id_int')[temporal_key].mean()
        train_ids = train_means.index.values        # sub_id_int values
        train_vals = train_means.values              # mean temporal values

        # For each patient-eye or visit, find the nearest training patient-eye
        independent_visits = self.args['dataset'].get('independent_visits', False)
        n_copied = 0
        if independent_visits:
            for idx in range(len(split_df)):
                split_val = float(split_df.iloc[idx][temporal_key])
                distances = np.abs(train_vals - split_val)
                nearest_train_id = train_ids[np.argmin(distances)]
                self.latents[split].data[idx] = self.latents['train'][nearest_train_id].data.clone()
                n_copied += 1
        else:
            split_means = split_df.groupby('sub_id_int')[temporal_key].mean()
            for split_sub_id, split_mean in split_means.items():
                distances = np.abs(train_vals - split_mean)
                nearest_train_id = train_ids[np.argmin(distances)]
                self.latents[split].data[split_sub_id] = self.latents['train'][nearest_train_id].data.clone()
                n_copied += 1

        print(f"[{split}_latent_init] Initialised {n_copied} {split} latents from nearest training latents "
              f"(by {temporal_key}).")

    def _init_val_latents_from_population_mean(self, split='val'):
        """
        Initialise each test-time latent (val or test) to the mean of all training latents.
        This provides a neutral starting point representing the average population anatomy.
        """
        mean_latent = self.latents['train'].data.mean(dim=0, keepdim=True)  # (1, C, H, W)
        self.latents[split].data.copy_(mean_latent.expand_as(self.latents[split].data))
        print(f"[{split}_latent_init] Initialised {split} latents from training population mean.")

    def run_time_sensitivity_probe(self, epoch_train):
        """
        Time sensitivity probe after training epochs: pass coords at t=-1 vs t=1
        to INR decoder with a fixed spatial latent, compute mean absolute difference
        in predictions, log to TB. Tells us if network is actually using time.
        """
        tb_writer = self.args.get('tb_writer', None)
        if tb_writer is None:
            return
        
        if self.latents['train'] is None or self.latents['train'].shape[0] == 0:
            return
        
        # Generate world grid coords
        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)
        
        # Pick the latent of the first subject
        latent_vec = self.latents['train'][0:1].detach()  # (1, C, H, W)
        
        # Condition vector of the first subject (or zeros if none)
        cond_dims = self.args['inr_decoder'].get('cond_dims', 0)
        condition_vec = torch.zeros((1, cond_dims), device=self.device)
        
        # Run inference at t = -1.0
        t_minus = torch.tensor([[-1.0]], device=self.device)
        self.inr_decoder['train'].eval()
        with torch.no_grad():
            rec_minus = self.inr_decoder['train'].inference(
                grid_coords, latent_vec, condition_vec, grid_shape, time_val=t_minus
            )
            # Run inference at t = 1.0
            t_plus = torch.tensor([[1.0]], device=self.device)
            rec_plus = self.inr_decoder['train'].inference(
                grid_coords, latent_vec, condition_vec, grid_shape, time_val=t_plus
            )
        
        # Compute mean absolute difference in intensity predictions
        sr_dims = sum(self.args['inr_decoder']['out_dim'][:-1])
        diff = torch.abs(rec_plus[..., :sr_dims] - rec_minus[..., :sr_dims])
        mean_diff = diff.mean().item()
        
        # Log to TensorBoard
        tb_writer.add_scalar("diagnostics/time_sensitivity_mean_diff", mean_diff, epoch_train)
        
        # Track segmentation sensitivity if applicable
        n_seg_channels = self.args['inr_decoder']['out_dim'][-1]
        if n_seg_channels > 0:
            seg_minus = rec_minus[..., sr_dims]
            seg_plus = rec_plus[..., sr_dims]
            seg_changed_fraction = (seg_minus != seg_plus).float().mean().item()
            tb_writer.add_scalar("diagnostics/time_sensitivity_seg_changed_fraction", seg_changed_fraction, epoch_train)
        
        # Log the difference map as an image if it's 2D
        if len(grid_shape) == 2:
            diff_img = diff[..., 0].cpu().numpy()
            max_val = diff_img.max()
            if max_val > 0:
                diff_img_norm = diff_img / max_val
            else:
                diff_img_norm = diff_img
            tb_writer.add_image("diagnostics/time_sensitivity_diff_map", diff_img_norm, epoch_train, dataformats='HW')
        
        print(f"[time_sensitivity_probe] Epoch {epoch_train}: mean absolute difference = {mean_diff:.6f}")

    def run_condition_sensitivity_probe(self, epoch_train, n_steps=9, split='train'):
        """Diagnose whether the temporal CONDITIONING variable actually drives the outputs.

        With time_as_input=false, the temporal condition (e.g. weeks_from_baseline) enters ONLY
        through the conditioning pathway (cond_encoding -> FiLM modulation). This probe fixes a
        single subject's (fitted) latent and sweeps the temporal condition across — and slightly
        beyond — its training range, measuring how much the predicted FAF and the predicted GA
        area change with time. A flat response means the model is ignoring time; a smooth,
        monotone GA-area trajectory is what we want for interpolation/extrapolation.

        Unlike run_time_sensitivity_probe (which varies the time *input* coordinate and is a no-op
        when time_as_input=false), this varies the actual conditioning variable used by the model.
        """
        tb_writer = self.args.get('tb_writer', None)
        if tb_writer is None:
            return

        temporal_key = self._temporal_key
        # Which conditions enter the conditioning path, in the same order as load_conditions builds them.
        enabled = [k for k, v in self.args['dataset'].get('conditions', {}).items() if v]
        if self.time_as_input and temporal_key in enabled:
            enabled.remove(temporal_key)
        if temporal_key not in enabled:
            print(f"[condition_probe] Temporal key '{temporal_key}' is not in the conditioning path "
                  f"(time_as_input={self.time_as_input}); skipping.")
            return
        t_idx = enabled.index(temporal_key)
        cond_dims = len(enabled)

        # Pick the first subject and use its trained latent.
        sub_ids = self._real_eye_subs(self.datasets[split].df)
        if not sub_ids:
            return
        sub_id = int(sub_ids[0])
        latent_vec = self.latents[split][sub_id:sub_id + 1].detach()

        # Normalisation: load_conditions maps weeks -> ((w - min)/(max - min)*2 - 1) * cond_scale.
        constraints = self.args['dataset'].get('constraints', {}).get(temporal_key, {})
        c_min = constraints.get('min', 0.0)
        c_max = constraints.get('max', 1.0)
        cond_scale = self.args.get('model_gen', {}).get('cond_scale', 1.0)

        # Sweep slightly beyond [-1, 1] to also probe extrapolation (future visits).
        norm_grid = np.linspace(-1.0, 1.3, n_steps)
        weeks_grid = ((norm_grid + 1.0) / 2.0) * (c_max - c_min) + c_min

        grid_coords, grid_shape = generate_world_grid(self.args, device=self.device)
        label_names = self.args['dataset'].get('label_names', [])
        # GA class index in the seg output. Accept the configured name ('GeographicAtrophy') or the
        # short alias ('GA'); fall back to the last seg class if neither is present.
        ga_aliases = ('GeographicAtrophy', 'GA')
        ga_label = next((label_names.index(a) for a in ga_aliases if a in label_names),
                        self.args['inr_decoder']['out_dim'][-1] - 1)

        self.inr_decoder[split].eval()
        faf_maps, ga_areas = [], []
        sr_dims = sum(self.args['inr_decoder']['out_dim'][:-1])
        with torch.no_grad():
            for c_norm in norm_grid:
                condition_vec = torch.zeros((1, cond_dims), device=self.device)
                condition_vec[0, t_idx] = float(c_norm) * cond_scale
                out = self.inr_decoder[split].inference(
                    grid_coords, latent_vec, condition_vec, grid_shape, time_val=None
                )
                faf = out[..., :sr_dims].detach().float().cpu().numpy()
                faf_maps.append(faf)
                seg_hard = out[..., sr_dims].detach().float().cpu().numpy()
                ga_areas.append(float((seg_hard == ga_label).sum()))

        # FAF sensitivity: mean |ΔFAF| between consecutive time points (per pixel, per step).
        faf_diffs = [np.mean(np.abs(faf_maps[i + 1] - faf_maps[i])) for i in range(len(faf_maps) - 1)]
        faf_sensitivity = float(np.mean(faf_diffs)) if faf_diffs else 0.0
        ga_min, ga_max = (min(ga_areas), max(ga_areas)) if ga_areas else (0.0, 0.0)

        tb_writer.add_scalar("condition_probe/faf_mean_abs_diff_per_step", faf_sensitivity, epoch_train)
        tb_writer.add_scalar("condition_probe/ga_area_range_px", ga_max - ga_min, epoch_train)
        tb_writer.add_scalar("condition_probe/ga_area_min_px", ga_min, epoch_train)
        tb_writer.add_scalar("condition_probe/ga_area_max_px", ga_max, epoch_train)

        # Log the GA-area-vs-time trajectory as a figure (the key readout for growth modelling).
        try:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(weeks_grid, ga_areas, marker='o')
            ax.axvspan(c_min, c_max, alpha=0.1, color='green', label='training range')
            ax.set_xlabel(f"{temporal_key}"); ax.set_ylabel("predicted GA area (grid px)")
            ax.set_title(f"Condition sweep (subject {sub_id}, epoch {epoch_train})")
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()
            tb_writer.add_image("condition_probe/ga_area_vs_time", fig_to_numpy(fig), epoch_train, dataformats='HWC')
            plt.close(fig)
        except Exception as e:
            print(f"[condition_probe] Figure logging failed: {e}")

        print(f"[condition_probe] Epoch {epoch_train}: FAF sensitivity/step = {faf_sensitivity:.6f}, "
              f"GA area {ga_min:.0f}->{ga_max:.0f}px over {temporal_key} "
              f"[{weeks_grid[0]:.1f}, {weeks_grid[-1]:.1f}] (train range [{c_min}, {c_max}]).")

    def re_init_latents(self, split='train'):
        self.latents[split].data.normal_(0, 0.01)
        # self.transformations[split].data.zero_()
        self.optimizers[split].zero_grad()

    def _init_optimizer(self, split='train'):

        params = [{'name': f'latents_{split}',
                   'params': self.latents[split],
                   'lr': self.args['optimizer']['lr_latent'],
                   'weight_decay': self.args['optimizer']['latent_weight_decay']}]

        # if self.args['inr_decoder']['tf_dim'] > 0:
        #    params.append({'name': f'transformations_{split}',
        #                   'params': self.transformations[split],
        #                   'lr': self.args['optimizer']['lr_tf'],
        #                   'weight_decay': self.args['optimizer']['tf_weight_decay']})
        if split == 'train':
            # Collect parameters. Note: inr_decoder.parameters() already includes
            # sr_net, modulator, and hashgrid (if active).
            params.append({'name': f'inr_decoder',
                           'params': self.inr_decoder[split].parameters(),
                           'lr': self.args['optimizer']['lr_inr'],
                           'weight_decay': self.args['optimizer']['inr_weight_decay']})
        # if split == 'val' and self.args['inr_decoder']['cond_dims'] > 0:
        #    params.append({'name': f'conditions_val',
        #                   'params': self.conditions_val,
        #                   'lr': self.args['optimizer']['lr_latent'],
        #                   'weight_decay': self.args['optimizer']['latent_weight_decay']})
        self.optimizers[split] = optim.AdamW(params)
        self.grad_scalers[split] = GradScaler() if self.args['amp'] else None
        if self.args['optimizer']['scheduler']['type'] == 'cosine':
            # The test split reuses the val inner-loop budget (epochs['val']).
            t_max_split = 'val' if split == 'test' else split
            self.schedulers[split] = CosineAnnealingLR(self.optimizers[split], T_max=self.args['epochs'][t_max_split],
                                                       eta_min=self.args['optimizer']['scheduler']['eta_min'])
        else:
            self.schedulers[split] = None

    def _update_scheduler(self, split='train'):
        if self.schedulers[split] is not None:
            self.schedulers[split].step()

    def _seed(self):
        random.seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        torch.cuda.manual_seed(self.args['seed'])
        torch.cuda.manual_seed_all(self.args['seed'])
        # Pin cuDNN so the same seed actually reproduces run-to-run on the same GPU.
        # NOTE: we deliberately do NOT call torch.use_deterministic_algorithms(True):
        # grid_sample backward (the latent-grid sampler) has no deterministic kernel and
        # would raise at runtime. deterministic+benchmark=False removes autotuning drift,
        # which is the dominant source of variance, without crashing.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


