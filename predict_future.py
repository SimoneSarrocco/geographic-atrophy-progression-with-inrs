import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import components from the repository
from build_atlas import AtlasBuilder
from utils import (
    generate_world_grid, 
    fig_to_numpy, 
    make_interleaved_figure, 
    load_2d_modality, 
    center_crop_2d
)

def parse_args():
    parser = argparse.ArgumentParser(description="GAP-INR Standalone Future Prediction using Latent Transition")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--transition_checkpoint", type=str, required=True, help="Path to trained transition model checkpoint .pth file")
    parser.add_argument("--offsets", type=int, nargs="+", default=[12, 24, 48], help="Future offsets in weeks")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Specific subject integer IDs to predict (default: first 5)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save TensorBoard logs and figures")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Device to use")
    return parser.parse_args()

def main():
    args_cmd = parse_args()
    
    if not os.path.exists(args_cmd.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args_cmd.checkpoint}")
        
    print(f"Loading checkpoint from: {args_cmd.checkpoint}")
    chkp = torch.load(args_cmd.checkpoint, map_location="cpu", weights_only=False)
    
    # Extract config from checkpoint
    args = chkp["args"]
    
    # Setup device
    if args_cmd.device is not None:
        args["device"] = args_cmd.device
    else:
        args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args["device"])
    print(f"Using device: {device}")
    
    # Configure AtlasBuilder settings for evaluation
    args["epochs"]["train"] = 0
    args["validate_every"] = 1
    chkp_dir = os.path.dirname(args_cmd.checkpoint)
    chkp_epoch = chkp["epoch"]
    args["load_model"] = {
        "path": chkp_dir,
        "epoch": chkp_epoch
    }
    
    # Remove latent transition activation from config since it's decoupled
    if "latent_transition" in args:
        del args["latent_transition"]
    
    # Setup output directory and TensorBoard
    if args_cmd.output_dir is not None:
        args["output_dir"] = args_cmd.output_dir
    else:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args["output_dir"] = os.path.join(chkp_dir, f"future_prediction_{time_stamp}")
        
    os.makedirs(args["output_dir"], exist_ok=True)
    tb_log_dir = os.path.join(args["output_dir"], "tb_logs")
    args["tb_writer"] = SummaryWriter(log_dir=tb_log_dir)
    print(f"Output directory: {args['output_dir']}")
    print(f"TensorBoard log directory: {tb_log_dir}")
    
    # Initialize AtlasBuilder
    atlas_builder = AtlasBuilder(args)
    
    # Load transition model checkpoint
    if not os.path.exists(args_cmd.transition_checkpoint):
        raise FileNotFoundError(f"Transition checkpoint not found at: {args_cmd.transition_checkpoint}")
    print(f"Loading transition model checkpoint from: {args_cmd.transition_checkpoint}")
    transition_chkp = torch.load(args_cmd.transition_checkpoint, map_location="cpu", weights_only=False)
    
    # Reconstruct the model
    model_type = transition_chkp.get('model_type', 'resnet')
    latent_shape = transition_chkp['latent_shape']
    
    if model_type == 'mlp':
        from models.latent_transition import LatentTransitionMLP
        transition_model = LatentTransitionMLP(
            latent_shape=latent_shape,
            hidden_size=transition_chkp.get('hidden_size', 512),
            num_layers=transition_chkp.get('num_layers', 3)
        )
    elif model_type == 'resnet':
        from models.latent_transition import LatentTransitionResNet
        transition_model = LatentTransitionResNet(
            latent_shape=latent_shape,
            time_emb_dim=transition_chkp.get('time_emb_dim', 64),
            num_blocks=transition_chkp.get('num_blocks', 3)
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")
        
    transition_model.load_state_dict(transition_chkp['model_state_dict'])
    transition_model = transition_model.to(device)
    atlas_builder.latent_transition_model = transition_model
    print("Successfully loaded standalone transition model.")
    
    # 1. Run validation latent optimization (TTA) on optimization visits
    print("\n--- Running Test-Time Adaptation (TTA) on Validation Set ---")
    atlas_builder._init_validation()
    
    # Separate optimization and evaluation indices
    holdout_cfg = args.get('validation', {})
    strategy = holdout_cfg.get('holdout_strategy', 'last')
    if strategy == 'specific':
        ho_pos = holdout_cfg.get('holdout_visit', None)
    elif strategy == 'none':
        ho_pos = 'none'
    else:
        ho_pos = None  # 'last'
        
    opt_idcs, eval_idcs = atlas_builder.datasets['val'].get_longitudinal_indices(holdout_position=ho_pos)
    atlas_builder.current_val_opt_idcs = opt_idcs
    atlas_builder.current_val_eval_idcs = eval_idcs
    
    # Temporarily set dataloader to subset of optimization visits
    orig_val_loader = atlas_builder.dataloaders['val']
    atlas_builder.dataloaders['val'] = atlas_builder.create_subset_dataloader('val', opt_idcs)
    
    epochs_val = args['epochs']['val']
    print(f"Optimizing validation latents for {epochs_val} epochs...")
    for epoch_val in range(epochs_val):
        loss, _ = atlas_builder.train_epoch(split='val', epoch=epoch_val, epoch_train=chkp_epoch)
        atlas_builder._update_scheduler(split='val')
        print(f"  TTA Epoch {epoch_val+1}/{epochs_val} | Optimization Loss: {loss:.4f}")
        
    # Restore original dataloader
    atlas_builder.dataloaders['val'] = orig_val_loader
    print("Test-Time Adaptation complete!\n")
    
    # 2. Perform Future Extrapolation using Latent Transition
    print("--- Running Future Extrapolation ---")
    val_df = atlas_builder.datasets['val'].df
    id_col = args['dataset'].get('id_column', 'subject_id')
    sampling_bbox = args['dataset'].get('sampling_bbox')
    temporal_key = atlas_builder._temporal_key
    
    # Determine which subjects to predict
    unique_subs = sorted(val_df['sub_id_int'].unique())
    if args_cmd.subjects is not None:
        picked_subs = [s for s in args_cmd.subjects if s in unique_subs]
    else:
        picked_subs = unique_subs[:5]
        
    print(f"Predicting for subjects: {picked_subs}")
    grid_coords, grid_shape = generate_world_grid(args, device=device)
    
    # Evaluate and gather actual optimized visits first for baseline/reference
    _, opt_data = atlas_builder._evaluate_visits(
        opt_idcs, 'val', grid_coords, grid_shape, epoch=chkp_epoch, tb_writer=args["tb_writer"]
    )
    
    atlas_builder.latent_transition_model.eval()
    
    for sub_id in picked_subs:
        sub_df = val_df[val_df['sub_id_int'] == sub_id]
        if sub_df.empty:
            continue
            
        # Get optimized visits for this subject-eye
        sub_opt_idcs = [idx for idx in opt_idcs if val_df.iloc[idx]['sub_id_int'] == sub_id]
        if len(sub_opt_idcs) == 0:
            print(f"Skipping subject {sub_id} (no optimized visits found)")
            continue
            
        sorted_sub_df = sub_df.sort_values(temporal_key)
        eye_id = str(sorted_sub_df.iloc[0].get(id_col, 'unknown'))
        patient_stats = atlas_builder._get_patient_stats('val', sub_id)
        
        # Chronologically sort the optimized visits
        sorted_opt_idcs = sorted(sub_opt_idcs, key=lambda idx: float(val_df.iloc[idx][temporal_key]))
        
        # 1. Collect reference optimized/GT visits
        gt_visits = []
        for idx in sorted_opt_idcs:
            row_dict = val_df.iloc[idx].to_dict()
            week = float(row_dict[temporal_key])
            
            # Load images
            gt_imgs = {}
            for mod_i, mod in enumerate(args['dataset']['modalities']):
                mod_path = atlas_builder.datasets['val'].resolve_path(row_dict, mod)
                is_seg = 'seg' in mod.lower() or 'mask' in mod.lower()
                img = load_2d_modality(mod_path, is_seg, patient_stats=patient_stats, mod_index=mod_i, args=args)
                img = center_crop_2d(img, sampling_bbox)
                gt_imgs[mod] = img
                
            gt_visits.append({
                'week': week,
                'label': f"Opt@{int(week)}w",
                'images': gt_imgs,
                'row_dict': row_dict,
                'is_gt': True
            })
            
        # 2. Extract last optimized visit as the starting point for transition
        last_opt_visit = gt_visits[-1]
        last_opt_idx = sorted_opt_idcs[-1]
        t_opt = last_opt_visit['week']
        z_opt = atlas_builder.latents['val'][last_opt_idx:last_opt_idx + 1] # shape (1, C, X, Y)
        
        print(f"Subject {eye_id} (ID: {sub_id}) | Last opt visit at week {t_opt}")
        
        # 3. Generate future predictions
        pred_visits = []
        actual_rows = [v['row_dict'] for v in gt_visits]
        
        for offset in args_cmd.offsets:
            t_future = t_opt + offset
            
            # Interpolate condition values/meta info for the future timepoint
            new_row = atlas_builder._get_interpolated_row_dict(actual_rows, t_future)
            
            with torch.no_grad():
                # Transition the latent code using the transition model
                delta_t_tensor = torch.tensor([[float(offset)]], device=device, dtype=torch.float32)
                z_future = atlas_builder.latent_transition_model(z_opt, delta_t_tensor)
                
                # Get conditions and time inputs
                conditions = atlas_builder.datasets['val'].load_conditions(new_row).to(device)
                time_val = (
                    atlas_builder.datasets['val'].load_time(new_row).to(device) 
                    if args['inr_decoder'].get('time_as_input', False) 
                    else None
                )
                
                # Reconstruct future visit using INR decoder
                inf_kwargs = {'time_val': time_val}
                volume_inf = atlas_builder.inr_decoder['val'].inference(
                    grid_coords, z_future, conditions, grid_shape, **inf_kwargs
                )
                
            # Extract modality images
            pred_imgs = atlas_builder._extract_modality_images(volume_inf)
            
            pred_visits.append({
                'week': t_future,
                'label': f"Transition@{int(t_future)}w (+{offset}w)",
                'images': pred_imgs,
                'is_gt': False
            })
            
        # 4. Log comparison figures for each modality
        for mod in args['dataset']['modalities']:
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
                
            # Align images to minimum dimensions
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
            
            # Make summary figure
            fig_summary = make_interleaved_figure(
                plot_images, plot_labels, plot_is_gts,
                title=f"{eye_id} - {mod} - Future Extrapolation via Latent Transition"
            )
            
            tag = f"future_transition/{eye_id}/{mod}_Extrapolation"
            args["tb_writer"].add_image(tag, fig_to_numpy(fig_summary), chkp_epoch, dataformats='HWC')
            plt.close(fig_summary)
            
    # Close TensorBoard SummaryWriter
    args["tb_writer"].close()
    print("\nFuture extrapolation and logging completed successfully!")

if __name__ == "__main__":
    main()
