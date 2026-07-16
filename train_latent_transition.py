import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add repo path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.latent_transition import LatentTransitionMLP, LatentTransitionResNet


def _decode_seg(decoder, latents, grid_coords, sr_dims, n_seg):
    """Decode a batch of latents (B,C,H,W) at grid_coords (N,2) through the frozen INR decoder.

    Returns:
        seg_logits: (B, N, n_seg)
        ga_area:    (B,)  mean GA-class probability over the grid (a soft lesion area).
    Assumes the Stage-1 decoder uses cond_dims 0 and time_as_input False (per-visit config),
    so no conditioning/time inputs are needed.
    """
    B, N = latents.shape[0], grid_coords.shape[0]
    coords_all = grid_coords.unsqueeze(0).expand(B, N, 2).reshape(B * N, 2)
    idcs = torch.arange(B, device=latents.device).repeat_interleave(N).unsqueeze(1)
    cond = torch.zeros(B * N, 0, device=latents.device, dtype=latents.dtype)
    out = decoder(coords_all, latents, cond, idcs_df=idcs, time_vals=None)  # (B*N, sr_dims+n_seg)
    seg_logits = out[:, sr_dims:sr_dims + n_seg].reshape(B, N, n_seg)
    ga_area = torch.softmax(seg_logits, dim=-1)[..., 1].mean(dim=1)  # (B,)
    return seg_logits, ga_area


class TransitionDataset(Dataset):
    def __init__(self, latents, pairs):
        self.latents = latents
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # Retrieve and clone optimized latents from the checkpoint tensor
        src_lat = self.latents[pair['src_idx']].detach().clone()
        tgt_lat = self.latents[pair['tgt_idx']].detach().clone()
        delta_t = torch.tensor([pair['delta_t']], dtype=torch.float32)
        return src_lat, delta_t, tgt_lat


def build_split_pairs(df, subjects, temporal_key='weeks_from_baseline'):
    # Group VISITS BY PATIENT-EYE so pairs connect visits of the SAME eye.
    # IMPORTANT: under independent_visits, sub_id_int is per-VISIT (unique per row), so
    # grouping by sub_id_int yields singletons -> zero pairs. Group by Eye_ID (the true
    # per-eye key) instead, and keep an eye if any of its rows' sub_id_int is in `subjects`
    # (so the existing sub_id_int-based split selection still works in both modes).
    group_col = 'Eye_ID' if 'Eye_ID' in df.columns else 'sub_id_int'
    subjects_set = set(subjects)
    sub_groups = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        if row['sub_id_int'] in subjects_set:
            key = row[group_col]
            sub_groups.setdefault(key, []).append(idx)

    pairs = []
    # For each group, sort chronologically and create forward pairs (t_j > t_i)
    for sub_id, indices in sub_groups.items():
        if len(indices) < 2:
            continue
        sorted_indices = sorted(indices, key=lambda idx: float(df.iloc[idx][temporal_key]))
        for i_pos, i in enumerate(sorted_indices):
            for j in sorted_indices[i_pos + 1:]:
                t_i = float(df.iloc[i][temporal_key])
                t_j = float(df.iloc[j][temporal_key])
                delta_t = t_j - t_i
                pairs.append({
                    'src_idx': i,
                    'tgt_idx': j,
                    'delta_t': delta_t
                })
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Transition Model Standalone")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained INR checkpoint (.pth)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Dataloader batch size")
    parser.add_argument("--type", type=str, choices=["mlp", "resnet"], default="resnet", help="Model architecture")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size for MLP model")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers for MLP model")
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Time embedding dimension for ResNet")
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of blocks for ResNet")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of subjects to hold out for validation")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="output_transition", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=1927, help="Random seed")
    # --- Decoder-space loss + monotonicity (gated; default OFF -> pure latent-MSE as before) ---
    # decoder_weight>0: also decode predicted vs target latent through the FROZEN INR decoder and
    #   penalise the SEGMENTATION difference -> optimises latents in the space we care about
    #   (a small latent error can decode to a large mask error). Self-contained: uses only the
    #   decoder + latents in the checkpoint (assumes the Stage-1 decoder has cond_dims 0 / no time).
    # mono_weight>0: penalise GA-area DECREASE from src->pred (Δt>0) -> GA grows monotonically.
    parser.add_argument("--decoder_weight", type=float, default=0.0, help="weight of decoder-space seg loss (0=off)")
    parser.add_argument("--mono_weight", type=float, default=0.0, help="weight of GA-area monotonicity penalty (0=off)")
    parser.add_argument("--recon_res", type=int, default=64, help="grid resolution for decoder-space loss")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 2. Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"INR checkpoint not found at: {args.checkpoint}")
    print(f"Loading INR checkpoint from: {args.checkpoint}")
    chkp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    latents = chkp['latents']  # shape (N_visits, C, X_1, X_2)
    df = chkp['dataset_df']
    inr_args = chkp['args']
    temporal_key = inr_args['dataset'].get('temporal_condition', 'weeks_from_baseline')

    print(f"Loaded {len(latents)} latent codes and {len(df)} visits.")

    # 4. Split subjects
    unique_subs = sorted(df['sub_id_int'].unique())
    random.shuffle(unique_subs)
    val_count = int(len(unique_subs) * args.val_split)
    
    val_subs = unique_subs[:val_count]
    train_subs = unique_subs[val_count:]
    
    print(f"Split unique subjects into {len(train_subs)} training and {len(val_subs)} validation patient-eyes.")

    # 5. Build pairs
    train_pairs = build_split_pairs(df, train_subs, temporal_key)
    val_pairs = build_split_pairs(df, val_subs, temporal_key)
    
    print(f"Built {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")
    
    if len(train_pairs) == 0:
        print("Error: No training pairs found! Make sure the dataset has longitudinal visits.")
        sys.exit(1)

    # 6. Datasets & Dataloaders
    train_dataset = TransitionDataset(latents, train_pairs)
    val_dataset = TransitionDataset(latents, val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 7. Create model
    latent_shape = latents.shape[1:]
    if args.type == 'mlp':
        model = LatentTransitionMLP(
            latent_shape=latent_shape,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        ).to(device)
    elif args.type == 'resnet':
        model = LatentTransitionResNet(
            latent_shape=latent_shape,
            time_emb_dim=args.time_emb_dim,
            num_blocks=args.num_blocks
        ).to(device)
    
    print(f"Instantiated {args.type.upper()} model for latent shape {latent_shape}.")

    # 7b. Optional: frozen INR decoder for decoder-space loss + monotonicity (gated)
    use_decoder = (args.decoder_weight > 0.0) or (args.mono_weight > 0.0)
    decoder = grid_coords_dec = None
    sr_dims = n_seg = 0
    seg_ce = nn.CrossEntropyLoss()
    if use_decoder:
        from models.inr_decoder import INR_Decoder
        out_dim = inr_args['inr_decoder']['out_dim']
        sr_dims, n_seg = int(sum(out_dim[:-1])), int(out_dim[-1])
        decoder = INR_Decoder(inr_args, device).to(device)
        decoder.load_state_dict(chkp['inr_decoder'])
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        r = args.recon_res
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, r), torch.linspace(-1, 1, r), indexing='ij')
        grid_coords_dec = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).to(device)  # (r*r, 2)
        print(f"Decoder-space loss ON: decoder_weight={args.decoder_weight}, mono_weight={args.mono_weight}, "
              f"grid={r}x{r}, sr_dims={sr_dims}, n_seg={n_seg}")

    def _extra_losses(src_lats, pred_lats, tgt_lats):
        """decoder-space seg loss (pred vs decoded target) + GA-area monotonicity. Returns a scalar."""
        extra = pred_lats.new_zeros(())
        pred_seg, area_pred = _decode_seg(decoder, pred_lats, grid_coords_dec, sr_dims, n_seg)
        if args.decoder_weight > 0.0:
            with torch.no_grad():
                tgt_seg, _ = _decode_seg(decoder, tgt_lats, grid_coords_dec, sr_dims, n_seg)
                tgt_lab = tgt_seg.argmax(dim=-1).reshape(-1)
            extra = extra + args.decoder_weight * seg_ce(pred_seg.reshape(-1, n_seg), tgt_lab)
        if args.mono_weight > 0.0:
            with torch.no_grad():
                _, area_src = _decode_seg(decoder, src_lats, grid_coords_dec, sr_dims, n_seg)
            extra = extra + args.mono_weight * torch.relu(area_src - area_pred).mean()
        return extra

    # 8. Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for src_lats, delta_ts, tgt_lats in train_loader:
            src_lats = src_lats.to(device)
            delta_ts = delta_ts.to(device)
            tgt_lats = tgt_lats.to(device)

            optimizer.zero_grad()
            pred_lats = model(src_lats, delta_ts)
            loss = criterion(pred_lats, tgt_lats)
            if use_decoder:
                loss = loss + _extra_losses(src_lats, pred_lats, tgt_lats)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)
        history['train_loss'].append(mean_train_loss)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for src_lats, delta_ts, tgt_lats in val_loader:
                src_lats = src_lats.to(device)
                delta_ts = delta_ts.to(device)
                tgt_lats = tgt_lats.to(device)

                pred_lats = model(src_lats, delta_ts)
                loss = criterion(pred_lats, tgt_lats)
                if use_decoder:
                    loss = loss + _extra_losses(src_lats, pred_lats, tgt_lats)
                val_losses.append(loss.item())

        mean_val_loss = np.mean(val_losses) if len(val_losses) > 0 else 0.0
        history['val_loss'].append(mean_val_loss)

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} | Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}")

        # Checkpoint saving
        state_dict = {
            'model_state_dict': model.state_dict(),
            'latent_shape': latent_shape,
            'model_type': args.type,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'time_emb_dim': args.time_emb_dim,
            'num_blocks': args.num_blocks,
            'epoch': epoch + 1
        }
        
        # Save best model
        if mean_val_loss < best_val_loss and len(val_losses) > 0:
            best_val_loss = mean_val_loss
            best_path = os.path.join(args.output_dir, "latent_transition_best.pth")
            torch.save(state_dict, best_path)

        # Save last model
        last_path = os.path.join(args.output_dir, "latent_transition_last.pth")
        torch.save(state_dict, last_path)

    # Save log history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("\n--- Training Complete ---")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Checkpoints and logs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
