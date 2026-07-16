import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LatentPairDataset(Dataset):
    def __init__(self, train_df, train_lats, val_df=None, val_lats=None, val_opt_idcs=None, temporal_key='weeks_from_baseline'):
        """
        Constructs all combinations of visits for the same patient-eye (i != j).
        """
        self.pairs = []
        self.train_lats = train_lats
        self.val_lats = val_lats

        # 1. Build training pairs (all training visits were optimized)
        self._build_pairs(train_df, allowed_idcs=None, lats_name='train', temporal_key=temporal_key)

        # 2. Build validation pairs (only using non-held-out/optimized visits)
        if val_df is not None and val_lats is not None and val_opt_idcs is not None:
            self._build_pairs(val_df, allowed_idcs=val_opt_idcs, lats_name='val', temporal_key=temporal_key)

    def _build_pairs(self, df, allowed_idcs, lats_name, temporal_key):
        # Group indices by sub_id_int
        sub_groups = {}
        allowed_set = set(allowed_idcs) if allowed_idcs is not None else None
        for idx in range(len(df)):
            if allowed_set is not None and idx not in allowed_set:
                continue
            sub_id = df.iloc[idx]['sub_id_int']
            if sub_id not in sub_groups:
                sub_groups[sub_id] = []
            sub_groups[sub_id].append(idx)

        # For each patient-eye group, create all possible pairs
        for sub_id, indices in sub_groups.items():
            if len(indices) < 2:
                continue
            # Sort indices chronologically
            sorted_indices = sorted(indices, key=lambda idx: float(df.iloc[idx][temporal_key]))
            for i in sorted_indices:
                for j in sorted_indices:
                    if i == j:
                        continue
                    t_i = float(df.iloc[i][temporal_key])
                    t_j = float(df.iloc[j][temporal_key])
                    delta_t = t_j - t_i
                    self.pairs.append({
                        'src_idx': i,
                        'tgt_idx': j,
                        'delta_t': delta_t,
                        'lats_name': lats_name
                    })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lats = self.train_lats if pair['lats_name'] == 'train' else self.val_lats
        
        # Retrieve and clone optimized latents
        src_lat = lats[pair['src_idx']].detach().clone()
        tgt_lat = lats[pair['tgt_idx']].detach().clone()
        delta_t = torch.tensor([pair['delta_t']], dtype=torch.float32)
        return src_lat, delta_t, tgt_lat


class LatentTransitionMLP(nn.Module):
    def __init__(self, latent_shape, hidden_size=512, num_layers=3):
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        self.flat_dim = int(np.prod(self.latent_shape))

        # Input: flattened source latent + 1 (delta_t)
        in_dim = self.flat_dim + 1

        layers = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_size
        
        layers.append(nn.Linear(current_dim, self.flat_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, src_latent, delta_t):
        batch_size = src_latent.shape[0]
        flat_src = src_latent.reshape(batch_size, -1)
        x = torch.cat([flat_src, delta_t], dim=-1)
        out = self.mlp(x)
        return out.reshape(batch_size, *self.latent_shape)


class ResBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, is_3d=False):
        super().__init__()
        conv_cls = nn.Conv3d if is_3d else nn.Conv2d
        
        self.conv1 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, channels)

    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.relu(h)
        
        # Project time embedding and match dimension of feature maps
        t_shift = self.time_proj(time_emb)
        for _ in range(x.ndim - 2):
            t_shift = t_shift.unsqueeze(-1)
            
        h = h + t_shift
        h = self.conv2(h)
        h = self.relu(h)
        return x + h


class LatentTransitionResNet(nn.Module):
    def __init__(self, latent_shape, time_emb_dim=64, num_blocks=3):
        super().__init__()
        self.latent_shape = tuple(latent_shape)
        channels = self.latent_shape[0]
        is_3d = (len(self.latent_shape) == 4)
        
        # Embed delta time scalar to vector representation
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.ModuleList([
            ResBlock(channels, time_emb_dim, is_3d=is_3d) for _ in range(num_blocks)
        ])
        
        conv_cls = nn.Conv3d if is_3d else nn.Conv2d
        self.final_conv = conv_cls(channels, channels, kernel_size=3, padding=1)

    def forward(self, src_latent, delta_t):
        t_emb = self.time_mlp(delta_t)
        
        h = src_latent
        for block in self.blocks:
            h = block(h, t_emb)
            
        out = self.final_conv(h)
        return out
