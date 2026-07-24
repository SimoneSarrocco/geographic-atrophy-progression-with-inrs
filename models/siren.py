import numpy as np
import torch
from torch import nn
from models.omega_scheduler import get_omega_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SineLayer(nn.Module):
    def __init__(self, in_feat, lat_feat, out_feat, bias=True, is_first=False, omega=30):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.in_features = in_feat
        self.out_features = out_feat
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        self.linear_lats = nn.Linear(lat_feat, out_feat * 2, bias=bias) if lat_feat > 0 else None
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega,
                                            np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        """
        input: (coords, latent_vec)
        """
        intermed = self.linear(input[0])
        if self.linear_lats is not None:
            lats = self.linear_lats(input[1])
            out = torch.sin((self.omega * intermed * lats[..., :self.out_features]) + lats[..., self.out_features:])
        else:
            out = torch.sin(self.omega * intermed)
        return out, input[1]


class Siren(nn.Module):
    def __init__(self, in_size, lat_size, sr_dims, n_seg_channels, hidden_size, num_layers, omega_0, omega_start,
                 omega_end, schedule_type,
                 outermost_linear, modulated_layers,
                 seg_head_num_layers=0, seg_head_hidden_size=0, seg_head_use_last_features=False,
                 seg_branch_activate=False, seg_branch_num_layers=2, seg_branch_hidden_size=256,
                 seg_branch_layer=-1, seg_branch_modulate=True, shared_output_layer=False):
    # def __init__(self, in_size, lat_size, out_size, hidden_size, num_layers, omega_0, omega_start, omega_end, schedule_type,
    #             outermost_linear, modulated_layers):
        super().__init__()
        l_in_mod = 0 in modulated_layers
        omegas = get_omega_schedule(omega_0, omega_start, omega_end, num_layers, schedule_type)
        self.net = [SineLayer(in_size, lat_size * l_in_mod, hidden_size, is_first=True, omega=omegas[0])]
        self.hidden_size = hidden_size
        for i in range(num_layers):
            l_in_mod = (i+1) in modulated_layers
            self.net.append(SineLayer(hidden_size, lat_size * l_in_mod, hidden_size, is_first=False, omega=omegas[i+1]))

        # if outermost_linear:
        #    self.final_linear = nn.Linear(hidden_size+lat_size, out_size, bias=True)
        #    with torch.no_grad():
        #        self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / omegas[-1],
        #                                        np.sqrt(6 / hidden_size) / omegas[-1])
        # else:
        # self.final_linear = SineLayer(hidden_size, 0, out_size, is_first=False, omega=omegas[-1])
        # self.net = nn.Sequential(*self.net)

        # Shared output layer (no segmentation head at all): a SINGLE final layer maps to
        # [recon | seg] together — the original GAP-INR behaviour. When enabled, the dedicated
        # reconstruction/segmentation heads and the seg branch are NOT built and seg_head_*/seg_branch
        # config is ignored. Channel order is preserved (sr_dims first, then n_seg_channels), so all
        # downstream slicing (output[..., :sr_dims] / output[..., sr_dims:]) is unchanged.
        self.shared_output = bool(shared_output_layer)

        # Define separate heads for Reconstruction and Segmentation
        self.seg_head_use_last_features = bool(seg_head_use_last_features) and outermost_linear and not self.shared_output
        # Option B: a dedicated segmentation BRANCH — a short SIREN sub-network that taps a mid-trunk
        # layer (default: penultimate) and is FiLM-modulated by the latent, then a final seg linear.
        # It shares the trunk up to the branch point but decodes labels through its own layers,
        # giving segmentation more capacity/decoupling than the penultimate-tap head while keeping the
        # MetaSeg shared-representation premise. Only supported in the outermost_linear setup.
        self.seg_branch_active = bool(seg_branch_activate) and outermost_linear and not self.shared_output
        n_trunk = num_layers + 1  # total SineLayers in self.net (input layer + num_layers hidden)
        if self.seg_branch_active:
            # Resolve the branch point (index into self.net whose OUTPUT feeds the branch).
            self.branch_layer_idx = (n_trunk - 2) if (seg_branch_layer is None or seg_branch_layer < 0) \
                else min(int(seg_branch_layer), n_trunk - 1)
            self.seg_branch_modulate = bool(seg_branch_modulate)
            branch_omega = omegas[-1]
            self.seg_branch_layers = nn.ModuleList()
            prev = hidden_size  # every trunk layer outputs hidden_size
            for _ in range(int(seg_branch_num_layers)):
                self.seg_branch_layers.append(
                    SineLayer(prev, lat_size if self.seg_branch_modulate else 0,
                              seg_branch_hidden_size, is_first=False, omega=branch_omega))
                prev = seg_branch_hidden_size

        if self.shared_output:
            # Single shared output layer producing [recon | seg] in one map (no seg head/branch).
            out_dims = sr_dims + n_seg_channels
            if outermost_linear:
                self.final_linear = nn.Linear(hidden_size + lat_size, out_dims, bias=True)
                with torch.no_grad():
                    self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_size) / omegas[-1],
                                                      np.sqrt(6 / hidden_size) / omegas[-1])
            else:
                self.final_linear = SineLayer(hidden_size, 0, out_dims, is_first=False, omega=omegas[-1])
        elif outermost_linear:
            self.final_linear_rec = nn.Linear(hidden_size + lat_size, sr_dims, bias=True)
            with torch.no_grad():
                self.final_linear_rec.weight.uniform_(-np.sqrt(6 / hidden_size) / omegas[-1],
                                                    np.sqrt(6 / hidden_size) / omegas[-1])

            if self.seg_branch_active:
                # Final seg linear sits on top of the branch output, with the latent re-injected.
                self.final_linear_seg = nn.Linear(prev + lat_size, n_seg_channels, bias=True)
                with torch.no_grad():
                    self.final_linear_seg.weight.uniform_(-np.sqrt(6 / branch_omega) / branch_omega,
                                                        np.sqrt(6 / branch_omega) / branch_omega)
            else:
                # Segmentation head. Input is the penultimate features (+ optionally the last-layer
                # features) plus the latents. If a single linear map is weak,
                # seg_head_num_layers > 0 builds a small ReLU MLP instead (more decoding capacity).
                seg_in = hidden_size + lat_size + (hidden_size if self.seg_head_use_last_features else 0)
                if seg_head_num_layers and seg_head_num_layers > 0:
                    seg_layers, d_in = [], seg_in
                    for _ in range(seg_head_num_layers):
                        seg_layers += [nn.Linear(d_in, seg_head_hidden_size), nn.ReLU(inplace=True)]
                        d_in = seg_head_hidden_size
                    seg_layers.append(nn.Linear(d_in, n_seg_channels))
                    self.final_linear_seg = nn.Sequential(*seg_layers)
                else:
                    self.final_linear_seg = nn.Linear(seg_in, n_seg_channels, bias=True)
                    with torch.no_grad():
                        self.final_linear_seg.weight.uniform_(-np.sqrt(6 / hidden_size) / omegas[-1],
                                                            np.sqrt(6 / hidden_size) / omegas[-1])
        else:
            self.final_linear_rec = SineLayer(hidden_size, 0, sr_dims, is_first=False, omega=omegas[-1])
            self.final_linear_seg = SineLayer(hidden_size, 0, n_seg_channels, is_first=False, omega=omegas[-1])

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        coords, latents = x

        # Run the trunk, collecting every layer's output so the seg branch can tap any depth.
        feats = []
        cur, lat = coords, latents
        for layer in self.net:
            cur, lat = layer((cur, lat))
            feats.append(cur)
        features = feats[-1]                                   # last-layer output
        penultimate_features = feats[-2] if len(feats) >= 2 else feats[-1]

        if self.shared_output:
            # Single shared output layer: one map -> [recon | seg] (no seg head/branch).
            if isinstance(self.final_linear, nn.Linear):
                return self.final_linear(torch.cat([features, latents], dim=-1))
            out, _ = self.final_linear((features, None))
            return out

        if isinstance(self.final_linear_rec, nn.Linear):
            rec_out = self.final_linear_rec(torch.cat([features, latents], dim=-1))

            if self.seg_branch_active:
                # Dedicated seg branch: tap the branch-point features, run the seg SineLayers
                # (FiLM-modulated by the latent), then the final seg linear with latent re-injected.
                seg_x = feats[self.branch_layer_idx]
                for bl in self.seg_branch_layers:
                    seg_x, _ = bl((seg_x, latents))
                seg_out = self.final_linear_seg(torch.cat([seg_x, latents], dim=-1))
            else:
                seg_inputs = [penultimate_features]
                if self.seg_head_use_last_features:
                    seg_inputs.append(features)
                seg_inputs.append(latents)
                seg_out = self.final_linear_seg(torch.cat(seg_inputs, dim=-1))
        else:
            rec_out, _ = self.final_linear_rec((features, None))
            seg_out, _ = self.final_linear_seg((penultimate_features, None))

        return torch.cat([rec_out, seg_out], dim=-1)


