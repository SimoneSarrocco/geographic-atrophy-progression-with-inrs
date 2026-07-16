import torch
import torch.nn as nn
import numpy as np
import math

PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

@torch.no_grad()
def fast_hash(ind: torch.Tensor, primes: torch.Tensor, hashmap_size: int):
    """Hashing function from:
    https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h#L76-L92
    """
    d = ind.shape[-1]
    ind = (ind * primes[:d]) & 0xffffffff  # uint32
    for i in range(1, d):
        ind[..., 0] ^= ind[..., i]
    return ind[..., 0] % hashmap_size


class _HashGrid(nn.Module):
    def __init__(self, dim: int, n_features: int, hashmap_size: int, resolution: float):
        super().__init__()
        self.dim = dim
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.resolution = resolution

        assert self.dim <= len(PRIMES), f"HashGrid only supports < {len(PRIMES)}-D inputs"

        self.embedding = nn.Embedding(hashmap_size, n_features)
        nn.init.uniform_(self.embedding.weight, a=-0.0001, b=0.0001)

        primes = torch.tensor(PRIMES, dtype=torch.int64)
        self.register_buffer('primes', primes, persistent=False)

        n_neigs = 1 << self.dim
        neigs = np.arange(n_neigs, dtype=np.int64).reshape((-1, 1))
        dims = np.arange(self.dim, dtype=np.int64).reshape((1, -1))
        bin_mask = torch.tensor(neigs & (1 << dims) == 0, dtype=bool)
        self.register_buffer('bin_mask', bin_mask, persistent=False)

    def forward(self, x: torch.Tensor):
        bdims = len(x.shape[:-1])
        x = x * self.resolution
        xi = x.long()
        xf = x - xi.float().detach()
        xi = xi.unsqueeze(dim=-2)
        xf = xf.unsqueeze(dim=-2)
        bin_mask = self.bin_mask.reshape((1,) * bdims + self.bin_mask.shape)
        inds = torch.where(bin_mask, xi, xi + 1)
        ws = torch.where(bin_mask, 1 - xf, xf)
        w = ws.prod(dim=-1, keepdim=True)
        hash_ids = fast_hash(inds, self.primes, self.hashmap_size)
        neig_data = self.embedding(hash_ids)
        return torch.sum(neig_data * w, dim=-2)


class HashGridEncoding(nn.Module):
    def __init__(self, in_dim: int, n_levels: int = 16, n_features_per_level: int = 2,
                 log2_hashmap_size: int = 15, base_resolution: int = 16, finest_resolution: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        
        b = math.exp((math.log(finest_resolution) - math.log(base_resolution)) / (n_levels - 1))
        levels = []
        for level_idx in range(n_levels):
            resolution = math.floor(base_resolution * (b ** level_idx))
            hashmap_size = min(resolution ** in_dim, 2 ** log2_hashmap_size)
            levels.append(_HashGrid(dim=in_dim, n_features=n_features_per_level, 
                                   hashmap_size=hashmap_size, resolution=resolution))
        self.levels = nn.ModuleList(levels)
        self.out_dim = n_levels * n_features_per_level

    def forward(self, x: torch.Tensor):
        # Shift coords from [-1, 1] to [0, 1] for hashgrid
        x = (x + 1.0) / 2.0
        return torch.cat([level(x) for level in self.levels], dim=-1)


class PosEncodingFourier(nn.Module):
    """Standard NeRF-style positional encoding with deterministic frequencies."""
    def __init__(self, in_dim: int, num_frequencies: int = 10, log_sampling: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        
        if log_sampling:
            freq_bands = 2.**torch.linspace(0., num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(num_frequencies - 1), num_frequencies)
            
        self.register_buffer('freq_bands', freq_bands)
        self.out_dim = in_dim + 2 * in_dim * num_frequencies

    def forward(self, x: torch.Tensor):
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)


class PosEncodingGaussian(nn.Module):
    """Random Fourier Features (Gaussian encoding)."""
    def __init__(self, in_dim: int, num_frequencies: int = 256, scale: float = 10.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        
        # B matrix maps input to random frequencies: y = cos(2*pi * Bx), sin(2*pi * Bx)
        self.register_buffer('B', torch.randn(in_dim, num_frequencies) * scale)
        self.out_dim = 2 * num_frequencies

    def forward(self, x: torch.Tensor):
        # x: (N, in_dim)
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class IdentityEncoding(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor):
        return x


class TimeEncoding(nn.Module):
    """Encodes a scalar temporal input (the time-as-input coordinate): Fourier features
    (kind='fourier' / num_frequencies>0), a bare raw scalar (num_frequencies=0), or a learned MLP
    embedding (kind='mlp')."""
    def __init__(self, num_frequencies=6, kind='fourier', mlp_hidden=16, mlp_out=16, mlp_layers=2):
        super().__init__()
        self.kind = (kind or 'fourier').lower()
        if self.kind == 'mlp':
            self.mlp = MLPConditionEncoding(1, hidden_dim=mlp_hidden, out_dim=mlp_out, n_layers=mlp_layers)
            self.out_dim = mlp_out
            self.num_frequencies = 0
        else:
            self.num_frequencies = num_frequencies
            if num_frequencies > 0:
                freq_bands = 2.**torch.linspace(0., num_frequencies - 1, num_frequencies)
                self.register_buffer('freq_bands', freq_bands)
                self.out_dim = 1 + 2 * num_frequencies  # raw + sin + cos
            else:
                self.out_dim = 1  # raw only (bare scalar)

    def forward(self, t):
        if self.kind == 'mlp':
            return self.mlp(t)
        if self.num_frequencies > 0:
            out = [t]
            for freq in self.freq_bands:
                out.append(torch.sin(t * freq * math.pi))
                out.append(torch.cos(t * freq * math.pi))
            return torch.cat(out, dim=-1)
        return t


class ConditionEncoding(nn.Module):
    """Encodes a multi-dimensional conditioning input with Fourier features."""
    def __init__(self, in_dim: int, num_frequencies: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        if num_frequencies > 0:
            freq_bands = 2.**torch.linspace(0., num_frequencies - 1, num_frequencies)
            self.register_buffer('freq_bands', freq_bands)
            self.out_dim = in_dim + 2 * in_dim * num_frequencies  # raw + sin + cos per dimension
        else:
            self.out_dim = in_dim
    
    def forward(self, x):
        if self.num_frequencies > 0:
            out = [x]
            for freq in self.freq_bands:
                out.append(torch.sin(x * freq * math.pi))
                out.append(torch.cos(x * freq * math.pi))
            return torch.cat(out, dim=-1)
        else:
            return x


class MLPConditionEncoding(nn.Module):
    """Learned smooth embedding of the conditioning variable via a small MLP.

    Unlike Fourier features it imposes no fixed high-frequency basis, so the learned
    temporal code stays smooth -> well-behaved interpolation and (especially) extrapolation,
    which matters for monotone biological progression (e.g. GA growth).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 16, n_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        n_layers = max(1, n_layers)
        if n_layers == 1:
            layers = [nn.Linear(in_dim, out_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SeparateMLPConditionEncoding(nn.Module):
    """One INDEPENDENT MLP per conditioning variable (each 1 -> out_dim), concatenated. Lets each
    temporal variable (e.g. AgeatVisit, weeks_from_baseline) learn its OWN embedding rather than
    sharing a single joint MLP over the stacked vector. out_dim_total = in_dim * out_dim."""
    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 16, n_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.mlps = nn.ModuleList([
            MLPConditionEncoding(1, hidden_dim=hidden_dim, out_dim=out_dim, n_layers=n_layers)
            for _ in range(in_dim)])
        self.out_dim = in_dim * out_dim

    def forward(self, x):
        # x: (..., in_dim) -> (..., in_dim * out_dim) = concat of per-variable embeddings
        return torch.cat([self.mlps[i](x[..., i:i + 1]) for i in range(self.in_dim)], dim=-1)


def get_condition_encoding(in_dim: int, kind: str = 'fourier', num_frequencies: int = 4,
                           mlp_hidden: int = 16, mlp_out: int = 16, mlp_layers: int = 2):
    """Factory for the conditioning-variable encoder.

    kind: 'fourier' (Fourier features), 'mlp' (single joint learned embedding over the stacked vector),
          'mlp_separate' (one MLP PER variable, concatenated), or 'raw'/'none'/'identity' (unchanged).
    """
    kind = (kind or 'fourier').lower()
    if in_dim <= 0:
        return ConditionEncoding(in_dim, 0)
    if kind in ('mlp_separate', 'mlp_sep', 'separate_mlp'):
        return SeparateMLPConditionEncoding(in_dim, hidden_dim=mlp_hidden, out_dim=mlp_out, n_layers=mlp_layers)
    if kind == 'mlp':
        return MLPConditionEncoding(in_dim, hidden_dim=mlp_hidden, out_dim=mlp_out, n_layers=mlp_layers)
    if kind in ('raw', 'none', 'identity'):
        return ConditionEncoding(in_dim, 0)
    return ConditionEncoding(in_dim, num_frequencies)  # default: 'fourier'


def get_encoding(args_inr, args_enc=None):
    """Factory function for coordinate encodings."""
    in_dim = args_inr['in_dim']
    if args_enc is None or not args_enc.get('activate', False):
        return IdentityEncoding(in_dim)
    
    enc_type = args_enc.get('type', 'hash').lower()
    
    if enc_type == 'hash':
        return HashGridEncoding(
            in_dim=in_dim,
            n_levels=args_enc.get('n_levels', 16),
            n_features_per_level=args_enc.get('n_features_per_level', 2),
            log2_hashmap_size=args_enc.get('log2_hashmap_size', 15),
            base_resolution=args_enc.get('base_resolution', 16),
            finest_resolution=args_enc.get('finest_resolution', 512)
        )
    elif enc_type == 'fourier':
        return PosEncodingFourier(
            in_dim=in_dim,
            num_frequencies=args_enc.get('num_frequencies', 10),
            log_sampling=args_enc.get('log_sampling', True)
        )
    elif enc_type == 'gaussian':
        return PosEncodingGaussian(
            in_dim=in_dim,
            num_frequencies=args_enc.get('num_frequencies', 256),
            scale=args_enc.get('scale', 10.0)
        )
    else:
        print(f"WARNING: Unknown encoding type '{enc_type}'. Falling back to Identity.")
        return IdentityEncoding(in_dim)
