import abc

import torch
from torch import nn
import numpy as np


class Layer(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, **kwargs):
        super(Layer, self).__init__()
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        self.in_size = in_size
        self.out_size = out_size

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class WIRE(Layer):
    '''
        Implicit representation with Gabor nonlinearity

        Inputs;
            in_size: Input features
            out_size; Output features
            bias: if True, enable bias for the linear operation
            omega_0: Legacy SIREN parameter
            omega: Frequency of Gabor sinusoid term
            scale: Scaling of Gabor Gaussian term
    '''

    def __init__(self, in_size, out_size, bias=True, **kwargs):
        super().__init__(in_size, out_size, **kwargs)
        self.omega_0 = kwargs.get("wire_omega_0", 10.0)  # Freq
        self.scale_0 = kwargs.get("wire_scale_0", 10.0)
        self.freqs = nn.Linear(in_size, out_size, bias=bias)
        self.scale = nn.Linear(in_size, out_size, bias=bias)

    def forward(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.scale(x) * self.scale_0
        x = torch.cos(omega) * torch.exp(-(scale * scale))
        if self.dropout is not None:
            x = self.dropout(x)
        return x