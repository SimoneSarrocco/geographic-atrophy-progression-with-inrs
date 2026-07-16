import math
import torch


def get_omega_schedule(omega_0, omega_start, omega_end, num_hidden_layers, schedule_type='linear'):
    """
    Compute per-layer omega values for a SIREN network.

    Produces a list of length (1 + num_hidden_layers):
      - omegas[0]  = omega_0  (first / input layer)
      - omegas[1..K] = scheduled values from omega_start to omega_end across hidden layers

    Inspired by medfuncta's w0_utils.get_w0s().

    Args:
        omega_0:  omega for the first (input) layer.
        omega_start: omega for the first hidden layer (hidden layer 1).
        omega_end:  omega for the last hidden layer (hidden layer K).
        num_hidden_layers:  number of hidden layers (K).
        schedule_type:  'linear', 'exponential', or 'constant'.

    Returns:
        List[float] of length (1 + num_hidden_layers).
    """
    if num_hidden_layers < 1:
        return [omega_0]

    # hidden layer omegas (indices 1..K)
    if schedule_type == 'linear':
        if num_hidden_layers == 1:
            hidden_omegas = [omega_end]
        else:
            # Linearly interpolate from omega_start to omega_end across the hidden layers
            hidden_omegas = [
                omega_start + (omega_end - omega_start) * i / (num_hidden_layers - 1)
                for i in range(num_hidden_layers)
            ]
    elif schedule_type == 'exponential':
        if num_hidden_layers == 1:
            hidden_omegas = [omega_end]
        else:
            # Exponential interpolation from omega_start to omega_end
            ratio = (omega_end / omega_start) ** (1.0 / (num_hidden_layers - 1))
            hidden_omegas = [omega_start * (ratio ** i) for i in range(num_hidden_layers)]
    elif schedule_type == 'constant':
        # All hidden layers use omega_end
        hidden_omegas = [omega_end] * num_hidden_layers
    else:
        raise ValueError(
            f"Unknown omega schedule type: '{schedule_type}'. "
            f"Choose from 'linear', 'exponential', or 'constant'."
        )

    # First layer always uses omega_0 exactly
    # Full list length: 1 (input) + num_hidden_layers (hidden)
    omegas = [omega_0] + hidden_omegas
    return omegas
