import torch
import numpy as np
from typing import Literal, Optional


def build_sigma_schedule(num_steps: int, sigma_min: float = 0.5, sigma_max: float = 1.0,
                         schedule: Literal['linear', 'cosine', 'quadratic', 'constant'] = 'linear',
                         device: str = 'cuda',
                         bind_alpha: bool = False,
                         alpha_values: Optional[torch.Tensor] = None) -> torch.Tensor:
                         
    if bind_alpha:
        if alpha_values is None:
            raise ValueError('bind_alpha=True requires alpha_values tensor')
        # Map alpha(t) in [0,1] to sigma range [sigma_min, sigma_max].
        # alpha_values is length T+1; we use first num_steps entries to align with noise steps.
        base = alpha_values[:num_steps]
        sigmas = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * base
    else:
        if schedule == 'constant':
            sigmas = torch.full((num_steps,), float(sigma_max))
        elif schedule == 'linear':
            sigmas = torch.linspace(float(sigma_min), float(sigma_max), num_steps)
        elif schedule == 'cosine':
            t = torch.linspace(0, 1, num_steps)
            base = 0.5 * (1 + torch.cos(np.pi * (1 - t)))
            sigmas = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * base
        elif schedule == 'quadratic':
            t = torch.linspace(0, 1, num_steps)
            base = t ** 2
            sigmas = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * base
        else:
            raise ValueError(f"Unknown sigma schedule: {schedule}")

    return sigmas.to(device)