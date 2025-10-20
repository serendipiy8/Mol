import torch
import numpy as np
from typing import Literal


def build_sigma_schedule(num_steps: int,
                         sigma_min: float = 0.5,
                         sigma_max: float = 1.0,
                         schedule: Literal['linear', 'cosine', 'quadratic', 'constant'] = 'linear',
                         device: str = 'cpu') -> torch.Tensor:
    """
    Build sigma (std) schedule matching SoftMaskTransform naming.
    """
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

