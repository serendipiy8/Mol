import torch
import numpy as np
from typing import Literal, Union


def build_alpha_schedule(num_steps: int,
                         schedule: Literal['linear', 'cosine', 'quadratic'] = 'linear',
                         device: str = 'cpu') -> torch.Tensor:
    """Build alpha(t) schedule with same naming as SoftMaskTransform."""
    if schedule == 'linear':
        alpha = torch.linspace(0, 1, num_steps + 1, device=device)
    elif schedule == 'cosine':
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        alpha = 0.5 * (1 + torch.cos(np.pi * (1 - t)))
    elif schedule == 'quadratic':
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        alpha = t ** 2
    else:
        raise ValueError(f'Unknown alpha schedule: {schedule}')
    return alpha


def compute_soft_mask(alpha_values: torch.Tensor,
                      tau: torch.Tensor,
                      t: Union[int, torch.Tensor],
                      kappa: float) -> torch.Tensor:
    """Compute s_t = sigmoid((alpha(t) - tau)/kappa)."""
    if isinstance(t, int):
        t_idx = torch.tensor(t, device=alpha_values.device, dtype=torch.long)
    else:
        t_idx = t.to(torch.long)
    alpha_t = alpha_values[t_idx]
    return torch.sigmoid((alpha_t - tau) / kappa)

