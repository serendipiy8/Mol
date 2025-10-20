import math
import torch
import torch.nn.functional as F
from typing import Tuple


class DiffusionLoss:
    """
    Loss functions for LO-MaskDiff training (supports single-/multi-modal).
    Implements weighted reconstruction loss and stable KL for tau params.
    """

    def __init__(self,
                 loss_type: str = 'mse',
                 reweighting: bool = True,
                 w_max: float = 1e3,
                 eps: float = 1e-8):
        self.loss_type = loss_type
        self.reweighting = reweighting
        self.w_max = w_max
        self.eps = eps

    def compute_loss(self,
                     x0: torch.Tensor,
                     x0_pred: torch.Tensor,
                     s_t: torch.Tensor,
                     sigma_t: torch.Tensor) -> torch.Tensor:
        """
        Weighted reconstruction loss per LO-MaskDiff: w = 1/(((1-s)^2)*sigma^2 + eps).
        """
        if self.loss_type == 'mse':
            base_loss = F.mse_loss(x0_pred, x0, reduction='none')
        elif self.loss_type == 'l1':
            base_loss = F.l1_loss(x0_pred, x0, reduction='none')
        elif self.loss_type == 'huber':
            base_loss = F.huber_loss(x0_pred, x0, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        base_loss = base_loss.sum(dim=-1)  # [N_nodes]

        if self.reweighting:
            weights = 1.0 / (((1.0 - s_t) ** 2) * (sigma_t ** 2) + self.eps)
            weights = torch.clamp(weights, max=self.w_max)
            weights = weights / (weights.mean() + self.eps)
            base_loss = base_loss * weights

        return base_loss.mean()

    def compute_loss_multi_modal(self,
                                 x0_coord: torch.Tensor,
                                 x0_pred: torch.Tensor,
                                 h0_feat: torch.Tensor,
                                 h0_pred: torch.Tensor,
                                 s_t: torch.Tensor,
                                 sigma_coord: torch.Tensor,
                                 sigma_feat: torch.Tensor,
                                 lambda_feat: float = 1.0) -> torch.Tensor:
        loss_coord = self.compute_loss(x0_coord, x0_pred, s_t, sigma_coord)
        loss_feat = self.compute_loss(h0_feat, h0_pred, s_t, sigma_feat)
        return loss_coord + lambda_feat * loss_feat

    def compute_kl_loss(self,
                        mu: torch.Tensor,
                        log_sigma: torch.Tensor,
                        prior_mu: float = 0.0,
                        prior_sigma: float = 1.0) -> torch.Tensor:
        """
        KL(N(mu, sigma) || N(prior_mu, prior_sigma)) in closed form.
        Uses clamped log_sigma and stable log(prior_sigma) handling.
        """
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=10.0)
        sigma = torch.exp(log_sigma)

        prior_sigma_t = torch.tensor(prior_sigma, device=mu.device, dtype=mu.dtype)
        prior_mu_t = torch.tensor(prior_mu, device=mu.device, dtype=mu.dtype)
        log_prior_sigma = torch.log(prior_sigma_t)

        kl_div = 0.5 * (
            (sigma / prior_sigma_t) ** 2 +
            ((mu - prior_mu_t) / prior_sigma_t) ** 2 -
            1.0 +
            2.0 * (log_prior_sigma - log_sigma)
        )

        return kl_div.mean()
