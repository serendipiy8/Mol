import inspect
import torch
from typing import Dict
from torch_geometric.data import Data

from .losses import DiffusionLoss
from .diffusion_process import SoftMaskDiffusionProcess


class DiffusionTrainer:
    """
    Training utilities for LO-MaskDiff diffusion process.
    Expects that batch already contains `tau` (recommended via SoftMaskDataTransform).
    """

    def __init__(self,
                 diffusion_process: SoftMaskDiffusionProcess,
                 loss_fn: DiffusionLoss,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda',
                 lambda_feat: float = 1.0):
        self.diffusion_process = diffusion_process
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.lambda_feat = lambda_feat

    def _assert_batch(self, batch: Data) -> None:
        if not hasattr(batch, 'ligand_pos') or batch.ligand_pos is None:
            raise ValueError('Batch must contain ligand_pos tensor')
        if not hasattr(batch, 'tau') or batch.tau is None:
            raise ValueError('Batch must contain tau; apply SoftMaskDataTransform before training')

    def training_step(self,
                     batch: Data,
                     model,
                     soft_mask_transform) -> Dict[str, float]:
        """
        Single training step. Assumes batch.tau exists.
        """
        self._assert_batch(batch)

        # Devices: align to trainer device
        x0 = batch.ligand_pos.to(self.device)
        tau = batch.tau.to(self.device)

        has_feat = hasattr(batch, 'ligand_atom_feature') and batch.ligand_atom_feature is not None

        # Sample random time steps (per node)
        t = torch.randint(0, self.diffusion_process.num_steps, (x0.size(0),), device=self.device, dtype=torch.long)

        if has_feat:
            h0 = batch.ligand_atom_feature.to(self.device)
            x_t, h_t, s_t = self.diffusion_process.forward_process_multi_modal(x0, h0, tau, t, soft_mask_transform)
            forward_fn = getattr(model, 'forward', model)
            sig = inspect.signature(forward_fn)
            if 'h_t' in sig.parameters:
                model_out = model(x_t, s_t, t, batch, h_t)
            else:
                model_out = model(x_t, s_t, t, batch)
            if isinstance(model_out, (tuple, list)) and len(model_out) == 2:
                x0_pred, h0_pred = model_out
            else:
                x0_pred, h0_pred = model_out, None
            sigma_coord = self.diffusion_process.sigmas[t]
            sigma_feat = self.diffusion_process.sigmas[t]
            if h0_pred is not None:
                diffusion_loss = self.loss_fn.compute_loss_multi_modal(x0, x0_pred, h0, h0_pred, s_t, sigma_coord, sigma_feat, self.lambda_feat)
            else:
                diffusion_loss = self.loss_fn.compute_loss(x0, x0_pred, s_t, sigma_coord)
        else:
            x_t, s_t = self.diffusion_process.forward_process(x0, tau, t, soft_mask_transform)
            x0_pred = model(x_t, s_t, t, batch)
            sigma_t = self.diffusion_process.sigmas[t]
            diffusion_loss = self.loss_fn.compute_loss(x0, x0_pred, s_t, sigma_t)

        # KL loss for τ parameters if present on batch
        if hasattr(batch, 'tau_mu') and hasattr(batch, 'tau_log_sigma') and batch.tau_mu is not None and batch.tau_log_sigma is not None:
            kl_loss = self.loss_fn.compute_kl_loss(batch.tau_mu.to(self.device), batch.tau_log_sigma.to(self.device))
        else:
            kl_loss = torch.tensor(0.0, device=self.device)

        total_loss = diffusion_loss + 0.1 * kl_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'kl_loss': kl_loss.item()
        }


