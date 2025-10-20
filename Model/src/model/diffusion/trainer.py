import inspect
import torch
from typing import Dict
from torch_geometric.data import Data

from .losses import DiffusionLoss
from .diffusion_process import SoftMaskDiffusionProcess


class DiffusionTrainer:
    def __init__(self, diffusion_process: SoftMaskDiffusionProcess, loss_fn: DiffusionLoss,
                 optimizer: torch.optim.Optimizer, device: str = 'cuda',
                 lambda_feat: float = 1.0):

        self.diffusion_process = diffusion_process
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.lambda_feat = lambda_feat

    def _assert_batch(self, batch: Data) -> None:
        if not hasattr(batch, 'ligand_pos') or batch.ligand_pos is None:
            raise ValueError('Batch must contain ligand_pos tensor')

    def _sample_tau_logistic_normal(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        # Logistic-Normal reparam: tau = sigmoid(mu + exp(clamp(log_sigma))*eps)
        eps = torch.randn_like(mu)
        sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0))
        u = mu + sigma * eps
        return torch.sigmoid(u)

    def _get_tau_params_from_model(self, model, batch: Data):
        # Prefer explicit tau param head if available
        if hasattr(model, 'predict_tau_params'):
            fn = getattr(model, 'predict_tau_params')
        elif hasattr(model, 'q_phi_head'):
            fn = getattr(model, 'q_phi_head')
        else:
            return None

        sig = inspect.signature(fn)
        if 'batch' in sig.parameters:
            out = fn(batch)
        else:
            return None

        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu, log_sigma = out
            return mu, log_sigma
        return None

    def _assert_shapes_pre(self, x0: torch.Tensor, tau: torch.Tensor, t: torch.Tensor) -> None:
        if x0.ndim != 2 or x0.size(-1) != 3:
            raise ValueError(f'ligand_pos must be [N,3], got {tuple(x0.shape)}')
        if tau.ndim != 1 or tau.size(0) != x0.size(0):
            raise ValueError(f'tau must be [N], got {tuple(tau.shape)} vs N={x0.size(0)}')
        if t.ndim != 1 or t.size(0) != x0.size(0):
            raise ValueError(f't must be [N], got {tuple(t.shape)} vs N={x0.size(0)}')
        dev = self.device
        for ten in (x0, tau, t):
            if ten.device.type != torch.device(dev).type:
                raise ValueError(f'Device mismatch: expected {dev}, got {ten.device}')

    def _assert_shapes_post_coord(self, x_t: torch.Tensor, s_t: torch.Tensor, x0: torch.Tensor) -> None:
        if x_t.shape != x0.shape:
            raise ValueError(f'x_t shape {tuple(x_t.shape)} must match x0 {tuple(x0.shape)}')
        if s_t.ndim != 1 or s_t.size(0) != x0.size(0):
            raise ValueError(f's_t must be [N], got {tuple(s_t.shape)} vs N={x0.size(0)}')
        for ten in (x_t, s_t):
            if ten.device != x0.device:
                raise ValueError('Post-forward device mismatch on coordinates path')

    def _assert_shapes_post_multi(self, h0: torch.Tensor, h_t: torch.Tensor, s_t: torch.Tensor, x0: torch.Tensor) -> None:
        if h0.ndim != 2 or h_t.ndim != 2 or h0.shape != h_t.shape:
            raise ValueError(f'h0/h_t must be [N,d] same shape, got {tuple(h0.shape)} vs {tuple(h_t.shape)}')
        if h0.size(0) != x0.size(0):
            raise ValueError('Feature N must equal coordinate N')
        if s_t.ndim != 1 or s_t.size(0) != x0.size(0):
            raise ValueError(f's_t must be [N], got {tuple(s_t.shape)}')
        for ten in (h0, h_t, s_t):
            if ten.device != x0.device:
                raise ValueError('Post-forward device mismatch on multi-modal path')

    def _assert_model_outputs(self, x0_pred: torch.Tensor, x0: torch.Tensor, h0_pred: torch.Tensor = None, h0: torch.Tensor = None) -> None:
        if x0_pred.shape != x0.shape:
            raise ValueError(f'x0_pred shape {tuple(x0_pred.shape)} must match x0 {tuple(x0.shape)}')
        if x0_pred.device != x0.device:
            raise ValueError('x0_pred device must match x0 device')
        if h0_pred is not None:
            if h0 is None:
                raise ValueError('h0_pred provided but h0 is None')
            if h0_pred.shape != h0.shape:
                raise ValueError(f'h0_pred shape {tuple(h0_pred.shape)} must match h0 {tuple(h0.shape)}')
            if h0_pred.device != h0.device:
                raise ValueError('h0_pred device must match h0 device')

    def training_step(self, batch: Data, model, soft_mask_transform) -> Dict[str, float]:
        
        self._assert_batch(batch)
        x0 = batch.ligand_pos.to(self.device)

        # Get tau from model q_phi if available; fallback to batch.tau if provided
        tau_mu = None
        tau_log_sigma = None
        tau_params = self._get_tau_params_from_model(model, batch)
        if tau_params is not None:
            tau_mu, tau_log_sigma = tau_params
            tau_mu = tau_mu.to(self.device)
            tau_log_sigma = tau_log_sigma.to(self.device)
            tau = self._sample_tau_logistic_normal(tau_mu, tau_log_sigma)
        else:
            if hasattr(batch, 'tau') and batch.tau is not None:
                tau = batch.tau.to(self.device)
            else:
                tau = torch.full((x0.size(0),), 0.5, device=self.device)

        has_feat = hasattr(batch, 'ligand_atom_feature') and batch.ligand_atom_feature is not None

        # Sample a single time step shared by all nodes
        t_scalar = torch.randint(0, self.diffusion_process.num_steps, (1,), device=self.device, dtype=torch.long).item()
        t = torch.full((x0.size(0),), t_scalar, device=self.device, dtype=torch.long)
        self._assert_shapes_pre(x0, tau, t)

        if has_feat:
            h0 = batch.ligand_atom_feature.to(self.device)
            x_t, h_t, s_t = self.diffusion_process.forward_process_multi_modal(x0, h0, tau, t, soft_mask_transform)
            self._assert_shapes_post_coord(x_t, s_t, x0)
            self._assert_shapes_post_multi(h0, h_t, s_t, x0)

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
            self._assert_model_outputs(x0_pred, x0, h0_pred, h0)

            sigma_coord = self.diffusion_process.sigmas[t]
            sigma_feat = self.diffusion_process.sigmas[t]
            
            if h0_pred is not None:
                diffusion_loss = self.loss_fn.compute_loss_multi_modal(x0, x0_pred, h0, h0_pred, s_t, sigma_coord, sigma_feat, self.lambda_feat)
            else:
                diffusion_loss = self.loss_fn.compute_loss(x0, x0_pred, s_t, sigma_coord)
        else:
            x_t, s_t = self.diffusion_process.forward_process(x0, tau, t, soft_mask_transform)
            self._assert_shapes_post_coord(x_t, s_t, x0)
            x0_pred = model(x_t, s_t, t, batch)
            self._assert_model_outputs(x0_pred, x0)
            sigma_t = self.diffusion_process.sigmas[t]
            diffusion_loss = self.loss_fn.compute_loss(x0, x0_pred, s_t, sigma_t)

        # KL loss for τ parameters: prefer model-provided; else fallback to batch fields
        if tau_mu is not None and tau_log_sigma is not None:
            kl_loss = self.loss_fn.compute_kl_loss(tau_mu, tau_log_sigma)
        elif hasattr(batch, 'tau_mu') and hasattr(batch, 'tau_log_sigma') and batch.tau_mu is not None and batch.tau_log_sigma is not None:
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


