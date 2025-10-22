import torch
import inspect
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data
from .noise_schedulers import build_sigma_schedule
from .soft_mask_scheduler import build_alpha_schedule

"""
Core diffusion process implementation for LO-MaskDiff
Implements the forward and reverse diffusion processes with soft masking
"""

class SoftMaskDiffusionProcess:

    def __init__(self, num_steps: int = 1000, sigma_min: float = 0.5,
                 sigma_max: float = 1.0, sigma_schedule: str = 'linear', device: str = 'cuda',
                 bind_sigma_to_alpha: bool = False, alpha_schedule: str = 'linear'):

        self.num_steps = num_steps
        self.device = device
        self.alpha_values = None
        if bind_sigma_to_alpha:
            self.alpha_values = build_alpha_schedule(num_steps=self.num_steps, schedule=alpha_schedule, device=device)
        self.sigmas = build_sigma_schedule(num_steps=self.num_steps, sigma_min=sigma_min,
                                           sigma_max=sigma_max, schedule=sigma_schedule,
                                           device=device, bind_alpha=bind_sigma_to_alpha,
                                           alpha_values=self.alpha_values)
    
    def get_sigma_t(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        if isinstance(t, int):
            t_idx = torch.tensor(t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t_idx = t.to(torch.long)
        else:
            raise TypeError("t must be int or Tensor")
        return self.sigmas[t_idx]
    
    
    def forward_process(self, x0: torch.Tensor, tau: torch.Tensor, t: Union[int, torch.Tensor],
                       soft_mask_transform) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(t, int):
            t_idx = torch.tensor(t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t_idx = t.to(torch.long)
        else:
            raise TypeError("t must be int or Tensor")

        s_t = soft_mask_transform.compute_soft_mask(tau, t_idx)
        # Align devices
        if s_t.device != x0.device:
            s_t = s_t.to(x0.device)

        sigma_t = self.sigmas[t_idx]
        if sigma_t.device != x0.device:
            sigma_t = sigma_t.to(x0.device)
        noise = torch.randn_like(x0)

        # Apply soft masking with scheduled noise: x_t = s_t * x_0 + (1 - s_t) * (sigma_t * ε)
        if sigma_t.ndim == 0:
            sigma_exp = sigma_t.view(1, 1)
        else:
            sigma_exp = sigma_t.unsqueeze(-1)  # [N_nodes, 1]
        x_t = s_t.unsqueeze(-1) * x0 + (1 - s_t.unsqueeze(-1)) * (sigma_exp * noise)
        
        return x_t, s_t
    
    def forward_process_multi_modal(self, x0_coord: torch.Tensor, h0_feat: torch.Tensor, tau: torch.Tensor,
                                    t: Union[int, torch.Tensor], soft_mask_transform, sigma_t_coord: Optional[torch.Tensor] = None, 
                                    sigma_t_feat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(t, int):
            t_idx = torch.tensor(t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t_idx = t.to(torch.long)
        else:
            raise TypeError("t must be int or Tensor")

        s_t = soft_mask_transform.compute_soft_mask(tau, t_idx)
        if s_t.device != x0_coord.device:
            s_t = s_t.to(x0_coord.device)
        if sigma_t_coord is None:
            sigma_t_coord = self.sigmas[t_idx]
        if sigma_t_feat is None:
            sigma_t_feat = self.sigmas[t_idx]
        if sigma_t_coord.device != x0_coord.device:
            sigma_t_coord = sigma_t_coord.to(x0_coord.device)
        if sigma_t_feat.device != h0_feat.device:
            sigma_t_feat = sigma_t_feat.to(h0_feat.device)

        sigma_c = sigma_t_coord.unsqueeze(-1) if sigma_t_coord.ndim > 0 else sigma_t_coord.view(1, 1)
        sigma_f = sigma_t_feat.unsqueeze(-1) if sigma_t_feat.ndim > 0 else sigma_t_feat.view(1, 1)

        noise_coord = torch.randn_like(x0_coord) * sigma_c
        noise_feat = torch.randn_like(h0_feat) * sigma_f

        x_t_coord = s_t.unsqueeze(-1) * x0_coord + (1 - s_t.unsqueeze(-1)) * noise_coord
        h_t_feat = s_t.unsqueeze(-1) * h0_feat + (1 - s_t.unsqueeze(-1)) * noise_feat

        return x_t_coord, h_t_feat, s_t
    
    def reverse_process_step(self, x_t: torch.Tensor,s_t: torch.Tensor, t: Union[int, torch.Tensor],
                           model_prediction: torch.Tensor, soft_mask_transform) -> torch.Tensor:
        """
        Single step of reverse diffusion process
        
        Args:
            x_t: Current noisy coordinates [N_nodes, 3]
            s_t: Soft mask values [N_nodes]
            t: Current time step
            model_prediction: Model prediction of x0 [N_nodes, 3]
            soft_mask_transform: Soft mask transform instance
            
        Returns:
            x_prev: Previous step coordinates [N_nodes, 3]
        """
        if isinstance(t, int):
            t = torch.tensor(t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t = t.to(torch.long)
        
        sigma_t = self.sigmas[t]
        if sigma_t.device != x_t.device:
            sigma_t = sigma_t.to(x_t.device)
        x0_pred = model_prediction

        # Soft-mask based reverse update
        if (t > 0).any():
            noise = torch.randn_like(x_t)
            sigma_exp = sigma_t.unsqueeze(-1) if sigma_t.ndim > 0 else sigma_t.view(1, 1)
            x_prev = s_t.unsqueeze(-1) * x0_pred + (1 - s_t.unsqueeze(-1)) * (sigma_exp * noise)
        else:
            x_prev = x0_pred
        
        return x_prev
    
    def reverse_process_step_multi_modal(self, x_t_coord: torch.Tensor, h_t_feat: torch.Tensor, s_t: torch.Tensor, 
                                         t: Union[int, torch.Tensor], model_prediction: Tuple[torch.Tensor, torch.Tensor], 
                                         soft_mask_transform, sigma_t_coord: Optional[torch.Tensor] = None,
                                         sigma_t_feat: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(t, int):
            t = torch.tensor(t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t = t.to(torch.long)

        if sigma_t_coord is None:
            sigma_t_coord = self.sigmas[t]
        if sigma_t_feat is None:
            sigma_t_feat = self.sigmas[t]
        if sigma_t_coord.device != x_t_coord.device:
            sigma_t_coord = sigma_t_coord.to(x_t_coord.device)
        if sigma_t_feat.device != h_t_feat.device:
            sigma_t_feat = sigma_t_feat.to(h_t_feat.device)

        x0_pred, h0_pred = model_prediction

        if (t > 0).any():
            noise_c = torch.randn_like(x_t_coord)
            noise_f = torch.randn_like(h_t_feat)
            sigma_c = sigma_t_coord.unsqueeze(-1) if sigma_t_coord.ndim > 0 else sigma_t_coord.view(1, 1)
            sigma_f = sigma_t_feat.unsqueeze(-1) if sigma_t_feat.ndim > 0 else sigma_t_feat.view(1, 1)
            x_prev = s_t.unsqueeze(-1) * x0_pred + (1 - s_t.unsqueeze(-1)) * (sigma_c * noise_c)
            h_prev = s_t.unsqueeze(-1) * h0_pred + (1 - s_t.unsqueeze(-1)) * (sigma_f * noise_f)
        else:
            x_prev = x0_pred
            h_prev = h0_pred

        return x_prev, h_prev
    
    def sample(self, model, shape: Tuple[int, int], tau: torch.Tensor,
               soft_mask_transform, num_steps: Optional[int] = None, guidance_scale: float = 1.0,
               batch_context: Optional[Data] = None, init_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the diffusion process
        
        Args:
            model: Denoising model
            shape: Shape of coordinates to generate [N_nodes, 3]
            tau: Unmask-time parameters [N_nodes]
            soft_mask_transform: Soft mask transform instance
            num_steps: Number of sampling steps (default: self.num_steps)
            guidance_scale: Guidance scale for conditional generation
            
        Returns:
            x0: Generated coordinates [N_nodes, 3]
        """
        
        if num_steps is None:
            num_steps = self.num_steps
        
        # Start from init or pure noise
        if init_x is not None:
            x = init_x.to(self.device)
        else:
            x = torch.randn(shape, device=self.device)
        
        # Reverse sampling using soft-mask updates
        for i in reversed(range(num_steps)):
            t = torch.tensor(i, device=self.device)
            
            s_t = soft_mask_transform.compute_soft_mask(tau, t)
            
            with torch.no_grad():
                # Prefer model(x, s_t, t, batch_context) if supported
                forward_fn = getattr(model, 'forward', model)
                sig = inspect.signature(forward_fn)
                if 'batch' in sig.parameters or 'batch_context' in sig.parameters:
                    model_pred = model(x, s_t, t, batch_context)
                else:
                    model_pred = model(x, s_t, t)
            
            # Apply guidance if specified
            if guidance_scale > 1.0:
                # This would require conditional model predictions
                # For now, just use the prediction as-is
                pass
            
            # Reverse step via soft-mask update
            x = self.reverse_process_step(x, s_t, t, model_pred, soft_mask_transform)
        
        return x
    
    def sample_multi_modal(self, model, shape_coord: Tuple[int, int], shape_feat: Tuple[int, int],
                           tau: torch.Tensor, soft_mask_transform, num_steps: Optional[int] = None,
                           guidance_scale: float = 1.0, batch_context: Optional[Data] = None,
                           init_x: Optional[torch.Tensor] = None, init_h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if num_steps is None:
            num_steps = self.num_steps

        x = init_x.to(self.device) if init_x is not None else torch.randn(shape_coord, device=self.device)
        h = init_h.to(self.device) if init_h is not None else torch.randn(shape_feat, device=self.device)

        for i in reversed(range(num_steps)):
            t = torch.tensor(i, device=self.device)
            s_t = soft_mask_transform.compute_soft_mask(tau, t)

            with torch.no_grad():
                forward_fn = getattr(model, 'forward', model)
                sig = inspect.signature(forward_fn)
                if 'h_t' in sig.parameters:
                    pred = model(x, s_t, t, batch_context, h)
                else:
                    pred = model(x, s_t, t, batch_context)

            x, h = self.reverse_process_step_multi_modal(x, h, s_t, t, pred, soft_mask_transform)

        return x, h



