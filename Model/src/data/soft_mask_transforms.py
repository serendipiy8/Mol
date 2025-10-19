import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from torch_geometric.data import Data

class SoftMaskTransform:
    """
    Core soft mask transform implementing the LO-MaskDiff mechanism
    """
    
    def __init__(self, num_steps: int = 1000, kappa: float = 0.1, alpha_schedule: str = 'linear', 
                 device: str = 'cuda', w_max: float = 1e3, eps: float = 1e-8):
        self.num_steps = num_steps
        self.kappa = kappa
        self.alpha_schedule = alpha_schedule
        self.device = device
        self.w_max = w_max  
        self.eps = eps   
        
        self.alpha_values = self._compute_alpha_schedule()
    
    # compute alpha schedule for time steps (t)
    def _compute_alpha_schedule(self) -> torch.Tensor:
        if self.alpha_schedule == 'linear':
            alpha = torch.linspace(0, 1, self.num_steps + 1, device=self.device)
        elif self.alpha_schedule == 'cosine':
            t = torch.linspace(0, 1, self.num_steps + 1, device=self.device)
            alpha = 0.5 * (1 + torch.cos(np.pi * (1 - t)))
        elif self.alpha_schedule == 'quadratic':
            t = torch.linspace(0, 1, self.num_steps + 1, device=self.device)
            alpha = t ** 2
        else:
            raise ValueError(f"Unknown alpha schedule: {self.alpha_schedule}")
        
        return alpha
    
    def _validate_inputs(self, tau: torch.Tensor, t: Union[int, torch.Tensor]):
        if tau.dim() != 1:
            raise ValueError(f"tau must be 1D tensor, got {tau.dim()}D")
        if torch.any(tau < 0) or torch.any(tau > 1):
            raise ValueError("tau must be in range [0, 1]")
        if isinstance(t, int) and (t < 0 or t > self.num_steps):
            raise ValueError(f"t must be in range [0, {self.num_steps}], got {t}")
    
    def compute_soft_mask(self, tau: torch.Tensor, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute soft mask s_t(i) = sigm((α(t) - τ_i) / κ)
        
        Args:
            tau: Unmask-time parameters [N_nodes] ∈ (0, 1)
            t: Current time step(s) ∈ [0, T]
            
        Returns:
            s_t: Soft mask values [N_nodes] ∈ (0, 1)
        """
        self._validate_inputs(tau, t)
        
        if isinstance(t, int):
            t_idx = torch.tensor(t, device=self.device, dtype=torch.long)
        elif isinstance(t, torch.Tensor):
            t_idx = torch.clamp(t, 0, self.num_steps).to(torch.long)
        else:
            raise TypeError("t must be int or Tensor")

        alpha_t = self.alpha_values[t_idx]

        # Compute soft mask: s_t(i) = sigm((α(t) - τ_i) / κ)
        soft_mask = torch.sigmoid((alpha_t - tau) / self.kappa)
        
        return soft_mask
    
    def forward_noise(self, x0: torch.Tensor, tau: torch.Tensor, 
                     t: Union[int, torch.Tensor], sigma_t: float = 1.0) -> torch.Tensor:
        """
        Apply forward noise with soft mask: x_t = s_t * x_0 + (1 - s_t) * η_t
        
        Args:
            x0: Clean coordinates [N_nodes, 3]
            tau: Unmask-time parameters [N_nodes]
            t: Current time step
            sigma_t: Noise standard deviation
            
        Returns:
            x_t: Noisy coordinates [N_nodes, 3]
        """

        s_t = self.compute_soft_mask(tau, t)
        noise = torch.randn_like(x0) * sigma_t
        
        # forward add noise: x_t = s_t * x_0 + (1 - s_t) * η_t
        x_t = s_t.unsqueeze(-1) * x0 + (1 - s_t.unsqueeze(-1)) * noise
        
        return x_t
    
    def forward_noise_multi_modal(self, x0_coord: torch.Tensor, h0_feat: torch.Tensor,
                                 tau: torch.Tensor, t: Union[int, torch.Tensor],
                                 sigma_t_coord: float = 1.0, sigma_t_feat: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward noise with soft mask for both coordinates and features
        
        Args:
            x0_coord: Clean coordinates [N_nodes, 3]
            h0_feat: Clean features [N_nodes, d_feat]
            tau: Unmask-time parameters [N_nodes]
            t: Current time step
            sigma_t_coord: Noise std for coordinates
            sigma_t_feat: Noise std for features
            
        Returns:
            x_t_coord: Noisy coordinates [N_nodes, 3]
            h_t_feat: Noisy features [N_nodes, d_feat]
        """

        s_t = self.compute_soft_mask(tau, t)
        
        # Coordinate noise
        noise_coord = torch.randn_like(x0_coord) * sigma_t_coord
        x_t_coord = s_t.unsqueeze(-1) * x0_coord + (1 - s_t.unsqueeze(-1)) * noise_coord
        
        # Feature noise
        noise_feat = torch.randn_like(h0_feat) * sigma_t_feat
        h_t_feat = s_t.unsqueeze(-1) * h0_feat + (1 - s_t.unsqueeze(-1)) * noise_feat
        
        return x_t_coord, h_t_feat
    
    def compute_soft_mask_sequence(self, tau: torch.Tensor) -> torch.Tensor:
        s_sequence = []
        for t in range(self.num_steps + 1):
            s_t = self.compute_soft_mask(tau, t)
            s_sequence.append(s_t)
        
        return torch.stack(s_sequence)
    
    # Reweighting function for loss calculation
    def compute_weights(self, s_t: torch.Tensor, t: Union[int, torch.Tensor], 
                       sigma_t: float = 1.0) -> torch.Tensor:
        """
        Compute weights according to theoretical framework
        w_t(i) = min(1/((1-s_t(i))^2 * σ_t^2), w_max)
        """

        weights = 1.0 / ((1 - s_t) ** 2 * sigma_t ** 2 + self.eps)
        
        weights = torch.clamp(weights, max=self.w_max)
        weights = weights / (weights.mean() + self.eps)
        
        return weights
    
    def compute_weights_multi_modal(self, s_t: torch.Tensor, t: Union[int, torch.Tensor],
                                   sigma_t_coord: float = 1.0, sigma_t_feat: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        w_coord = self.compute_weights(s_t, t, sigma_t_coord)
        w_feat = self.compute_weights(s_t, t, sigma_t_feat)
        
        return w_coord, w_feat


class LogisticNormalReparameterization:
    """
    Logistic-Normal reparameterization for τ sampling
    Enhanced with numerical stability
    """
    
    def __init__(self, device: str = 'cuda', eps: float = 1e-8):
        self.device = device
        self.eps = eps
    
    def sample_tau(self, mu: torch.Tensor, log_sigma: torch.Tensor,
                   epsilon: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample τ using reparameterization trick
        
        Args:
            mu: Mean parameters [N_nodes]
            log_sigma: Log standard deviation [N_nodes]
            epsilon: Optional noise for deterministic sampling
            
        Returns:
            tau: Sampled unmask-time parameters [N_nodes] ∈ (0, 1)
        """
        if epsilon is None:
            epsilon = torch.randn_like(mu)
        
        # Sample from logistic-normal: u = μ + σ * ε
        u = mu + torch.exp(torch.clamp(log_sigma, min=-10, max=10)) * epsilon
        
        # Transform to (0, 1): τ = sigmoid(u)
        tau = torch.sigmoid(u)
        
        return tau
    
    # the prior distribution kl divergence loss
    def compute_kl_divergence(self, mu: torch.Tensor, log_sigma: torch.Tensor,
                             prior_mu: float = 0.0, prior_sigma: float = 1.0) -> torch.Tensor:
        """
        Compute KL divergence KL(q(τ) || p(τ))
        
        Args:
            mu: Mean parameters [N_nodes]
            log_sigma: Log standard deviation [N_nodes]
            prior_mu: Prior mean
            prior_sigma: Prior standard deviation
            
        Returns:
            kl_div: KL divergence [N_nodes]
        """
        sigma = torch.exp(torch.clamp(log_sigma, min=-10, max=10))
        
        # KL divergence for normal distributions
        kl_div = 0.5 * (
            (sigma / prior_sigma) ** 2 + 
            ((mu - prior_mu) / prior_sigma) ** 2 - 
            1 + 
            2 * (torch.log(prior_sigma + self.eps) - log_sigma)
        )
        
        return kl_div


class SoftMaskDataTransform:
    """
    Data transform that applies soft mask to ProteinLigandData
    Enhanced with multi-modal support and error handling
    """
    
    def __init__(self, soft_mask_transform: SoftMaskTransform,
                 reparameterization: LogisticNormalReparameterization):

        self.soft_mask_transform = soft_mask_transform
        self.reparameterization = reparameterization
    
    def __call__(self, data: Data) -> Data:
        """
        Apply soft mask transform to data
        
        Args:
            data: ProteinLigandData instance
        
        Returns:
            data: Modified data with soft mask information
        """

        if not hasattr(data, 'ligand_pos'):
            raise ValueError("Data must contain ligand_pos")
        if not hasattr(data, 'ligand_atom_feature'):
            raise ValueError("Data must contain ligand_atom_feature")
        

        num_ligand_atoms = data.ligand_pos.size(0)
        if not hasattr(data, 'tau_mu') or not hasattr(data, 'tau_log_sigma'):
            data.tau_mu = torch.zeros(num_ligand_atoms, device=data.ligand_pos.device)
            data.tau_log_sigma = torch.zeros(num_ligand_atoms, device=data.ligand_pos.device)
        
        tau = self.reparameterization.sample_tau(
            data.tau_mu, 
            data.tau_log_sigma)
        data.tau = tau

        s_sequence = self.soft_mask_transform.compute_soft_mask_sequence(tau)
        data.s_sequence = s_sequence
        
        data.num_steps = self.soft_mask_transform.num_steps
        data.kappa = self.soft_mask_transform.kappa
        data.has_coord = True
        data.has_feat = hasattr(data, 'ligand_atom_feature')
        
        return data


def create_soft_mask_transforms(num_steps: int = 1000, kappa: float = 0.1,
                               alpha_schedule: str = 'linear', device: str = 'cuda',
                               w_max: float = 1e3, eps: float = 1e-8) -> SoftMaskDataTransform:
    soft_mask_transform = SoftMaskTransform(
        num_steps=num_steps,
        kappa=kappa,
        alpha_schedule=alpha_schedule,
        device=device,
        w_max=w_max,
        eps=eps)
    
    reparameterization = LogisticNormalReparameterization(device=device, eps=eps)
    
    return SoftMaskDataTransform(soft_mask_transform, reparameterization)