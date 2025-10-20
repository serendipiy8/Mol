#!/usr/bin/env python3
"""
最小训练脚本：接入 SoftMaskDataTransform 与 DiffusionTrainer，单步训练验证。
"""

import os
import sys
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from torch import nn
from torch_geometric.data import Data

from src.data.soft_mask_transforms import create_soft_mask_transforms
from src.model.diffusion.diffusion_process import SoftMaskDiffusionProcess
from src.model.diffusion.trainer import DiffusionTrainer
from src.model.diffusion.losses import DiffusionLoss


class TinyDenoiser(nn.Module):
    def __init__(self, hidden_dim: int = 64, has_feat: bool = True):
        super().__init__()
        self.has_feat = has_feat
        self.coord_mlp = nn.Sequential(
            nn.Linear(3 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        if has_feat:
            self.feat_mlp = nn.Sequential(
                nn.Linear(32 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 32)
            )
            # q_phi heads for tau parameters
            self.tau_mu_head = nn.Sequential(
                nn.Linear(32, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.tau_log_sigma_head = nn.Sequential(
                nn.Linear(32, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x_t, s_t, t, batch, h_t=None):
        t_norm = (t.float() / max(1, getattr(batch, 'num_steps', 1000))).unsqueeze(-1)
        inp = torch.cat([x_t, s_t.unsqueeze(-1)], dim=-1)
        x0_pred = self.coord_mlp(inp)
        if self.has_feat and h_t is not None:
            hf = torch.cat([h_t, s_t.unsqueeze(-1)], dim=-1)
            h0_pred = self.feat_mlp(hf)
            return x0_pred, h0_pred
        return x0_pred

    def predict_tau_params(self, batch: Data):
        if not hasattr(batch, 'ligand_atom_feature') or batch.ligand_atom_feature is None:
            return None
        h = batch.ligand_atom_feature
        mu = self.tau_mu_head(h).squeeze(-1)
        log_sigma = self.tau_log_sigma_head(h).squeeze(-1)
        return mu, log_sigma


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构造一个小的 Data
    Np, Nl, d_feat = 12, 8, 32
    data = Data()
    data.protein_pos = torch.randn(Np, 3)
    data.ligand_pos = torch.randn(Nl, 3)
    data.protein_atom_feature = torch.randn(Np, d_feat)
    data.ligand_atom_feature = torch.randn(Nl, d_feat)

    # 变换：SoftMaskDataTransform
    transform = create_soft_mask_transforms(num_steps=100, kappa=0.1, device=device)
    data = transform(data)

    # 组件
    diffusion = SoftMaskDiffusionProcess(num_steps=300, sigma_min=0.5, sigma_max=1.0, device=device)
    loss_fn = DiffusionLoss()
    model = TinyDenoiser(has_feat=True).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(diffusion, loss_fn, optim, device=device, lambda_feat=0.5)

    # 多步训练并观察每节点 tau 的变化
    data = data.to(device)
    model.train()

    def sample_tau_from_mu_logsigma(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0))
        return torch.sigmoid(mu + sigma * eps)

    num_steps = 1000
    for step in range(num_steps):
        with torch.no_grad():
            tau_mu_logsigma = model.predict_tau_params(data)
            if tau_mu_logsigma is not None:
                mu, log_sigma = tau_mu_logsigma
                mu = mu.to(device)
                log_sigma = log_sigma.to(device)
                tau_sample = sample_tau_from_mu_logsigma(mu, log_sigma)
                print(f"[step {step}] tau stats: mean={tau_sample.mean().item():.4f}, min={tau_sample.min().item():.4f}, max={tau_sample.max().item():.4f}")
                print(f"[step {step}] tau[:5] = {tau_sample[:5].detach().cpu().numpy()}")
            else:
                # 回退到数据中的 tau
                tau_sample = getattr(data, 'tau', torch.full((data.ligand_pos.size(0),), 0.5, device=device))
                print(f"[step {step}] (fallback) tau mean={tau_sample.mean().item():.4f}")

        metrics = trainer.training_step(data, model, transform.soft_mask_transform)
        print({k: float(v) for k, v in metrics.items()})


if __name__ == '__main__':
    main()


