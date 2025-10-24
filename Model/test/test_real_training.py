#!/usr/bin/env python3
"""
最小可跑训练闭环（DataLoader → SoftMaskDiffusion → Trainer）
 - 加载 CrossDocked 数据（train/test）
 - 用 TinyDenoiser 演示 q_φ 接入 + 扩散训练单步
 - 打印 total/diffusion/KL 损失与 tau 统计
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn


# 确保可以 import src/*
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
SRC_ROOT = os.path.join(PROJ_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from src.data.data_loader import get_data_loaders
from src.model.diffusion.diffusion_process import SoftMaskDiffusionProcess
from src.model.diffusion.losses import DiffusionLoss
from src.model.diffusion.trainer import DiffusionTrainer
from src.data.soft_mask_transforms import create_soft_mask_transforms


class TinyDenoiser(nn.Module):
    """最小去噪器
    支持 (x_t, s_t, t, batch[, h_t]) 接口；提供 predict_tau_params 作为 q_φ 头。
    """
    def __init__(self, hidden_dim: int = 128, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim
        in_dim = 3 + 1 + time_dim  # x(3) + s(1) + t_emb
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.h_head = nn.Identity()

    def _time_embed(self, t: torch.Tensor, N: int, T: int, device: str):
        if t.ndim == 0:
            t = t.view(1)
        t = t.to(torch.long)
        t_float = t.float() / max(T, 1)
        if t_float.numel() == 1:
            t_float = t_float.repeat(N)
        return t_float.view(N, 1).repeat(1, self.time_dim).to(device)

    def forward(self, x_t: torch.Tensor, s_t: torch.Tensor, t: torch.Tensor, batch=None, h_t: torch.Tensor = None):
        device = x_t.device
        N = x_t.size(0)
        T = int(t.max().item()) + 1 if t.numel() > 0 else 1000
        t_emb = self._time_embed(t, N, T, device)
        x_in = torch.cat([x_t, s_t.view(N, 1).to(device), t_emb], dim=-1)
        x0_pred = self.mlp(x_in)
        if h_t is not None:
            h0_pred = self.h_head(h_t)
            return x0_pred, h0_pred
        return x0_pred

    @torch.no_grad()
    def predict_tau_params(self, batch) -> tuple:
        # 简易 q_φ：返回每个配体节点的一组 (mu, log_sigma)
        if hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
            N = batch.ligand_pos.size(0)
            device = batch.ligand_pos.device
        else:
            N = 1
            device = 'cpu'
        mu = torch.zeros(N, device=device)
        log_sigma = torch.full((N,), float(np.log(0.1)), device=device)
        return mu, log_sigma


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # DataLoader
    dataset_path = 'src/data/crossdocked_v1.1_rmsd1.0_processed'
    split_file = 'src/data/split_by_name.pt'
    train_loader, test_loader = get_data_loaders(
        dataset_path=dataset_path,
        batch_size=16,
        num_workers=0,     
        split_file=split_file,
        shuffle_train=True,
        shuffle_test=False,
        require_protein=True,
        require_ligand=True,
        require_feat=False,
    )
    if train_loader is None:
        print('❌ train_loader 不可用')
        return

    # 组件
    sm_data_transform = create_soft_mask_transforms(
        num_steps=1000, kappa=0.1, alpha_schedule='linear', device=device,
        kappa_start=0.2, kappa_end=0.05, kappa_schedule='linear'
    )
    diffusion = SoftMaskDiffusionProcess(
        num_steps=1000, sigma_min=0.0, sigma_max=1.0, sigma_schedule='linear', device=device,
        bind_sigma_to_alpha=False, alpha_schedule='linear'
    )
    loss_fn = DiffusionLoss(loss_type='mse', reweighting=True, w_max=1e3, eps=1e-8)

    model = TinyDenoiser(hidden_dim=128, time_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(diffusion_process=diffusion, loss_fn=loss_fn, optimizer=optimizer, device=device, lambda_feat=1.0)

    # 训练若干步
    max_steps = 50
    step = 0
    t0 = time.time()
    for batch in train_loader:
        if batch is None:
            continue

        logs = trainer.training_step(batch, model, sm_data_transform.soft_mask_transform)

        # 打印监控
        tau = getattr(batch, 'tau', None)
        if isinstance(tau, torch.Tensor):
            tau = tau.detach().float()
            print(f"step {step:04d} | loss={logs['total_loss']:.4f} (dif={logs['diffusion_loss']:.4f}, kl={logs['kl_loss']:.4f}) | tau μ={tau.mean():.3f} σ={tau.std():.3f}")
        else:
            print(f"step {step:04d} | loss={logs['total_loss']:.4f} (dif={logs['diffusion_loss']:.4f}, kl={logs['kl_loss']:.4f})")

        step += 1
        if step >= max_steps:
            break

    print(f"✅ 训练完成: {step} steps, 用时 {time.time()-t0:.2f}s")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n⏹️ 中断')

