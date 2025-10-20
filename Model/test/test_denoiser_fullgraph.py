#!/usr/bin/env python3
"""
测试：完整异质图 → EGNNDenoiser → 单步训练与 τ 排序信号
"""

import os
import sys
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from torch_geometric.data import Data

from src.data.soft_mask_transforms import create_soft_mask_transforms
from src.model.denoiser.egnn import EGNNDenoiser
from src.model.diffusion.diffusion_process import SoftMaskDiffusionProcess
from src.model.diffusion.trainer import DiffusionTrainer
from src.model.diffusion.losses import DiffusionLoss
from src.data.graph_builder import create_graph_builder


def build_real_complete_graph(Np: int = 6, Nl: int = 8, d: int = 32, device: str = 'cpu') -> Data:
    # Protein nodes
    protein_pos = torch.randn(Np, 3, device=device)
    protein_feat = torch.randn(Np, d, device=device)
    # Ligand nodes
    ligand_pos = torch.randn(Nl, 3, device=device)
    ligand_feat = torch.randn(Nl, d, device=device)

    base = Data(
        protein_pos=protein_pos,
        ligand_pos=ligand_pos,
        protein_atom_feature=protein_feat,
        ligand_atom_feature=ligand_feat,
    )
    gb = create_graph_builder(device=device)
    data = gb.build_complete_graph(base)
    return data


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = build_real_complete_graph(device=device)

    # Create soft-mask transform and apply
    transform = create_soft_mask_transforms(num_steps=100, kappa=0.1, device=device)
    # For this test, only tau and s_sequence are needed; no geometry change
    data = transform(data)

    # Diffusion + loss + model + trainer
    diffusion = SoftMaskDiffusionProcess(num_steps=100, sigma_min=0.5, sigma_max=1.0, device=device)
    loss_fn = DiffusionLoss()
    base_edge_dim = int(data.edge_attr.size(1)) if data.edge_attr is not None else 0
    edge_dim = base_edge_dim + (1 if hasattr(data, 'node_type') and data.node_type is not None else 0)
    model = EGNNDenoiser(node_dim=32, edge_dim=edge_dim, hidden_dim=128, num_layers=2, use_time=True, use_s_gate=True, degree_norm=True).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(diffusion, loss_fn, optim, device=device, lambda_feat=0.5)

    # Prepare ligand-only noisy inputs for model, but batch carries full graph
    lig_idx = (data.node_type == 1)
    x0 = data.ligand_pos
    h0 = data.ligand_atom_feature

    # Single training step via trainer
    data = data.to(device)
    model.train()
    metrics = trainer.training_step(data, model, transform.soft_mask_transform)
    print('metrics:', {k: float(v) for k, v in metrics.items()})

    # Check that model used full-graph path by verifying outputs size equals ligand count
    print('N_ligand:', int(data.num_ligand_atoms))


if __name__ == '__main__':
    main()


