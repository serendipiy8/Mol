import os
import sys
import torch
from torch import optim


# 让 from src... 可导入
CUR_DIR = os.path.dirname(__file__)  # .../Model
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from src.model.utils.repro import set_seed
from src.model.denoiser.egnn_denoiser import EGNNDenoiser
from src.model.encoders.protein_encoder import ProteinEncoder
from src.model.encoders.ligand_encoder import LigandEncoder
from src.model.diffusion.trainer import DiffusionTrainer
from src.model.diffusion.diffusion_process import SoftMaskDiffusionProcess
from src.model.diffusion.losses import DiffusionLoss
from src.data.soft_mask_transforms import SoftMaskTransform
from src.data.data_loader import get_data_loaders


def _enrich_batch_for_cross_edges(batch):
    """在 PyG Batch 上补充异构所需字段，启用蛋白→配体跨边与上下文注入。"""
    prot_pos = getattr(batch, 'protein_pos', None)
    lig_pos = getattr(batch, 'ligand_pos', None)
    if prot_pos is None or lig_pos is None:
        return batch
    Np = int(prot_pos.size(0))
    Nl = int(lig_pos.size(0))
    try:
        batch.num_protein_atoms = Np
        batch.pos = torch.cat([prot_pos, lig_pos], dim=0)
        batch.edge_index = torch.empty(2, 0, dtype=torch.long, device=batch.pos.device)
        batch.node_type = torch.cat([
            torch.zeros(Np, dtype=torch.long, device=batch.pos.device),
            torch.ones(Nl, dtype=torch.long, device=batch.pos.device)
        ], dim=0)
    except Exception:
        pass
    return batch


def _build_radius_edges(pos: torch.Tensor, radius: float) -> torch.Tensor:
    N = int(pos.size(0))
    if N == 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    D = torch.cdist(pos, pos)
    mask = (D > 0) & (D < radius)
    i, j = torch.nonzero(mask, as_tuple=True)
    if i.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    # 双向
    edge_ij = torch.stack([i, j], dim=0)
    edge_ji = torch.stack([j, i], dim=0)
    return torch.cat([edge_ij, edge_ji], dim=1)


def main():
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据路径（指向 processed 目录）
    data_root = os.path.abspath(os.path.join(CUR_DIR, 'src', 'data', 'crossdocked_v1.1_rmsd1.0_processed'))

    # DataLoader（Windows 建议 num_workers=0）
    train_loader, _ = get_data_loaders(
        dataset_path=data_root,
        split_file=None,
        batch_size=1,
        num_workers=0,
        shuffle_train=True,
        shuffle_test=False,
        require_ligand=True,
        require_feat=True,
        require_protein=False,
    )

    # 取一个 batch 以确定特征维度
    sample_batch = None
    for b in train_loader:
        if b is not None:
            sample_batch = b
            break
    if sample_batch is None:
        raise RuntimeError('No valid batch from DataLoader')

    lig_feat = getattr(sample_batch, 'ligand_atom_feature', None)
    if lig_feat is None:
        raise RuntimeError('Batch has no ligand_atom_feature')
    node_dim = int(lig_feat.size(1))

    # 模型 & 扩散组件（编码器先创建以备接入）
    # 让 g_P 维度与 denoiser.hidden_dim 对齐（128），让配体编码维度与 node_dim 对齐
    prot_encoder = ProteinEncoder(in_dim=node_dim, hidden_dim=128, num_layers=3, edge_dim=0).to(device)
    lig_encoder = LigandEncoder(in_dim=node_dim, hidden_dim=node_dim, num_layers=3, edge_dim=0).to(device)
    model = EGNNDenoiser(
        node_dim=node_dim,
        hidden_dim=128,
        num_layers=4,
        use_time=True,
        use_s_gate=True,
        use_cross_mp=True,
        cross_mp_bidirectional=False,
        cross_mp_every=1,
        use_protein_context=True,
        use_dynamic_cross_edges=True,
        cross_topk=24,
        cross_radius=6.0,
        use_bond_head=True,
        bond_hidden=128,
        bond_classes=2,
        bond_radius=2.2,
    ).to(device)

    diffusion = SoftMaskDiffusionProcess(
        num_steps=1000,
        sigma_min=0.2,
        sigma_max=1.0,
        sigma_schedule='linear',
        device=device,
        bind_sigma_to_alpha=False,
        alpha_schedule='linear',
    )
    # 扩散前向所需的 soft mask 核心（包含 compute_soft_mask）
    soft_mask_transform = SoftMaskTransform(
        num_steps=1000, kappa=0.1, alpha_schedule='linear', device=device,
    )
    loss_fn = DiffusionLoss(loss_type='mse', reweighting=True, w_max=1e2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer = DiffusionTrainer(diffusion, loss_fn, optimizer, device=device, lambda_feat=1.0, grad_clip_norm=1.0)

    # 训练若干步
    model.train()
    max_steps = 200
    step = 0
    while step < max_steps:
        for batch in train_loader:
            if batch is None:
                continue
            # 补充异构所需字段
            batch = _enrich_batch_for_cross_edges(batch)
            # 迁移到设备（trainer 内会检查 device 一致）
            for name in ['ligand_pos', 'ligand_atom_feature', 'protein_pos', 'protein_atom_feature', 'pos', 'edge_index', 'edge_attr', 'node_type']:
                ten = getattr(batch, name, None)
                if isinstance(ten, torch.Tensor) and ten.device.type != device:
                    setattr(batch, name, ten.to(device))
            # 使用简单半径图给 protein/ligand 编码器（不影响 denoiser 内部跨边策略）
            pp_edge_index = _build_radius_edges(getattr(batch, 'protein_pos', torch.empty(0, 3, device=device)), radius=5.0)
            ll_edge_index = _build_radius_edges(getattr(batch, 'ligand_pos', torch.empty(0, 3, device=device)), radius=2.2)
            # 编码器前向，得到 g_P 与 h_L，并注入到 batch 供 denoiser 使用
            if getattr(batch, 'protein_pos', None) is not None and getattr(batch, 'protein_atom_feature', None) is not None:
                h_P, g_P = prot_encoder(batch.protein_pos, batch.protein_atom_feature, pp_edge_index)
                setattr(batch, 'g_P', g_P.detach())
            if getattr(batch, 'ligand_pos', None) is not None and getattr(batch, 'ligand_atom_feature', None) is not None:
                h_L = lig_encoder(batch.ligand_pos, batch.ligand_atom_feature, ll_edge_index)
                # 用编码后的配体节点嵌入作为 h_t
                batch.h_t_override = h_L.detach()
            stats = trainer.training_step(batch, model, soft_mask_transform)
            step += 1
            if step % 20 == 0 or step == 1:
                print(f"step {step:04d} | total {stats['total_loss']:.6f} | diff {stats['diffusion_loss']:.6f} | kl {stats['kl_loss']:.6f}")
            if step >= max_steps:
                break


if __name__ == '__main__':
    main()


