import os
import sys
import torch


# 确保可以以 `from src...` 导入（需要把 Model 目录加到 sys.path）
CUR_DIR = os.path.dirname(__file__)  # .../Model
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from src.model.denoiser.egnn_denoiser import EGNNDenoiser
from src.model.utils.repro import set_seed


def main():
    set_seed(0)

    # 构造简单的蛋白/配体示例
    device = torch.device("cpu")
    Np = 32
    Nl = 16
    node_dim = 32

    prot_pos = torch.randn(Np, 3, device=device)
    lig_pos = torch.randn(Nl, 3, device=device) + torch.tensor([2.0, 0.0, 0.0])

    # 扩散步与 soft mask
    t = torch.tensor(10, device=device)
    s_t = torch.sigmoid(torch.randn(Nl, device=device))

    # 配体节点特征（作为 h_t）
    h_t = torch.randn(Nl, node_dim, device=device)
    x_t = lig_pos.clone()

    # 组一个最小 batch（属性式），触发“异构路径”：node_type + pos + edge_index
    class B: ...
    batch = B()
    batch.num_protein_atoms = Np
    # 合并坐标，满足 forward 的分支判定
    batch.pos = torch.cat([prot_pos, lig_pos], dim=0)
    # 空边（2,0）即可通过分支
    batch.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    # 提供 node_type：0=protein, 1=ligand
    batch.node_type = torch.cat([
        torch.zeros(Np, dtype=torch.long, device=device),
        torch.ones(Nl, dtype=torch.long, device=device)
    ], dim=0)
    # 供动态跨边重建使用
    batch.protein_pos = prot_pos
    batch.ligand_pos = lig_pos

    # 实例化模型：开启动态跨边与蛋白上下文注入，关闭配体内KNN（默认未实现）
    model = EGNNDenoiser(
        node_dim=node_dim,
        hidden_dim=64,
        num_layers=3,
        use_time=True,
        use_s_gate=True,
        use_cross_mp=True,
        cross_mp_bidirectional=False,
        cross_mp_every=1,
        use_protein_context=True,
        use_dynamic_cross_edges=True,
        cross_topk=16,
        cross_radius=6.0,
        use_bond_head=True,
        bond_hidden=64,
        bond_classes=2,
        bond_radius=2.2,
    ).to(device)

    model.eval()
    with torch.no_grad():
        x0_pred, h0_pred = model(x_t=x_t, s_t=s_t, t=t, batch=batch, h_t=h_t)

    print("x0_pred:", tuple(x0_pred.shape))
    print("h0_pred:", tuple(h0_pred.shape))
    # 检查成键预测是否挂到 batch
    pred_bonds = getattr(batch, 'pred_bond_index', None)
    pred_logits = getattr(batch, 'pred_bond_logits', None)
    if pred_bonds is not None and pred_logits is not None:
        print("pred_bond_index:", tuple(pred_bonds.shape))
        print("pred_bond_logits:", tuple(pred_logits.shape))
    else:
        print("no bond predictions attached")


if __name__ == "__main__":
    main()


