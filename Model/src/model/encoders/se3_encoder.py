import torch
import torch.nn as nn
from typing import Optional

try:
    from e3nn import o3
    E3NN_AVAILABLE = True
except Exception:
    E3NN_AVAILABLE = False

from .egnn import EGNNEncoder


class SE3ProteinEncoder(nn.Module):
    """
    SE(3)-aware protein encoder.
    - If e3nn is available, use a lightweight equivariant layer to get scalar and vector contexts;
      otherwise fallback to EGNN 。
    Returns (h_nodes, g_P_scalar, g_vec_direction[3] or None).
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4,
                 edge_dim: int = 0, aggr: str = 'mean'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_vec_context = True
        if E3NN_AVAILABLE:
            self.lin_in = nn.Linear(in_dim, hidden_dim)
            self.rbf_dim = 8
            self.cutoff = 8.0
            self.msg_mlp = nn.Sequential(
                nn.Linear(hidden_dim + self.rbf_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.upd = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.backbone = EGNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim,
                                        num_layers=num_layers, edge_dim=edge_dim, aggr=aggr)
            self.pool = nn.AdaptiveAvgPool1d(1)

    def _rbf(self, d: torch.Tensor) -> torch.Tensor:
        centers = torch.linspace(0, self.cutoff, self.rbf_dim, device=d.device, dtype=d.dtype)
        gamma = 1.0 / (2 * (self.cutoff / max(self.rbf_dim, 1)) ** 2)
        return torch.exp(-gamma * (d.unsqueeze(-1) - centers) ** 2)

    def forward(self, node_feat: torch.Tensor, pos: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None):
        if not E3NN_AVAILABLE:
            h, _ = self.backbone(node_feat, pos, edge_index, edge_attr)
            if h.size(0) > 0:
                g_P = self.pool(h.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(-1)
            else:
                g_P = torch.zeros(h.size(1), device=h.device)
            g_vec = None
            if self.use_vec_context and pos.size(0) > 3:
                mu = pos.mean(dim=0, keepdim=True)
                x = pos - mu
                cov = x.t().mm(x) / (pos.size(0) - 1)
                try:
                    _, vecs = torch.linalg.eigh(cov)
                    g_vec = vecs[:, -1].detach()
                except Exception:
                    g_vec = None
            return h, g_P, g_vec

        # e3nn 可用：简化的等变消息传递（标量通道 + Y1 向量上下文）
        h = self.lin_in(node_feat)
        N = h.size(0)
        if edge_index is None or edge_index.numel() == 0:
            g_P = self.pool(h.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(-1) if h.size(0) > 0 else torch.zeros(h.size(1), device=h.device)
            return h, g_P, None

        src, dst = edge_index
        rij = pos[dst] - pos[src]
        dij = torch.norm(rij, dim=-1)
        dir_unit = rij / (dij.unsqueeze(-1) + 1e-8)  # 近似 Y1 分量
        rbf = self._rbf(dij)

        h_src = h[src]
        msg_in = torch.cat([h_src, rbf], dim=-1)
        m = self.msg_mlp(msg_in)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        h = self.upd(torch.cat([h, agg], dim=-1))
        h = self.norm(h)

        # 向量上下文：按标量门控加权的方向和
        gate = torch.sigmoid((h_src).sum(dim=-1, keepdim=True))
        v_edge = gate * dir_unit
        v_node = torch.zeros(N, 3, device=h.device, dtype=h.dtype)
        v_node.index_add_(0, dst, v_edge)

        g_P = self.pool(h.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(-1) if h.size(0) > 0 else torch.zeros(h.size(1), device=h.device)
        g_vec = None
        if self.use_vec_context:
            g_vec = v_node.mean(dim=0)
            g_vec = g_vec / (g_vec.norm() + 1e-8)
        return h, g_P, g_vec


