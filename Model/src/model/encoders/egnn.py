import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max


class EGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0, coord_dim: int = 3, aggr: str = 'mean',
                 use_attention: bool = True, attn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.aggr = aggr
        self.use_attention = use_attention

        h_in = in_dim * 2 + 1 + edge_dim

        if use_attention:
            a_hidden = attn_hidden if attn_hidden is not None else out_dim
            self.value_mlp = nn.Sequential(
                nn.Linear(h_in, out_dim), nn.SiLU(),
                nn.Linear(out_dim, out_dim), nn.SiLU())
            self.attn_mlp = nn.Sequential(
                nn.Linear(h_in, a_hidden), nn.SiLU(),
                nn.Linear(a_hidden, 1))
        else:
            self.phi_e = nn.Sequential(
                nn.Linear(h_in, out_dim), nn.SiLU(),
                nn.Linear(out_dim, out_dim), nn.SiLU())

        self.phi_x = nn.Sequential(nn.Linear(out_dim, 1, bias=False))

        self.phi_h = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim), nn.SiLU(),
            nn.Linear(out_dim, out_dim))

        self.residual = (in_dim == out_dim)
        self.dropout = nn.Dropout(dropout) if (isinstance(dropout, float) and dropout > 0.0) else nn.Identity()

    def forward(self, x_h: torch.Tensor, x_pos: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):

        N = x_h.size(0)
        src, dst = edge_index
        rij = x_pos[dst] - x_pos[src]
        dij = torch.norm(rij, dim=-1, keepdim=True)

        if edge_attr is not None:
            e_in = torch.cat([x_h[src], x_h[dst], dij, edge_attr], dim=-1)
        else:
            e_in = torch.cat([x_h[src], x_h[dst], dij], dim=-1)

        if self.use_attention:
            v_ij = self.value_mlp(e_in)
            logits = self.attn_mlp(e_in).squeeze(-1)
            max_per_dst, _ = scatter_max(logits, dst, dim=0, dim_size=N)
            stable = logits - max_per_dst[dst]
            weights = torch.exp(stable)
            denom = scatter_add(weights, dst, dim=0, dim_size=N)
            alpha = weights / (denom[dst] + 1e-9)
            m_ij = v_ij * alpha.unsqueeze(-1)
        else:
            m_ij = self.phi_e(e_in)

        m_ij = self.dropout(m_ij)

        coord_coef = self.phi_x(m_ij)
        dx = rij * coord_coef

        h_agg = torch.zeros(N, m_ij.size(-1), device=x_h.device)
        x_agg = torch.zeros_like(x_pos)
        h_agg.index_add_(0, dst, m_ij)
        x_agg.index_add_(0, dst, dx)

        if self.aggr == 'mean':
            deg = torch.bincount(dst, minlength=N).clamp(min=1).view(-1, 1)
            h_agg = h_agg / deg
            x_agg = x_agg / deg

        h_in = torch.cat([x_h, h_agg], dim=-1)
        h_out = self.phi_h(h_in)

        if self.residual:
            h_out = h_out + x_h

        x_out = x_pos + x_agg

        return h_out, x_out


class EGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4, edge_dim: int = 0, aggr: str = 'mean',
                 use_attention: bool = True, attn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim, edge_dim=edge_dim, aggr=aggr,
                      use_attention=use_attention, attn_hidden=attn_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_feat: torch.Tensor, x_pos: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        h = self.input_proj(x_feat)
        x = x_pos

        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        h = self.norm(h)

        return h, x