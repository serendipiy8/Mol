import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max


class CrossGraphMessagePassing(nn.Module):
    def __init__(self, prot_dim: int, lig_dim: int, hidden_dim: int, bidirectional: bool = False,
                 use_distance: bool = True, rbf_dim: int = 0, use_direction: bool = False,
                 distance_sigma: float = 4.0, dropout: float = 0.0, use_angle: bool = True, num_heads: int = 4):
        super().__init__()

        self.bidirectional = bidirectional
        self.use_distance = use_distance
        self.use_direction = use_direction
        self.use_angle = use_angle
        self.rbf_dim = int(rbf_dim) if rbf_dim is not None else 0
        self.distance_sigma = distance_sigma
        self.num_heads = max(1, int(num_heads))
        self.dropout = nn.Dropout(dropout) if (isinstance(dropout, float) and dropout > 0.0) else nn.Identity()
        self.norm_l = nn.LayerNorm(lig_dim)
        self.norm_p = nn.LayerNorm(prot_dim)

        geom_extra = (1 if use_distance else 0) + (3 if use_direction else 0) + self.rbf_dim + (2 if self.use_angle else 0)
        in_dim = prot_dim + lig_dim + geom_extra

        # multi-head by simple grouped projections
        self.proj_p2l = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, lig_dim))
            for _ in range(self.num_heads)
        ])

        self.att_p2l = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            for _ in range(self.num_heads)
        ])

        if bidirectional:
            self.proj_l2p = nn.ModuleList([
                nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, prot_dim))
                for _ in range(self.num_heads)
            ])
            self.att_l2p = nn.ModuleList([
                nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
                for _ in range(self.num_heads)
            ])

    def _rbf(self, d: torch.Tensor, K: int, cutoff: float = 8.0) -> torch.Tensor:
        if K <= 0:
            return d.new_zeros(d.size(0), 0)
        centers = torch.linspace(0, cutoff, K, device=d.device, dtype=d.dtype)
        d_exp = d.unsqueeze(-1)
        gamma = 1.0 / (2 * (cutoff / max(K, 1)) ** 2)
        return torch.exp(-gamma * (d_exp - centers) ** 2)

    def _build_cat(self, hp: torch.Tensor, hl: torch.Tensor, rp: torch.Tensor, rl: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        parts = [hp, hl]
        diff = rl - rp
        dist = torch.norm(diff, dim=-1, keepdim=True)
        if self.use_distance:
            parts.append(dist)
        if self.use_direction:
            dir_unit = diff / (dist + 1e-8)
            parts.append(dir_unit)
        if self.rbf_dim > 0:
            parts.append(self._rbf(dist.squeeze(-1), self.rbf_dim))
        if self.use_angle:
            # angle features: cos(theta_l), cos(theta_p) by projecting to neighbor directions (simple approx)
            # here we reuse dir_unit against a fixed axis as a proxy to avoid extra neighborhoods
            ux = diff / (dist + 1e-8)
            cos_l = ux[:, 0:1]
            cos_p = ux[:, 1:2]
            parts.extend([cos_l, cos_p])
        cat = torch.cat(parts, dim=-1)
        return cat, dist.squeeze(-1)

    def forward(self, h_prot: torch.Tensor, h_lig: torch.Tensor, cross_edges: torch.Tensor,
                prot_pos: torch.Tensor = None, lig_pos: torch.Tensor = None,
                edge_attr: torch.Tensor = None, s_mask: torch.Tensor = None, t_bias: torch.Tensor = None,
                use_softmax: bool = True, use_dist_decay: bool = True):
        if cross_edges.size(1) == 0:
            return h_prot, h_lig

        p_idx, l_idx = cross_edges
        hp = h_prot[p_idx]
        hl = h_lig[l_idx]

        if prot_pos is not None and lig_pos is not None:
            rp = prot_pos[p_idx]
            rl = lig_pos[l_idx]
            cat, d = self._build_cat(hp, hl, rp, rl)
        else:
            cat = torch.cat([hp, hl], dim=-1)
            d = None

        if edge_attr is not None:
            cat = torch.cat([cat, edge_attr], dim=-1)

        # multi-head attention aggregation
        agg_l = torch.zeros_like(h_lig)
        for h in range(self.num_heads):
            logits = self.att_p2l[h](cat).squeeze(-1)  # [E]
            if t_bias is not None:
                logits = logits + t_bias.squeeze(-1)
            if use_softmax:
                max_per_l, _ = scatter_max(logits, l_idx, dim=0, dim_size=h_lig.size(0))
                stable = logits - max_per_l[l_idx]
                weights = torch.exp(stable)
                denom = scatter_add(weights, l_idx, dim=0, dim_size=h_lig.size(0))
                alpha = weights / (denom[l_idx] + 1e-9)
            else:
                alpha = torch.sigmoid(logits)
            if use_dist_decay and d is not None:
                decay = torch.exp(-(d ** 2) / (2 * (self.distance_sigma ** 2)))
                alpha = alpha * decay
            if s_mask is not None:
                alpha = alpha * s_mask[l_idx]
            v_ij = self.proj_p2l[h](cat)
            msg_pl = v_ij * alpha.unsqueeze(-1)
            msg_pl = self.dropout(msg_pl)
            agg_l.index_add_(0, l_idx, msg_pl)

        h_l_out = self.norm_l(h_lig + agg_l)
        h_p_out = h_prot

        if self.bidirectional:
            agg_p = torch.zeros_like(h_prot)
            for h in range(self.num_heads):
                logits_lp = self.att_l2p[h](cat).squeeze(-1)
                if use_softmax:
                    max_per_p, _ = scatter_max(logits_lp, p_idx, dim=0, dim_size=h_prot.size(0))
                    stable_p = logits_lp - max_per_p[p_idx]
                    weights_p = torch.exp(stable_p)
                    denom_p = scatter_add(weights_p, p_idx, dim=0, dim_size=h_prot.size(0))
                    alpha_lp = weights_p / (denom_p[p_idx] + 1e-9)
                else:
                    alpha_lp = torch.sigmoid(logits_lp)
                v_lp = self.proj_l2p[h](cat)
                msg_lp = v_lp * alpha_lp.unsqueeze(-1)
                msg_lp = self.dropout(msg_lp)
                agg_p.index_add_(0, p_idx, msg_lp)
            h_p_out = self.norm_p(h_prot + agg_p)

        return h_p_out, h_l_out


