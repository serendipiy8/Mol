import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class EGNNLayer(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int, degree_norm: bool = True):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),)

        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),)
        self.degree_norm = degree_norm

    def forward(self, x, pos, edge_index, edge_attr=None, s: torch.Tensor = None):
        src, dst = edge_index
        x_i, x_j = x[dst], x[src]
        diff = pos[dst] - pos[src]
        r2 = (diff ** 2).sum(dim=-1, keepdim=True)

        if edge_attr is None:
            edge_attr = torch.zeros(diff.size(0), 0, device=x.device)
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, r2, edge_attr], dim=-1))
        if s is not None:
            gate = s[src].unsqueeze(-1)
            m_ij = m_ij * gate

        # node update
        m_sum = scatter_add(m_ij, dst, dim=0, dim_size=x.size(0))
        if self.degree_norm:
            ones = torch.ones(m_ij.size(0), 1, device=x.device, dtype=m_ij.dtype)
            deg = scatter_add(ones, dst, dim=0, dim_size=x.size(0))
            m_sum = m_sum / (deg.clamp_min(1.0))
        x = self.node_mlp(torch.cat([x, m_sum], dim=-1))

        # coord update (equivariant)
        gamma = self.coord_mlp(m_ij)  # [E,1]
        coord_update = scatter_add(gamma * diff, dst, dim=0, dim_size=pos.size(0))
        if self.degree_norm:
            ones = torch.ones(m_ij.size(0), 1, device=pos.device, dtype=pos.dtype)
            deg = scatter_add(ones, dst, dim=0, dim_size=pos.size(0))
            coord_update = coord_update / (deg.clamp_min(1.0))
        pos = pos + coord_update
        return x, pos


class EGNNDenoiser(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int = 0, hidden_dim: int = 128, num_layers: int = 3,
                 time_dim: int = 16, use_time: bool = True, use_s_gate: bool = True, degree_norm: bool = True):
        super().__init__()
        self.use_time = use_time
        self.use_s_gate = use_s_gate
        self.time_dim = time_dim
        in_dim = node_dim + 1 + (time_dim if use_time else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim, hidden_dim, degree_norm=degree_norm) for _ in range(num_layers)
        ])
        self.out_coord = nn.Linear(hidden_dim, 3)
        self.out_feat = nn.Linear(hidden_dim, node_dim)

    def _time_embed(self, t_scalar: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        if not self.use_time:
            return torch.zeros(num_nodes, 0, device=device)
        dim = self.time_dim
        t = t_scalar.float().unsqueeze(0)
        half = dim // 2
        freqs = torch.exp(torch.linspace(0, 1, half, device=device) * (-4.0))
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < dim:
            emb = torch.cat([emb, torch.zeros(1, dim - emb.size(-1), device=device)], dim=-1)
        return emb.repeat(num_nodes, 1)

    def forward(self, x_t, s_t, t, batch, h_t=None):
        # x_t: [N_l,3], h_t: [N_l,d], s_t: [N_l]
        if h_t is None:
            raise ValueError('EGNNDenoiser requires node features h_t')

        node_type = getattr(batch, 'node_type', None)
        num_protein = getattr(batch, 'num_protein_atoms', None)
        pos_all = getattr(batch, 'pos', None)
        x_all = getattr(batch, 'x', None)
        edge_index = getattr(batch, 'edge_index', None)
        edge_attr = getattr(batch, 'edge_attr', None)

        if node_type is not None and num_protein is not None and pos_all is not None and edge_index is not None:

            Np = int(num_protein)
            Nl = int(h_t.size(0))

            if x_all is not None and x_all.size(0) == (Np + Nl) and x_all.size(1) >= h_t.size(1):
                h_all = x_all.clone().to(h_t.device)
                h_all[Np:Np+Nl, :h_t.size(1)] = h_t
            else:
                prot_feat = getattr(batch, 'protein_atom_feature', None)
                if prot_feat is None:
                    prot_feat = torch.zeros(Np, h_t.size(1), device=h_t.device, dtype=h_t.dtype)
                elif prot_feat.size(1) != h_t.size(1):
                    d = h_t.size(1)
                    if prot_feat.size(1) < d:
                        pad = torch.zeros(Np, d - prot_feat.size(1), device=prot_feat.device, dtype=prot_feat.dtype)
                        prot_feat = torch.cat([prot_feat, pad], dim=1)
                    else:
                        prot_feat = prot_feat[:, :d]
                h_all = torch.cat([prot_feat.to(h_t.device), h_t], dim=0)

            if pos_all is not None and pos_all.size(0) == (Np + Nl):
                pos = pos_all.clone().to(x_t.device)
                pos[Np:Np+Nl] = x_t
            else:
                prot_pos = getattr(batch, 'protein_pos', None)
                if prot_pos is None:
                    prot_pos = torch.zeros(Np, 3, device=x_t.device, dtype=x_t.dtype)
                pos = torch.cat([prot_pos.to(x_t.device), x_t], dim=0)

            s_all = torch.zeros(Np + Nl, device=h_all.device, dtype=s_t.dtype)
            s_all[Np:Np+Nl] = s_t


            if edge_index is not None:
                src, dst = edge_index
                if node_type is not None:
                    # cross indicator to edge_attr, 1 if cross, 0 if not
                    cross = (node_type[src] != node_type[dst]).float().unsqueeze(-1)
                    if edge_attr is None:
                        edge_attr = cross
                    else:
                        edge_attr = torch.cat([edge_attr, cross.to(edge_attr.device)], dim=-1)

            if isinstance(t, torch.Tensor):
                t_scalar = t[0] if t.numel() > 0 else torch.tensor(0, device=h_all.device)
            else:
                t_scalar = torch.tensor(float(t), device=h_all.device)
            t_emb = self._time_embed(t_scalar, h_all.size(0), h_all.device)
            node_in = torch.cat([h_all, s_all.unsqueeze(-1), t_emb], dim=-1)
            xhid = F.silu(self.input_proj(node_in))

            if edge_index is None:
                N = xhid.size(0)
                src = torch.arange(N, device=xhid.device).repeat_interleave(N)
                dst = torch.arange(N, device=xhid.device).repeat(N)
                mask = src != dst
                edge_index = torch.stack([src[mask], dst[mask]], dim=0)

            for layer in self.layers:
                xhid, pos = layer(xhid, pos, edge_index, edge_attr, s=s_all if self.use_s_gate else None)

            x_lig = xhid[Np:Np+Nl]
            x0_pred = self.out_coord(x_lig)
            h0_pred = self.out_feat(x_lig)
            return x0_pred, h0_pred

        # Fallback: ligand-only graph path
        if isinstance(t, torch.Tensor):
            t_scalar = t[0] if t.numel() > 0 else torch.tensor(0, device=h_t.device)
        else:
            t_scalar = torch.tensor(float(t), device=h_t.device)
        t_emb = self._time_embed(t_scalar, h_t.size(0), h_t.device)
        node_in = torch.cat([h_t, s_t.unsqueeze(-1), t_emb], dim=-1)
        x = F.silu(self.input_proj(node_in))
        pos = x_t

        if edge_index is None:
            N = x.size(0)
            src = torch.arange(N, device=x.device).repeat_interleave(N)
            dst = torch.arange(N, device=x.device).repeat(N)
            mask = src != dst
            edge_index = torch.stack([src[mask], dst[mask]], dim=0)

        for layer in self.layers:
            x, pos = layer(x, pos, edge_index, edge_attr, s=s_t if self.use_s_gate else None)

        x0_pred = self.out_coord(x)
        h0_pred = self.out_feat(x)

        return x0_pred, h0_pred

