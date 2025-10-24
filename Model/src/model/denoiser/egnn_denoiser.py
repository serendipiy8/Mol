import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from src.model.encoders.cross_mp import CrossGraphMessagePassing


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
        # unit direction to stabilize coordinate updates
        r = torch.sqrt(r2 + 1e-8)
        dir_ij = diff / r

        if edge_attr is None:
            edge_attr = torch.zeros(diff.size(0), 0, device=x.device)
        else:
            edge_attr = edge_attr.to(x.device)
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
        coord_update = scatter_add(gamma * dir_ij, dst, dim=0, dim_size=pos.size(0))
        if self.degree_norm:
            ones = torch.ones(m_ij.size(0), 1, device=pos.device, dtype=pos.dtype)
            deg = scatter_add(ones, dst, dim=0, dim_size=pos.size(0))
            coord_update = coord_update / (deg.clamp_min(1.0))
        pos = pos + coord_update

        return x, pos


class EGNNDenoiser(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int = 1, hidden_dim: int = 128, num_layers: int = 3,
                 time_dim: int = 16, use_time: bool = True, use_s_gate: bool = True, degree_norm: bool = True,
                 use_cross_mp: bool = True, cross_mp_hidden: int = None, cross_mp_bidirectional: bool = False,
                 cross_mp_use_distance: bool = True, cross_mp_rbf_dim: int = 0, cross_mp_use_direction: bool = False,
                 cross_mp_distance_sigma: float = 4.0, cross_mp_dropout: float = 0.0, cross_mp_every: int = 1,
                 use_protein_context: bool = True, context_mode: str = 'film', context_dropout: float = 0.0,
                 use_dynamic_cross_edges: bool = True, cross_topk: int = 32, cross_radius: float = 6.0,
                 use_bond_head: bool = True, bond_hidden: int = 128, bond_classes: int = 2, bond_radius: float = 2.2,
                 use_atom_type_head: bool = True, atom_type_classes: int = 10,
                 use_dual_branch: bool = True, coord_use_tanh: bool = False, coord_alpha_min: float = 0.1,
                 debug_graph: bool = False):
        super().__init__()
        self.use_time = use_time
        self.use_s_gate = use_s_gate
        self.time_dim = time_dim
        in_dim = node_dim + 1 + (time_dim if use_time else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.use_dual_branch = bool(use_dual_branch)
        self.coord_use_tanh = bool(coord_use_tanh)
        self.coord_alpha_min = float(coord_alpha_min)
        self.debug_graph = bool(debug_graph)
        if self.use_dual_branch:
            # 更强坐标更新：坐标分支关闭度归一化，增强位移与梯度；特征分支保留度归一化稳定特征学习
            self.coord_layers = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, hidden_dim, degree_norm=False) for _ in range(num_layers)
            ])
            self.feat_layers = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, hidden_dim, degree_norm=degree_norm) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                EGNNLayer(hidden_dim, edge_dim, hidden_dim, degree_norm=degree_norm) for _ in range(num_layers)
            ])
        self.out_coord = nn.Linear(hidden_dim, 3)
        self.out_feat = nn.Linear(hidden_dim, node_dim)

        # local knn fallback to ensure connectivity
        def _knn_graph(pos: torch.Tensor, k: int = 8) -> torch.Tensor:
            if not isinstance(pos, torch.Tensor) or pos.ndim != 2 or pos.size(0) == 0:
                return torch.empty(2, 0, dtype=torch.long, device=pos.device if isinstance(pos, torch.Tensor) else 'cpu')
            with torch.no_grad():
                D = torch.cdist(pos, pos, p=2)
                N = D.size(0)
                D.fill_diagonal_(float('inf'))
                _, idx = torch.topk(-D, k=k, dim=-1)  # smallest distances
                src = torch.arange(N, device=pos.device).unsqueeze(1).repeat(1, k).reshape(-1)
                dst = idx.reshape(-1)
            return torch.stack([src, dst], dim=0)
        self._knn_graph = _knn_graph

        # cross-graph message passing
        self.use_cross_mp = use_cross_mp
        self.cross_mp_every = max(1, int(cross_mp_every))
        if use_cross_mp:
            self.cross_mp = CrossGraphMessagePassing(
                prot_dim=hidden_dim,
                lig_dim=hidden_dim,
                hidden_dim=cross_mp_hidden if cross_mp_hidden is not None else hidden_dim,
                bidirectional=cross_mp_bidirectional,
                use_distance=cross_mp_use_distance,
                rbf_dim=cross_mp_rbf_dim,
                use_direction=cross_mp_use_direction,
                distance_sigma=cross_mp_distance_sigma,
                dropout=cross_mp_dropout,
            )
        else:
            self.cross_mp = None

        # protein context injection (FiLM)
        self.use_protein_context = use_protein_context
        self.context_mode = context_mode
        self.context_dropout = nn.Dropout(context_dropout) if (isinstance(context_dropout, float) and context_dropout > 0.0) else nn.Identity()
        if self.use_protein_context and self.context_mode == 'film':
            self.ctx_gamma = nn.Linear(hidden_dim, hidden_dim)
            self.ctx_beta = nn.Linear(hidden_dim, hidden_dim)
            # optional vector context (3D) projection
            self.ctx_vec_proj = nn.Linear(3, hidden_dim)

        # dynamic cross-edge recomputation
        self.use_dynamic_cross_edges = use_dynamic_cross_edges
        self.cross_topk = int(cross_topk)
        self.cross_radius = float(cross_radius)

        # ligand bond prediction head (optional)
        self.use_bond_head = use_bond_head
        self.bond_radius = float(bond_radius)
        if self.use_bond_head:
            # input: [h_i, h_j, ||r||, dir] -> logits
            self.bond_head = nn.Sequential(
                nn.Linear(2 * hidden_dim + 1 + 3, bond_hidden), nn.SiLU(),
                nn.Linear(bond_hidden, bond_classes)
            )

        # epsilon prediction heads (optional usage in trainer)
        self.eps_coord_head = nn.Linear(hidden_dim, 3)
        self.eps_feat_head = nn.Linear(hidden_dim, node_dim)

        # atom type prediction head (element classes)
        self.use_atom_type_head = use_atom_type_head
        self.atom_type_classes = int(atom_type_classes)
        if self.use_atom_type_head:
            self.atom_type_head = nn.Linear(hidden_dim, self.atom_type_classes)

        # q_phi: tau parameter head (predict mu, log_sigma per ligand node)
        # Uses ligand raw features (node_dim) and global protein context (hidden_dim)
        self.qphi_feat_proj = nn.Linear(node_dim, hidden_dim)
        self.qphi_ctx_proj = nn.Linear(hidden_dim, hidden_dim)
        self.qphi_out = nn.Linear(hidden_dim, 2)

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

    def _get_cross_edges_local(self, batch, Np: int, Nl: int, node_type: torch.Tensor, edge_index: torch.Tensor):
        cross_edges = getattr(batch, 'cross_edges', None)
        if cross_edges is not None and cross_edges.size(1) > 0:
            return cross_edges
        if node_type is None or edge_index is None:
            return torch.empty(2, 0, dtype=torch.long, device=node_type.device if node_type is not None else edge_index.device)
        src, dst = edge_index
        mask = (node_type[src] == 0) & (node_type[dst] == 1)
        if not mask.any():
            return torch.empty(2, 0, dtype=torch.long, device=src.device)
        p_idx_local = src[mask]
        l_idx_local = dst[mask] - Np
        return torch.stack([p_idx_local, l_idx_local], dim=0)

    @staticmethod
    def _radius_graph(pos: torch.Tensor, radius: float, topk: int = 8) -> torch.Tensor:
        if pos is None or pos.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=pos.device if isinstance(pos, torch.Tensor) else 'cpu')
        D = torch.cdist(pos, pos)
        mask = (D < radius) & (D > 0)
        N = pos.size(0)
        edges_i = []
        edges_j = []
        for i in range(N):
            nbrs = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if nbrs.numel() == 0:
                continue
            dists = D[i, nbrs]
            k = min(int(topk), int(nbrs.numel()))
            _, idx = torch.topk(dists, k, largest=False)
            sel = nbrs[idx]
            edges_i.append(torch.full((k,), i, device=pos.device, dtype=torch.long))
            edges_j.append(sel)
        if not edges_i:
            return torch.empty(2, 0, dtype=torch.long, device=pos.device)
        ei = torch.cat(edges_i)
        ej = torch.cat(edges_j)
        # make directed both ways
        return torch.cat([torch.stack([ei, ej], dim=0), torch.stack([ej, ei], dim=0)], dim=1)

    def _recompute_cross_edges(self, prot_pos: torch.Tensor, lig_pos: torch.Tensor) -> torch.Tensor:
        if prot_pos is None or lig_pos is None or prot_pos.numel() == 0 or lig_pos.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=lig_pos.device if lig_pos is not None else prot_pos.device)
        D = torch.cdist(prot_pos, lig_pos)
        mask = D < self.cross_radius
        p_idx_all, l_idx_all = torch.nonzero(mask, as_tuple=True)
        if p_idx_all.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=lig_pos.device)
        if self.cross_topk > 0:
            edges_list = []
            for j in range(lig_pos.size(0)):
                j_mask = (l_idx_all == j)
                if not j_mask.any():
                    continue
                p_idx_j = p_idx_all[j_mask]
                d_j = D[p_idx_j, j]
                k = min(self.cross_topk, p_idx_j.numel())
                _, topk_idx = torch.topk(d_j, k, largest=False)
                pj_top = p_idx_j[topk_idx]
                lj_top = torch.full((k,), j, device=lig_pos.device, dtype=torch.long)
                edges_list.append(torch.stack([pj_top, lj_top], dim=0))
            if edges_list:
                return torch.cat(edges_list, dim=1)
            return torch.empty(2, 0, dtype=torch.long, device=lig_pos.device)
        else:
            return torch.stack([p_idx_all, l_idx_all], dim=0)

    def _inject_protein_context(self, xhid: torch.Tensor, Np: int, Nl: int, s_all: torch.Tensor,
                                g_P_override: torch.Tensor = None, g_P_vec_override: torch.Tensor = None):
        if not self.use_protein_context or Np <= 0 or Nl <= 0:
            return xhid
        if self.context_mode == 'film':
            if g_P_override is not None:
                g_P = g_P_override.to(xhid.device, dtype=xhid.dtype)
            else:
                g_P = xhid[:Np].mean(dim=0)
            gamma = self.ctx_gamma(g_P)
            beta = self.ctx_beta(g_P)
            # vector context: project 3D unit vector if provided
            vec_term = 0.0
            if g_P_vec_override is not None:
                v = g_P_vec_override.to(xhid.device, dtype=xhid.dtype)
                if v.dim() == 1 and v.numel() == 3:
                    v = v / (v.norm() + 1e-8)
                    vec_term = self.ctx_vec_proj(v)
            h_lig = xhid[Np:Np+Nl]
            if self.use_s_gate and s_all is not None:
                gate = s_all[Np:Np+Nl].unsqueeze(-1)
            else:
                gate = 1.0
            mod = (gamma.unsqueeze(0) + vec_term.unsqueeze(0)) * h_lig + beta.unsqueeze(0)
            mod = self.context_dropout(mod)
            xhid = xhid.clone()
            xhid[Np:Np+Nl] = h_lig + gate * mod
        return xhid

    def _predict_bonds(self, h_lig: torch.Tensor, pos_lig: torch.Tensor):
        Nl = pos_lig.size(0)
        if Nl == 0:
            return torch.empty(2, 0, dtype=torch.long, device=pos_lig.device), torch.empty(0, 2, device=pos_lig.device)
        # candidate pairs within radius (upper triangle)
        D = torch.cdist(pos_lig, pos_lig)
        mask = (D < self.bond_radius) & (D > 0)
        idx_i, idx_j = torch.nonzero(mask, as_tuple=True)
        keep = idx_i < idx_j
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]
        if idx_i.numel() == 0:
            return torch.empty(2, 0, dtype=torch.long, device=pos_lig.device), torch.empty(0, 2, device=pos_lig.device)
        hi = h_lig[idx_i]
        hj = h_lig[idx_j]
        rij = pos_lig[idx_j] - pos_lig[idx_i]
        dij = torch.norm(rij, dim=-1, keepdim=True)
        dir_ij = rij / (dij + 1e-8)
        feat = torch.cat([hi, hj, dij, dir_ij], dim=-1)
        logits = self.bond_head(feat)
        edge_index = torch.stack([idx_i, idx_j], dim=0)
        return edge_index, logits

    def predict_tau_params(self, batch) -> tuple:
        """Predict tau parameters (mu, log_sigma) for ligand nodes.
        Expects batch to contain ligand_atom_feature [Nl, node_dim].
        Optionally uses batch.g_P as global protein context; falls back to zeros if missing.
        """
        lig_feat = getattr(batch, 'ligand_atom_feature', None)
        if lig_feat is None:
            raise ValueError('predict_tau_params requires batch.ligand_atom_feature')

        device = self.qphi_feat_proj.weight.device
        g_P = getattr(batch, 'g_P', None)
        if g_P is None:
            g_P = torch.zeros(self.qphi_ctx_proj.in_features if hasattr(self.qphi_ctx_proj, 'in_features') else self.input_proj.out_features,
                              device=device)
        # Make sure projection weights and inputs are co-located on same device/dtype
        lig_feat = lig_feat.to(device, dtype=self.qphi_feat_proj.weight.dtype)
        # Align ligand feature dimension to expected input dim of qphi_feat_proj
        in_req = self.qphi_feat_proj.in_features
        if lig_feat.dim() == 1:
            lig_feat = lig_feat.view(1, -1)
        if lig_feat.size(-1) != in_req:
            if lig_feat.size(-1) < in_req:
                pad = torch.zeros(lig_feat.size(0), in_req - lig_feat.size(-1), device=lig_feat.device, dtype=lig_feat.dtype)
                lig_feat = torch.cat([lig_feat, pad], dim=-1)
            else:
                lig_feat = lig_feat[:, :in_req]
        g_P = g_P.to(device, dtype=self.qphi_ctx_proj.weight.dtype)
        h = F.silu(self.qphi_feat_proj(lig_feat))
        g_proj = F.silu(self.qphi_ctx_proj(g_P))
        h = h + g_proj.unsqueeze(0).expand_as(h)
        out = self.qphi_out(h)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        return mu, log_sigma

    def forward(self, x_t, s_t, t, batch, h_t=None):
        if h_t is None:
            raise ValueError('EGNNDenoiser requires node features h_t')

        # Edge-free sampling path: no graph/message passing, direct residual
        if bool(getattr(batch, '_edge_free', False)):
            if isinstance(t, torch.Tensor):
                t_tensor = t.to(h_t.device)
                t_scalar = t_tensor if t_tensor.dim() == 0 else t_tensor.reshape(-1)[0]
            else:
                t_scalar = torch.tensor(float(t), device=h_t.device)
            t_emb = self._time_embed(t_scalar, h_t.size(0), h_t.device)
            node_in = torch.cat([h_t, s_t.unsqueeze(-1), t_emb], dim=-1)
            x = F.silu(self.input_proj(node_in))
            delta = self.out_coord(x)
            x0_pred = x_t + delta
            h0_pred = self.out_feat(x)
            setattr(batch, 'eps_coord_pred', self.eps_coord_head(x))
            setattr(batch, 'eps_feat_pred', self.eps_feat_head(x))
            if self.use_atom_type_head:
                setattr(batch, 'pred_atom_type_logits', self.atom_type_head(x))
            return x0_pred, h0_pred

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
                    cross = (node_type[src] != node_type[dst]).float().unsqueeze(-1)
                    if edge_attr is None:
                        edge_attr = cross
                    else:
                        edge_attr = torch.cat([edge_attr, cross.to(edge_attr.device)], dim=-1)

            if isinstance(t, torch.Tensor):
                t_tensor = t.to(h_all.device)
                t_scalar = t_tensor if t_tensor.dim() == 0 else t_tensor.reshape(-1)[0]
            else:
                t_scalar = torch.tensor(float(t), device=h_all.device)
            t_emb = self._time_embed(t_scalar, h_all.size(0), h_all.device)
            node_in = torch.cat([h_all, s_all.unsqueeze(-1), t_emb], dim=-1)
            xhid = F.silu(self.input_proj(node_in))

            g_P_override = getattr(batch, 'g_P', None)
            g_P_vec_override = getattr(batch, 'g_P_vec', None)
            xhid = self._inject_protein_context(xhid, Np, Nl, s_all if self.use_s_gate else None,
                                                g_P_override=g_P_override, g_P_vec_override=g_P_vec_override)

            if self.use_cross_mp and self.cross_mp is not None:
                h_prot_cur = xhid[:Np]
                h_lig_cur = xhid[Np:Np+Nl]
                prot_pos_cur = getattr(batch, 'protein_pos', None)
                lig_pos_cur = getattr(batch, 'ligand_pos', None)
                if prot_pos_cur is None or lig_pos_cur is None:
                    prot_pos_cur = pos[:Np]
                    lig_pos_cur = pos[Np:Np+Nl]
                if self.use_dynamic_cross_edges:
                    cross_edges_local = self._recompute_cross_edges(prot_pos_cur, lig_pos_cur)
                else:
                    cross_edges_local = self._get_cross_edges_local(batch, Np, Nl, node_type, edge_index)
                s_mask = s_all[Np:Np+Nl] if self.use_s_gate else None
                h_p_out, h_l_out = self.cross_mp(
                    h_prot=h_prot_cur,
                    h_lig=h_lig_cur,
                    cross_edges=cross_edges_local,
                    prot_pos=prot_pos_cur,
                    lig_pos=lig_pos_cur,
                    edge_attr=None,
                    s_mask=s_mask,
                    t_bias=None,
                    use_softmax=True,
                    use_dist_decay=True,
                )
                xhid = xhid.clone()
                if h_p_out is not None and h_p_out.size(0) == Np:
                    xhid[:Np] = h_p_out
                xhid[Np:Np+Nl] = h_l_out

            if edge_index is None or edge_index.numel() == 0:
                # Build pp/ll/cross edges by radius and merge
                prot_pos_cur = getattr(batch, 'protein_pos', None)
                lig_pos_cur = getattr(batch, 'ligand_pos', None)
                if prot_pos_cur is None or lig_pos_cur is None:
                    prot_pos_cur = pos[:Np]
                    lig_pos_cur = pos[Np:Np+Nl]
                pp = self._radius_graph(prot_pos_cur, radius=max(4.0, self.cross_radius), topk=12)
                ll = self._radius_graph(lig_pos_cur, radius=max(2.0, self.bond_radius), topk=6)
                if ll.numel() == 0:
                    ll = self._knn_graph(lig_pos_cur, k=6)
                if self.debug_graph:
                    pp_n = int(pp.size(1)) if isinstance(pp, torch.Tensor) else 0
                    ll_n = int(ll.size(1)) if isinstance(ll, torch.Tensor) else 0
                # offset to global indices
                if pp.numel() > 0:
                    pp_global = pp.clone()
                else:
                    pp_global = torch.empty(2, 0, dtype=torch.long, device=pos.device)
                if ll.numel() > 0:
                    ll_global = ll + torch.tensor([[Np],[Np]], device=ll.device)
                else:
                    ll_global = torch.empty(2, 0, dtype=torch.long, device=pos.device)
                # cross edges from radius recomputation
                cross_local = self._recompute_cross_edges(prot_pos_cur, lig_pos_cur)
                if cross_local.numel() > 0:
                    cross_global = torch.stack([cross_local[0], cross_local[1] + Np], dim=0)
                    # add both directions
                    cross_global = torch.cat([cross_global, torch.stack([cross_global[1], cross_global[0]], dim=0)], dim=1)
                else:
                    cross_global = torch.empty(2, 0, dtype=torch.long, device=pos.device)
                if self.debug_graph:
                    cr_n = int(cross_global.size(1)) if isinstance(cross_global, torch.Tensor) else 0
                    tot_n = int(pp_global.size(1) + ll_global.size(1) + cr_n)
                    print(f"[GRAPH] pp={pp_n} ll={ll_n} cross={cr_n} total={tot_n}")
                edge_index = torch.cat([pp_global, ll_global, cross_global], dim=1) if (pp_global.numel() + ll_global.numel() + cross_global.numel()) > 0 else torch.empty(2, 0, dtype=torch.long, device=pos.device)
                # edge_attr: mark cross edges with 1, others 0
                num_pp = pp_global.size(1)
                num_ll = ll_global.size(1)
                num_cr = cross_global.size(1)
                if (num_pp + num_ll + num_cr) > 0:
                    attr = torch.zeros(num_pp + num_ll + num_cr, 1, device=pos.device, dtype=xhid.dtype)
                    if num_cr > 0:
                        attr[num_pp + num_ll:] = 1.0
                    edge_attr = attr
                else:
                    edge_attr = torch.zeros(0, 1, device=pos.device, dtype=xhid.dtype)

            if not self.use_dual_branch:
                branches = [('single', xhid, pos)]
            else:
                branches = []
                xhid_c = xhid.clone()
                xhid_f = xhid.clone()
                pos_c = pos
                pos_f = pos
                # run coord branch
                for layer_idx, layer in enumerate(self.coord_layers):
                    xhid_c, pos_c = layer(xhid_c, pos_c, edge_index, edge_attr, s=s_all if self.use_s_gate else None)
                    if self.use_cross_mp and self.cross_mp is not None and ((layer_idx + 1) % self.cross_mp_every == 0):
                        h_prot_cur = xhid_c[:Np]
                        h_lig_cur = xhid_c[Np:Np+Nl]
                        prot_pos_cur = getattr(batch, 'protein_pos', None) or pos_c[:Np]
                        lig_pos_cur = getattr(batch, 'ligand_pos', None) or pos_c[Np:Np+Nl]
                        cross_edges_local = self._recompute_cross_edges(prot_pos_cur, lig_pos_cur) if self.use_dynamic_cross_edges else self._get_cross_edges_local(batch, Np, Nl, node_type, edge_index)
                        s_mask = s_all[Np:Np+Nl] if self.use_s_gate else None
                        h_p_out, h_l_out = self.cross_mp(
                            h_prot=h_prot_cur,
                            h_lig=h_lig_cur,
                            cross_edges=cross_edges_local,
                            prot_pos=prot_pos_cur,
                            lig_pos=lig_pos_cur,
                            edge_attr=None,
                            s_mask=s_mask,
                            t_bias=None,
                            use_softmax=True,
                            use_dist_decay=True,
                        )
                        xhid_c = xhid_c.clone()
                        if h_p_out is not None and h_p_out.size(0) == Np:
                            xhid_c[:Np] = h_p_out
                        xhid_c[Np:Np+Nl] = h_l_out
                # run feat branch (ignore its pos updates)
                for layer_idx, layer in enumerate(self.feat_layers):
                    xhid_f, _ = layer(xhid_f, pos_f, edge_index, edge_attr, s=s_all if self.use_s_gate else None)
                    if self.use_cross_mp and self.cross_mp is not None and ((layer_idx + 1) % self.cross_mp_every == 0):
                        h_prot_cur = xhid_f[:Np]
                        h_lig_cur = xhid_f[Np:Np+Nl]
                        prot_pos_cur = getattr(batch, 'protein_pos', None) or pos[:Np]
                        lig_pos_cur = getattr(batch, 'ligand_pos', None) or pos[Np:Np+Nl]
                        cross_edges_local = self._recompute_cross_edges(prot_pos_cur, lig_pos_cur) if self.use_dynamic_cross_edges else self._get_cross_edges_local(batch, Np, Nl, node_type, edge_index)
                        s_mask = s_all[Np:Np+Nl] if self.use_s_gate else None
                        h_p_out, h_l_out = self.cross_mp(
                            h_prot=h_prot_cur,
                            h_lig=h_lig_cur,
                            cross_edges=cross_edges_local,
                            prot_pos=prot_pos_cur,
                            lig_pos=lig_pos_cur,
                            edge_attr=None,
                            s_mask=s_mask,
                            t_bias=None,
                            use_softmax=True,
                            use_dist_decay=True,
                        )
                        xhid_f = xhid_f.clone()
                        if h_p_out is not None and h_p_out.size(0) == Np:
                            xhid_f[:Np] = h_p_out
                        xhid_f[Np:Np+Nl] = h_l_out
                branches.append(('coord_feat', (xhid_c, pos_c, xhid_f)))

            if not self.use_dual_branch:
                # original single branch
                for layer_idx, layer in enumerate(self.layers):
                    xhid, pos = layer(xhid, pos, edge_index, edge_attr, s=s_all if self.use_s_gate else None)
                    if self.use_cross_mp and self.cross_mp is not None and ((layer_idx + 1) % self.cross_mp_every == 0):
                        h_prot_cur = xhid[:Np]
                        h_lig_cur = xhid[Np:Np+Nl]
                    prot_pos_cur = getattr(batch, 'protein_pos', None)
                    lig_pos_cur = getattr(batch, 'ligand_pos', None)
                    if prot_pos_cur is None or lig_pos_cur is None:
                        prot_pos_cur = pos[:Np]
                        lig_pos_cur = pos[Np:Np+Nl]
                    if self.use_dynamic_cross_edges:
                        cross_edges_local = self._recompute_cross_edges(prot_pos_cur, lig_pos_cur)
                    else:
                        cross_edges_local = self._get_cross_edges_local(batch, Np, Nl, node_type, edge_index)
                    s_mask = s_all[Np:Np+Nl] if self.use_s_gate else None
                    h_p_out, h_l_out = self.cross_mp(
                        h_prot=h_prot_cur,
                        h_lig=h_lig_cur,
                        cross_edges=cross_edges_local,
                        prot_pos=prot_pos_cur,
                        lig_pos=lig_pos_cur,
                        edge_attr=None,
                        s_mask=s_mask,
                        t_bias=None,
                        use_softmax=True,
                        use_dist_decay=True,
                    )
                    xhid = xhid.clone()
                    if h_p_out is not None and h_p_out.size(0) == Np:
                        xhid[:Np] = h_p_out
                    xhid[Np:Np+Nl] = h_l_out

            if not self.use_dual_branch:
                x_lig_c = xhid[Np:Np+Nl]
                x_lig_f = x_lig_c
            else:
                xhid_c, pos_c, xhid_f = branches[-1][1]
                x_lig_c = xhid_c[Np:Np+Nl]
                x_lig_f = xhid_f[Np:Np+Nl]
            # removed verbose forward debug prints
            # Predict coordinates as residual on x_t with step-dependent scale and limiting
            # 强力残差头：直接对 x_t 回归 Δ，增强坐标梯度
            x0_pred = x_t + self.out_coord(x_lig_c)
            # optional tiny residual refinement (disabled by default):
            # x0_pred = x0_pred + 0.1 * self.out_coord(x_lig_c)
            h0_pred = self.out_feat(x_lig_f)

            # reduce per-iteration prints

            # epsilon predictions for supervision (optional)
            eps_coord_pred = self.eps_coord_head(x_lig_c)
            eps_feat_pred = self.eps_feat_head(x_lig_f)
            setattr(batch, 'eps_coord_pred', eps_coord_pred)
            setattr(batch, 'eps_feat_pred', eps_feat_pred)

            # atom type predictions (optional)
            if self.use_atom_type_head:
                atom_logits = self.atom_type_head(x_lig_f)
                setattr(batch, 'pred_atom_type_logits', atom_logits)

            # optional ligand bond prediction (attach to batch)
            if self.use_bond_head:
                pos_lig = pos[Np:Np+Nl]
                edge_ll, bond_logits = self._predict_bonds(h_lig=x_lig_c, pos_lig=pos_lig)
                setattr(batch, 'pred_bond_index', edge_ll)
                setattr(batch, 'pred_bond_logits', bond_logits)

            return x0_pred, h0_pred

        if isinstance(t, torch.Tensor):
            t_tensor = t.to(h_t.device)
            t_scalar = t_tensor if t_tensor.dim() == 0 else t_tensor.reshape(-1)[0]
        else:
            t_scalar = torch.tensor(float(t), device=h_t.device)
        t_emb = self._time_embed(t_scalar, h_t.size(0), h_t.device)
        node_in = torch.cat([h_t, s_t.unsqueeze(-1), t_emb], dim=-1)
        x = F.silu(self.input_proj(node_in))
        pos = x_t

        if edge_index is None or edge_index.numel() == 0:
            # Build ligand-ligand radius graph for message passing
            ll = self._radius_graph(pos, radius=max(2.0, self.bond_radius), topk=6)
            if ll.numel() == 0:
                ll = self._knn_graph(pos, k=6)
            edge_index = ll if ll.numel() > 0 else torch.empty(2, 0, dtype=torch.long, device=x.device)
            edge_attr = torch.zeros(edge_index.size(1), 1, device=x.device, dtype=x.dtype) if edge_index.numel() > 0 else torch.zeros(0, 1, device=x.device, dtype=x.dtype)

        if not self.use_dual_branch:
            for layer in self.layers:
                x, pos = layer(x, pos, edge_index, edge_attr, s=s_t if self.use_s_gate else None)
            x_c = x
            x_f = x
        else:
            x_c = x.clone()
            x_f = x.clone()
            pos_c = pos
            for layer in self.coord_layers:
                x_c, pos_c = layer(x_c, pos_c, edge_index, edge_attr, s=s_t if self.use_s_gate else None)
            for layer in self.feat_layers:
                x_f, _ = layer(x_f, pos, edge_index, edge_attr, s=s_t if self.use_s_gate else None)


        # 强力残差头（ligand-only）：对 x_t 回归 Δ
        x0_pred = x_t + self.out_coord(x_c)
        # optional tiny residual: x0_pred = x0_pred + 0.1 * self.out_coord(x_c)
        # reduce per-iteration prints

        h0_pred = self.out_feat(x_f)
        
        # optional ligand bond prediction on ligand-only path (if positions available)
        if self.use_bond_head:
            edge_ll, bond_logits = self._predict_bonds(h_lig=x, pos_lig=pos)
            setattr(batch, 'pred_bond_index', edge_ll)
            setattr(batch, 'pred_bond_logits', bond_logits)

        # epsilon prediction heads on ligand-only path
        setattr(batch, 'eps_coord_pred', self.eps_coord_head(x_c))
        setattr(batch, 'eps_feat_pred', self.eps_feat_head(x_f))

        if self.use_atom_type_head:
            setattr(batch, 'pred_atom_type_logits', self.atom_type_head(x_f))

        return x0_pred, h0_pred

