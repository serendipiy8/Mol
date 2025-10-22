import torch
import torch.nn as nn
from .egnn import EGNNEncoder


class ProteinEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4, edge_dim: int = 0, aggr: str = 'mean'):
        super().__init__()
        self.egnn = EGNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim, aggr=aggr)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, protein_pos: torch.Tensor, protein_atom_feature: torch.Tensor, pp_edge_index: torch.Tensor, pp_edge_attr: torch.Tensor = None):
        h_prot, _ = self.egnn(protein_atom_feature, protein_pos, pp_edge_index, pp_edge_attr)
        if h_prot.size(0) > 0:
            g_P = self.pool(h_prot.transpose(0, 1).unsqueeze(0)).squeeze(0).squeeze(-1)
        else:
            g_P = torch.zeros(h_prot.size(1), device=h_prot.device)
        return h_prot, g_P


