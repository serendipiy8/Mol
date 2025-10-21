import torch
import torch.nn as nn
from .egnn import EGNNEncoder


class ProteinEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4, edge_dim: int = 0, aggr: str = 'mean'):
        super().__init__()
        self.egnn = EGNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim, aggr=aggr)

    def forward(self, protein_pos: torch.Tensor, protein_atom_feature: torch.Tensor, pp_edge_index: torch.Tensor, pp_edge_attr: torch.Tensor = None):
        h_prot, _ = self.egnn(protein_atom_feature, protein_pos, pp_edge_index, pp_edge_attr)
        g_P = h_prot.mean(dim=0) if h_prot.size(0) > 0 else torch.zeros(h_prot.size(1), device=h_prot.device)
        return h_prot, g_P


