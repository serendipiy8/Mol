import torch
import torch.nn as nn
from .egnn import EGNNEncoder


class LigandEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 4, edge_dim: int = 0, aggr: str = 'mean'):
        super().__init__()
        self.egnn = EGNNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, edge_dim=edge_dim, aggr=aggr)

    def forward(self, ligand_pos: torch.Tensor, ligand_atom_feature: torch.Tensor, ll_edge_index: torch.Tensor, ll_edge_attr: torch.Tensor = None):
        h_lig, _ = self.egnn(ligand_atom_feature, ligand_pos, ll_edge_index, ll_edge_attr)
        return h_lig


