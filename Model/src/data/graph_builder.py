from pickle import FALSE
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops
import torch_geometric.transforms as T

"""
The graph builder for the protein-ligand complexes.
Build the graph with protein-protein, ligand-ligand, and cross edges.
"""

# Constants
EDGE_GEOM_DIM = 7  # rel_pos(3) + distance(1) + direction(3)
RBF_DIM = 16  # Gaussian RBF encoding dimension
EDGE_TYPE_DIM = 3  # protein-protein, ligand-ligand, cross


class HeteroGraphBuilder:
    def __init__(self, protein_protein_cutoff: float = 4.5, ligand_ligand_cutoff: float = 2.0,
                 cross_cutoff: float = 6.0, use_self_loops: bool = True, device: str = 'cuda',
                 max_cross_edges_per_ligand: int = 32): 
        """
        Initialize graph builder
        
        Args:
            protein_protein_cutoff: Distance cutoff for protein-protein edges (Å)
            ligand_ligand_cutoff: Distance cutoff for ligand-ligand edges (Å)
            cross_cutoff: Distance cutoff for protein-ligand cross edges (Å)
            max_cross_edges_per_ligand: Maximum cross edges per ligand atom

        In fact, the cutoff distances we follows the previous work, and the ligand-ligand cutoff is 
        always true chemical bond, and the others are based on the distance between the atoms.
        """


        self.protein_protein_cutoff = protein_protein_cutoff
        self.ligand_ligand_cutoff = ligand_ligand_cutoff
        self.cross_cutoff = cross_cutoff
        self.use_self_loops = use_self_loops
        self.device = device
        self.max_cross_edges_per_ligand = max_cross_edges_per_ligand
    
    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)
    
    def build_protein_protein_edges(self, protein_pos: torch.Tensor) -> torch.Tensor:
        """
        Build protein-protein edges based on Cα-Cα distances
        
        Args:
            protein_pos: Protein atom positions [N_protein, 3]
            
        Returns:
            edge_index: Protein-protein edge indices [2, N_edges] (local indices)
        """

        protein_pos = self._to_device(protein_pos)
        
        distances = torch.cdist(protein_pos, protein_pos)
        upper_tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        edge_mask = (distances < self.protein_protein_cutoff) & upper_tri
        
        edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
        
        edge_index = to_undirected(edge_index)
        
        if self.use_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=protein_pos.size(0))
        
        return edge_index
    
    def build_ligand_ligand_edges(self, ligand_pos: torch.Tensor, ligand_bond_index: Optional[torch.Tensor] = None,
                                 ligand_bond_type: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build ligand-ligand edges based on chemical bonds and spatial proximity
        """
        ligand_pos = self._to_device(ligand_pos)
        edges = []
        edge_types = []
        

        # Add chemical bonds if provided
        if ligand_bond_index is not None and ligand_bond_index.size(1) > 0:
            edges.append(ligand_bond_index)
            if ligand_bond_type is not None:
                edge_types.append(ligand_bond_type)
        
        if ligand_bond_index is not None:
            existing_edges = set()
            for i in range(ligand_bond_index.size(1)):
                src, tgt = ligand_bond_index[:, i].tolist()
                existing_edges.add((min(src, tgt), max(src, tgt)))
        else:
            existing_edges = set()
        
        # Spatial edges
        distances = torch.cdist(ligand_pos, ligand_pos)
        spatial_mask = (distances < self.ligand_ligand_cutoff) & (distances > 0)
        
        new_edges = []
        for i in range(ligand_pos.size(0)):
            for j in range(i + 1, ligand_pos.size(0)):
                if spatial_mask[i, j]:
                    edge_key = (i, j)
                    if edge_key not in existing_edges:
                        new_edges.append([i, j])
        
        if new_edges:
            spatial_edges = torch.tensor(new_edges).t()
            edges.append(spatial_edges)
            edge_types.append(torch.zeros(spatial_edges.size(1), dtype=torch.long, device=ligand_pos.device))
        
        if edges:
            edge_index = torch.cat(edges, dim=1)
            edge_index = to_undirected(edge_index)
            
            if edge_types:
                edge_type = torch.cat(edge_types)
                edge_type = torch.cat([edge_type, edge_type])
            else:
                edge_type = None
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=ligand_pos.device)
            edge_type = None
        
        if self.use_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=ligand_pos.size(0))
            if edge_type is not None:
                self_loop_type = torch.zeros(ligand_pos.size(0), dtype=torch.long, device=ligand_pos.device)
                edge_type = torch.cat([edge_type, self_loop_type])
        
        return edge_index, edge_type
    
    def build_cross_edges(self, protein_pos: torch.Tensor,
                         ligand_pos: torch.Tensor) -> torch.Tensor:
        """
        Build protein-ligand cross edges based on distance
        
        Args:
            protein_pos: Protein atom positions [N_protein, 3]
            ligand_pos: Ligand atom positions [N_ligand, 3]
            
        Returns:
            edge_index: Cross edge indices [2, N_edges] (protein -> ligand, local indices)
        """

        protein_pos = self._to_device(protein_pos)
        ligand_pos = self._to_device(ligand_pos)
        
        distances = torch.cdist(protein_pos, ligand_pos)
        edge_mask = distances < self.cross_cutoff
        
        protein_indices, ligand_indices = torch.nonzero(edge_mask, as_tuple=True)
        
        if self.max_cross_edges_per_ligand > 0:
            limited_edges = []
            for lig_idx in range(ligand_pos.size(0)):
                lig_mask = ligand_indices == lig_idx
                if lig_mask.any():
                    prot_indices = protein_indices[lig_mask]
                    lig_distances = distances[prot_indices, lig_idx]
                    
                    k = min(self.max_cross_edges_per_ligand, len(prot_indices))
                    _, top_k_indices = torch.topk(lig_distances, k, largest=False)
                    
                    limited_edges.append(torch.stack([
                        prot_indices[top_k_indices],
                        torch.full((k,), lig_idx, device=ligand_pos.device)
                    ]))
            
            if limited_edges:
                edge_index = torch.cat(limited_edges, dim=1)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=ligand_pos.device)
        else:
            edge_index = torch.stack([protein_indices, ligand_indices])
        
        return edge_index
    
    def build_hetero_graph(self, protein_pos: torch.Tensor, ligand_pos: torch.Tensor,
                          ligand_bond_index: Optional[torch.Tensor] = None,
                          ligand_bond_type: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        pp_edges = self.build_protein_protein_edges(protein_pos)
        ll_edges, ll_edge_type = self.build_ligand_ligand_edges(ligand_pos, ligand_bond_index, ligand_bond_type)
        cross_edges = self.build_cross_edges(protein_pos, ligand_pos)
        
        return {
            'protein_protein_edges': pp_edges,  
            'ligand_ligand_edges': ll_edges,  
            'ligand_ligand_edge_type': ll_edge_type,  
            'cross_edges': cross_edges,       
            'num_protein_atoms': protein_pos.size(0),
            'num_ligand_atoms': ligand_pos.size(0)
        }
    
    def compute_rbf_features(self, distances: torch.Tensor, num_rbf: int = RBF_DIM, 
                           rbf_cutoff: float = 5.0) -> torch.Tensor:
        """
        Compute RBF features for distances
        
        Args:
            distances: Distance values [N_edges]
            num_rbf: Number of RBF features
            rbf_cutoff: RBF cutoff distance
            
        Returns:
            rbf_features: RBF features [N_edges, num_rbf]
        """
        rbf_centers = torch.linspace(0, rbf_cutoff, num_rbf, device=distances.device)
        
        distances_expanded = distances.unsqueeze(-1)  # [N_edges, 1]
        rbf_centers_expanded = rbf_centers.unsqueeze(0)  # [1, num_rbf]
        
        gamma = 1.0 / (2 * (rbf_cutoff / num_rbf) ** 2)
        rbf_features = torch.exp(-gamma * (distances_expanded - rbf_centers_expanded) ** 2)
        
        return rbf_features
    
    def compute_edge_features(self, edge_index: torch.Tensor, pos: torch.Tensor,
                             edge_type: str = 'spatial', bond_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute edge features based on geometry
        """

        if edge_index.size(1) == 0:
            return torch.empty((0, EDGE_GEOM_DIM + RBF_DIM + EDGE_TYPE_DIM + 5), device=pos.device)
        
        src_pos = pos[edge_index[0]]
        tgt_pos = pos[edge_index[1]]
        rel_pos = tgt_pos - src_pos
        
        
        distance = torch.norm(rel_pos, dim=1, keepdim=True)
        direction = rel_pos / (distance + 1e-8)
        
        # Compute RBF features
        rbf_features = self.compute_rbf_features(distance.squeeze(-1))
        
        # Edge type one-hot encoding
        edge_type_map = {'protein_protein': 0, 'ligand_ligand': 1, 'cross': 2}
        edge_type_idx = edge_type_map.get(edge_type, 0)
        edge_type_onehot = torch.zeros(edge_index.size(1), EDGE_TYPE_DIM, device=pos.device)
        edge_type_onehot[:, edge_type_idx] = 1.0
        
        # Bond type features (if available)
        if bond_type is not None and len(bond_type) == edge_index.size(1):
            bond_type_onehot = torch.zeros(edge_index.size(1), 5, device=pos.device)
            bond_type_onehot[torch.arange(edge_index.size(1)), bond_type] = 1.0
        else:
            bond_type_onehot = torch.zeros(edge_index.size(1), 5, device=pos.device)
        
        edge_features = torch.cat([
            rel_pos,           # 3
            distance,          # 1
            direction,         # 3
            rbf_features,      # RBF_DIM
            edge_type_onehot,  # EDGE_TYPE_DIM
            bond_type_onehot   # 5
        ], dim=1)
        
        return edge_features
    
    def build_complete_graph(self, protein_pos: torch.Tensor, ligand_pos: torch.Tensor,
                           protein_atom_feature: Optional[torch.Tensor] = None,
                           ligand_atom_feature: Optional[torch.Tensor] = None,
                           ligand_bond_index: Optional[torch.Tensor] = None,
                           ligand_bond_type: Optional[torch.Tensor] = None) -> Data:

        protein_pos = self._to_device(protein_pos)
        ligand_pos = self._to_device(ligand_pos)
        

        graph_dict = self.build_hetero_graph(protein_pos, ligand_pos, ligand_bond_index, ligand_bond_type)
        
        # graph edges
        pp_edges = graph_dict['protein_protein_edges'] 
        ll_edges = graph_dict['ligand_ligand_edges']   
        cross_edges = graph_dict['cross_edges']     
        
        num_protein = protein_pos.size(0)
        num_ligand = ligand_pos.size(0)
        
        if ll_edges.size(1) > 0:
            ll_edges_global = ll_edges + num_protein
        else:
            ll_edges_global = ll_edges
        

        if cross_edges.size(1) > 0:
            cross_edges_global = torch.stack([cross_edges[0],  cross_edges[1] + num_protein])
        else:
            cross_edges_global = cross_edges
        
        all_edges = []
        if pp_edges.size(1) > 0:
            all_edges.append(pp_edges)
        if ll_edges_global.size(1) > 0:
            all_edges.append(ll_edges_global)
        if cross_edges_global.size(1) > 0:
            all_edges.append(cross_edges_global)
        
        if all_edges:
            edge_index = torch.cat(all_edges, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=protein_pos.device)


        # graph nodes 
        pos = torch.cat([protein_pos, ligand_pos], dim=0) 
        
        if protein_atom_feature is not None:
            protein_atom_feature = self._to_device(protein_atom_feature)
        else:
            protein_atom_feature = torch.zeros(num_protein, 5, device=protein_pos.device)
        
        if ligand_atom_feature is not None:
            ligand_atom_feature = self._to_device(ligand_atom_feature)
        else:
            ligand_atom_feature = torch.zeros(num_ligand, 5, device=ligand_pos.device)
        
        # Ensure both features have the same dimension
        protein_dim = protein_atom_feature.shape[1]
        ligand_dim = ligand_atom_feature.shape[1]
        max_dim = max(protein_dim, ligand_dim)
        
        # Pad features to the same dimension
        if protein_dim < max_dim:
            padding = torch.zeros(num_protein, max_dim - protein_dim, device=protein_atom_feature.device)
            protein_atom_feature = torch.cat([protein_atom_feature, padding], dim=1)
        
        if ligand_dim < max_dim:
            padding = torch.zeros(num_ligand, max_dim - ligand_dim, device=ligand_atom_feature.device)
            ligand_atom_feature = torch.cat([ligand_atom_feature, padding], dim=1)
        
        x = torch.cat([protein_atom_feature, ligand_atom_feature], dim=0) 
        
        # Create node type indicator (0=protein, 1=ligand)
        node_type = torch.cat([
            torch.zeros(num_protein, dtype=torch.long, device=protein_pos.device),
            torch.ones(num_ligand, dtype=torch.long, device=ligand_pos.device)
        ])
        
        if edge_index.size(1) > 0:
            # For mixed edge types, we don't have specific bond types
            edge_attr = self.compute_edge_features(edge_index, pos, 'mixed', None)
        else:
            edge_attr = torch.empty((0, EDGE_GEOM_DIM + RBF_DIM + EDGE_TYPE_DIM + 5), device=pos.device)
        
        # Create complete Data object
        data = Data(
            x=x,                    # Node features [N_total, feature_dim]
            pos=pos,                # Node positions [N_total, 3]
            edge_index=edge_index,  # Edge indices [2, E_total]
            edge_attr=edge_attr,    # Edge features [E_total, edge_feature_dim]
            node_type=node_type,    # Node type indicator [N_total]
            num_protein_atoms=num_protein,
            num_ligand_atoms=num_ligand
        )
        
        # Add graph structure information
        data.protein_protein_edges = pp_edges
        data.ligand_ligand_edges = ll_edges
        data.cross_edges = cross_edges
        data.ligand_ligand_edge_type = graph_dict.get('ligand_ligand_edge_type')
        
        # Preserve original positions and features
        data.protein_pos = protein_pos
        data.ligand_pos = ligand_pos
        
        # Preserve original features if they exist
        if protein_atom_feature is not None:
            data.protein_atom_feature = protein_atom_feature
        if ligand_atom_feature is not None:
            data.ligand_atom_feature = ligand_atom_feature
        if ligand_bond_index is not None:
            data.ligand_bond_index = ligand_bond_index
        if ligand_bond_type is not None:
            data.ligand_bond_type = ligand_bond_type
        
        return data


class GraphDataTransform:
    """
    Data transform that builds protein-ligand graphs
    """

    def __init__(self, graph_builder: HeteroGraphBuilder):
        self.graph_builder = graph_builder
    
    def __call__(self, data: Data) -> Data:
        data.protein_pos = self.graph_builder._to_device(data.protein_pos)      
        data.ligand_pos = self.graph_builder._to_device(data.ligand_pos)
        
        graph_dict = self.graph_builder.build_hetero_graph(
            data.protein_pos,
            data.ligand_pos,
            getattr(data, 'ligand_bond_index', None),
            getattr(data, 'ligand_bond_type', None)
        )
        
        for key, value in graph_dict.items():
            setattr(data, key, value)
        
        if data.protein_protein_edges.size(1) > 0:
            data.protein_protein_edge_features = self.graph_builder.compute_edge_features(
                data.protein_protein_edges, data.protein_pos, 'protein_protein'
            )
        
        if data.ligand_ligand_edges.size(1) > 0:
            # Get bond type, but only use it if length matches
            bond_type = getattr(data, 'ligand_ligand_edge_type', None)
            if bond_type is not None and len(bond_type) != data.ligand_ligand_edges.size(1):
                bond_type = None  # Don't use if length doesn't match
            data.ligand_ligand_edge_features = self.graph_builder.compute_edge_features(
                data.ligand_ligand_edges, data.ligand_pos, 'ligand_ligand', bond_type
            )
        
        if data.cross_edges.size(1) > 0:
            combined_pos = torch.cat([data.protein_pos, data.ligand_pos], dim=0)
            data.cross_edge_features = self.graph_builder.compute_edge_features(
                data.cross_edges, combined_pos, 'cross'
            )
        
        return data
    
    def build_complete_graph(self, data: Data) -> Data:
        protein_pos = data.protein_pos
        ligand_pos = data.ligand_pos
        protein_atom_feature = getattr(data, 'protein_atom_feature', None)
        ligand_atom_feature = getattr(data, 'ligand_atom_feature', None)
        ligand_bond_index = getattr(data, 'ligand_bond_index', None)
        ligand_bond_type = getattr(data, 'ligand_bond_type', None)
        
        complete_data = self.graph_builder.build_complete_graph(
            protein_pos=protein_pos,
            ligand_pos=ligand_pos,
            protein_atom_feature=protein_atom_feature,
            ligand_atom_feature=ligand_atom_feature,
            ligand_bond_index=ligand_bond_index,
            ligand_bond_type=ligand_bond_type
        )
        
        for key, value in data.__dict__.items():
            if key not in ['x', 'pos', 'edge_index', 'edge_attr', 'node_type', 'num_protein_atoms', 'num_ligand_atoms']:
                setattr(complete_data, key, value)
        
        return complete_data


def create_graph_builder(protein_protein_cutoff: float = 4.5, ligand_ligand_cutoff: float = 2.0,
                        cross_cutoff: float = 6.0, use_self_loops: bool = True,
                        device: str = 'cuda', max_cross_edges_per_ligand: int = 32) -> GraphDataTransform:
    graph_builder = HeteroGraphBuilder(
        protein_protein_cutoff=protein_protein_cutoff,
        ligand_ligand_cutoff=ligand_ligand_cutoff,
        cross_cutoff=cross_cutoff,
        use_self_loops=use_self_loops,
        device=device,
        max_cross_edges_per_ligand=max_cross_edges_per_ligand
    )
    
    return GraphDataTransform(graph_builder)