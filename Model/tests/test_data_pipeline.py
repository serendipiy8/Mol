"""
Test data pipeline for LO-MaskDiff
Tests the integration of soft mask transforms and graph building
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.soft_mask_transforms import SoftMaskTransform, LogisticNormalReparameterization, create_soft_mask_transforms
from src.data.graph_builder import HeteroGraphBuilder, create_graph_builder
from src.data.utils import ProteinLigandData
from src.data.dataset import CrossDockedDataset


def test_soft_mask_transform():
    """Test soft mask transform functionality"""
    print("Testing Soft Mask Transform...")
    
    # Create soft mask transform
    soft_mask_transform = SoftMaskTransform(
        num_steps=100,
        kappa=0.1,
        alpha_schedule='linear',
        device='cpu'
    )
    
    # Test data
    num_nodes = 10
    tau = torch.rand(num_nodes)  # Random unmask-time parameters
    
    # Test soft mask computation
    s_t = soft_mask_transform.compute_soft_mask(tau, t=50)
    assert s_t.shape == (num_nodes,), f"Expected shape ({num_nodes},), got {s_t.shape}"
    assert torch.all((s_t >= 0) & (s_t <= 1)), "Soft mask values should be in [0, 1]"
    
    # Test forward noise
    x0 = torch.randn(num_nodes, 3)
    x_t = soft_mask_transform.forward_noise(x0, tau, t=50, sigma_t=1.0)
    assert x_t.shape == x0.shape, f"Expected shape {x0.shape}, got {x_t.shape}"
    
    # Test sequence computation
    s_sequence = soft_mask_transform.compute_soft_mask_sequence(tau)
    assert s_sequence.shape == (101, num_nodes), f"Expected shape (101, {num_nodes}), got {s_sequence.shape}"
    
    print("✓ Soft mask transform tests passed!")


def test_reparameterization():
    """Test logistic-normal reparameterization"""
    print("Testing Logistic-Normal Reparameterization...")
    
    reparam = LogisticNormalReparameterization(device='cpu')
    
    # Test parameters
    num_nodes = 10
    mu = torch.randn(num_nodes)
    log_sigma = torch.randn(num_nodes)
    
    # Test sampling
    tau = reparam.sample_tau(mu, log_sigma)
    assert tau.shape == (num_nodes,), f"Expected shape ({num_nodes},), got {tau.shape}"
    assert torch.all((tau > 0) & (tau < 1)), "τ values should be in (0, 1)"
    
    # Test KL divergence
    kl_div = reparam.compute_kl_divergence(mu, log_sigma)
    assert kl_div.shape == (num_nodes,), f"Expected shape ({num_nodes},), got {kl_div.shape}"
    assert torch.all(kl_div >= 0), "KL divergence should be non-negative"
    
    print("✓ Reparameterization tests passed!")


def test_graph_builder():
    """Test heterogeneous graph building"""
    print("Testing Graph Builder...")
    
    # Create graph builder
    graph_builder = HeteroGraphBuilder(
        protein_protein_cutoff=4.5,
        ligand_ligand_cutoff=2.0,
        cross_cutoff=6.0,
        use_self_loops=True,  # Fixed parameter name
        device='cpu',
        max_cross_edges_per_ligand=10
    )
    
    # Test data
    protein_pos = torch.randn(20, 3)  # 20 protein atoms
    ligand_pos = torch.randn(10, 3)   # 10 ligand atoms
    ligand_bond_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # One bond
    ligand_bond_type = torch.tensor([1, 1], dtype=torch.long)  # Single bonds
    
    # Test protein-protein edges
    pp_edges = graph_builder.build_protein_protein_edges(protein_pos)
    assert pp_edges.shape[0] == 2, f"Expected 2 rows, got {pp_edges.shape[0]}"
    # Check indices are in valid range
    assert torch.all(pp_edges >= 0) and torch.all(pp_edges < 20), "PP edge indices out of range"
    
    # Test ligand-ligand edges
    ll_edges, ll_edge_type = graph_builder.build_ligand_ligand_edges(ligand_pos, ligand_bond_index, ligand_bond_type)
    assert ll_edges.shape[0] == 2, f"Expected 2 rows, got {ll_edges.shape[0]}"
    # Check indices are in valid range (local indices)
    assert torch.all(ll_edges >= 0) and torch.all(ll_edges < 10), "LL edge indices out of range"
    assert ll_edge_type is not None, "Edge type should be returned"
    
    # Test cross edges
    cross_edges = graph_builder.build_cross_edges(protein_pos, ligand_pos)
    assert cross_edges.shape[0] == 2, f"Expected 2 rows, got {cross_edges.shape[0]}"
    # Check indices are in valid range (local indices)
    assert torch.all(cross_edges[0] >= 0) and torch.all(cross_edges[0] < 20), "Cross edge protein indices out of range"
    assert torch.all(cross_edges[1] >= 0) and torch.all(cross_edges[1] < 10), "Cross edge ligand indices out of range"
    
    # Test complete graph building
    graph_dict = graph_builder.build_hetero_graph(protein_pos, ligand_pos, ligand_bond_index, ligand_bond_type)
    assert 'protein_protein_edges' in graph_dict
    assert 'ligand_ligand_edges' in graph_dict
    assert 'cross_edges' in graph_dict
    assert 'ligand_ligand_edge_type' in graph_dict
    assert graph_dict['num_protein_atoms'] == 20
    assert graph_dict['num_ligand_atoms'] == 10
    
    # Test edge features computation
    if pp_edges.size(1) > 0:
        pp_features = graph_builder.compute_edge_features(pp_edges, protein_pos, 'protein_protein')
        expected_dim = 7 + 16 + 3 + 5  # geom + rbf + edge_type + bond_type
        assert pp_features.shape[1] == expected_dim, f"Expected {expected_dim} features, got {pp_features.shape[1]}"
    
    if ll_edges.size(1) > 0:
        ll_features = graph_builder.compute_edge_features(ll_edges, ligand_pos, 'ligand_ligand', ll_edge_type)
        assert ll_features.shape[1] == expected_dim, f"Expected {expected_dim} features, got {ll_features.shape[1]}"
    
    if cross_edges.size(1) > 0:
        combined_pos = torch.cat([protein_pos, ligand_pos], dim=0)
        cross_features = graph_builder.compute_edge_features(cross_edges, combined_pos, 'cross')
        assert cross_features.shape[1] == expected_dim, f"Expected {expected_dim} features, got {cross_features.shape[1]}"
    
    print("✓ Graph builder tests passed!")


def test_integration():
    """Test integration of all components"""
    print("Testing Integration...")
    
    # Create transforms
    soft_mask_transform = create_soft_mask_transforms(
        num_steps=100,
        kappa=0.1,
        device='cpu'
    )
    
    graph_transform = create_graph_builder(
        protein_protein_cutoff=4.5,
        ligand_ligand_cutoff=2.0,
        cross_cutoff=6.0,
        device='cpu'
    )
    
    # Create mock data
    data = ProteinLigandData()
    data.protein_pos = torch.randn(20, 3)
    data.ligand_pos = torch.randn(10, 3)
    data.ligand_bond_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.ligand_element = torch.randint(1, 10, (10,))
    data.protein_element = torch.randint(1, 10, (20,))
    
    # Apply transforms
    data = graph_transform(data)
    data = soft_mask_transform(data)
    
    # Check that all required attributes are present
    required_attrs = [
        'protein_protein_edges', 'ligand_ligand_edges', 'cross_edges',
        'tau', 's_sequence', 'num_steps', 'kappa'
    ]
    
    for attr in required_attrs:
        assert hasattr(data, attr), f"Missing attribute: {attr}"
    
    print("✓ Integration tests passed!")


def test_monotonicity():
    """Test monotonicity of soft mask"""
    print("Testing Soft Mask Monotonicity...")
    
    soft_mask_transform = SoftMaskTransform(
        num_steps=100,
        kappa=0.1,
        alpha_schedule='linear',
        device='cpu'
    )
    
    # Test with fixed tau
    tau = torch.tensor([0.3, 0.5, 0.7])
    
    # Compute soft mask for different time steps
    s_values = []
    for t in range(0, 101, 10):
        s_t = soft_mask_transform.compute_soft_mask(tau, t)
        s_values.append(s_t)
    
    s_values = torch.stack(s_values)
    
    # Check monotonicity: s_t should increase with t for fixed tau
    for i in range(s_values.shape[1]):  # For each node
        s_sequence = s_values[:, i]
        # Check that s_t is monotonically increasing
        diffs = s_sequence[1:] - s_sequence[:-1]
        assert torch.all(diffs >= -1e-6), f"Soft mask not monotonically increasing for node {i}"
    
    print("✓ Monotonicity tests passed!")


def main():
    """Run all tests"""
    print("Running LO-MaskDiff Data Pipeline Tests...")
    print("=" * 50)
    
    try:
        test_soft_mask_transform()
        test_reparameterization()
        test_graph_builder()
        test_integration()
        test_monotonicity()
        
        print("=" * 50)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
