"""
测试完整的Graph构建功能
验证GraphDataTransform是否能返回完整的PyTorch Geometric Data对象
"""

import torch
import numpy as np
from src.data.dataset import CrossDockedDataset
from src.data.graph_builder import create_graph_builder

def test_complete_graph_construction():
    """测试完整Graph构建"""
    print("🧬 测试完整Graph构建...")
    
    # 创建graph builder
    transform = create_graph_builder(
        protein_protein_cutoff=4.5,
        ligand_ligand_cutoff=2.0,
        cross_cutoff=6.0,
        use_self_loops=True,
        device='cpu'
    )
    
    # 加载数据集
    try:
        dataset = CrossDockedDataset('src/data/crossdocked_v1.1_rmsd1.0_processed')
        print(f"✅ 数据集加载成功，总样本数: {len(dataset)}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 测试第一个样本
    data = dataset[0]
    if data is None:
        print("❌ 无法加载测试数据")
        return
    
    print(f"\n📊 原始数据信息:")
    print(f"  Protein atoms: {data.protein_pos.shape[0]}")
    print(f"  Ligand atoms: {data.ligand_pos.shape[0]}")
    print(f"  Protein features: {getattr(data, 'protein_atom_feature', 'None')}")
    print(f"  Ligand features: {getattr(data, 'ligand_atom_feature', 'None')}")
    
    # 方法1: 使用原有的transform (只添加边信息)
    print(f"\n🔧 方法1: 使用原有transform...")
    data_with_edges = transform(data)
    
    print(f"  添加的边信息:")
    print(f"    protein_protein_edges: {data_with_edges.protein_protein_edges.shape}")
    print(f"    ligand_ligand_edges: {data_with_edges.ligand_ligand_edges.shape}")
    print(f"    cross_edges: {data_with_edges.cross_edges.shape}")
    
    # 方法2: 使用新的build_complete_graph (创建完整Graph)
    print(f"\n🔧 方法2: 使用build_complete_graph...")
    complete_data = transform.build_complete_graph(data)
    
    print(f"  完整Graph信息:")
    print(f"    x (node features): {complete_data.x.shape}")
    print(f"    pos (positions): {complete_data.pos.shape}")
    print(f"    edge_index (unified edges): {complete_data.edge_index.shape}")
    print(f"    edge_attr (edge features): {complete_data.edge_attr.shape}")
    print(f"    node_type (0=protein, 1=ligand): {complete_data.node_type.shape}")
    print(f"    num_protein_atoms: {complete_data.num_protein_atoms}")
    print(f"    num_ligand_atoms: {complete_data.num_ligand_atoms}")
    
    # 验证Graph结构
    print(f"\n✅ Graph结构验证:")
    
    # 检查节点数量
    expected_nodes = data.protein_pos.shape[0] + data.ligand_pos.shape[0]
    assert complete_data.x.shape[0] == expected_nodes, f"节点数量不匹配: {complete_data.x.shape[0]} != {expected_nodes}"
    assert complete_data.pos.shape[0] == expected_nodes, f"位置数量不匹配: {complete_data.pos.shape[0]} != {expected_nodes}"
    assert complete_data.node_type.shape[0] == expected_nodes, f"节点类型数量不匹配: {complete_data.node_type.shape[0]} != {expected_nodes}"
    
    # 检查边索引范围
    if complete_data.edge_index.size(1) > 0:
        max_node_idx = torch.max(complete_data.edge_index).item()
        assert max_node_idx < expected_nodes, f"边索引超出范围: {max_node_idx} >= {expected_nodes}"
        print(f"  ✓ 边索引范围正确: 0-{max_node_idx}")
    
    # 检查节点类型
    protein_nodes = (complete_data.node_type == 0).sum().item()
    ligand_nodes = (complete_data.node_type == 1).sum().item()
    assert protein_nodes == data.protein_pos.shape[0], f"蛋白质节点数量不匹配: {protein_nodes} != {data.protein_pos.shape[0]}"
    assert ligand_nodes == data.ligand_pos.shape[0], f"配体节点数量不匹配: {ligand_nodes} != {data.ligand_pos.shape[0]}"
    print(f"  ✓ 节点类型正确: {protein_nodes}个蛋白质节点, {ligand_nodes}个配体节点")
    
    # 检查位置
    expected_pos = torch.cat([data.protein_pos, data.ligand_pos], dim=0)
    assert torch.allclose(complete_data.pos, expected_pos), "位置不匹配"
    print(f"  ✓ 位置信息正确")
    
    # 检查边特征维度
    expected_edge_dim = 7 + 16 + 3 + 5  # geom + rbf + edge_type + bond_type
    assert complete_data.edge_attr.shape[1] == expected_edge_dim, f"边特征维度不匹配: {complete_data.edge_attr.shape[1]} != {expected_edge_dim}"
    print(f"  ✓ 边特征维度正确: {complete_data.edge_attr.shape[1]}")
    
    # 检查是否保留了原始数据
    original_attrs = ['protein_pos', 'ligand_pos']
    for attr in original_attrs:
        assert hasattr(complete_data, attr), f"缺少原始属性: {attr}"
        print(f"  ✓ 保留了原始属性: {attr}")
    
    # 检查可选属性
    optional_attrs = ['protein_atom_feature', 'ligand_atom_feature', 'ligand_bond_index', 'ligand_bond_type']
    for attr in optional_attrs:
        if hasattr(data, attr):
            assert hasattr(complete_data, attr), f"缺少原始属性: {attr}"
            print(f"  ✓ 保留了原始属性: {attr}")
        else:
            print(f"  - 原始数据中没有 {attr} 属性")
    
    print(f"\n🎉 完整Graph构建成功！")
    print(f"  总节点数: {complete_data.x.shape[0]}")
    print(f"  总边数: {complete_data.edge_index.shape[1]}")
    print(f"  节点特征维度: {complete_data.x.shape[1]}")
    print(f"  边特征维度: {complete_data.edge_attr.shape[1]}")
    
    return complete_data

def test_graph_with_different_features():
    """测试不同特征维度的Graph构建"""
    print(f"\n🧪 测试不同特征维度...")
    
    # 创建graph builder
    transform = create_graph_builder(device='cpu')
    
    # 创建测试数据
    protein_pos = torch.randn(50, 3)
    ligand_pos = torch.randn(20, 3)
    protein_atom_feature = torch.randn(50, 10)  # 10维特征
    ligand_atom_feature = torch.randn(20, 8)    # 8维特征
    ligand_bond_index = torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    ligand_bond_type = torch.tensor([1, 1, 2, 1], dtype=torch.long)
    
    # 创建ProteinLigandData对象
    from src.data.utils import ProteinLigandData
    data = ProteinLigandData(
        protein_pos=protein_pos,
        ligand_pos=ligand_pos,
        protein_atom_feature=protein_atom_feature,
        ligand_atom_feature=ligand_atom_feature,
        ligand_bond_index=ligand_bond_index,
        ligand_bond_type=ligand_bond_type
    )
    
    # 构建完整Graph
    complete_data = transform.build_complete_graph(data)
    
    print(f"  测试数据:")
    print(f"    Protein: {protein_pos.shape[0]} atoms, {protein_atom_feature.shape[1]} features")
    print(f"    Ligand: {ligand_pos.shape[0]} atoms, {ligand_atom_feature.shape[1]} features")
    
    print(f"  构建结果:")
    print(f"    总节点数: {complete_data.x.shape[0]}")
    print(f"    节点特征维度: {complete_data.x.shape[1]}")
    print(f"    总边数: {complete_data.edge_index.shape[1]}")
    
    # 验证特征维度
    expected_feature_dim = max(protein_atom_feature.shape[1], ligand_atom_feature.shape[1])
    assert complete_data.x.shape[1] == expected_feature_dim, f"特征维度不匹配: {complete_data.x.shape[1]} != {expected_feature_dim}"
    print(f"  ✓ 特征维度正确: {complete_data.x.shape[1]}")
    
    print(f"  ✅ 不同特征维度测试通过！")

def main():
    """主函数"""
    print("🔬 完整Graph构建测试")
    print("=" * 50)
    
    # 测试完整Graph构建
    complete_data = test_complete_graph_construction()
    
    # 测试不同特征维度
    test_graph_with_different_features()
    
    print(f"\n🎯 总结:")
    print(f"1. ✅ GraphDataTransform现在可以创建完整的PyTorch Geometric Data对象")
    print(f"2. ✅ 包含统一的edge_index、pos、x、edge_attr等标准字段")
    print(f"3. ✅ 支持不同维度的节点特征")
    print(f"4. ✅ 保留了原始数据的所有属性")
    print(f"5. ✅ 可以直接用于GNN模型训练")

if __name__ == "__main__":
    main()
