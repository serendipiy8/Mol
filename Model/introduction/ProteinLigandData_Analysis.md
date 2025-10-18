# ProteinLigandData 数据结构详细分析

## 📋 **核心概念**

**是的，每一个 `ProteinLigandData` 都是一个完整的 protein-ligand 复合体！**

每个数据样本包含：
- **1个蛋白质分子** (完整的蛋白质结构)
- **1个配体分子** (与该蛋白质结合的配体)
- **它们之间的相互作用信息**

## 🧬 **ProteinLigandData 包含的所有字段**

### **1. 蛋白质相关字段 (protein_*)**

```python
# 蛋白质原子信息
protein_element          # 原子类型 (原子序数: 6=C, 7=N, 8=O, 1=H, 16=S, 34=Se)
protein_pos              # 原子3D坐标 [N_atoms, 3]
protein_atom_feature     # 原子特征 [N_atoms, feature_dim]
protein_atom_name        # 原子名称 ['CA', 'CB', 'N', 'O', ...]
protein_is_backbone      # 是否为骨架原子 [N_atoms]
protein_atom_to_aa_type  # 原子所属氨基酸类型 [N_atoms]

# 蛋白质残基信息
protein_residues         # 残基列表
protein_amino_acid       # 氨基酸类型 [N_residues]
protein_center_of_mass   # 残基质心 [N_residues, 3]
protein_pos_CA           # CA原子位置 [N_residues, 3]
protein_pos_C            # C原子位置 [N_residues, 3]
protein_pos_N            # N原子位置 [N_residues, 3]
protein_pos_O            # O原子位置 [N_residues, 3]

# 蛋白质文件信息
protein_file             # 蛋白质PDB文件路径
protein_molecule_name    # 蛋白质分子名称
```

### **2. 配体相关字段 (ligand_*)**

```python
# 配体原子信息
ligand_element           # 原子类型 (原子序数)
ligand_pos               # 原子3D坐标 [N_ligand_atoms, 3]
ligand_atom_feature      # 原子特征 [N_ligand_atoms, feature_dim]
ligand_atom_name         # 原子名称

# 配体键信息
ligand_bond_index        # 键连接 [2, N_bonds] - 原子索引对
ligand_bond_type         # 键类型 [N_bonds] - 1=单键, 2=双键, 3=三键, 4=芳香键
ligand_bond_feature      # 键特征 [N_bonds, bond_feature_dim]

# 配体分子信息
ligand_smiles            # SMILES字符串
ligand_mol_weight        # 分子量
ligand_num_atoms         # 原子数量
ligand_num_bonds         # 键数量

# 配体分解信息 (DecompDiff特有)
ligand_decomp_centers    # 分解中心点 [N_arms, 3]
ligand_arms              # 臂信息
ligand_scaffold          # 支架信息
ligand_fc_bond_type      # 全连接键类型

# 配体文件信息
ligand_file              # 配体SDF文件路径
ligand_filename          # 配体文件名
```

### **3. 相互作用和上下文信息**

```python
# 蛋白质-配体相互作用
interaction_distance     # 相互作用距离
interaction_type         # 相互作用类型 (氢键、疏水、π-π等)

# 上下文信息
ligand_context_element   # 上下文原子类型
ligand_context_pos       # 上下文原子位置
ligand_context_bond_index # 上下文键连接

# 掩码信息
ligand_masked_element    # 掩码原子类型
mask_ctx_edge_index_0    # 上下文边索引0
mask_ctx_edge_index_1    # 上下文边索引1
mask_compose_edge_index_0 # 组合边索引0
mask_compose_edge_index_1 # 组合边索引1
```

### **4. 邻接和拓扑信息**

```python
# 邻接列表
ligand_nbh_list          # 配体原子邻接列表 {atom_idx: [neighbor_indices]}
protein_nbh_list         # 蛋白质原子邻接列表

# 图结构信息
edge_index               # 边索引 (来自torch_geometric.Data)
edge_attr                # 边属性
```

### **5. 先验和引导信息**

```python
# 先验信息 (DecompDiff特有)
ligand_atom_prior_mask   # 配体原子先验掩码
protein_pocket_prior_mask # 蛋白质口袋先验掩码
ligand_center_prior      # 配体中心先验

# 引导信息
guidance_info            # 引导信息
```

### **6. 元数据和标识符**

```python
# 文件路径
protein_file             # 蛋白质PDB文件路径
ligand_file              # 配体SDF文件路径
src_ligand_filename      # 源配体文件名

# 标识符
sample_id                # 样本ID
complex_id               # 复合体ID
```

## 🔍 **数据结构示例**

```python
# 一个典型的ProteinLigandData样本
sample = ProteinLigandData(
    # 蛋白质信息
    protein_element=torch.tensor([6, 7, 8, 1, 6, ...]),  # C, N, O, H, C, ...
    protein_pos=torch.tensor([[10.5, 20.3, 15.7], ...]), # 3D坐标
    protein_atom_feature=torch.tensor([[1, 0, 0, ...], ...]), # 原子特征
    
    # 配体信息
    ligand_element=torch.tensor([6, 6, 7, 8, ...]),       # C, C, N, O, ...
    ligand_pos=torch.tensor([[12.1, 18.9, 16.2], ...]),  # 3D坐标
    ligand_bond_index=torch.tensor([[0, 1, 1, 2], [1, 2, 3, 3]]), # 键连接
    ligand_bond_type=torch.tensor([1, 1, 2, 1]),         # 单键, 单键, 双键, 单键
    ligand_smiles="CC(=O)Nc1ccc(O)cc1",                  # SMILES
    
    # 文件信息
    protein_file="1a2b_protein.pdb",
    ligand_file="1a2b_ligand.sdf",
    
    # 其他信息...
)
```

## 📊 **数据维度统计**

基于我们的分析，典型的数据维度：

- **蛋白质**: 平均 ~1000-5000 个原子
- **配体**: 平均 ~20-50 个原子
- **蛋白质键**: 通常很多 (蛋白质内部键)
- **配体键**: 平均 ~20-40 个键
- **相互作用**: 蛋白质-配体之间的接触

## 🎯 **关键特点**

1. **一对一关系**: 每个样本 = 1个蛋白质 + 1个配体
2. **完整结构**: 包含完整的3D坐标和化学信息
3. **相互作用**: 包含蛋白质-配体结合信息
4. **分解支持**: 支持分子分解为arms和scaffolds
5. **图结构**: 基于PyTorch Geometric的图数据结构

## 🔬 **科学意义**

这种数据结构支持：
- **结构基药物设计**: 基于蛋白质-配体复合体结构
- **分子生成**: 为特定蛋白质生成配体
- **相互作用预测**: 预测结合亲和力
- **分子分解**: 理解分子组成部分的作用

每个 `ProteinLigandData` 都是一个完整的、真实的蛋白质-配体复合体，包含了进行结构基药物设计所需的所有信息！
