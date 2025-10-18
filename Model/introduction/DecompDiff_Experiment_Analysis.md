# DecompDiff 实验设置详细分析

## 🧪 实验测试内容

### 1. **Jensen-Shannon Divergence (JSD) 键长分布评估**

#### **实验目的**
评估生成分子的键长分布是否与真实分子（参考分子）的键长分布一致。

#### **实验设置**
```python
# 从 eval_bond_length.py 可以看到：
def eval_bond_length_profile(bond_length_profile: BondLengthProfile) -> Dict[str, Optional[float]]:
    metrics = {}
    
    # Jensen-Shannon distances
    for bond_type, gt_distribution in eval_bond_length_config.EMPIRICAL_DISTRIBUTIONS.items():
        if bond_type not in bond_length_profile:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = None
        else:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(
                gt_distribution, bond_length_profile[bond_type])
```

#### **评估的键类型**
- **C-C** (单键、双键、芳香键)
- **C-N** (单键、双键、芳香键)  
- **C-O** (单键、双键)
- **C:C** (芳香键)
- **C:N** (芳香键)

#### **实验流程**
1. **收集参考分子键长**: 从训练集中的真实分子提取各种键类型的长度分布
2. **收集生成分子键长**: 从模型生成的分子提取相同键类型的长度分布
3. **计算JSD**: 使用Jensen-Shannon散度比较两个分布
4. **JSD值越小**: 表示生成分子的键长分布越接近真实分子

### 2. **分子性质比较实验**

#### **评估指标**
从 `evaluate_mol_from_meta_full.py` 可以看到：

```python
# 化学性质评估
chem_results = scoring_func.get_chem(mol)

# 对接评估
if args.docking_mode in ['vina_full', 'vina_score']:
    vina_results = {
        'score_only': score_only_results,
        'minimize': minimize_results,
        'dock': dock_results  # 如果使用vina_full模式
    }
```

#### **具体评估指标**

1. **QED (Quantitative Estimate of Drug-likeness)**
   - **目的**: 评估分子的药物相似性
   - **范围**: 0-1，越高越好
   - **计算**: `np.mean(qed), np.median(qed)`

2. **SA (Synthetic Accessibility)**
   - **目的**: 评估分子的合成可达性
   - **范围**: 1-10，越低越好（越容易合成）
   - **计算**: `np.mean(sa), np.median(sa)`

3. **Vina Score (分子对接分数)**
   - **Vina Score**: 快速对接评估
   - **Vina Min**: 能量最小化后的对接分数
   - **Vina Dock**: 完整对接评估
   - **范围**: 负值，越低越好（结合越强）

4. **High Affinity Rate**
   - **定义**: 对接分数 < -6.0 kcal/mol 的分子比例
   - **意义**: 评估生成高质量结合分子的能力

5. **Success Rate**
   - **定义**: 成功生成有效分子结构的比例
   - **意义**: 评估模型的生成成功率

## 📊 实验数据来源

### **参考分子（Ground Truth）**
```python
# 从训练集中提取的键长分布作为参考
EMPIRICAL_DISTRIBUTIONS = {
    (6, 6, 1): [...],  # C-C单键分布
    (6, 6, 2): [...],  # C-C双键分布
    (6, 6, 4): [...],  # C-C芳香键分布
    # ... 其他键类型
}
```

### **生成分子**
```python
# 从模型生成结果中提取
def bond_distance_from_mol(mol):
    # 从生成的分子的3D结构中提取键长
    pos = mol.GetConformer().GetPositions()
    all_distances = []
    for bond in mol.GetBonds():
        # 计算键长并分类
        distance = pdist[s_idx, e_idx]
        all_distances.append(((s_sym, e_sym, bond_type), distance))
    return all_distances
```

## 🔬 实验设置细节

### **测试集选择**
- **蛋白质数量**: 100个不同的蛋白质
- **每个蛋白质**: 生成多个配体候选
- **关键**: 这些蛋白质在训练集中从未见过

### **生成过程**
```python
# 从 sample_diffusion_decomp.py:
def sample_diffusion_ligand_decomp(model, data, ...):
    # 为给定蛋白质口袋生成配体
    # 使用扩散模型逐步生成分子结构
    # 包括原子类型、位置、键类型
```

### **评估流程**
1. **生成阶段**: 为测试集蛋白质生成配体分子
2. **化学验证**: 检查分子有效性（SMILES解析、3D结构）
3. **性质计算**: 计算QED、SA等化学性质
4. **对接评估**: 使用AutoDock Vina进行分子对接
5. **统计分析**: 计算各种指标的均值和分位数

## 📈 实验结果解读

### **JSD键长分布结果**
```
| Bond | liGAN | GraphBP | AR | Pocket2Mol | TargetDiff | Ours |
|------|-------|---------|----|------------|------------|------|
| C-C  | 0.601 | 0.368   | 0.609 | 0.496      | 0.369      | 0.359 |
| C-N  | 0.634 | 0.456   | 0.474 | 0.416      | 0.363      | 0.344 |
```

**解读**: DecompDiff在C-C和C-N键长分布上表现最好（JSD最小），说明生成的分子具有更真实的化学结构。

### **主要评估结果**
```
| Methods    | Vina Score | Vina Min | Vina Dock | High Affinity | QED | SA | Success Rate |
|------------|------------|----------|-----------|---------------|-----|----|--------------|
| Reference  | -6.46      | -6.49    | -7.26     | -             | 0.47| 0.74| 25.0%        |
| DecompDiff | -6.04      | -7.09    | -8.43     | 71.0%         | 0.43| 0.60| 24.5%        |
```

**解读**: 
- **Vina Min**: -7.09（最好），说明生成的分子与蛋白质结合很强
- **High Affinity**: 71.0%（最高），说明大部分生成分子都有良好的结合亲和力
- **Success Rate**: 24.5%，接近参考分子，说明生成质量很高

## 🎯 实验的科学意义

### **1. 结构真实性验证**
- JSD评估确保生成的分子具有化学上合理的键长
- 这是分子生成的基础要求

### **2. 药物性质评估**
- QED和SA评估分子的药物开发潜力
- 确保生成的分子具有实际应用价值

### **3. 生物活性验证**
- Vina对接分数评估分子与蛋白质的结合能力
- High Affinity Rate评估生成高质量药物候选的能力

### **4. 泛化能力测试**
- 使用训练集未见过的蛋白质进行测试
- 验证模型对新靶点的配体设计能力

## 🔍 实验的创新点

1. **分解先验**: 将分子分解为arms和scaffolds，提高生成质量
2. **结构约束**: 使用蛋白质口袋约束，确保生成的分子能够结合
3. **多尺度评估**: 从化学结构到生物活性的全方位评估
4. **真实世界验证**: 使用实际药物发现中的评估标准

这些实验设置确保了DecompDiff不仅在理论上先进，在实际药物发现中也有应用价值。
