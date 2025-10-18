#!/usr/bin/env python3
"""
分析CrossDocked数据集的结构
查看蛋白质-配体的对应关系
"""

import sys
import os
sys.path.append('.')

from src.data.dataset import ModelCrossDockedDataset
import torch
from collections import defaultdict, Counter

def analyze_protein_ligand_mapping():
    """分析蛋白质-配体的映射关系"""
    
    print("=" * 60)
    print("CrossDocked数据集结构分析")
    print("=" * 60)
    
    # 加载数据集
    dataset = ModelCrossDockedDataset('./src/data/crossdocked_v1.1_rmsd1.0')
    
    print(f"总样本数: {len(dataset)}")
    
    # 分析前100个样本
    protein_ligand_pairs = []
    protein_to_ligands = defaultdict(list)
    ligand_to_proteins = defaultdict(list)
    
    print("\n分析前100个样本...")
    for i in range(min(100, len(dataset))):
        sample = dataset[i]
        
        # 获取蛋白质和配体信息
        protein_file = sample.get('protein_file', 'unknown')
        ligand_file = sample.get('ligand_file', 'unknown')
        protein_name = sample.get('protein_molecule_name', 'unknown')
        ligand_smiles = sample.get('ligand_smiles', 'unknown')
        
        # 提取蛋白质ID（通常是PDB ID）
        if '/' in protein_file:
            protein_id = protein_file.split('/')[-1].split('_')[0]  # 提取PDB ID
        else:
            protein_id = protein_file.split('_')[0] if '_' in protein_file else protein_file
        
        # 提取配体ID
        if '/' in ligand_file:
            ligand_id = ligand_file.split('/')[-1].split('_')[0]
        else:
            ligand_id = ligand_file.split('_')[0] if '_' in ligand_file else ligand_file
        
        protein_ligand_pairs.append((protein_id, ligand_id))
        protein_to_ligands[protein_id].append(ligand_id)
        ligand_to_proteins[ligand_id].append(protein_id)
    
    print(f"\n前100个样本分析结果:")
    print(f"唯一蛋白质数量: {len(protein_to_ligands)}")
    print(f"唯一配体数量: {len(ligand_to_proteins)}")
    
    # 分析每个蛋白质对应的配体数量
    ligands_per_protein = [len(ligands) for ligands in protein_to_ligands.values()]
    proteins_per_ligand = [len(proteins) for proteins in ligand_to_proteins.values()]
    
    print(f"\n每个蛋白质对应的配体数量统计:")
    print(f"  平均: {sum(ligands_per_protein) / len(ligands_per_protein):.2f}")
    print(f"  最大: {max(ligands_per_protein)}")
    print(f"  最小: {min(ligands_per_protein)}")
    print(f"  分布: {dict(Counter(ligands_per_protein))}")
    
    print(f"\n每个配体对应的蛋白质数量统计:")
    print(f"  平均: {sum(proteins_per_ligand) / len(proteins_per_ligand):.2f}")
    print(f"  最大: {max(proteins_per_ligand)}")
    print(f"  最小: {min(proteins_per_ligand)}")
    print(f"  分布: {dict(Counter(proteins_per_ligand))}")
    
    # 显示一些具体的例子
    print(f"\n具体例子:")
    print("蛋白质 -> 配体映射 (前5个):")
    for i, (protein, ligands) in enumerate(list(protein_to_ligands.items())[:5]):
        print(f"  {protein}: {ligands[:3]}{'...' if len(ligands) > 3 else ''}")
    
    print("\n配体 -> 蛋白质映射 (前5个):")
    for i, (ligand, proteins) in enumerate(list(ligand_to_proteins.items())[:5]):
        print(f"  {ligand}: {proteins[:3]}{'...' if len(proteins) > 3 else ''}")
    
    return protein_to_ligands, ligand_to_proteins

def analyze_specific_protein(protein_id, protein_to_ligands):
    """分析特定蛋白质的所有配体"""
    
    print(f"\n" + "=" * 60)
    print(f"分析蛋白质 {protein_id} 的配体")
    print("=" * 60)
    
    if protein_id not in protein_to_ligands:
        print(f"蛋白质 {protein_id} 在样本中未找到")
        return
    
    ligands = protein_to_ligands[protein_id]
    print(f"蛋白质 {protein_id} 对应的配体数量: {len(ligands)}")
    print(f"配体列表: {ligands}")
    
    return ligands

def analyze_dataset_structure():
    """分析数据集的整体结构"""
    
    print(f"\n" + "=" * 60)
    print("数据集结构分析")
    print("=" * 60)
    
    # 加载数据集
    dataset = ModelCrossDockedDataset('./src/data/crossdocked_v1.1_rmsd1.0')
    
    # 分析一个样本的详细信息
    sample = dataset[0]
    
    print("样本包含的字段:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape {value.shape}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} length {len(value)}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # 分析分子分解信息
    print(f"\n分子分解信息:")
    print(f"  Arms数量: {sample.get('num_arms', 'N/A')}")
    print(f"  Scaffold数量: {sample.get('num_scaffold', 'N/A')}")
    
    # 分析原子信息
    print(f"\n原子信息:")
    if 'protein_pos' in sample:
        print(f"  蛋白质原子数: {sample['protein_pos'].shape[0]}")
    if 'ligand_pos' in sample:
        print(f"  配体原子数: {sample['ligand_pos'].shape[0]}")
    
    return sample

if __name__ == "__main__":
    try:
        # 分析数据集结构
        sample = analyze_dataset_structure()
        
        # 分析蛋白质-配体映射
        protein_to_ligands, ligand_to_proteins = analyze_protein_ligand_mapping()
        
        # 让用户选择要分析的蛋白质
        print(f"\n" + "=" * 60)
        print("交互式分析")
        print("=" * 60)
        
        # 显示一些可用的蛋白质
        available_proteins = list(protein_to_ligands.keys())[:10]
        print("可用的蛋白质ID (前10个):")
        for i, protein in enumerate(available_proteins):
            print(f"  {i+1}. {protein} ({len(protein_to_ligands[protein])} 配体)")
        
        # 分析第一个蛋白质作为示例
        if available_proteins:
            example_protein = available_proteins[0]
            analyze_specific_protein(example_protein, protein_to_ligands)
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
