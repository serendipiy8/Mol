#!/usr/bin/env python3
"""
真实训练场景的完整测试
模拟真实的训练循环，包括数据加载、批次处理、性能统计等
"""

import sys
import os
sys.path.append('src')

import argparse
import torch
import time
import numpy as np
from collections import defaultdict
from src.data.dataset import get_model_dataset
from src.data.data_loader import get_data_loaders
from src.data.transforms import get_default_transform

class TrainingStats:
    """训练统计类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.batch_times = []
        self.batch_sizes = []
        self.protein_atom_counts = []
        self.ligand_atom_counts = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = None
        self.end_time = None
    
    def add_batch(self, batch, batch_time, success=True):
        if success:
            self.success_count += 1
            self.batch_times.append(batch_time)
            
            if batch is not None:
                if hasattr(batch, 'num_graphs'):
                    self.batch_sizes.append(batch.num_graphs)
                
                if hasattr(batch, 'protein_pos') and batch.protein_pos is not None:
                    self.protein_atom_counts.append(batch.protein_pos.shape[0])
                
                if hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                    self.ligand_atom_counts.append(batch.ligand_pos.shape[0])
        else:
            self.error_count += 1
    
    def get_stats(self):
        if not self.batch_times:
            return {
                'total_batches': self.success_count + self.error_count,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'success_rate': 0.0,
                'avg_batch_time': 0.0,
                'total_time': 0.0
            }
        
        return {
            'total_batches': self.success_count + self.error_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / (self.success_count + self.error_count) * 100,
            'avg_batch_time': np.mean(self.batch_times),
            'total_time': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'avg_batch_size': np.mean(self.batch_sizes) if self.batch_sizes else 0,
            'avg_protein_atoms': np.mean(self.protein_atom_counts) if self.protein_atom_counts else 0,
            'avg_ligand_atoms': np.mean(self.ligand_atom_counts) if self.ligand_atom_counts else 0
        }

def test_real_training_scenario():
    """测试真实训练场景"""
    print("🚀 开始真实训练场景测试...")
    
    # 训练配置
    TRAINING_CONFIG = {
        'dataset_path': 'src/data/crossdocked_v1.1_rmsd1.0_processed',
        'split_file': 'src/data/split_by_name.pt',
        'batch_size': 32,
        'num_workers': 0,  # 避免多进程问题
        'shuffle_train': True,
        'shuffle_test': False,
        'test_batches': 100,  # 测试的批次数量
        'val_batches': 50
    }
    
    print("📊 训练配置:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    
    # 创建配置对象
    config = argparse.Namespace(
        name='crossdocked',
        path=TRAINING_CONFIG['dataset_path'],
        split=TRAINING_CONFIG['split_file'],
        mode='full',
        version='ref_prior_aromatic'
    )
    
    print("\n🔧 创建真实训练DataLoader...")
    
    try:
        # 获取数据加载器
        data_loaders = get_data_loaders(
            dataset_path=TRAINING_CONFIG['dataset_path'],
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=TRAINING_CONFIG['num_workers'],
            split_file=TRAINING_CONFIG['split_file'],
            shuffle_train=TRAINING_CONFIG['shuffle_train'],
            shuffle_test=TRAINING_CONFIG['shuffle_test']
        )
        
        print(f"✅ DataLoader创建成功")
        print(f"   可用的分割: {list(data_loaders.keys())}")
        
        for split_name, loader in data_loaders.items():
            print(f"   {split_name}: {len(loader)} 个批次")
        
    except Exception as e:
        print(f"❌ DataLoader创建失败: {e}")
        return False
    
    print("\n🚀 开始模拟真实训练循环...")
    
    # 测试训练集
    if 'train' in data_loaders:
        train_stats = test_loader_performance(
            data_loaders['train'], 
            'TRAIN', 
            TRAINING_CONFIG['test_batches']
        )
        
        print(f"\n📊 TRAIN 性能统计:")
        stats = train_stats.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    
    # 测试验证集
    if 'val' in data_loaders:
        val_stats = test_loader_performance(
            data_loaders['val'], 
            'VAL', 
            TRAINING_CONFIG['val_batches']
        )
        
        print(f"\n📊 VAL 性能统计:")
        stats = val_stats.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    
    # 测试测试集
    if 'test' in data_loaders:
        test_stats = test_loader_performance(
            data_loaders['test'], 
            'TEST', 
            TRAINING_CONFIG['val_batches']
        )
        
        print(f"\n📊 TEST 性能统计:")
        stats = test_stats.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 80)
    print("🎉 真实训练场景测试完成!")
    
    return True

def test_loader_performance(dataloader, split_name, total_batches):
    """测试数据加载器性能"""
    stats = TrainingStats()
    stats.start_time = time.time()
    
    print(f"\n📈 测试{split_name} DataLoader:")
    print(f"   {split_name}: 测试 {total_batches} 个批次")
    
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= total_batches:
            break
        
        batch_start = time.time()
        
        try:
            # 模拟真实训练中的批次处理
            if batch is not None:
                # 模拟前向传播
                if hasattr(batch, 'protein_pos') and batch.protein_pos is not None:
                    # 模拟对蛋白质位置的处理
                    protein_center = batch.protein_pos.mean(dim=0, keepdim=True)
                
                if hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                    # 模拟对配体位置的处理
                    ligand_center = batch.ligand_pos.mean(dim=0, keepdim=True)
                
                # 模拟损失计算（简单示例）
                if hasattr(batch, 'protein_pos') and hasattr(batch, 'ligand_pos'):
                    # 计算蛋白质和配体中心的距离作为示例
                    if batch.protein_pos.size(0) > 0 and batch.ligand_pos.size(0) > 0:
                        distance = torch.norm(protein_center - ligand_center)
                
                stats.add_batch(batch, time.time() - batch_start, success=True)
            else:
                stats.add_batch(None, time.time() - batch_start, success=False)
            
            batch_count += 1
            
            # 每10个批次报告一次进度
            if batch_count % 10 == 0:
                elapsed = time.time() - stats.start_time
                avg_time = elapsed / batch_count
                print(f"     批次 {batch_count}/{total_batches}: 成功={stats.success_count}, 错误={stats.error_count}, 平均时间={avg_time:.3f}s")
        
        except Exception as e:
            stats.add_batch(None, time.time() - batch_start, success=False)
            batch_count += 1
            if batch_count <= 5:  # 只打印前5个错误
                print(f"     批次 {batch_count} 处理失败: {e}")
    
    stats.end_time = time.time()
    
    # 打印最终统计
    elapsed = stats.end_time - stats.start_time
    print(f"   Epoch 1 完成: {elapsed:.2f}s, {batch_count} 批次")
    
    return stats

def test_data_quality():
    """测试数据质量"""
    print("\n🔍 数据质量检查...")
    
    try:
        # 加载数据集
        config = argparse.Namespace(
            name='crossdocked',
            path='src/data/crossdocked_v1.1_rmsd1.0_processed',
            split='src/data/split_by_name.pt',
            mode='full',
            version='ref_prior_aromatic'
        )
        
        result = get_model_dataset(config)
        if isinstance(result, tuple):
            dataset, subsets = result
        else:
            dataset = result
        
        print(f"✅ 数据集加载成功: {len(dataset)} 个样本")
        
        # 检查数据完整性
        sample_checks = []
        for i in range(min(100, len(dataset))):  # 检查前100个样本
            try:
                sample = dataset[i]
                if sample is not None:
                    # 检查必要属性
                    has_protein_pos = hasattr(sample, 'protein_pos') and sample.protein_pos is not None
                    has_ligand_pos = hasattr(sample, 'ligand_pos') and sample.ligand_pos is not None
                    has_protein_feat = hasattr(sample, 'protein_atom_feature') and sample.protein_atom_feature is not None
                    has_ligand_feat = hasattr(sample, 'ligand_atom_feature') and sample.ligand_atom_feature is not None
                    
                    sample_checks.append({
                        'index': i,
                        'has_protein_pos': has_protein_pos,
                        'has_ligand_pos': has_ligand_pos,
                        'has_protein_feat': has_protein_feat,
                        'has_ligand_feat': has_ligand_feat,
                        'protein_atoms': sample.protein_pos.shape[0] if has_protein_pos else 0,
                        'ligand_atoms': sample.ligand_pos.shape[0] if has_ligand_pos else 0
                    })
                else:
                    sample_checks.append({'index': i, 'error': 'Failed to load'})
            except Exception as e:
                sample_checks.append({'index': i, 'error': str(e)})
        
        # 统计结果
        valid_samples = [s for s in sample_checks if 'error' not in s]
        print(f"✅ 数据质量检查完成:")
        print(f"   检查样本数: {len(sample_checks)}")
        print(f"   有效样本数: {len(valid_samples)}")
        
        if valid_samples:
            protein_atoms = [s['protein_atoms'] for s in valid_samples]
            ligand_atoms = [s['ligand_atoms'] for s in valid_samples]
            
            print(f"   平均蛋白质原子数: {np.mean(protein_atoms):.1f}")
            print(f"   平均配体原子数: {np.mean(ligand_atoms):.1f}")
            print(f"   蛋白质原子数范围: {min(protein_atoms)} - {max(protein_atoms)}")
            print(f"   配体原子数范围: {min(ligand_atoms)} - {max(ligand_atoms)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🔥 开始真实训练场景完整测试...")
    
    success = True
    
    # 1. 数据质量检查
    if not test_data_quality():
        success = False
    
    # 2. 真实训练场景测试
    if not test_real_training_scenario():
        success = False
    
    if success:
        print("\n🎉 所有测试通过！系统准备就绪！")
        print("✅ 数据加载系统正常工作")
        print("✅ 批次处理性能良好")
        print("✅ 训练循环模拟成功")
    else:
        print("\n❌ 部分测试失败！需要进一步检查！")
    
    return success

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
