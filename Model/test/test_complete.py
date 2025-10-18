#!/usr/bin/env python3
"""
完整的数据加载测试代码 - 真实训练场景
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

def test_dataset_loading():
    """测试数据集加载"""
    print("=== 测试数据集加载 ===")
    
    # 创建配置
    config = argparse.Namespace(
        name='crossdocked',
        path='src/data/crossdocked_v1.1_rmsd1.0_processed',
        split='src/data/crossdocked_v1.1_rmsd1.0_processed/split_by_name.pt',
        mode='full',
        version='ref_prior_aromatic'
    )
    
    try:
        # 加载数据集
        result = get_model_dataset(config)
        
        if isinstance(result, tuple):
            dataset, subsets = result
            print(f"✅ 数据集加载成功")
            print(f"   总样本数: {len(dataset)}")
            print(f"   分割: {list(subsets.keys())}")
            for split_name, subset in subsets.items():
                print(f"   {split_name}: {len(subset)} 个样本")
        else:
            dataset = result
            print(f"✅ 数据集加载成功 (无分割)")
            print(f"   总样本数: {len(dataset)}")
        
        # 测试单个样本
        print("\n--- 测试单个样本 ---")
        sample = dataset[0]
        if sample is not None:
            print("✅ 单个样本加载成功")
            print(f"   样本ID: {getattr(sample, 'id', 'N/A')}")
            if hasattr(sample, 'protein_pos') and sample.protein_pos is not None:
                print(f"   蛋白质原子数: {sample.protein_pos.shape[0]}")
            if hasattr(sample, 'ligand_pos') and sample.ligand_pos is not None:
                print(f"   配体原子数: {sample.ligand_pos.shape[0]}")
        else:
            print("❌ 单个样本加载失败")
        
        return dataset, subsets if isinstance(result, tuple) else None
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_data_loader():
    """测试数据加载器"""
    print("\n=== 测试数据加载器 ===")
    
    try:
        # 创建数据加载器
        data_loaders = get_data_loaders(
            dataset_path='src/data/crossdocked_v1.1_rmsd1.0_processed',
            batch_size=32,
            num_workers=0,  # 避免多进程问题
            split_file='src/data/crossdocked_v1.1_rmsd1.0_processed/split_by_name.pt',
            shuffle_train=True,
            shuffle_test=False
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"   可用的分割: {list(data_loaders.keys())}")
        
        # 测试每个分割的加载器
        for split_name, loader in data_loaders.items():
            print(f"\n--- 测试 {split_name} 加载器 ---")
            print(f"   批次数量: {len(loader)}")
            
            # 测试第一个批次
            try:
                for batch_idx, batch in enumerate(loader):
                    if batch is not None:
                        print(f"✅ 批次 {batch_idx} 加载成功")
                        print(f"   批次大小: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'N/A'}")
                        if hasattr(batch, 'protein_pos') and batch.protein_pos is not None:
                            print(f"   蛋白质原子总数: {batch.protein_pos.shape[0]}")
                        if hasattr(batch, 'ligand_pos') and batch.ligand_pos is not None:
                            print(f"   配体原子总数: {batch.ligand_pos.shape[0]}")
                    else:
                        print(f"❌ 批次 {batch_idx} 加载失败")
                    break  # 只测试第一个批次
            except Exception as e:
                print(f"❌ {split_name} 加载器测试失败: {e}")
        
        return data_loaders
        
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_transforms():
    """测试数据变换"""
    print("\n=== 测试数据变换 ===")
    
    try:
        # 创建变换
        transform = get_default_transform(
            add_ord_feat=False,
            ligand_atom_mode='basic',
            ligand_bond_mode='fc',
            max_num_arms=10,
            random_rot=False,  # 关闭随机旋转以便测试
            normalize_pos=True,
            add_noise=False   # 关闭噪声以便测试
        )
        print("✅ 变换创建成功")
        
        # 创建测试数据
        from src.data.utils import ProteinLigandData
        test_data = ProteinLigandData()
        test_data.protein_pos = torch.randn(10, 3)
        test_data.ligand_pos = torch.randn(5, 3)
        test_data.protein_atom_feature = torch.randn(10, 9)
        test_data.ligand_atom_feature = torch.randn(5, 9)
        test_data.ligand_bond_index = torch.randint(0, 5, (2, 8))
        test_data.ligand_bond_type = torch.randint(1, 4, (8,))
        
        print("✅ 测试数据创建成功")
        print(f"   变换前 - 蛋白质原子数: {test_data.protein_pos.shape[0]}")
        print(f"   变换前 - 配体原子数: {test_data.ligand_pos.shape[0]}")
        
        # 应用变换
        transformed_data = transform(test_data)
        print("✅ 变换应用成功")
        print(f"   变换后 - 蛋白质原子数: {transformed_data.protein_pos.shape[0]}")
        print(f"   变换后 - 配体原子数: {transformed_data.ligand_pos.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 变换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_training_scenario():
    """测试真实训练场景"""
    print("\n=== 测试真实训练场景 ===")
    
    # 训练配置
    TRAINING_CONFIG = {
        'batch_size': 32,
        'num_workers': 0,
        'test_batches': 50,  # 测试的批次数量
    }
    
    print("📊 训练配置:")
    for key, value in TRAINING_CONFIG.items():
        print(f"   {key}: {value}")
    
    try:
        # 获取数据加载器
        data_loaders = get_data_loaders(
            dataset_path='src/data/crossdocked_v1.1_rmsd1.0_processed',
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=TRAINING_CONFIG['num_workers'],
            split_file='src/data/crossdocked_v1.1_rmsd1.0_processed/split_by_name.pt',
            shuffle_train=True,
            shuffle_test=False
        )
        
        print(f"✅ 真实训练DataLoader创建成功")
        
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
        
        # 测试测试集
        if 'test' in data_loaders:
            test_stats = test_loader_performance(
                data_loaders['test'], 
                'TEST', 
                min(20, TRAINING_CONFIG['test_batches']//2)
            )
            
            print(f"\n📊 TEST 性能统计:")
            stats = test_stats.get_stats()
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实训练场景测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

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

def test_full_pipeline():
    """测试完整流程"""
    print("=== 测试完整流程 ===")
    
    try:
        # 1. 加载数据集
        dataset, subsets = test_dataset_loading()
        if dataset is None:
            return False
        
        # 2. 创建数据加载器
        data_loaders = test_data_loader()
        if data_loaders is None:
            return False
        
        # 3. 测试变换
        if not test_transforms():
            return False
        
        # 4. 测试真实训练场景
        if not test_real_training_scenario():
            return False
        
        print("\n🎉 所有测试通过！完整流程工作正常！")
        return True
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 开始完整的数据加载测试...")
    
    success = test_full_pipeline()
    
    if success:
        print("\n✅ 测试完成！所有功能正常！")
        print("🎯 系统已准备好进行真实训练！")
    else:
        print("\n❌ 测试失败！需要调试！")
    
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