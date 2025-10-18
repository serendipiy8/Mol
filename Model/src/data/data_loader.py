import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.data import Batch
import argparse
from .dataset import get_model_dataset
from .transforms import get_default_transform


def _default_collate_fn(batch):
    """默认的collate函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
        return Batch.from_data_list(batch)
    except Exception as e:
        print(f"Batch collation failed: {e}")
        return batch[0] if batch else None


def collate_fn_protein_ligand(batch):
    """蛋白质-配体数据的collate函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
        return Batch.from_data_list(batch)
    except Exception as e:
        print(f"Protein-ligand batch collation failed: {e}")
        return batch[0] if batch else None


def get_data_loaders(dataset_path, split_file=None, batch_size=32, num_workers=0, 
                    shuffle_train=True, shuffle_test=False, collate_fn=None):
    """
    获取训练和测试数据加载器
    
    Args:
        dataset_path: 数据集路径
        split_file: 分割文件路径
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle_train: 是否打乱训练数据
        shuffle_test: 是否打乱测试数据
        collate_fn: collate函数
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 创建配置对象
    config = argparse.Namespace()
    config.path = dataset_path
    config.split_file = split_file
    
    # 获取数据集
    result = get_model_dataset(config)
    if isinstance(result, tuple):
        dataset, (train_subset, test_subset) = result
    else:
        dataset = result
        train_subset = test_subset = None
    
    # 设置collate函数
    if collate_fn is None:
        collate_fn = collate_fn_protein_ligand
    
    # 创建数据加载器
    train_loader = None
    test_loader = None
    
    if train_subset is not None:
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    if test_subset is not None:
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, test_loader


def get_balanced_data_loaders(dataset_path, split_file=None, batch_size=32, 
                            num_workers=0, collate_fn=None):
    """
    获取平衡的数据加载器（使用SubsetRandomSampler）
    
    Args:
        dataset_path: 数据集路径
        split_file: 分割文件路径
        batch_size: 批次大小
        num_workers: 工作进程数
        collate_fn: collate函数
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    # 创建配置对象
    config = argparse.Namespace()
    config.path = dataset_path
    config.split_file = split_file
    
    # 获取数据集
    result = get_model_dataset(config)
    if isinstance(result, tuple):
        dataset, (train_subset, test_subset) = result
    else:
        dataset = result
        train_subset = test_subset = None
    
    # 设置collate函数
    if collate_fn is None:
        collate_fn = collate_fn_protein_ligand
    
    # 创建数据加载器
    train_loader = None
    test_loader = None
    
    if train_subset is not None:
        train_sampler = SubsetRandomSampler(train_subset.indices)
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    if test_subset is not None:
        test_sampler = SubsetRandomSampler(test_subset.indices)
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, test_loader


def get_transform_configs_for_splits(config):
    """为不同分割获取变换配置"""
    train_config = argparse.Namespace()
    test_config = argparse.Namespace()
    
    # 复制原始配置
    for attr in dir(config):
        if not attr.startswith('_'):
            setattr(train_config, attr, getattr(config, attr))
            setattr(test_config, attr, getattr(config, attr))
    
    # 训练集使用默认变换
    train_config.transform = get_default_transform()
    
    # 测试集不使用变换或使用更简单的变换
    test_config.transform = None
    
    return train_config, test_config


def get_data_loaders_with_split_transforms(config, batch_size=32, num_workers=0):
    """
    获取带有分割特定变换的数据加载器
    
    Args:
        config: 配置对象
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    train_config, test_config = get_transform_configs_for_splits(config)
    
    train_loader, _ = get_data_loaders(
        train_config.path,
        train_config.split_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        shuffle_test=False
    )
    
    _, test_loader = get_data_loaders(
        test_config.path,
        test_config.split_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=False,
        shuffle_test=False
    )
    
    return train_loader, test_loader


def get_proper_data_loaders(config, batch_size=32, num_workers=0):
    """
    获取合适的数据加载器（推荐使用）
    
    Args:
        config: 配置对象
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    return get_data_loaders(
        config.path,
        getattr(config, 'split_file', None),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        shuffle_test=False
    )