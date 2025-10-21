import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.utils.data as tud
from torch.utils.data import SubsetRandomSampler, Subset
from torch_geometric.data import Batch
import argparse
from .dataset import get_model_dataset
from .transforms import get_default_transform
from .utils import ProteinLigandDataLoader


def _default_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
        return Batch.from_data_list(batch)
    except Exception as e:
        print(f"Batch collation failed: {e}")
        return batch[0] if batch else None


def collate_fn_protein_ligand(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
        return Batch.from_data_list(batch)
    except Exception as e:
        print(f"Protein-ligand batch collation failed: {e}")
        return batch[0] if batch else None


def _filter_indices_by_predicate(dataset, indices, predicate_fn):
    """Return filtered indices that satisfy predicate(dataset[i])."""
    filtered = []
    for idx in indices:
        sample = dataset[idx]
        try:
            if predicate_fn(sample):
                filtered.append(idx)
        except Exception:
            continue
    return filtered


def worker_init_reconnect_db(worker_id: int):
    """Top-level worker init: reconnect LMDB inside each worker (Windows-safe)."""
    try:
        info = tud.get_worker_info()
    except Exception:
        info = None
    if info is None:
        return
    ds = info.dataset
    try:
        while isinstance(ds, Subset):
            ds = ds.dataset
    except Exception:
        pass
    try:
        if hasattr(ds, '_connect_db'):
            ds._connect_db()
    except Exception:
        pass


def get_data_loaders(dataset_path, split_file=None, batch_size=32, num_workers=0, 
                    shuffle_train=True, shuffle_test=False, collate_fn=None,
                    require_ligand: bool = False, require_feat: bool = False,
                    require_protein: bool = False):
    """
    Get training and test data loaders
    """

    config = argparse.Namespace()
    config.path = dataset_path
    config.split_file = split_file
    
    result = get_model_dataset(config)
    if isinstance(result, tuple):
        dataset, (train_subset, test_subset) = result
    else:
        dataset = result
        train_subset = test_subset = None
    
    train_loader = None
    test_loader = None
    
    # (kept for compatibility, no longer used)
    def _resolve_base_dataset(ds):
        return ds.dataset if isinstance(ds, Subset) else ds

    # Optional: filter subsets by presence of ligand_pos(/features)
    def _pred_ok(sample):
        if sample is None:
            return False
        ok = True
        if require_protein:
            ok = ok and hasattr(sample, 'protein_pos') and sample.protein_pos is not None and sample.protein_pos.numel() > 0
        if require_ligand:
            ok = ok and hasattr(sample, 'ligand_pos') and sample.ligand_pos is not None and sample.ligand_pos.numel() > 0
        if require_feat:
            ok = ok and hasattr(sample, 'ligand_atom_feature') and sample.ligand_atom_feature is not None and sample.ligand_atom_feature.numel() > 0
        return ok

    if train_subset is not None:
        if hasattr(train_subset, 'indices') and len(train_subset.indices) == 0:
            train_loader = None
        else:
            train_loader = ProteinLigandDataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers and num_workers > 0 else False,
                prefetch_factor=2 if num_workers and num_workers > 0 else None,
                worker_init_fn=worker_init_reconnect_db,
                collate_fn=collate_fn_protein_ligand
            )
    
    if test_subset is not None:
        if hasattr(test_subset, 'indices') and len(test_subset.indices) == 0:
            test_loader = None
        else:
            test_loader = ProteinLigandDataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=shuffle_test,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True if num_workers and num_workers > 0 else False,
                prefetch_factor=2 if num_workers and num_workers > 0 else None,
                worker_init_fn=worker_init_reconnect_db,
                collate_fn=collate_fn_protein_ligand
            )
    
    return train_loader, test_loader


def get_balanced_data_loaders(dataset_path, split_file=None, batch_size=32, 
                            num_workers=0, collate_fn=None):
    # Create config object
    config = argparse.Namespace()
    config.path = dataset_path
    config.split_file = split_file
    
    # Get dataset
    result = get_model_dataset(config)
    if isinstance(result, tuple):
        dataset, (train_subset, test_subset) = result
    else:
        dataset = result
        train_subset = test_subset = None
    
    # Create data loaders (use PyG ProteinLigandDataLoader)
    train_loader = None
    test_loader = None
    
    if train_subset is not None:
        train_sampler = SubsetRandomSampler(train_subset.indices)
        train_loader = ProteinLigandDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    if test_subset is not None:
        test_sampler = SubsetRandomSampler(test_subset.indices)
        test_loader = ProteinLigandDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, test_loader


def get_transform_configs_for_splits(config):
    """Get transform configurations for different splits"""
    train_config = argparse.Namespace()
    test_config = argparse.Namespace()
    
    # Copy original config
    for attr in dir(config):
        if not attr.startswith('_'):
            setattr(train_config, attr, getattr(config, attr))
            setattr(test_config, attr, getattr(config, attr))
    
    # Training set uses default transform
    train_config.transform = get_default_transform()
    
    # Test set uses no transform or simpler transform
    test_config.transform = None
    
    return train_config, test_config


def get_data_loaders_with_split_transforms(config, batch_size=32, num_workers=0):
    """
    Get data loaders with split-specific transforms
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
    Get proper data loaders (recommended)
    """
    return get_data_loaders(
        config.path,
        getattr(config, 'split_file', None),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        shuffle_test=False
    )