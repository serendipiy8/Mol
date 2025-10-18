import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import random
from torch_geometric.data import Data
from .utils import ProteinLigandData

MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

BOND_TYPES = [1, 2, 3, 4]

def get_atomic_number_from_index(index: int) -> int:
    reverse_map = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
    return reverse_map.get(index, 6)

def get_atom_type_index(atomic_num: int, hybridization: str, is_aromatic: bool) -> int:
    key = (atomic_num, hybridization, is_aromatic)
    return MAP_ATOM_TYPE_FULL_TO_INDEX.get(key, 1)

class AddOrdinalFeatures:
    
    def __init__(self):
        pass
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if hasattr(data, 'protein_atom_feature') and data.protein_atom_feature is not None:
            num_protein_atoms = data.protein_atom_feature.shape[0]
            ordinal_features = torch.arange(num_protein_atoms, dtype=torch.float).unsqueeze(1)
            data.protein_atom_feature = torch.cat([data.protein_atom_feature, ordinal_features], dim=1)
        
        if hasattr(data, 'ligand_atom_feature') and data.ligand_atom_feature is not None:
            num_ligand_atoms = data.ligand_atom_feature.shape[0]
            ordinal_features = torch.arange(num_ligand_atoms, dtype=torch.float).unsqueeze(1)
            data.ligand_atom_feature = torch.cat([data.ligand_atom_feature, ordinal_features], dim=1)
        
        return data

class LigandAtomModeTransform:
    
    def __init__(self, mode: str = 'full'):
        self.mode = mode
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if hasattr(data, 'ligand_atom_feature') and data.ligand_atom_feature is not None:
            if self.mode == 'basic':
                data.ligand_atom_feature = data.ligand_atom_feature[:, :1]
            elif self.mode == 'full':
                pass
            else:
                raise ValueError(f"Unknown ligand atom mode: {self.mode}")
        
        return data

class LigandBondModeTransform:
    
    def __init__(self, mode: str = 'fc'):
        self.mode = mode
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if hasattr(data, 'ligand_bond_type') and data.ligand_bond_type is not None:
            if self.mode == 'fc':
                data.ligand_bond_type = torch.ones_like(data.ligand_bond_type)
            elif self.mode == 'bond_type':
                pass
            else:
                raise ValueError(f"Unknown ligand bond mode: {self.mode}")
        
        return data

class AddDecompositionIndicators:
    
    def __init__(self, max_num_arms: int = 10):
        self.max_num_arms = max_num_arms
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if hasattr(data, 'ligand_atom_feature') and data.ligand_atom_feature is not None:
            num_ligand_atoms = data.ligand_atom_feature.shape[0]
            
            arm_indicators = torch.zeros(num_ligand_atoms, self.max_num_arms)
            scaffold_indicator = torch.zeros(num_ligand_atoms, 1)
            
            data.ligand_atom_feature = torch.cat([
                data.ligand_atom_feature,
                arm_indicators,
                scaffold_indicator
            ], dim=1)
            
            data.max_num_arms = self.max_num_arms
            
        return data

class RandomRotation:
    
    def __init__(self, apply: bool = True):
        self.apply = apply
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if not self.apply:
            return data
        
        if hasattr(data, 'ligand_pos') and data.ligand_pos is not None:
            rotation_matrix = self._random_rotation_matrix()
            data.ligand_pos = torch.matmul(data.ligand_pos, rotation_matrix.T)
        
        return data
    
    def _random_rotation_matrix(self):
        alpha = random.uniform(0, 2 * np.pi)
        beta = random.uniform(0, 2 * np.pi)
        gamma = random.uniform(0, 2 * np.pi)
        
        Rz1 = torch.tensor([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ], dtype=torch.float)
        
        Ry = torch.tensor([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ], dtype=torch.float)
        
        Rz2 = torch.tensor([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ], dtype=torch.float)
        
        rotation_matrix = torch.matmul(torch.matmul(Rz2, Ry), Rz1)
        return rotation_matrix

class NormalizePositions:
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if hasattr(data, 'protein_pos') and data.protein_pos is not None:
            data.protein_pos = data.protein_pos - data.protein_pos.mean(dim=0, keepdim=True)
        
        if hasattr(data, 'ligand_pos') and data.ligand_pos is not None:
            data.ligand_pos = data.ligand_pos - data.ligand_pos.mean(dim=0, keepdim=True)
        
        return data

class AddNoise:
    
    def __init__(self, noise_std: float = 0.1):
        self.noise_std = noise_std
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        if hasattr(data, 'protein_pos') and data.protein_pos is not None:
            noise = torch.randn_like(data.protein_pos) * self.noise_std
            data.protein_pos = data.protein_pos + noise
        
        if hasattr(data, 'ligand_pos') and data.ligand_pos is not None:
            noise = torch.randn_like(data.ligand_pos) * self.noise_std
            data.ligand_pos = data.ligand_pos + noise
        
        return data

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data: ProteinLigandData) -> ProteinLigandData:
        for transform in self.transforms:
            data = transform(data)
        return data

def get_ligand_transform(
    add_ord_feat: bool = False,
    ligand_atom_mode: str = 'basic',
    ligand_bond_mode: str = 'fc',
    max_num_arms: int = 10,
    random_rot: bool = False,
    normalize_pos: bool = True,
    add_noise: bool = False,
    noise_std: float = 0.1
) -> Compose:
    transforms = []
    
    if add_ord_feat:
        transforms.append(AddOrdinalFeatures())
    
    transforms.append(LigandAtomModeTransform(mode=ligand_atom_mode))
    transforms.append(LigandBondModeTransform(mode=ligand_bond_mode))
    transforms.append(AddDecompositionIndicators(max_num_arms=max_num_arms))
    
    if random_rot:
        transforms.append(RandomRotation(apply=True))
    
    if normalize_pos:
        transforms.append(NormalizePositions())
    
    if add_noise:
        transforms.append(AddNoise(noise_std=noise_std))
    
    return Compose(transforms)

def get_protein_transform(
    normalize_pos: bool = True,
    add_noise: bool = False,
    noise_std: float = 0.1
) -> Compose:
    transforms = []
    
    if normalize_pos:
        transforms.append(NormalizePositions())
    
    if add_noise:
        transforms.append(AddNoise(noise_std=noise_std))
    
    return Compose(transforms)

def get_default_transform(
    add_ord_feat: bool = False,
    ligand_atom_mode: str = 'basic',
    ligand_bond_mode: str = 'fc',
    max_num_arms: int = 10,
    random_rot: bool = False,
    normalize_pos: bool = True,
    add_noise: bool = False,
    noise_std: float = 0.1
) -> Compose:
    transforms = []
    
    if add_ord_feat:
        transforms.append(AddOrdinalFeatures())
    
    transforms.append(LigandAtomModeTransform(mode=ligand_atom_mode))
    transforms.append(LigandBondModeTransform(mode=ligand_bond_mode))
    transforms.append(AddDecompositionIndicators(max_num_arms=max_num_arms))
    
    if random_rot:
        transforms.append(RandomRotation(apply=True))
    
    if normalize_pos:
        transforms.append(NormalizePositions())
    
    if add_noise:
        transforms.append(AddNoise(noise_std=noise_std))
    
    return Compose(transforms)