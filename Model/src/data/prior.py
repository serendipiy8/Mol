# Copyright 2023 Model Project
# Prior computation for Model project

import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist


def compute_golden_prior_from_data(data):
    """
    Compute golden prior information from data
    """
    # Compute ligand atom prior masks
    if hasattr(data, 'ligand_atom_mask') and hasattr(data, 'ligand_pos'):
        ligand_prior_masks = compute_ligand_prior_masks(data)
        data.ligand_prior_masks = ligand_prior_masks
    
    # Compute protein pocket prior masks
    if hasattr(data, 'pocket_atom_masks') and hasattr(data, 'protein_pos'):
        pocket_prior_masks = compute_pocket_prior_masks(data)
        data.pocket_prior_masks = pocket_prior_masks
    
    # Compute decomposition center prior information
    if hasattr(data, 'ligand_pos') and hasattr(data, 'ligand_atom_mask'):
        decomp_centers = compute_decomp_centers(data)
        data.ligand_decomp_centers = decomp_centers
    
    return data


def compute_ligand_prior_masks(data):
    """
    Compute ligand atom prior masks
    """
    num_atoms = data.ligand_pos.size(0)
    num_arms = data.num_arms
    
    # Initialize prior masks
    prior_masks = torch.zeros(num_atoms, num_arms + 1, dtype=torch.float32)
    
    # Assign prior probabilities for each arm
    for arm_idx in range(num_arms):
        arm_atoms = (data.ligand_atom_mask == arm_idx)
        if arm_atoms.any():
            # Prior probability for arm atoms
            prior_masks[arm_atoms, arm_idx] = 1.0
    
    # Prior probability for scaffold atoms
    scaffold_atoms = (data.ligand_atom_mask == -1)
    if scaffold_atoms.any():
        prior_masks[scaffold_atoms, num_arms] = 1.0
    
    return prior_masks


def compute_pocket_prior_masks(data):
    """
    Compute protein pocket prior masks
    """
    if not hasattr(data, 'pocket_atom_masks'):
        return None
    
    num_protein_atoms = data.protein_pos.size(0)
    num_arms = data.num_arms
    
    # Initialize pocket prior masks
    pocket_prior_masks = torch.zeros(num_protein_atoms, num_arms + 1, dtype=torch.float32)
    
    # Assign prior probabilities for each arm corresponding pocket
    for arm_idx in range(num_arms):
        if arm_idx < data.pocket_atom_masks.size(0):
            pocket_mask = data.pocket_atom_masks[arm_idx]
            pocket_prior_masks[pocket_mask, arm_idx] = 1.0
    
    # Prior probability for non-pocket atoms (marked as scaffold)
    non_pocket_atoms = ~data.pocket_atom_masks.any(0)
    if non_pocket_atoms.any():
        pocket_prior_masks[non_pocket_atoms, num_arms] = 1.0
    
    return pocket_prior_masks


def compute_decomp_centers(data):
    """
    Compute decomposition center points
    """
    if not hasattr(data, 'ligand_atom_mask'):
        return None
    
    num_arms = data.num_arms
    decomp_centers = []
    
    # Compute center of mass for each arm
    for arm_idx in range(num_arms):
        arm_atoms = (data.ligand_atom_mask == arm_idx)
        if arm_atoms.any():
            arm_positions = data.ligand_pos[arm_atoms]
            center = arm_positions.mean(dim=0)
            decomp_centers.append(center)
        else:
            # If no atoms, use zero vector
            decomp_centers.append(torch.zeros(3))
    
    # Compute scaffold center of mass
    scaffold_atoms = (data.ligand_atom_mask == -1)
    if scaffold_atoms.any():
        scaffold_positions = data.ligand_pos[scaffold_atoms]
        scaffold_center = scaffold_positions.mean(dim=0)
        decomp_centers.append(scaffold_center)
    else:
        decomp_centers.append(torch.zeros(3))
    
    return torch.stack(decomp_centers)


def compute_distance_prior(data, max_distance=10.0):
    """
    Compute prior information based on distance
    """
    if not (hasattr(data, 'protein_pos') and hasattr(data, 'ligand_pos')):
        return None
    
    protein_pos = data.protein_pos.numpy()
    ligand_pos = data.ligand_pos.numpy()
    
    # Compute protein-ligand distance matrix
    distances = cdist(protein_pos, ligand_pos)
    
    # Distance-based prior probability (closer distance, higher probability)
    distance_prior = torch.exp(-distances / max_distance)
    
    return distance_prior


def compute_contact_prior(data, contact_cutoff=4.0):
    """
    Compute prior information based on contact
    """
    if not (hasattr(data, 'protein_pos') and hasattr(data, 'ligand_pos')):
        return None
    
    protein_pos = data.protein_pos
    ligand_pos = data.ligand_pos
    
    # Compute distances
    protein_expanded = protein_pos.unsqueeze(1)  # [N_protein, 1, 3]
    ligand_expanded = ligand_pos.unsqueeze(0)    # [1, N_ligand, 3]
    distances = torch.norm(protein_expanded - ligand_expanded, dim=2)
    
    # Contact mask
    contact_mask = (distances < contact_cutoff).float()
    
    return contact_mask


def compute_geometric_prior(data):
    """
    Compute geometric prior information
    """
    prior_info = {}
    
    if hasattr(data, 'ligand_pos') and hasattr(data, 'ligand_atom_mask'):
        # Compute geometric center for each arm
        num_arms = data.num_arms
        arm_centers = []
        
        for arm_idx in range(num_arms):
            arm_atoms = (data.ligand_atom_mask == arm_idx)
            if arm_atoms.any():
                arm_pos = data.ligand_pos[arm_atoms]
                center = arm_pos.mean(dim=0)
                arm_centers.append(center)
            else:
                arm_centers.append(torch.zeros(3))
        
        prior_info['arm_centers'] = torch.stack(arm_centers)
        
        # Compute scaffold center
        scaffold_atoms = (data.ligand_atom_mask == -1)
        if scaffold_atoms.any():
            scaffold_pos = data.ligand_pos[scaffold_atoms]
            scaffold_center = scaffold_pos.mean(dim=0)
            prior_info['scaffold_center'] = scaffold_center
    
    return prior_info


def compute_chemical_prior(data):
    """
    Compute chemical prior information
    """
    prior_info = {}
    
    if hasattr(data, 'ligand_element') and hasattr(data, 'ligand_atom_mask'):
        # Statistics of atom type distribution for each arm
        num_arms = data.num_arms
        arm_atom_types = []
        
        for arm_idx in range(num_arms):
            arm_atoms = (data.ligand_atom_mask == arm_idx)
            if arm_atoms.any():
                arm_elements = data.ligand_element[arm_atoms]
                atom_type_counts = torch.bincount(arm_elements, minlength=118)  # 118 element types
                arm_atom_types.append(atom_type_counts.float())
            else:
                arm_atom_types.append(torch.zeros(118))
        
        prior_info['arm_atom_types'] = torch.stack(arm_atom_types)
        
        # Scaffold atom types
        scaffold_atoms = (data.ligand_atom_mask == -1)
        if scaffold_atoms.any():
            scaffold_elements = data.ligand_element[scaffold_atoms]
            scaffold_atom_types = torch.bincount(scaffold_elements, minlength=118).float()
            prior_info['scaffold_atom_types'] = scaffold_atom_types
    
    return prior_info


def apply_prior_noise(prior_masks, noise_std=0.1):
    """
    Add noise to prior masks
    """
    if prior_masks is None:
        return None
    
    # Add Gaussian noise
    noise = torch.randn_like(prior_masks) * noise_std
    
    # Ensure probabilities are in [0, 1] range
    noisy_prior = torch.clamp(prior_masks + noise, 0, 1)
    
    # Re-normalize
    noisy_prior = F.softmax(noisy_prior, dim=-1)
    
    return noisy_prior


def compute_beta_prior(data, beta=1.0):
    """
    Compute Beta prior distribution
    """
    if not hasattr(data, 'ligand_atom_mask'):
        return None
    
    num_atoms = data.ligand_pos.size(0)
    num_arms = data.num_arms
    
    # Initialize Beta prior
    alpha = torch.ones(num_atoms, num_arms + 1) * beta
    beta_param = torch.ones(num_atoms, num_arms + 1) * beta
    
    # Adjust parameters based on actual decomposition
    for arm_idx in range(num_arms):
        arm_atoms = (data.ligand_atom_mask == arm_idx)
        if arm_atoms.any():
            alpha[arm_atoms, arm_idx] += 1.0  # Increase success probability
    
    scaffold_atoms = (data.ligand_atom_mask == -1)
    if scaffold_atoms.any():
        alpha[scaffold_atoms, num_arms] += 1.0
    
    return alpha, beta_param


def compute_prior_statistics(data):
    """
    Compute prior statistics
    """
    stats = {}
    
    if hasattr(data, 'ligand_prior_masks'):
        stats['ligand_prior_shape'] = data.ligand_prior_masks.shape
        stats['ligand_prior_mean'] = data.ligand_prior_masks.mean().item()
        stats['ligand_prior_std'] = data.ligand_prior_masks.std().item()
    
    if hasattr(data, 'pocket_prior_masks'):
        stats['pocket_prior_shape'] = data.pocket_prior_masks.shape
        stats['pocket_prior_mean'] = data.pocket_prior_masks.mean().item()
        stats['pocket_prior_std'] = data.pocket_prior_masks.std().item()
    
    if hasattr(data, 'ligand_decomp_centers'):
        stats['decomp_centers_shape'] = data.ligand_decomp_centers.shape
        stats['decomp_centers_mean'] = data.ligand_decomp_centers.mean().item()
        stats['decomp_centers_std'] = data.ligand_decomp_centers.std().item()
    
    return stats