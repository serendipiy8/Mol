from typing import Tuple

import torch
from torch_geometric.data import Data

from .conditional_sampler import ConditionalSampler


def sample_multi_modal(model, diffusion_process, soft_mask_transform, batch: Data, num_samples: int = 1,
                       device: str = 'cuda', out_dir: str = None, prefix: str = 'lig') -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified sampler: wraps ConditionalSampler to generate coordinates/features and optional SDF writing.

    Returns the last sample's (x0, h0) tensors for convenience.
    """
    sampler = ConditionalSampler(diffusion_process=diffusion_process, soft_mask_transform=soft_mask_transform, device=device)
    last_x = None
    last_h = None
    if out_dir is not None:
        sampler.sample_and_write(model, batch, out_dir=out_dir, num_samples=num_samples, prefix=prefix, use_multi_modal=True)
        # also return one sample tensors
        x0, h0 = diffusion_process.sample_multi_modal(model=model,
                                                      shape_coord=(sampler._infer_num_ligand_atoms(batch), 3),
                                                      shape_feat=(sampler._infer_num_ligand_atoms(batch), getattr(model, 'hidden_dim', 64)),
                                                      tau=sampler._predict_tau(model, batch, sampler._infer_num_ligand_atoms(batch)),
                                                      soft_mask_transform=soft_mask_transform,
                                                      batch_context=batch)
        last_x, last_h = x0, h0
    else:
        x0, h0 = diffusion_process.sample_multi_modal(model=model,
                                                      shape_coord=(sampler._infer_num_ligand_atoms(batch), 3),
                                                      shape_feat=(sampler._infer_num_ligand_atoms(batch), getattr(model, 'hidden_dim', 64)),
                                                      tau=sampler._predict_tau(model, batch, sampler._infer_num_ligand_atoms(batch)),
                                                      soft_mask_transform=soft_mask_transform,
                                                      batch_context=batch)
        last_x, last_h = x0, h0
    return last_x, last_h


