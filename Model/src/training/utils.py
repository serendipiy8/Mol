import os
from typing import List

import torch


def move_to_device(batch, device: str):
    # unwrap common wrappers
    if isinstance(batch, dict):
        for key in ('data', 'batch'):
            if key in batch and batch[key] is not None:
                batch = batch[key]
                break
        else:
            # still dict: move tensors in-place
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            return batch
    try:
        keys = list(batch.keys) if hasattr(batch, 'keys') else None
    except Exception:
        keys = None
    keys = keys or [k for k in dir(batch) if not k.startswith('_')]
    for k in keys:
        try:
            v = getattr(batch, k)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            try:
                setattr(batch, k, v.to(device))
            except Exception:
                pass
    return batch


def sdf_dir_to_smiles(sdf_dir_path: str) -> List[str]: 
    smiles = []
    try:
        from rdkit import Chem
    except Exception:
        return smiles
    for fname in os.listdir(sdf_dir_path):
        if not fname.lower().endswith('.sdf'):
            continue
        fpath = os.path.join(sdf_dir_path, fname)
        try:
            suppl = Chem.SDMolSupplier(fpath, removeHs=True, sanitize=True)
            for mol in suppl:
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol, kekuleSmiles=True)
                if smi:
                    smiles.append(smi)
        except Exception:
            continue
    return smiles


