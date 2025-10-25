import os
from typing import Optional, List, Tuple, Any

import torch
from torch_geometric.data import Data

try:
    from rdkit import Chem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

from ...evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf
from .proposer import Proposer


class ConditionalSampler:

    def __init__(self, diffusion_process, soft_mask_transform, device: str = 'cuda'):
        self.diffusion_process = diffusion_process
        self.soft_mask_transform = soft_mask_transform
        self.device = device
        self.proposer = Proposer(device=device)

    @staticmethod
    def _get_field(obj: Any, name: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        try:
            return getattr(obj, name)
        except Exception:
            return None

    @staticmethod
    def _unwrap(obj: Any):
        if isinstance(obj, dict):
            for key in ('data', 'batch'):
                if key in obj and obj[key] is not None:
                    return obj[key]
            for v in obj.values():
                if hasattr(v, 'keys'):
                    return v
            return obj
        if isinstance(obj, (list, tuple)):
            for it in obj:
                if it is not None:
                    return it
        return obj

    @staticmethod
    def _infer_num_ligand_atoms(batch: Data) -> int:
        lig_pos = ConditionalSampler._get_field(batch, 'ligand_pos')
        if isinstance(lig_pos, torch.Tensor):
            return int(lig_pos.size(0))
        lig_el = ConditionalSampler._get_field(batch, 'ligand_element')
        if isinstance(lig_el, torch.Tensor):
            return int(lig_el.numel())
        lig_feat = ConditionalSampler._get_field(batch, 'ligand_atom_feature')
        if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim >= 1:
            return int(lig_feat.size(0))
        raise ValueError('Cannot infer ligand atom count: need ligand_pos or ligand_element')

    @staticmethod
    def _infer_elements(batch: Data, n_atoms: int) -> List[str]:
        symbols: List[str] = []
        lig_el = ConditionalSampler._get_field(batch, 'ligand_element')
        if isinstance(lig_el, torch.Tensor):
            arr = lig_el.detach().cpu().long().tolist()
            if RDKit_AVAILABLE:
                pt = Chem.GetPeriodicTable()
                for z in arr:
                    z_safe = max(1, int(z))
                    symbols.append(pt.GetElementSymbol(z_safe))
            else:
                # Fallback: map common organic elements
                mapping = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
                for z in arr:
                    symbols.append(mapping.get(int(z), 'C'))
            return symbols
        lig_feat = ConditionalSampler._get_field(batch, 'ligand_atom_feature')
        if isinstance(lig_feat, torch.Tensor) and lig_feat.ndim == 2 and lig_feat.size(1) >= 1:
            arr = lig_feat[:, 0].detach().cpu().long().tolist()
            if RDKit_AVAILABLE:
                pt = Chem.GetPeriodicTable()
                for z in arr:
                    z_safe = max(1, int(z))
                    symbols.append(pt.GetElementSymbol(z_safe))
            else:
                mapping = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}
                for z in arr:
                    symbols.append(mapping.get(int(z), 'C'))
            return symbols
        # Default to carbon if elements missing
        return ['C'] * n_atoms

    @staticmethod
    def _map_classes_to_elements(class_ids: torch.Tensor) -> List[str]:

    # Safe palette: H, C, N, O, F, P, S, Cl, Br, I
        palette_safe = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        try:
            from rdkit import Chem
            pt = Chem.GetPeriodicTable()
            syms = []
            for c in class_ids.detach().cpu().long().tolist():
                # Clamp to valid palette range
                idx = max(0, min(c, len(palette_safe) - 1))
                z = palette_safe[idx]
                syms.append(pt.GetElementSymbol(int(z)))
            return syms
        except Exception:
            # Fallback mapping
            mapping = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
            out = []
            for c in class_ids.detach().cpu().long().tolist():
                idx = max(0, min(c, len(palette_safe) - 1))
                z = palette_safe[idx]
                out.append(mapping.get(int(z), 'C'))
            return out

    @staticmethod
    def _guess_bonds_by_distance(elements: List[str], coords_np, scale: float = 1.15):
        radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
        }
        max_deg = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1}
        import numpy as np
        coords = np.asarray(coords_np, dtype=float)
        N = len(elements)
        if N <= 1:
            return None
        # build all candidate pairs with distance
        cand = []
        for i in range(N):
            ri = radii.get(elements[i], 0.8)
            for j in range(i + 1, N):
                rj = radii.get(elements[j], 0.8)
                cutoff = scale * (ri + rj)
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                if 0.1 < dij <= cutoff:
                    cand.append((dij, i, j))
        if not cand:
            return None
        cand.sort(key=lambda x: x[0])
        deg = [0] * N
        pairs = []
        for _, i, j in cand:
            if deg[i] >= max_deg.get(elements[i], 4):
                continue
            if deg[j] >= max_deg.get(elements[j], 4):
                continue
            pairs.append((i, j))
            deg[i] += 1
            deg[j] += 1
        if not pairs:
            return None
        ei = torch.tensor([[a for a, b in pairs], [b for a, b in pairs]], dtype=torch.long)
        return ei

    @staticmethod
    def _prune_by_valence(edge_index: torch.Tensor, edge_type: torch.Tensor, elements: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Greedy prune edges exceeding max valence based on simple rules
        max_valence = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1}
        order_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1}  # 0: none, 1:single,2:double,3:triple,4:aromatic(≈1)
        if edge_index.numel() == 0:
            return edge_index, edge_type
        ei = edge_index.clone()
        et = edge_type.clone()
        N = len(elements)
        # compute weighted degree
        def degrees(ei_t, et_t):
            deg = torch.zeros(N, dtype=torch.float32)
            for k in range(ei_t.size(1)):
                a = int(ei_t[0, k].item())
                b = int(ei_t[1, k].item())
                w = float(order_map.get(int(et_t[k].item()), 1))
                deg[a] += w
                deg[b] += w
            return deg
        deg = degrees(ei, et)
        changed = True
        while changed:
            changed = False
            for k in range(ei.size(1)-1, -1, -1):
                a = int(ei[0, k].item())
                b = int(ei[1, k].item())
                va = max_valence.get(elements[a], 4)
                vb = max_valence.get(elements[b], 4)
                if deg[a] > va or deg[b] > vb:
                    # remove this edge and update degrees
                    w = float(order_map.get(int(et[k].item()), 1))
                    deg[a] -= w
                    deg[b] -= w
                    ei = torch.cat([ei[:, :k], ei[:, k+1:]], dim=1)
                    et = torch.cat([et[:k], et[k+1:]], dim=0)
                    changed = True
                    break
        return ei, et

    @torch.no_grad()
    def _predict_tau(self, model, batch: Data, n_atoms: int) -> torch.Tensor:
        if hasattr(model, 'predict_tau_params'):
            try:
                mu, log_sigma = model.predict_tau_params(batch)
                mu = mu.to(self.device)
                log_sigma = log_sigma.to(self.device)
                eps = torch.randn_like(mu)
                sigma = torch.exp(torch.clamp(log_sigma, min=-10.0, max=10.0))
                u = mu + sigma * eps
                return torch.sigmoid(u)
            except Exception:
                pass
        return torch.full((n_atoms,), 0.5, device=self.device)

    @torch.no_grad()
    def sample_once(self, model, batch: Data, use_multi_modal: bool = False) -> torch.Tensor:
        batch = self._unwrap(batch)
        n_atoms = self._infer_num_ligand_atoms(batch)
        print("DEBUG sample_once -> n_atoms:", n_atoms)
        tau = torch.full((n_atoms,), 0.5, device=self.device)

        # generate coordinates (and optional features)
        # auto-enable multi-modal if model expects h_t
        try:
            import inspect
            sig = inspect.signature(getattr(model, 'forward', model)).parameters
            requires_h = 'h_t' in sig
        except Exception:
            requires_h = False
        if requires_h:
            use_multi_modal = True

        if use_multi_modal:
            # feature dim must match model's expected node_dim (input_proj in_features = node_dim + 1 + time_dim)
            if hasattr(model, 'out_feat') and hasattr(model.out_feat, 'out_features'):
                feat_dim = int(model.out_feat.out_features)
            else:
                feat_dim = int(getattr(model, 'hidden_dim', 64))
            x0, _ = self.diffusion_process.sample_multi_modal(
                model=model,
                shape_coord=(n_atoms, 3),
                shape_feat=(n_atoms, feat_dim),
                tau=tau,
                soft_mask_transform=self.soft_mask_transform,
                batch_context=batch,
            )
        else:
            x0 = self.diffusion_process.sample(
                model=model,
                shape=(n_atoms, 3),
                tau=tau,
                soft_mask_transform=self.soft_mask_transform,
                batch_context=batch,
            )
        return x0

    @torch.no_grad()
    def sample_and_write(self, model, batch: Data, out_dir: str, num_samples: int = 1,
                         prefix: str = 'lig', use_multi_modal: bool = False) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        sdf_paths: List[str] = []

        batch = self._unwrap(batch)
        for name in ('protein_pos', 'protein_atom_feature', 'ligand_pos', 'ligand_atom_feature', 'ligand_element', 'protein_element'):
            val = getattr(batch, name, None)
            if isinstance(val, torch.Tensor):
                setattr(batch, name, val.to(self.device))

        n_atoms = self._infer_num_ligand_atoms(batch)
        elements = self._infer_elements(batch, n_atoms)
        print("DEBUG sample_and_write -> n_atoms:", n_atoms)
        print("DEBUG sample_and_write -> initial elements:", elements)

        for i in range(num_samples):
            h_last = None
            if use_multi_modal:
                tau = torch.full((n_atoms,), 0.5, device=self.device)
                x0, h_last = self.diffusion_process.sample_multi_modal(
                    model=model,
                    shape_coord=(n_atoms, 3),
                    shape_feat=(n_atoms, getattr(model, 'hidden_dim', 64)),
                    tau=tau,
                    soft_mask_transform=self.soft_mask_transform,
                    batch_context=batch,
                )
            else:
                x0 = self.sample_once(model, batch, use_multi_modal=False)
            coords = x0.detach().cpu().numpy()
            print(f"DEBUG sample_and_write -> sample {i} coords shape:", coords.shape)

            if use_multi_modal and h_last is not None and hasattr(model, 'atom_type_head'):
                logits = model.atom_type_head(h_last.to(self.device))
                print("DEBUG sample_and_write -> atom_type_head logits shape:", logits.shape)
                if logits.size(0) != n_atoms:
                    print("WARNING: logits batch size does not match n_atoms")
                elements = self._map_classes_to_elements(logits.argmax(dim=-1))

<<<<<<< HEAD
            # Build RDKit Mol from coords, then add reasonable bonds by distance/valence heuristics
            from rdkit import Chem
            mol = build_rdkit_mol_from_coords(elements, coords)
            # Guess bonds
            try:
                ei = self._guess_bonds_by_distance(elements, coords, scale=1.15)
            except Exception:
                ei = None
            if isinstance(ei, torch.Tensor) and ei.numel() > 0 and ei.size(0) == 2:
                try:
                    edge_index = ei
                    edge_type = torch.ones(edge_index.size(1), dtype=torch.long)
                    # prune by simple valence constraints
                    edge_index, edge_type = self._prune_by_valence(edge_index, edge_type, elements)
                    # add SINGLE bonds to mol
                    from rdkit.Chem import rdchem
                    rw = Chem.RWMol(mol)
                    E = edge_index.size(1)
                    for k in range(E):
                        a = int(edge_index[0, k].item())
                        b = int(edge_index[1, k].item())
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        if a < 0 or b < 0 or a >= rw.GetNumAtoms() or b >= rw.GetNumAtoms():
                            continue
                        if rw.GetBondBetweenAtoms(a, b) is not None:
                            continue
                        try:
                            rw.AddBond(a, b, rdchem.BondType.SINGLE)
                        except Exception:
                            continue
                    mol = rw.GetMol()
                except Exception:
                    pass
            sdf_path = os.path.join(out_dir, f"{prefix}_{i:05d}.sdf")
=======
            sdf_path = os.path.join(out_dir, f"{prefix}_{i:05d}.sdf")
            from rdkit import Chem
            mol = build_rdkit_mol_from_coords(elements, coords)
>>>>>>> parent of 9f06ac1 (train and sample)
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                print("WARNING: SanitizeMol failed:", e)
            write_rdkit_mol_sdf(mol, sdf_path)
            sdf_paths.append(sdf_path)
        return sdf_paths

    @torch.no_grad()
    def decode_types_and_bonds(self, batch: Data, atom_type_logits: torch.Tensor = None,
                               bond_logits: torch.Tensor = None, bond_edge_index: torch.Tensor = None,
                               valence_mask: bool = True) -> dict:
        """Decode element types and multi-class bonds with optional valence masking.
        Returns a dict with 'elements', 'edge_index', 'edge_type'.
        """
        out = {}
        if isinstance(atom_type_logits, torch.Tensor):
            elem_cls = atom_type_logits.argmax(dim=-1)
            out['elements'] = elem_cls.detach().cpu()
        if isinstance(bond_logits, torch.Tensor) and isinstance(bond_edge_index, torch.Tensor):
            edge_type = bond_logits.argmax(dim=-1)
            # optional: apply simple valence mask (skip here; placeholder for future)
            out['edge_index'] = bond_edge_index.detach().cpu()
            out['edge_type'] = edge_type.detach().cpu()
        return out

    @torch.no_grad()
    def propose_and_sample(self, model, protein_pos: torch.Tensor, num_atoms: int,
                           element_prior: Optional[List[int]] = None, sigma_init: float = 4.0,
                           use_multi_modal: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pos_init, elem_init = self.proposer.propose_from_receptor(protein_pos=protein_pos, num_atoms=num_atoms,
                                                                  sigma=sigma_init, element_prior=element_prior)
        # Run reverse diffusion initialized from proposer
        tau = torch.full((num_atoms,), 0.5, device=self.device)
        if use_multi_modal:
            x0, h0 = self.diffusion_process.sample_multi_modal(
                model=model,
                shape_coord=(num_atoms, 3),
                shape_feat=(num_atoms, getattr(model, 'hidden_dim', 64)),
                tau=tau,
                soft_mask_transform=self.soft_mask_transform,
                batch_context=None,
                init_x=pos_init,
                init_h=None
            )
            return x0, h0
        else:
            x0 = self.diffusion_process.sample(
                model=model,
                shape=(num_atoms, 3),
                tau=tau,
                soft_mask_transform=self.soft_mask_transform,
                batch_context=None,
                init_x=pos_init
            )
            return x0, None

