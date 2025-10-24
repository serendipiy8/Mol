import os
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


def _load_receptor_from_pdb(pdb_path: str, sanitize: bool = False) -> Optional[Chem.Mol]:
    try:
        mol = Chem.MolFromPDBFile(pdb_path, sanitize=sanitize, removeHs=False)
        if mol is None:
            return None
        if mol.GetNumConformers() == 0:
            conf = Chem.Conformer(mol.GetNumAtoms())
            mol.AddConformer(conf, assignId=True)
        return mol
    except Exception:
        return None


def _get_coords(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        coords[i] = [p.x, p.y, p.z]
    return coords


def _is_aromatic(atom: Chem.Atom) -> bool:
    try:
        return atom.GetIsAromatic()
    except Exception:
        return False


def _ring_centers(mol: Chem.Mol) -> np.ndarray:
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    coords = _get_coords(mol)
    centers = []
    for ring in rings:
        if all(_is_aromatic(mol.GetAtomWithIdx(i)) for i in ring):
            pts = coords[np.array(ring, dtype=int)]
            centers.append(pts.mean(axis=0))
    return np.array(centers, dtype=np.float32) if centers else np.zeros((0, 3), dtype=np.float32)


def _donors_acceptors(mol: Chem.Mol) -> Tuple[List[int], List[int]]:
    donors = rdMolDescriptors.CalcNumHBD(mol)
    acceptors = rdMolDescriptors.CalcNumHBA(mol)
    patt_d = Chem.MolFromSmarts('[!$([#6,H0,-,-2,-3])]')
    patt_a = Chem.MolFromSmarts('[$([O,S;H1;v2]),$([O,S;H0;v2;!$(*-*=[O,N,P,S])]),$([N;v3;H0;$(Nc)])]')
    d_idx = set()
    a_idx = set()
    if patt_d is not None:
        for m in mol.GetSubstructMatches(patt_d):
            d_idx.add(m[0])
    if patt_a is not None:
        for m in mol.GetSubstructMatches(patt_a):
            a_idx.add(m[0])
    return list(d_idx), list(a_idx)


def compute_contacts(ligand: Chem.Mol, receptor: Chem.Mol, cutoff: float = 4.0) -> int:
    lc = _get_coords(ligand)
    rc = _get_coords(receptor)
    if lc.size == 0 or rc.size == 0:
        return 0
    dists = np.sqrt(((lc[:, None, :] - rc[None, :, :]) ** 2).sum(axis=-1))
    return int((dists < cutoff).sum())


def compute_hbonds(ligand: Chem.Mol, receptor: Chem.Mol, max_dist: float = 3.5) -> int:
    # crude donor-acceptor distance check
    ld, la = _donors_acceptors(ligand)
    rd, ra = _donors_acceptors(receptor)
    if not (ld or la) or not (rd or ra):
        return 0
    lc = _get_coords(ligand)
    rc = _get_coords(receptor)
    cnt = 0
    for i in ld:
        for j in ra:
            dij = np.linalg.norm(lc[i] - rc[j])
            if dij <= max_dist:
                cnt += 1
    for i in la:
        for j in rd:
            dij = np.linalg.norm(lc[i] - rc[j])
            if dij <= max_dist:
                cnt += 1
    return cnt


def compute_pi_interactions(ligand: Chem.Mol, receptor: Chem.Mol, max_center_dist: float = 5.0) -> int:
    lcent = _ring_centers(ligand)
    rcent = _ring_centers(receptor)
    if lcent.size == 0 or rcent.size == 0:
        return 0
    dists = np.sqrt(((lcent[:, None, :] - rcent[None, :, :]) ** 2).sum(axis=-1))
    return int((dists < max_center_dist).sum())


def run_interactions_evaluation(generated_sdf_dir: str, receptor_pdb_path: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    receptor = _load_receptor_from_pdb(receptor_pdb_path, sanitize=False)
    if receptor is None:
        return {'error': 'Failed to load receptor PDB', 'success': False}

    from rdkit import Chem
    contact_counts = []
    hbond_counts = []
    pi_counts = []

    for fname in os.listdir(generated_sdf_dir):
        if not fname.lower().endswith('.sdf'):
            continue
        fpath = os.path.join(generated_sdf_dir, fname)
        try:
            suppl = Chem.SDMolSupplier(fpath, removeHs=False, sanitize=True)
            for lig in suppl:
                if lig is None:
                    continue
                if lig.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(lig)
                contact_counts.append(compute_contacts(lig, receptor, cutoff=4.0))
                hbond_counts.append(compute_hbonds(lig, receptor, max_dist=3.5))
                pi_counts.append(compute_pi_interactions(lig, receptor, max_center_dist=5.0))
        except Exception:
            continue

    summary = {
        'success': True,
        'num_molecules': len(contact_counts),
        'contacts_mean': float(np.mean(contact_counts)) if contact_counts else 0.0,
        'contacts_std': float(np.std(contact_counts)) if contact_counts else 0.0,
        'hbonds_mean': float(np.mean(hbond_counts)) if hbond_counts else 0.0,
        'hbonds_std': float(np.std(hbond_counts)) if hbond_counts else 0.0,
        'pi_mean': float(np.mean(pi_counts)) if pi_counts else 0.0,
        'pi_std': float(np.std(pi_counts)) if pi_counts else 0.0,
    }

    with open(os.path.join(output_dir, 'interactions_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    import pandas as pd
    df = pd.DataFrame({
        'contacts': contact_counts,
        'hbonds': hbond_counts,
        'pi_interactions': pi_counts,
    })
    df.to_csv(os.path.join(output_dir, 'interactions_details.csv'), index=False)

    return summary


