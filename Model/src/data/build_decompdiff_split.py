import os
import pickle
import lmdb
import random
from typing import Dict, List, Tuple, Optional

import torch


def _get_processed_dir(dataset_root: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(dataset_root)), 'crossdocked_v1.1_rmsd1.0_processed'))


def _open_lmdb(lmdb_path: str):
    return lmdb.open(
        lmdb_path,
        map_size=10 * (1024 * 1024 * 1024),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )


def _extract_rmsd(raw: dict) -> Optional[float]:
    cand = ['rmsd', 'dock_rmsd', 'pose_rmsd', 'rmsd_to_crystal', 'rmsd_ref']
    for k in cand:
        if k in raw:
            try:
                return float(raw[k])
            except Exception:
                continue
    return None


def _extract_protein_id(raw: dict) -> Optional[str]:
    cand = ['protein_name', 'protein_id', 'target', 'receptor_name', 'receptor_id', 'pdb_id', 'protein_file', 'receptor_pdb']
    for k in cand:
        if k in raw:
            try:
                v = raw[k]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode('utf-8', errors='ignore')
                v = str(v)
                base = os.path.splitext(os.path.basename(v))[0]
                return base if len(base) > 0 else v
            except Exception:
                continue
    # fallback to basename of protein path-like fields
    for k in raw.keys():
        if 'protein' in k or 'receptor' in k:
            try:
                v = str(raw[k])
                base = os.path.splitext(os.path.basename(v))[0]
                if base:
                    return base
            except Exception:
                continue
    return None


def _extract_ligand_id(raw: dict) -> Optional[str]:
    cand = ['ligand_name', 'ligand_id', 'ligand_file', 'ligand_sdf', 'ligand_path']
    for k in cand:
        if k in raw:
            try:
                v = raw[k]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode('utf-8', errors='ignore')
                v = str(v)
                base = os.path.splitext(os.path.basename(v))[0]
                return base if len(base) > 0 else v
            except Exception:
                continue
    return None


def _extract_protein_seq(raw: dict) -> Optional[str]:
    cand = ['protein_sequence', 'sequence', 'aa_sequence', 'receptor_sequence']
    for k in cand:
        if k in raw:
            try:
                v = raw[k]
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode('utf-8', errors='ignore')
                s = ''.join([c for c in str(v) if c.isalpha()])
                if len(s) >= 30:
                    return s
            except Exception:
                continue
    return None


def _seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    # very rough: align to min length and compute identity
    m = min(la, lb)
    eq = sum(1 for i in range(m) if a[i] == b[i])
    return eq / float(m)


def _select_diverse_proteins(candidates: List[Tuple[str, Optional[str]]], max_n: int, max_identity: float, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: List[Tuple[str, Optional[str]]] = []
    names: List[str] = []
    for pid, seq in candidates:
        ok = True
        for _, s2 in selected:
            if seq and s2:
                if _seq_identity(seq, s2) > max_identity:
                    ok = False
                    break
            else:
                # if no sequences, treat only exact same id as non-diverse
                pass
        if ok:
            selected.append((pid, seq))
            names.append(pid)
            if len(names) >= max_n:
                break
    return names


def build_split(dataset_root: str,
                out_dir: Optional[str] = None,
                rmsd_threshold: float = 1.0,
                num_train_complexes: int = 100_000,
                num_test_proteins: int = 100,
                max_seq_identity: float = 0.30,
                seed: int = 2024) -> Dict:
    processed_dir = _get_processed_dir(dataset_root)
    lmdb_path = None
    # locate lmdb
    for fn in os.listdir(processed_dir):
        if fn.endswith('.lmdb') and 'crossdocked' in fn:
            lmdb_path = os.path.join(processed_dir, fn)
            break
    if lmdb_path is None:
        raise FileNotFoundError('LMDB not found in processed dir: ' + processed_dir)

    env = _open_lmdb(lmdb_path)
    keys: List[str] = []
    complexes: List[Tuple[int, str, str, Optional[str]]] = []  # (idx, protein_id, ligand_id, protein_seq)
    with env.begin(buffers=True) as txn:
        for i, key in enumerate(txn.cursor().iternext(values=False)):
            try:
                keys.append(key.decode())
                raw = txn.get(key)
                rd = pickle.loads(bytes(raw)) if raw is not None else None
                if rd is None:
                    continue
                rmsd = _extract_rmsd(rd)
                if rmsd is None or rmsd > rmsd_threshold:
                    continue
                pid = _extract_protein_id(rd) or f'prot_{i}'
                lid = _extract_ligand_id(rd) or keys[-1]
                pseq = _extract_protein_seq(rd)
                complexes.append((i, pid, lid, pseq))
            except Exception:
                continue

    if len(complexes) == 0:
        raise RuntimeError('No complexes passed RMSD filter; check RMSD keys or threshold')

    # collect protein candidates (pid, seq)
    pid_seq: Dict[str, Optional[str]] = {}
    for _, pid, _, pseq in complexes:
        if pid not in pid_seq:
            pid_seq[pid] = pseq
    cand_list = list(pid_seq.items())
    test_pids = _select_diverse_proteins(cand_list, max_n=num_test_proteins, max_identity=max_seq_identity, seed=seed)

    # split complexes
    rng = random.Random(seed)
    train_indices: List[int] = []
    test_indices: List[int] = []
    train_pairs: List[Tuple[str, str]] = []
    test_pairs: List[Tuple[str, str]] = []

    for idx, pid, lid, _ in complexes:
        if pid in test_pids:
            test_indices.append(idx)
            test_pairs.append((pid, lid))
        else:
            train_indices.append(idx)
            train_pairs.append((pid, lid))

    # downsample train to requested number
    if len(train_indices) > num_train_complexes:
        sel = rng.sample(list(range(len(train_indices))), k=num_train_complexes)
        train_indices = [train_indices[i] for i in sel]
        train_pairs = [train_pairs[i] for i in sel]

    # write outputs
    if out_dir is None:
        out_dir = processed_dir
    os.makedirs(out_dir, exist_ok=True)
    split_by_name = {
        'train': train_pairs,
        'test': test_pairs,
    }
    split_by_idx = {
        'train_indices': train_indices,
        'test_indices': test_indices,
    }
    torch.save(split_by_name, os.path.join(out_dir, 'split_by_name.pt'))
    torch.save(split_by_idx, os.path.join(out_dir, 'split_by_index.pt'))

    # emit test ligand references
    try:
        from src.evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf
        ref_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'experiments', 'reference')
        ref_dir = os.path.abspath(ref_dir)
        os.makedirs(ref_dir, exist_ok=True)
        written = 0
        with env.begin(buffers=True) as txn:
            for i, key in enumerate(keys):
                if i not in test_indices:
                    continue
                raw = txn.get(key.encode())
                rd = pickle.loads(bytes(raw)) if raw is not None else None
                if rd is None:
                    continue
                lig_pos = rd.get('ligand_pos', None) or rd.get('ligand_context_pos', None)
                lig_el = rd.get('ligand_element', None)
                if lig_pos is None:
                    continue
                import numpy as np
                coords = np.asarray(lig_pos, dtype=float)
                symbols = None
                if lig_el is not None:
                    try:
                        from rdkit import Chem
                        pt = Chem.GetPeriodicTable()
                        z = list(map(int, list(lig_el)))
                        symbols = [pt.GetElementSymbol(max(1, zi)) for zi in z]
                    except Exception:
                        pass
                if symbols is None:
                    symbols = ['C'] * coords.shape[0]
                mol = build_rdkit_mol_from_coords(symbols, coords)
                write_rdkit_mol_sdf(mol, os.path.join(ref_dir, f'reference_{written:05d}.sdf'))
                written += 1
                if written >= num_test_proteins:
                    break
    except Exception:
        pass

    return {
        'train_pairs': len(train_pairs),
        'test_pairs': len(test_pairs),
        'train_indices': len(train_indices),
        'test_indices': len(test_indices),
        'out_dir': out_dir,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', type=str, required=True, help='Path under which processed dir resides')
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--rmsd_threshold', type=float, default=1.0)
    ap.add_argument('--num_train', type=int, default=100000)
    ap.add_argument('--num_test_proteins', type=int, default=100)
    ap.add_argument('--max_seq_identity', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    stats = build_split(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        rmsd_threshold=args.rmsd_threshold,
        num_train_complexes=args.num_train,
        num_test_proteins=args.num_test_proteins,
        max_seq_identity=args.max_seq_identity,
        seed=args.seed,
    )
    print('Done:', stats)


