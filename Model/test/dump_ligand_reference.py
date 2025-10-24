import os
import sys
import argparse
from typing import Optional

import torch


def add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


add_src_to_path()

from src.data.data_loader import get_data_loaders  # noqa: E402
from src.evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf  # noqa: E402


def unwrap(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ('data', 'batch'):
            if k in obj and obj[k] is not None:
                return obj[k]
        for v in obj.values():
            try:
                if hasattr(v, 'ligand_pos') or hasattr(v, 'pos'):
                    return v
            except Exception:
                continue
        return obj
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if it is not None:
                return unwrap(it)
    return obj


def z_to_symbols(z: torch.Tensor):
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        pt = Chem.GetPeriodicTable()
        zz = z.detach().cpu().long().tolist()
        return [pt.GetElementSymbol(int(max(1, zi))) if int(zi) > 0 else 'C' for zi in zz]
    except Exception:
        mapping = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        zz = z.detach().cpu().long().tolist()
        return [mapping.get(int(zi), 'C') for zi in zz]


def main():
    parser = argparse.ArgumentParser(description='Dump a reference ligand SDF from dataset (no training)')
    parser.add_argument('--dataset_path', type=str, default='src/data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--split_file', type=str, default='src/data/split_by_name.pt')
    parser.add_argument('--out_dir', type=str, default='experiments/reference')
    parser.add_argument('--index', type=int, default=0, help='sample index within loader to dump')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader = get_data_loaders(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        batch_size=1,
        num_workers=0,
        shuffle_train=False,
        shuffle_test=False,
        require_ligand=True,
        require_feat=True,
        require_protein=True,
    )

    loader = test_loader or train_loader
    if loader is None:
        raise RuntimeError('Failed to build data loader')

    # get nth sample
    batch = None
    for i, b in enumerate(loader):
        if b is None:
            continue
        if i == int(args.index):
            batch = b
            break
    if batch is None:
        raise RuntimeError(f'No batch found at index {args.index}')

    bb = unwrap(batch)

    lig_pos = getattr(bb, 'ligand_pos', None)
    lig_el = getattr(bb, 'ligand_element', None)
    bond_index = getattr(bb, 'ligand_bond_index', None)
    bond_type = getattr(bb, 'ligand_bond_type', None)

    # Option 1: Copy dataset-provided ligand file if available
    for key in ('ligand_file', 'src_ligand_filename', 'sub_ligand_file'):
        lf = getattr(bb, key, None)
        if isinstance(lf, str) and os.path.isfile(lf):
            dst = os.path.join(args.out_dir, f'reference_from_dataset_{args.index:05d}.sdf')
            try:
                import shutil
                shutil.copyfile(lf, dst)
                print(f'Copied dataset ligand file to {dst}')
            except Exception as e:
                print(f'Warning: failed to copy dataset ligand file: {e}')
            break

    # Option 2: Reconstruct from fields and write two SDFs: coords-only and (if available) with bonds
    if isinstance(lig_pos, torch.Tensor) and lig_pos.ndim == 2 and lig_pos.size(1) == 3 and lig_pos.size(0) > 0:
        coords = lig_pos.detach().cpu().numpy()
        if isinstance(lig_el, torch.Tensor) and lig_el.numel() == lig_pos.size(0):
            symbols = z_to_symbols(lig_el)
        else:
            symbols = ['C'] * coords.shape[0]
        mol = build_rdkit_mol_from_coords(symbols, coords)
        out1 = os.path.join(args.out_dir, f'reference_coords_only_{args.index:05d}.sdf')
        write_rdkit_mol_sdf(mol, out1)
        print(f'Wrote coords-only reference to {out1}')

        # with bonds if available (deduplicate edges)
        if isinstance(bond_index, torch.Tensor) and bond_index.ndim == 2 and bond_index.size(0) == 2 and bond_index.numel() > 0:
            try:
                from rdkit import Chem
                from rdkit import RDLogger
                RDLogger.DisableLog('rdApp.*')
                rw = Chem.RWMol()
                atom_indices = []
                for sym in symbols:
                    a = Chem.Atom(sym)
                    atom_indices.append(rw.AddAtom(a))
                type_map = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
                E = bond_index.size(1)
                seen = set()
                for k in range(E):
                    a = int(bond_index[0, k].item())
                    b = int(bond_index[1, k].item())
                    if a == b:
                        continue
                    lo = a if a < b else b
                    hi = b if a < b else a
                    key = (lo, hi)
                    if key in seen:
                        continue
                    seen.add(key)
                    bt = 1
                    if isinstance(bond_type, torch.Tensor) and bond_type.numel() == E:
                        bt = int(bond_type[k].item())
                    bond_t = type_map.get(bt, Chem.rdchem.BondType.SINGLE)
                    if rw.GetBondBetweenAtoms(lo, hi) is None:
                        try:
                            rw.AddBond(lo, hi, bond_t)
                        except Exception:
                            continue
                m2 = rw.GetMol()
                # set conformer
                conf = Chem.Conformer(len(atom_indices))
                for i, pos in enumerate(coords):
                    conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
                m2.RemoveAllConformers()
                m2.AddConformer(conf, assignId=True)
                try:
                    Chem.SanitizeMol(m2)
                except Exception:
                    pass
                out2 = os.path.join(args.out_dir, f'reference_with_bonds_{args.index:05d}.sdf')
                w = Chem.SDWriter(out2)
                try:
                    w.write(m2)
                finally:
                    w.close()
                print(f'Wrote bonds reference to {out2}')
            except Exception as e:
                print(f'Warning: failed to write bonds SDF: {e}')
    else:
        print('Warning: ligand_pos missing or invalid; nothing written')


if __name__ == '__main__':
    main()

import os
import sys
import argparse
from typing import Optional

import torch


def add_src_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src_path = os.path.join(root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


add_src_to_path()

from src.data.data_loader import get_data_loaders  # noqa: E402
from src.evaluation.utils_pdb_writer import build_rdkit_mol_from_coords, write_rdkit_mol_sdf  # noqa: E402


def unwrap(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ('data', 'batch'):
            if k in obj and obj[k] is not None:
                return obj[k]
        for v in obj.values():
            try:
                if hasattr(v, 'ligand_pos') or hasattr(v, 'pos'):
                    return v
            except Exception:
                continue
        return obj
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if it is not None:
                return unwrap(it)
    return obj


def z_to_symbols(z: torch.Tensor):
    try:
        from rdkit import Chem
        pt = Chem.GetPeriodicTable()
        zz = z.detach().cpu().long().tolist()
        return [pt.GetElementSymbol(int(max(1, zi))) if int(zi) > 0 else 'C' for zi in zz]
    except Exception:
        mapping = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
        zz = z.detach().cpu().long().tolist()
        return [mapping.get(int(zi), 'C') for zi in zz]


def main():
    parser = argparse.ArgumentParser(description='Dump a reference ligand SDF from dataset (no training)')
    parser.add_argument('--dataset_path', type=str, default='src/data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--split_file', type=str, default='src/data/split_by_name.pt')
    parser.add_argument('--out_dir', type=str, default='experiments/reference')
    parser.add_argument('--index', type=int, default=0, help='sample index within loader to dump')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader = get_data_loaders(
        dataset_path=args.dataset_path,
        split_file=args.split_file,
        batch_size=1,
        num_workers=0,
        shuffle_train=False,
        shuffle_test=False,
        require_ligand=True,
        require_feat=True,
        require_protein=True,
    )

    loader = test_loader or train_loader
    if loader is None:
        raise RuntimeError('Failed to build data loader')

    # get nth sample
    batch = None
    for i, b in enumerate(loader):
        if b is None:
            continue
        if i == int(args.index):
            batch = b
            break
    if batch is None:
        raise RuntimeError(f'No batch found at index {args.index}')

    bb = unwrap(batch)

    lig_pos = getattr(bb, 'ligand_pos', None)
    lig_el = getattr(bb, 'ligand_element', None)
    bond_index = getattr(bb, 'ligand_bond_index', None)
    bond_type = getattr(bb, 'ligand_bond_type', None)

    # Option 1: Copy dataset-provided ligand file if available
    copied = False
    for key in ('ligand_file', 'src_ligand_filename', 'sub_ligand_file'):
        lf = getattr(bb, key, None)
        if isinstance(lf, str) and os.path.isfile(lf):
            dst = os.path.join(args.out_dir, f'reference_from_dataset_{args.index:05d}.sdf')
            try:
                import shutil
                shutil.copyfile(lf, dst)
                print(f'Copied dataset ligand file to {dst}')
                copied = True
                break
            except Exception as e:
                print(f'Warning: failed to copy dataset ligand file: {e}')

    # Option 2: Reconstruct from fields and write two SDFs: coords-only and (if available) with bonds
    if isinstance(lig_pos, torch.Tensor) and lig_pos.ndim == 2 and lig_pos.size(1) == 3 and lig_pos.size(0) > 0:
        coords = lig_pos.detach().cpu().numpy()
        if isinstance(lig_el, torch.Tensor) and lig_el.numel() == lig_pos.size(0):
            symbols = z_to_symbols(lig_el)
        else:
            symbols = ['C'] * coords.shape[0]
        mol = build_rdkit_mol_from_coords(symbols, coords)
        out1 = os.path.join(args.out_dir, f'reference_coords_only_{args.index:05d}.sdf')
        write_rdkit_mol_sdf(mol, out1)
        print(f'Wrote coords-only reference to {out1}')

        # with bonds if available
        if isinstance(bond_index, torch.Tensor) and bond_index.ndim == 2 and bond_index.size(0) == 2 and bond_index.numel() > 0:
            try:
                from rdkit import Chem
                rw = Chem.RWMol()
                atom_indices = []
                for sym in symbols:
                    a = Chem.Atom(sym)
                    atom_indices.append(rw.AddAtom(a))
                type_map = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
                E = bond_index.size(1)
                for k in range(E):
                    a = int(bond_index[0, k].item())
                    b = int(bond_index[1, k].item())
                    bt = 1
                    if isinstance(bond_type, torch.Tensor) and bond_type.numel() == E:
                        bt = int(bond_type[k].item())
                    bond_t = type_map.get(bt, Chem.rdchem.BondType.SINGLE)
                    if a != b:
                        try:
                            rw.AddBond(int(a), int(b), bond_t)
                        except Exception:
                            continue
                m2 = rw.GetMol()
                # set conformer
                conf = Chem.Conformer(len(atom_indices))
                for i, pos in enumerate(coords):
                    conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
                m2.RemoveAllConformers()
                m2.AddConformer(conf, assignId=True)
                try:
                    Chem.SanitizeMol(m2)
                except Exception:
                    pass
                out2 = os.path.join(args.out_dir, f'reference_with_bonds_{args.index:05d}.sdf')
                w = Chem.SDWriter(out2)
                try:
                    w.write(m2)
                finally:
                    w.close()
                print(f'Wrote bonds reference to {out2}')
            except Exception as e:
                print(f'Warning: failed to write bonds SDF: {e}')
    else:
        print('Warning: ligand_pos missing or invalid; nothing written')


if __name__ == '__main__':
    main()


