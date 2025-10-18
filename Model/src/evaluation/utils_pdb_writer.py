#!/usr/bin/env python3
import os
from typing import List, Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


ELEMENT_SYMBOLS = [
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr'
]


def write_protein_pdb(atom_symbols: List[str], positions: np.ndarray, out_pdb_path: str) -> None:
    """将蛋白原子坐标写为简单PDB（仅ATOM行，供转换为PDBQT使用）。
    positions: (N,3)
    """
    os.makedirs(os.path.dirname(out_pdb_path), exist_ok=True)
    with open(out_pdb_path, 'w') as f:
        for idx, (sym, pos) in enumerate(zip(atom_symbols, positions), start=1):
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            name = sym.rjust(2)
            line = (
                f"ATOM  {idx:5d} {name:>3s} RES A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {sym:>2s}\n"
            )
            f.write(line)
        f.write("END\n")


def build_rdkit_mol_from_coords(atom_symbols: List[str], positions: np.ndarray) -> Chem.Mol:
    """用原子元素+三维坐标构建一个无键的 RDKit Mol（用于写 SDF 或后续构键）。
    这里不尝试自动推断键，只保证构象存在。
    """
    rw = Chem.RWMol()
    atom_indices: List[int] = []
    for sym in atom_symbols:
        a = Chem.Atom(sym)
        atom_indices.append(rw.AddAtom(a))
    mol = rw.GetMol()
    conf = Chem.Conformer(len(atom_indices))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def write_rdkit_mol_sdf(mol: Chem.Mol, out_sdf_path: str) -> None:
    os.makedirs(os.path.dirname(out_sdf_path), exist_ok=True)
    w = Chem.SDWriter(out_sdf_path)
    w.write(mol)
    w.close()


