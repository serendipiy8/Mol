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
    try:
        # 防止显式价态未计算导致的异常
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return mol


def write_rdkit_mol_sdf(mol: Chem.Mol, out_sdf_path: str) -> None:
    os.makedirs(os.path.dirname(out_sdf_path), exist_ok=True)
    # 尝试宽松更新属性与部分清理，避免 "getExplicitValence" 报错
    m = Chem.Mol(mol)
    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        # 只做轻量级的属性相关清理，避免严格校验失败
        flags = Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
        Chem.SanitizeMol(m, sanitizeOps=flags)
    except Exception:
        # 忽略清理失败，尽量写出用于后处理
        pass
    # 先尝试常规写出；失败则进行降级处理
    try:
        w = Chem.SDWriter(out_sdf_path)
        try:
            w.write(m)
        finally:
            w.close()
        return
    except Exception:
        pass

    # 降级方案1：修正非法元素为碳并重新计算属性
    try:
        rw = Chem.RWMol(m)
        for a in rw.GetAtoms():
            if a.GetAtomicNum() <= 0:
                a.SetAtomicNum(6)
        m2 = rw.GetMol()
        try:
            m2.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        try:
            flags = Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            Chem.SanitizeMol(m2, sanitizeOps=flags)
        except Exception:
            pass
        w = Chem.SDWriter(out_sdf_path)
        try:
            w.write(m2)
        finally:
            w.close()
        return
    except Exception:
        pass

    # 降级方案2：构造“仅原子、无键”的分子并禁止隐式价态，然后写出
    try:
        rw2 = Chem.RWMol()
        amap = {}
        for a in m.GetAtoms():
            na = Chem.Atom(int(a.GetAtomicNum()) if a.GetAtomicNum() > 0 else 6)
            na.SetFormalCharge(0)
            na.SetNoImplicit(True)
            na.SetNumExplicitHs(0)
            idx = rw2.AddAtom(na)
            amap[a.GetIdx()] = idx
        m_atom = rw2.GetMol()
        # 复制坐标
        if m.GetNumConformers() > 0:
            conf_src = m.GetConformer()
            conf_dst = Chem.Conformer(m_atom.GetNumAtoms())
            for i in range(m_atom.GetNumAtoms()):
                p = conf_src.GetAtomPosition(i)
                conf_dst.SetAtomPosition(i, p)
            m_atom.RemoveAllConformers()
            m_atom.AddConformer(conf_dst, assignId=True)
        try:
            m_atom.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        w = Chem.SDWriter(out_sdf_path)
        try:
            w.write(m_atom)
        finally:
            w.close()
        return
    except Exception:
        # 最终失败则抛出，让上层决定是否跳过该分子
        raise


