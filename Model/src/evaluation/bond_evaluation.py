import os
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
from rdkit import Chem


def _bond_type_to_str(bt: Chem.BondType) -> str:
    if bt == Chem.BondType.SINGLE:
        return 'single'
    if bt == Chem.BondType.DOUBLE:
        return 'double'
    if bt == Chem.BondType.TRIPLE:
        return 'triple'
    if bt == Chem.BondType.AROMATIC:
        return 'aromatic'
    return 'other'


def run_bond_evaluation(generated_sdf_dir: str, output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    per_mol_stats = []
    total_counts = {
        'single': 0,
        'double': 0,
        'triple': 0,
        'aromatic': 0,
        'other': 0,
        'ring_bonds': 0,
        'total_bonds': 0,
        'sanitize_fail': 0,
    }

    n_mols = 0

    for fname in os.listdir(generated_sdf_dir):
        if not fname.lower().endswith('.sdf'):
            continue
        fpath = os.path.join(generated_sdf_dir, fname)
        try:
            suppl = Chem.SDMolSupplier(fpath, removeHs=False, sanitize=False)
        except Exception:
            continue

        for mol in suppl:
            if mol is None:
                continue
            n_mols += 1

            stats = {
                'file': fname,
                'single': 0,
                'double': 0,
                'triple': 0,
                'aromatic': 0,
                'other': 0,
                'ring_bonds': 0,
                'total_bonds': 0,
                'sanitize_fail': 0,
                'num_atoms': mol.GetNumAtoms(),
            }

            # Try sanitization to catch valence/chemistry issues
            mol_work = Chem.Mol(mol)
            try:
                Chem.SanitizeMol(mol_work)
            except Exception:
                stats['sanitize_fail'] = 1

            # Count bonds on original mol (robust to sanitize failure)
            for b in mol.GetBonds():
                btype = _bond_type_to_str(b.GetBondType())
                if btype in stats:
                    stats[btype] += 1
                else:
                    stats['other'] += 1
                stats['total_bonds'] += 1
                if b.IsInRing():
                    stats['ring_bonds'] += 1

            per_mol_stats.append(stats)

    df = pd.DataFrame(per_mol_stats)
    if len(df) > 0:
        # aggregate
        for key in total_counts.keys():
            if key in df.columns:
                total_counts[key] = int(df[key].sum())
        total_counts['num_molecules'] = int(len(df))
        total_counts['avg_bonds_per_mol'] = float(df['total_bonds'].mean())
        total_counts['aromatic_ratio'] = float((df['aromatic'].sum() / max(total_counts['total_bonds'], 1)))
        total_counts['ring_bond_ratio'] = float((df['ring_bonds'].sum() / max(total_counts['total_bonds'], 1)))
        total_counts['sanitize_fail_rate'] = float(df['sanitize_fail'].mean())
    else:
        total_counts['num_molecules'] = 0
        total_counts['avg_bonds_per_mol'] = 0.0
        total_counts['aromatic_ratio'] = 0.0
        total_counts['ring_bond_ratio'] = 0.0
        total_counts['sanitize_fail_rate'] = 0.0

    # save
    df.to_csv(os.path.join(output_dir, 'bond_details.csv'), index=False)
    with open(os.path.join(output_dir, 'bond_summary.json'), 'w') as f:
        json.dump(total_counts, f, indent=2)

    return {
        'summary': total_counts,
        'details_path': os.path.join(output_dir, 'bond_details.csv')
    }


