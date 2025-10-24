import os
import sys
import argparse


def load_first_mol(sdf_path: str):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    # try strict first
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    for m in suppl:
        if m is not None:
            return m
    # fallback: relaxed sanitize
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
    for m in suppl:
        if m is None:
            continue
        try:
            m.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            pass
        return m
    return None


def main():
    parser = argparse.ArgumentParser(description='Visualize ligand SDF: output SMILES and PNG')
    parser.add_argument('--sdf', type=str, default='experiments/outputs/main/samples_conditional/lig_00000.sdf', help='input SDF path')
    parser.add_argument('--out_png', type=str, default='experiments/reference/reference_vis_00000.png', help='output PNG path')
    args = parser.parse_args()

    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem import Draw

    sdf_path = args.sdf
    if not os.path.isfile(sdf_path):
        # fallback to coords-only
        alt = 'experiments/reference/reference_coords_only_00000.sdf'
        if os.path.isfile(alt):
            sdf_path = alt
        else:
            print(f'Error: SDF not found: {args.sdf} (and fallback {alt} missing)')
            sys.exit(1)

    mol = load_first_mol(sdf_path)
    if mol is None:
        print(f'Error: failed to read any molecule from {sdf_path}')
        sys.exit(1)

    # try smiles
    try:
        smiles = Chem.MolToSmiles(mol)
    except Exception:
        # compute valence-less smiles as fallback
        try:
            tmp = Chem.RemoveHs(Chem.Mol(mol))
            smiles = Chem.MolToSmiles(tmp)
        except Exception:
            smiles = '[SMILES_FAILED]'

    print('SMILES:', smiles)

    # 2D depiction
    try:
        from rdkit.Chem import AllChem
        mol2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol2d)
        os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
        Draw.MolToFile(mol2d, args.out_png, size=(600, 450))
        print('Saved PNG to:', args.out_png)
    except Exception as e:
        print('Warning: failed to render PNG:', e)


if __name__ == '__main__':
    main()


