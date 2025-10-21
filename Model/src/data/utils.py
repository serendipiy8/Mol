import os 
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import ChemicalFeatures, rdchem
from rdkit import RDConfig
import torch
from openbabel import openbabel as ob
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# ATOM_FAMILIES
ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 
                 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}

# BOND_TYPES
BOND_TYPES = {
    rdchem.BondType.UNSPECIFIED: 0,
    rdchem.BondType.SINGLE: 1,
    rdchem.BondType.DOUBLE: 2,
    rdchem.BondType.TRIPLE: 3,
    rdchem.BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}

# HYBRIDIZATION_TYPE
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}

# 仅在确实需要时启用 follow_batch；默认置空以避免缺失字段引发装配错误
FOLLOW_BATCH = tuple()

def convert_sdf_to_pdb(sdf_path, pdb_path):
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")

    mol = ob.OBMol()
    obConversion.ReadFile(mol, sdf_path)
    obConversion.WriteFile(mol, pdb_path)

class PDBProtein(object):

    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())}

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode = 'auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        self.title = None
        
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []

        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain_id': line[21],
                    'res_id': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'element': element_symb,
                    'occupancy': float(line[54:60]) if line[54:60].strip() else 0.0,
                    'b_factor': float(line[60:66]) if line[60:66].strip() else 0.0,
                }
        
    def _parse(self):
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []

        for atom_line in self._enum_formatted_atom_lines():
            atom_id = atom_line['atom_id']
            atom_name = atom_line['atom_name']
            res_name = atom_line['res_name']
            chain_id = atom_line['chain_id']
            res_id = atom_line['res_id']
            x = atom_line['x']
            y = atom_line['y']
            z = atom_line['z']
            element = atom_line['element']
            occupancy = atom_line['occupancy']
            b_factor = atom_line['b_factor']

            try:
                atomic_num = self.ptable.GetAtomicNumber(element)
            except:
                atomic_num = 0

            is_backbone = atom_name in self.BACKBONE_NAMES

            aa_type = self.AA_NAME_NUMBER.get(res_name, -1)

            atom_info = {
                'atom_id': atom_id,
                'atom_name': atom_name,
                'res_name': res_name,
                'res_id': res_id,
                'chain_id': chain_id,
                'element': element,
                'atomic_num': atomic_num,
                'x': x,
                'y': y,
                'z': z,
                'occupancy': occupancy,
                'b_factor': b_factor,
                'is_backbone': is_backbone,
                'aa_type': aa_type,
            }

            self.atoms.append(atom_info)
            self.element.append(atomic_num)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_num))
            self.pos.append([x, y, z])
            self.atom_name.append(atom_name)
            self.is_backbone.append(is_backbone)
            self.atom_to_aa_type.append(aa_type)

        self.element = np.array(self.element)
        self.atomic_weight = np.array(self.atomic_weight)
        self.pos = np.array(self.pos)
        self.is_backbone = np.array(self.is_backbone)
        self.atom_to_aa_type = np.array(self.atom_to_aa_type)

    def to_dict_atom(self):
        return {
            'element': self.element,
            'pos': self.pos,
            'is_backbone': self.is_backbone,
            'atom_to_aa_type': self.atom_to_aa_type,
            'atomic_weight': self.atomic_weight,
            'atom_name': self.atom_name,
        }
    
    def to_dict_residue(self):
        return {}

def parse_sdf_file(path, kekulize=True):
    if isinstance(path, str):
        rdmol = next(iter(Chem.SDMolSupplier(path, removeHs = True, sanitize = True)))
    elif isinstance(path, Chem.Mol):
        rdmol = path
    else:
        raise ValueError(type(path))

    if kekulize:
        Chem.Kekulize(rdmol)

    data = process_from_mol(rdmol)
    data['smiles'] = Chem.MolToSmiles(rdmol, kekuleSmiles = kekulize)
    return data

def process_from_mol(rdmol):
    atom_features = []
    atom_positions = []

    for atom in rdmol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        hybridization = atom.GetHybridization()
        hybridization_type = str(hybridization)        
        is_aromatic = atom.GetIsAromatic()
        formal_charge = atom.GetFormalCharge()
        degree = atom.GetDegree()
        total_valence = atom.GetTotalValence()
        
        atom_feature = [
            atomic_num,
            HYBRIDIZATION_TYPE_ID.get(hybridization_type, 0),
            int(is_aromatic),
            formal_charge,
            degree,
            total_valence,
        ]
        atom_features.append(atom_feature)

        # Atom position
        conformer = rdmol.GetConformer()
        pos = conformer.GetAtomPosition(atom.GetIdx())
        atom_positions.append([pos.x, pos.y, pos.z])

    bond_features = []
    bond_index = []

    for bond in rdmol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_index.extend([[i, j], [j, i]])

        bond_type = BOND_TYPES.get(bond.GetBondType(), 0)
        bond_feature = [bond_type]
        is_conjugated = bond.GetIsConjugated()
        bond_feature.append(int(is_conjugated))
        is_in_ring = bond.IsInRing()
        bond_feature.append(int(is_in_ring))
        
        bond_features.extend([bond_feature, bond_feature])

    return {
        'rdmol': rdmol,
        'element': np.array([f[0] for f in atom_features]),
        'hybridization': np.array([f[1] for f in atom_features]),
        'is_aromatic': np.array([f[2] for f in atom_features]),
        'formal_charge': np.array([f[3] for f in atom_features]),
        'degree': np.array([f[4] for f in atom_features]),
        'total_valence': np.array([f[5] for f in atom_features]),
        'pos': np.array(atom_positions),
        'bond_index': np.array(bond_index).T if bond_index else np.array([]).reshape(2, 0),
        'bond_type': np.array([f[0] for f in bond_features]) if bond_features else np.array([]),
        'bond_is_conjugated': np.array([f[1] for f in bond_features]) if bond_features else np.array([]),
        'bond_is_in_ring': np.array([f[2] for f in bond_features]) if bond_features else np.array([]),
    }


class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)
        
        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        if 'bond_index' in ligand_dict and ligand_dict['bond_index'].size > 0:
            instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if
                                                          instance.ligand_bond_index[0, k].item() == i] for i in
                                               instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            if hasattr(self, 'ligand_pos') and self.ligand_pos is not None:
                return self.ligand_pos.size(0)
        elif key == 'ligand_context_bond_index':
            if hasattr(self, 'ligand_context_pos') and self.ligand_context_pos is not None:
                return self.ligand_context_pos.size(0)
        elif key == 'mask_ctx_edge_index_0':
            if hasattr(self, 'ligand_masked_pos') and self.ligand_masked_pos is not None:
                return self.ligand_masked_pos.size(0)
        elif key == 'mask_ctx_edge_index_1':
            if hasattr(self, 'ligand_context_pos') and self.ligand_context_pos is not None:
                return self.ligand_context_pos.size(0)
        elif key == 'mask_compose_edge_index_0':
            if hasattr(self, 'ligand_masked_pos') and self.ligand_masked_pos is not None:
                return self.ligand_masked_pos.size(0)
        elif key == 'mask_compose_edge_index_1':
            if hasattr(self, 'ligand_context_pos') and self.ligand_context_pos is not None:
                return self.ligand_context_pos.size(0)
        return super().__inc__(key, value)

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Edge indices are concatenated along dim=1; node-level tensors along dim=0
        if key in (
            'ligand_bond_index',
            'ligand_context_bond_index',
            'mask_ctx_edge_index_0',
            'mask_ctx_edge_index_1',
            'mask_compose_edge_index_0',
            'mask_compose_edge_index_1',
            'protein_protein_edges',
            'ligand_ligand_edges',
            'cross_edges',
        ):
            return 1
        return 0

class ProteinLigandDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=FOLLOW_BATCH, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)

def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=FOLLOW_BATCH)


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def get_batch_connectivity_matrix(ligand_batch, ligand_bond_index, ligand_bond_type, ligand_bond_batch):
    import torch_scatter
    
    batch_ligand_size = torch_scatter.segment_coo(
        torch.ones_like(ligand_batch),
        ligand_batch,
        reduce='sum',
    )
    batch_index_offset = torch.cumsum(batch_ligand_size, 0) - batch_ligand_size
    batch_size = len(batch_index_offset)
    batch_connectivity_matrix = []
    
    for batch_index in range(batch_size):
        start_index, end_index = ligand_bond_index[:, ligand_bond_batch == batch_index]
        start_index -= batch_index_offset[batch_index]
        end_index -= batch_index_offset[batch_index]
        bond_type = ligand_bond_type[ligand_bond_batch == batch_index]
        
        connectivity_matrix = torch.zeros(
            batch_ligand_size[batch_index], 
            batch_ligand_size[batch_index],
            dtype=torch.int
        )
        
        for s, e, t in zip(start_index, end_index, bond_type):
            connectivity_matrix[s, e] = connectivity_matrix[e, s] = t
            
        batch_connectivity_matrix.append(connectivity_matrix)
    
    return batch_connectivity_matrix