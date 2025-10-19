import os
import pickle
import lmdb
from typing import Optional, Union, List, Tuple
import torch
from torch_geometric.data import Data, Batch
from .utils import ProteinLigandData


class CrossDockedDataset:
    
    def __init__(self, root: str, split: Optional[str] = None):
        """
        Initialize CrossDocked dataset
        """
        self.root = os.path.abspath(root)
        self.split = split
        
        # Determine processed directory path
        self.processed_dir = os.path.join(os.path.dirname(self.root), 'crossdocked_v1.1_rmsd1.0_processed')
        self.processed_dir = os.path.abspath(self.processed_dir)
        
        # Find LMDB and name2id files
        self.lmdb_path, self.name2id_path = self._find_lmdb_and_name2id_files()
        
        # Initialize database connection
        self.db = None
        self.keys = []
        self.name2id = {}
        
        # Load data
        self._load_name2id()
        self._connect_db()
        
        # Load split information
        self.train_indices = []
        self.test_indices = []
        self._load_split()
    
    def _find_lmdb_and_name2id_files(self) -> Tuple[str, str]:
        """Find LMDB and name2id files"""
        
        # Possible filename patterns
        lmdb_patterns = [
            'crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic.lmdb',
            'crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic.lmdb-lock',
            '*.lmdb'
        ]
        
        name2id_patterns = [
            'crossdocked_v1.1_rmsd1.0_processed_full_ref_prior_aromatic_name2id.pt',
            'name2id.pt',
            '*_name2id.pt'
        ]
        
        lmdb_path = None
        name2id_path = None
        
        # Find LMDB file
        for pattern in lmdb_patterns:
            if '*' not in pattern:
                full_path = os.path.join(self.processed_dir, pattern)
                if os.path.exists(full_path):
                    lmdb_path = os.path.abspath(full_path)
                    break
            else:
                # Use glob to find
                import glob
                matches = glob.glob(os.path.join(self.processed_dir, pattern))
                if matches:
                    lmdb_path = os.path.abspath(matches[0])
                    break
        
        # Find name2id file
        for pattern in name2id_patterns:
            if '*' not in pattern:
                full_path = os.path.join(self.processed_dir, pattern)
                if os.path.exists(full_path):
                    name2id_path = os.path.abspath(full_path)
                    break
            else:
                # Use glob to find
                import glob
                matches = glob.glob(os.path.join(self.processed_dir, pattern))
                if matches:
                    name2id_path = os.path.abspath(matches[0])
                    break
        
        if lmdb_path is None:
            raise FileNotFoundError(f"LMDB file not found in: {self.processed_dir}")
        if name2id_path is None:
            raise FileNotFoundError(f"name2id file not found in: {self.processed_dir}")
            
        return lmdb_path, name2id_path
    
    def _connect_db(self):
        """Connect to LMDB database"""
        if self.db is None:
            self.db = lmdb.open(
                self.lmdb_path,
                map_size=10*(1024*1024*1024),   # 10GB
                create=False,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.db.begin() as txn:
                self.keys = [key.decode() for key in txn.cursor().iternext(values=False)]
    
    def _load_name2id(self):
        """Load name2id mapping"""
        if os.path.exists(self.name2id_path):
            self.name2id = torch.load(self.name2id_path)
        else:
            raise FileNotFoundError(f"name2id file not found: {self.name2id_path}")
    
    def _load_split(self):
        """Load dataset split"""
        split_file = os.path.join(self.processed_dir, 'split_by_name.pt')
        
        if os.path.exists(split_file):
            split_data = torch.load(split_file)
            
            if isinstance(split_data, dict):
                train_list = split_data.get('train', [])
                test_list = split_data.get('test', [])
            else:
                # If list format, assume first 80% is train set, last 20% is test set
                split_idx = int(len(split_data) * 0.8)
                train_list = split_data[:split_idx]
                test_list = split_data[split_idx:]
            
            # Match keys to indices
            self.train_indices = []
            self.test_indices = []
            
            for protein_name, ligand_name in train_list:
                ligand_key = ligand_name.replace('.sdf', '')
                
                key_variants = [
                    ligand_key,
                    protein_name,
                    ligand_name,
                    os.path.basename(protein_name),
                    os.path.basename(ligand_name),
                    os.path.splitext(os.path.basename(protein_name))[0],
                    os.path.splitext(os.path.basename(ligand_name))[0]
                ]
                
                found_idx = None
                for variant in key_variants:
                    if variant in self.name2id:
                        found_idx = self.name2id[variant]
                        break
                
                if found_idx is not None:
                    self.train_indices.append(found_idx)
            
            for protein_name, ligand_name in test_list:
                ligand_key = ligand_name.replace('.sdf', '')
                
                key_variants = [
                    ligand_key,
                    protein_name,
                    ligand_name,
                    os.path.basename(protein_name),
                    os.path.basename(ligand_name),
                    os.path.splitext(os.path.basename(protein_name))[0],
                    os.path.splitext(os.path.basename(ligand_name))[0]
                ]
                
                found_idx = None
                for variant in key_variants:
                    if variant in self.name2id:
                        found_idx = self.name2id[variant]
                        break
                
                if found_idx is not None:
                    self.test_indices.append(found_idx)
        else:
            # If no split file, use all data as training set
            self.train_indices = list(range(len(self.keys)))
            self.test_indices = []
    
    def __len__(self) -> int:
        """Return dataset size"""
        if self.split == 'train':
            return len(self.train_indices)
        elif self.split == 'test':
            return len(self.test_indices)
        else:
            return len(self.keys)
    
    def _get_single_item(self, idx: Union[int, List, Tuple]) -> Optional[ProteinLigandData]:
        """Get single data item"""
        if self.db is None:
            self._connect_db()
        
        if isinstance(idx, (list, tuple)):
            idx = idx[0] if idx else 0
        
        try:
            key = self.keys[idx]
            if isinstance(key, str):
                key = key.encode()
            raw_data = pickle.loads(self.db.begin().get(key))
            
            # Convert to ProteinLigandData object
            data = ProteinLigandData()
            
            # Protein data
            if 'protein_pos' in raw_data:
                data.protein_pos = torch.tensor(raw_data['protein_pos'], dtype=torch.float32)
            if 'protein_atom_feature' in raw_data:
                data.protein_atom_feature = torch.tensor(raw_data['protein_atom_feature'], dtype=torch.float32)
            if 'protein_atom_name' in raw_data:
                data.protein_atom_name = raw_data['protein_atom_name']
            if 'protein_atom_to_aa_type' in raw_data:
                data.protein_atom_to_aa_type = torch.tensor(raw_data['protein_atom_to_aa_type'], dtype=torch.long)
            
            # Ligand data
            if 'ligand_pos' in raw_data:
                data.ligand_pos = torch.tensor(raw_data['ligand_pos'], dtype=torch.float32)
            if 'ligand_bond_type' in raw_data:
                data.ligand_bond_type = torch.tensor(raw_data['ligand_bond_type'], dtype=torch.long)
            if 'ligand_atom_feature' in raw_data:
                data.ligand_atom_feature = torch.tensor(raw_data['ligand_atom_feature'], dtype=torch.float32)
            if 'ligand_bond_index' in raw_data:
                data.ligand_bond_index = torch.tensor(raw_data['ligand_bond_index'], dtype=torch.long)
            
            # Other attributes
            if 'ligand_context_pos' in raw_data:
                data.ligand_context_pos = torch.tensor(raw_data['ligand_context_pos'], dtype=torch.float32)
            if 'ligand_context_feature' in raw_data:
                data.ligand_context_feature = torch.tensor(raw_data['ligand_context_feature'], dtype=torch.float32)
            if 'ligand_context_bond_index' in raw_data:
                data.ligand_context_bond_index = torch.tensor(raw_data['ligand_context_bond_index'], dtype=torch.long)
            if 'ligand_context_bond_type' in raw_data:
                data.ligand_context_bond_type = torch.tensor(raw_data['ligand_context_bond_type'], dtype=torch.long)
            
            return data
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return None
    
    def __getitem__(self, idx: Union[int, List, Tuple]) -> Optional[ProteinLigandData]:
        """Get data item"""
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        elif isinstance(idx, tuple):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)


class PDBBindDataset:
    
    def __init__(self, root: str, split: Optional[str] = None):
        """
        Initialize PDBBind dataset
        """
        self.root = os.path.abspath(root)
        self.split = split
        
        # Determine processed directory path
        self.processed_dir = os.path.join(os.path.dirname(self.root), 'pdbbind_processed')
        self.processed_dir = os.path.abspath(self.processed_dir)
        
        # Find LMDB and name2id files
        self.lmdb_path, self.name2id_path = self._find_lmdb_and_name2id_files()
        
        # Initialize database connection
        self.db = None
        self.keys = []
        self.name2id = {}
        
        # Load data
        self._load_name2id()
        self._connect_db()
        
        # Load split information
        self.train_indices = []
        self.test_indices = []
        self._load_split()
    
    def _find_lmdb_and_name2id_files(self) -> Tuple[str, str]:
        """Find LMDB and name2id files"""
        
        # Possible filename patterns
        lmdb_patterns = [
            'pdbbind_processed_full_ref_prior_aromatic.lmdb',
            '*.lmdb'
        ]
        
        name2id_patterns = [
            'pdbbind_processed_full_ref_prior_aromatic_name2id.pt',
            'name2id.pt',
            '*_name2id.pt'
        ]
        
        lmdb_path = None
        name2id_path = None
        
        # Find LMDB file
        for pattern in lmdb_patterns:
            if '*' not in pattern:
                full_path = os.path.join(self.processed_dir, pattern)
                if os.path.exists(full_path):
                    lmdb_path = os.path.abspath(full_path)
                    break
            else:
                # Use glob to find
                import glob
                matches = glob.glob(os.path.join(self.processed_dir, pattern))
                if matches:
                    lmdb_path = os.path.abspath(matches[0])
                    break
        
        # Find name2id file
        for pattern in name2id_patterns:
            if '*' not in pattern:
                full_path = os.path.join(self.processed_dir, pattern)
                if os.path.exists(full_path):
                    name2id_path = os.path.abspath(full_path)
                    break
            else:
                # Use glob to find
                import glob
                matches = glob.glob(os.path.join(self.processed_dir, pattern))
                if matches:
                    name2id_path = os.path.abspath(matches[0])
                    break
        
        if lmdb_path is None:
            raise FileNotFoundError(f"LMDB file not found in: {self.processed_dir}")
        if name2id_path is None:
            raise FileNotFoundError(f"name2id file not found in: {self.processed_dir}")
            
        return lmdb_path, name2id_path
    
    def _connect_db(self):
        """Connect to LMDB database"""
        if self.db is None:
            self.db = lmdb.open(
                self.lmdb_path,
                map_size=10*(1024*1024*1024),   # 10GB
                create=False,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.db.begin() as txn:
                self.keys = [key.decode() for key in txn.cursor().iternext(values=False)]
    
    def _load_name2id(self):
        """Load name2id mapping"""
        if os.path.exists(self.name2id_path):
            self.name2id = torch.load(self.name2id_path)
        else:
            raise FileNotFoundError(f"name2id file not found: {self.name2id_path}")
    
    def _load_split(self):
        """Load dataset split"""
        split_file = os.path.join(self.processed_dir, 'split_by_name.pt')
        
        if os.path.exists(split_file):
            split_data = torch.load(split_file)
            
            if isinstance(split_data, dict):
                train_list = split_data.get('train', [])
                test_list = split_data.get('test', [])
            else:
                # If list format, assume first 80% is train set, last 20% is test set
                split_idx = int(len(split_data) * 0.8)
                train_list = split_data[:split_idx]
                test_list = split_data[split_idx:]
            
            # Match keys to indices
            self.train_indices = []
            self.test_indices = []
            
            for protein_name, ligand_name in train_list:
                ligand_key = ligand_name.replace('.sdf', '')
                
                key_variants = [
                    ligand_key,
                    protein_name,
                    ligand_name,
                    os.path.basename(protein_name),
                    os.path.basename(ligand_name),
                    os.path.splitext(os.path.basename(protein_name))[0],
                    os.path.splitext(os.path.basename(ligand_name))[0]
                ]
                
                found_idx = None
                for variant in key_variants:
                    if variant in self.name2id:
                        found_idx = self.name2id[variant]
                        break
                
                if found_idx is not None:
                    self.train_indices.append(found_idx)
            
            for protein_name, ligand_name in test_list:
                ligand_key = ligand_name.replace('.sdf', '')
                
                key_variants = [
                    ligand_key,
                    protein_name,
                    ligand_name,
                    os.path.basename(protein_name),
                    os.path.basename(ligand_name),
                    os.path.splitext(os.path.basename(protein_name))[0],
                    os.path.splitext(os.path.basename(ligand_name))[0]
                ]
                
                found_idx = None
                for variant in key_variants:
                    if variant in self.name2id:
                        found_idx = self.name2id[variant]
                        break
                
                if found_idx is not None:
                    self.test_indices.append(found_idx)
        else:
            # If no split file, use all data as training set
            self.train_indices = list(range(len(self.keys)))
            self.test_indices = []
    
    def __len__(self) -> int:
        """Return dataset size"""
        if self.split == 'train':
            return len(self.train_indices)
        elif self.split == 'test':
            return len(self.test_indices)
        else:
            return len(self.keys)
    
    def _get_single_item(self, idx: Union[int, List, Tuple]) -> Optional[ProteinLigandData]:
        """Get single data item"""
        if self.db is None:
            self._connect_db()
        
        if isinstance(idx, (list, tuple)):
            idx = idx[0] if idx else 0
        
        try:
            key = self.keys[idx]
            if isinstance(key, str):
                key = key.encode()
            raw_data = pickle.loads(self.db.begin().get(key))
            
            # Convert to ProteinLigandData object
            data = ProteinLigandData()
            
            # Protein data
            if 'protein_pos' in raw_data:
                data.protein_pos = torch.tensor(raw_data['protein_pos'], dtype=torch.float32)
            if 'protein_atom_feature' in raw_data:
                data.protein_atom_feature = torch.tensor(raw_data['protein_atom_feature'], dtype=torch.float32)
            if 'protein_atom_name' in raw_data:
                data.protein_atom_name = raw_data['protein_atom_name']
            if 'protein_atom_to_aa_type' in raw_data:
                data.protein_atom_to_aa_type = torch.tensor(raw_data['protein_atom_to_aa_type'], dtype=torch.long)
            
            # Ligand data
            if 'ligand_pos' in raw_data:
                data.ligand_pos = torch.tensor(raw_data['ligand_pos'], dtype=torch.float32)
            if 'ligand_bond_type' in raw_data:
                data.ligand_bond_type = torch.tensor(raw_data['ligand_bond_type'], dtype=torch.long)
            if 'ligand_atom_feature' in raw_data:
                data.ligand_atom_feature = torch.tensor(raw_data['ligand_atom_feature'], dtype=torch.float32)
            if 'ligand_bond_index' in raw_data:
                data.ligand_bond_index = torch.tensor(raw_data['ligand_bond_index'], dtype=torch.long)
            
            # Other attributes
            if 'ligand_context_pos' in raw_data:
                data.ligand_context_pos = torch.tensor(raw_data['ligand_context_pos'], dtype=torch.float32)
            if 'ligand_context_feature' in raw_data:
                data.ligand_context_feature = torch.tensor(raw_data['ligand_context_feature'], dtype=torch.float32)
            if 'ligand_context_bond_index' in raw_data:
                data.ligand_context_bond_index = torch.tensor(raw_data['ligand_context_bond_index'], dtype=torch.long)
            if 'ligand_context_bond_type' in raw_data:
                data.ligand_context_bond_type = torch.tensor(raw_data['ligand_context_bond_type'], dtype=torch.long)
            
            return data
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return None
    
    def __getitem__(self, idx: Union[int, List, Tuple]) -> Optional[ProteinLigandData]:
        """Get data item"""
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        elif isinstance(idx, tuple):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)


def get_model_dataset(config):
    """
    Get model dataset
    """
    dataset_path = config.path
    split_file = getattr(config, 'split_file', None)
    
    # Determine dataset type based on path
    if 'crossdocked' in dataset_path.lower():
        dataset = CrossDockedDataset(dataset_path)
    elif 'pdbbind' in dataset_path.lower():
        dataset = PDBBindDataset(dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_path}")
    
    # Ensure name2id is loaded
    if not hasattr(dataset, 'name2id') or not dataset.name2id:
        dataset._load_name2id()
    
    # Handle split
    if split_file and os.path.exists(split_file):
        split_data = torch.load(split_file)
        
        if isinstance(split_data, dict):
            train_list = split_data.get('train', [])
            test_list = split_data.get('test', [])
        else:
            # If list format, assume first 80% is train set, last 20% is test set
            split_idx = int(len(split_data) * 0.8)
            train_list = split_data[:split_idx]
            test_list = split_data[split_idx:]
        
        # Match keys to indices
        train_indices = []
        test_indices = []
        
        for protein_name, ligand_name in train_list:
            ligand_key = ligand_name.replace('.sdf', '')
            
            key_variants = [
                ligand_key,
                protein_name,
                ligand_name,
                os.path.basename(protein_name),
                os.path.basename(ligand_name),
                os.path.splitext(os.path.basename(protein_name))[0],
                os.path.splitext(os.path.basename(ligand_name))[0]
            ]
            
            found_idx = None
            for variant in key_variants:
                if variant in dataset.name2id:
                    found_idx = dataset.name2id[variant]
                    break
            
            if found_idx is not None:
                train_indices.append(found_idx)
        
        for protein_name, ligand_name in test_list:
            ligand_key = ligand_name.replace('.sdf', '')
            
            key_variants = [
                ligand_key,
                protein_name,
                ligand_name,
                os.path.basename(protein_name),
                os.path.basename(ligand_name),
                os.path.splitext(os.path.basename(protein_name))[0],
                os.path.splitext(os.path.basename(ligand_name))[0]
            ]
            
            found_idx = None
            for variant in key_variants:
                if variant in dataset.name2id:
                    found_idx = dataset.name2id[variant]
                    break
            
            if found_idx is not None:
                test_indices.append(found_idx)
        
        # Create subsets
        from torch.utils.data import Subset
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        return dataset, (train_subset, test_subset)
    else:
        # No split file, return complete dataset
        return dataset