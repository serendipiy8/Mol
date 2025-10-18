import argparse
import os
from typing import Dict, List, Any, Optional


# Experiment Config
class ExperimentConfig:
    
    def __init__(self):
        self.base_config = self._get_base_config()
        self.evaluation_config = self._get_evaluation_config()
        self.baseline_config = self._get_baseline_config()
    
    def _get_base_config(self) -> Dict[str, Any]:
        return {
            # dataset config
            'dataset_name': 'crossdocked',
            'dataset_path': 'src/data/crossdocked_v1.1_rmsd1.0_processed',
            'split_file': 'src/data/split_by_name.pt',
            
            # model parameters
            'model_type': 'model', 
            'num_samples_per_test': 1000,
            'batch_size': 32,
            'num_workers': 0,
            
            # experiment output
            'output_dir': 'experiments/outputs',
            'save_intermediate': True,
            'save_final_results': True,
            
            # seed
            'random_seed': 42,
            
            # GPU parameters
            'device': 'cuda',
            'use_gpu': True
        }
    
    def _get_evaluation_config(self) -> Dict[str, Any]:
        return {
            # Molecular Properties Evaluation
            'molecular_properties': {
                'qed': True,
                'sa_score': True,
                'logp': True,
                'molecular_weight': True,
                'num_atoms': True,
                'num_rings': True,
                'num_aromatic_rings': True,
                'num_rotatable_bonds': True
            },
            
            # Geometric Evaluation
            'geometric_evaluation': {
                'bond_length_distribution': True,
                'bond_angle_distribution': True,
                'torsion_angle_distribution': True,
                'js_divergence': True
            },
            
            # Docking Evaluation
            'docking_evaluation': {
                'vina_score': True,
                'high_affinity_rate': True,
                'success_rate': True, 
                'binding_pose_analysis': True
            },
            
            # Chemical Evaluation
            'chemical_evaluation': {
                'validity_rate': True,
                'uniqueness_rate': True,
                'novelty_rate': True,
                'scaffold_diversity': True,
                'functional_group_analysis': True
            },
            
            # Similarity Evaluation
            'similarity_evaluation': {
                'tanimoto_similarity': True,
                'morgan_fingerprint': True,
                'molecular_descriptor_similarity': True
            }
        }
    
    # Baseline Config
    def _get_baseline_config(self) -> Dict[str, Any]:
        return {
            'baselines': [
                'DecompDiff',
                'TargetDiff', 
                'Pocket2Mol',
                'AR',
                'GraphBP',
                'LiGAN',
                'FLAG',
                'DrugGPS',
                'AutoFragmentDiff'
            ],
            
            'baseline_results_path': 'experiments/baseline_results',
            
            'baseline_configs': {
                'DecompDiff': {
                    'model_path': None, 
                    'num_samples': 1000,
                    'temperature': 1.0
                },
                'GraphAF': {
                    'model_path': None,
                    'num_samples': 1000,
                    'temperature': 1.0
                },
            }
        }
    
    def get_experiment_config(self, experiment_type: str) -> Dict[str, Any]:
        configs = {
            'molecular_properties': self._get_molecular_properties_config(),
            'geometric_evaluation': self._get_geometric_evaluation_config(),
            'docking_evaluation': self._get_docking_evaluation_config(),
            'chemical_evaluation': self._get_chemical_evaluation_config(),
            'similarity_evaluation': self._get_similarity_evaluation_config(),
            'full_evaluation': self._get_full_evaluation_config()
        }
        
        return configs.get(experiment_type, self.base_config)
    
    def _get_molecular_properties_config(self) -> Dict[str, Any]:
        config = self.base_config.copy()
        config.update({
            'experiment_name': 'molecular_properties_evaluation',
            'output_file': 'molecular_properties_results.json',
            'evaluation_metrics': [
                'qed', 'sa_score', 'logp', 'molecular_weight',
                'num_atoms', 'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds'
            ],
            'statistical_tests': ['ks_test', 'mann_whitney_u'],
            'visualization': True
        })
        return config
    
    def _get_geometric_evaluation_config(self) -> Dict[str, Any]:
        config = self.base_config.copy()
        config.update({
            'experiment_name': 'geometric_evaluation',
            'output_file': 'geometric_evaluation_results.json',
            'evaluation_metrics': [
                'bond_length_js_divergence',
                'bond_angle_js_divergence', 
                'torsion_angle_js_divergence',
                'bond_length_distribution',
                'bond_angle_distribution',
                'torsion_angle_distribution'
            ],
            'bins': 50,
            'visualization': True,
            'save_distributions': True
        })
        return config
    
    def _get_docking_evaluation_config(self) -> Dict[str, Any]:
        config = self.base_config.copy()
        config.update({
            'experiment_name': 'docking_evaluation',
            'output_file': 'docking_evaluation_results.json',
            'evaluation_metrics': [
                'vina_score_mean',
                'vina_score_std',
                'high_affinity_rate',
                'success_rate',
                'binding_pose_rmsd'
            ],
            'vina_config': {
                'exhaustiveness': 32,
                'num_modes': 9,
                'energy_range': 3.0
            },
            'high_affinity_threshold': -7.0,  # kcal/mol
            'success_rate_threshold': -6.0
        })
        return config
    
    def _get_chemical_evaluation_config(self) -> Dict[str, Any]:
        config = self.base_config.copy()
        config.update({
            'experiment_name': 'chemical_evaluation',
            'output_file': 'chemical_evaluation_results.json',
            'evaluation_metrics': [
                'validity_rate',
                'uniqueness_rate', 
                'novelty_rate',
                'scaffold_diversity',
                'functional_group_coverage'
            ],
            'reference_dataset': 'crossdocked',
            'validity_checks': ['smiles_validity', 'chemical_validity'],
            'uniqueness_threshold': 0.95
        })
        return config
    
    def _get_similarity_evaluation_config(self) -> Dict[str, Any]:
        config = self.base_config.copy()
        config.update({
            'experiment_name': 'similarity_evaluation',
            'output_file': 'similarity_evaluation_results.json',
            'evaluation_metrics': [
                'tanimoto_similarity_mean',
                'tanimoto_similarity_std',
                'morgan_fingerprint_similarity',
                'molecular_descriptor_similarity'
            ],
            'fingerprint_radius': 2,
            'fingerprint_bits': 2048
        })
        return config
    
    def _get_full_evaluation_config(self) -> Dict[str, Any]:
        config = self.base_config.copy()
        config.update({
            'experiment_name': 'full_evaluation',
            'output_file': 'full_evaluation_results.json',
            'include_all_metrics': True,
            'run_all_experiments': True,
            'generate_report': True,
            'report_format': 'html'
        })
        return config

def create_experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='ModelA experiment evaluation')
    

    parser.add_argument('--experiment_type', type=str, default='full_evaluation',
                       choices=['molecular_properties', 'geometric_evaluation', 
                               'docking_evaluation', 'chemical_evaluation',
                               'similarity_evaluation', 'full_evaluation'])
    
    parser.add_argument('--dataset_path', type=str, default='src/data/crossdocked_v1.1_rmsd1.0_processed')
    parser.add_argument('--output_dir', type=str, default='experiments/outputs')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random_seed', type=int, default=42)
    
    # Evaluation Parameters
    parser.add_argument('--enable_visualization', action='store_true')
    parser.add_argument('--save_intermediate', action='store_true')
    parser.add_argument('--run_baselines', action='store_true')
    
    # Specific Evaluation Parameters
    parser.add_argument('--high_affinity_threshold', type=float, default=-7.0)   
    parser.add_argument('--js_divergence_bins', type=int, default=50)
    parser.add_argument('--fingerprint_radius', type=int, default=2)
    
    return parser

def load_experiment_config(config_path: Optional[str] = None) -> ExperimentConfig:
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        pass
    
    return ExperimentConfig()