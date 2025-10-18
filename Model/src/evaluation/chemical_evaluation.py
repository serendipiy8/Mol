import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Set, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import hashlib

class ChemicalEvaluator:
    
    def __init__(self, reference_smiles: Optional[List[str]] = None):
        self.reference_smiles = reference_smiles or []
        self.reference_scaffolds = self._extract_scaffolds(self.reference_smiles)
        self.reference_fingerprints = self._compute_fingerprints(self.reference_smiles)
    
    def evaluate_validity(self, smiles_list: List[str]) -> Dict[str, Any]:
        total_molecules = len(smiles_list)
        valid_smiles = []
        invalid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
            else:
                invalid_smiles.append(smiles)
        
        validity_rate = len(valid_smiles) / total_molecules if total_molecules > 0 else 0
        
        return {
            'total_molecules': total_molecules,
            'valid_molecules': len(valid_smiles),
            'invalid_molecules': len(invalid_smiles),
            'validity_rate': validity_rate,
            'valid_smiles': valid_smiles,
            'invalid_smiles': invalid_smiles
        }
    
    def evaluate_uniqueness(self, smiles_list: List[str]) -> Dict[str, Any]:
        unique_smiles = list(set(smiles_list))
        
        uniqueness_rate = len(unique_smiles) / len(smiles_list) if len(smiles_list) > 0 else 0
        
        smiles_counter = Counter(smiles_list)
        duplicates = {smiles: count for smiles, count in smiles_counter.items() if count > 1}
        
        return {
            'total_molecules': len(smiles_list),
            'unique_molecules': len(unique_smiles),
            'uniqueness_rate': uniqueness_rate,
            'duplicates': duplicates,
            'max_duplicates': max(smiles_counter.values()) if smiles_counter else 0,
            'avg_duplicates': np.mean(list(smiles_counter.values())) if smiles_counter else 0
        }
    
    def  evaluate_novelty(self, generated_smiles: List[str], 
                        reference_smiles: Optional[List[str]] = None) -> Dict[str, Any]:
        if reference_smiles is None:
            reference_smiles = self.reference_smiles
        
        if not reference_smiles:
            return {
                'total_generated': len(generated_smiles),
                'novel_molecules': len(generated_smiles),
                'novelty_rate': 1.0,
                'novel_smiles': generated_smiles
            }
        
        reference_set = set(reference_smiles)
        novel_smiles = [smiles for smiles in generated_smiles if smiles not in reference_set]
        
        novelty_rate = len(novel_smiles) / len(generated_smiles) if len(generated_smiles) > 0 else 0
        
        return {
            'total_generated': len(generated_smiles),
            'reference_molecules': len(reference_set),
            'novel_molecules': len(novel_smiles),
            'novelty_rate': novelty_rate,
            'novel_smiles': novel_smiles
        }
    
    def evaluate_scaffold_diversity(self, smiles_list: List[str]) -> Dict[str, Any]:
        scaffolds = self._extract_scaffolds(smiles_list)
        
        scaffold_counter = Counter(scaffolds)
        unique_scaffolds = len(scaffold_counter)
        total_molecules = len(smiles_list)
        
        scaffold_diversity = unique_scaffolds / total_molecules if total_molecules > 0 else 0
        
        gini_coefficient = self._calculate_gini_coefficient(list(scaffold_counter.values()))
        
        most_common_scaffolds = scaffold_counter.most_common(10)
        
        return {
            'total_molecules': total_molecules,
            'unique_scaffolds': unique_scaffolds,
            'scaffold_diversity': scaffold_diversity,
            'gini_coefficient': gini_coefficient,
            'most_common_scaffolds': most_common_scaffolds,
            'scaffold_distribution': dict(scaffold_counter)
        }
    
    def evaluate_functional_groups(self, smiles_list: List[str]) -> Dict[str, Any]:
        functional_groups = {}
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            groups = self._detect_functional_groups(mol)
            for group in groups:
                functional_groups[group] = functional_groups.get(group, 0) + 1
        
        total_molecules = len([s for s in smiles_list if Chem.MolFromSmiles(s) is not None])
        
        coverage = {group: count / total_molecules for group, count in functional_groups.items()}
        
        return {
            'total_molecules': total_molecules,
            'functional_groups': functional_groups,
            'coverage': coverage,
            'unique_groups': len(functional_groups)
        }
    
    def _extract_scaffolds(self, smiles_list: List[str]) -> List[str]:
        scaffolds = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    scaffolds.append(scaffold_smiles)
                except:
                    scaffolds.append('invalid')
            else:
                scaffolds.append('invalid')
        
        return scaffolds
    
    def _compute_fingerprints(self, smiles_list: List[str]) -> List[Any]:
        fingerprints = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fingerprints.append(fp)
                except:
                    fingerprints.append(None)
            else:
                fingerprints.append(None)
        
        return fingerprints
    
    def _calculate_gini_coefficient(self, values: List[int]) -> float:
        if len(values) == 0:
            return 0.0
        
        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _detect_functional_groups(self, mol: Chem.Mol) -> List[str]:
        groups = []
        
        functional_group_patterns = {
            'hydroxyl': '[OH]',
            'carbonyl': '[CX3]=[OX1]',
            'carboxyl': '[CX3](=[OX1])[OX2H1]',
            'amine': '[NX3;H2,H1;!$(NC=O)]',
            'amide': '[NX3][CX3](=[OX1])',
            'ester': '[CX3](=[OX1])[OX2H0][#6]',
            'ether': '[OD2]([#6])[#6]',
            'thiol': '[SH]',
            'sulfide': '[SD2]([#6])[#6]',
            'nitro': '[NX3+](=O)[O-]',
            'halide': '[F,Cl,Br,I]',
            'benzene': '[c]1[c][c][c][c][c]1',
            'pyridine': '[n]1[c][c][c][c][c]1'
        }
        
        for group_name, pattern in functional_group_patterns.items():
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is not None:
                if mol.HasSubstructMatch(pattern_mol):
                    groups.append(group_name)
        
        return groups
    
    def comprehensive_evaluation(self, generated_smiles: List[str]) -> Dict[str, Any]:
        print("Starting comprehensive chemical evaluation...")
        
        print("Evaluating molecular validity...")
        validity_results = self.evaluate_validity(generated_smiles)
        
        print("Evaluating molecular uniqueness...")
        uniqueness_results = self.evaluate_uniqueness(generated_smiles)
        
        print("Evaluating molecular novelty...")
        novelty_results = self.evaluate_novelty(generated_smiles)
        
        print("Evaluating scaffold diversity...")
        scaffold_results = self.evaluate_scaffold_diversity(generated_smiles)
        
        print("Evaluating functional group distribution...")
        functional_group_results = self.evaluate_functional_groups(generated_smiles)
        
        return {
            'validity': validity_results,
            'uniqueness': uniqueness_results,
            'novelty': novelty_results,
            'scaffold_diversity': scaffold_results,
            'functional_groups': functional_group_results
        }
    
    def plot_chemical_evaluation(self, evaluation_results: Dict[str, Any],
                                save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        validity_data = [
            evaluation_results['validity']['validity_rate'],
            1 - evaluation_results['validity']['validity_rate']
        ]
        axes[0, 0].pie(validity_data, labels=['Valid', 'Invalid'], autopct='%1.1f%%')
        axes[0, 0].set_title('Molecular Validity')
        
        uniqueness_data = [
            evaluation_results['uniqueness']['uniqueness_rate'],
            1 - evaluation_results['uniqueness']['uniqueness_rate']
        ]
        axes[0, 1].pie(uniqueness_data, labels=['Unique', 'Duplicate'], autopct='%1.1f%%')
        axes[0, 1].set_title('Molecular Uniqueness')
        
        novelty_data = [
            evaluation_results['novelty']['novelty_rate'],
            1 - evaluation_results['novelty']['novelty_rate']
        ]
        axes[0, 2].pie(novelty_data, labels=['Novel', 'Known'], autopct='%1.1f%%')
        axes[0, 2].set_title('Molecular Novelty')
        
        # scaffold distribution (top 10)
        scaffold_data = evaluation_results['scaffold_diversity']['most_common_scaffolds'][:10]
        scaffold_names = [f'Scaffold {i+1}' for i in range(len(scaffold_data))]
        scaffold_counts = [count for _, count in scaffold_data]
        
        axes[1, 0].bar(scaffold_names, scaffold_counts, alpha=0.7)
        axes[1, 0].set_title('Top 10 Scaffolds')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # functional group distribution (top 10)
        functional_data = evaluation_results['functional_groups']['functional_groups']
        sorted_groups = sorted(functional_data.items(), key=lambda x: x[1], reverse=True)[:10]
        group_names = [name for name, _ in sorted_groups]
        group_counts = [count for _, count in sorted_groups]
        
        axes[1, 1].bar(group_names, group_counts, alpha=0.7, color='green')
        axes[1, 1].set_title('Top 10 Functional Groups')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # evaluation metrics summary
        metrics = [
            evaluation_results['validity']['validity_rate'],
            evaluation_results['uniqueness']['uniqueness_rate'],
            evaluation_results['novelty']['novelty_rate'],
            evaluation_results['scaffold_diversity']['scaffold_diversity']
        ]
        metric_names = ['Validity', 'Uniqueness', 'Novelty', 'Scaffold Diversity']
        
        bars = axes[1, 2].bar(metric_names, metrics, alpha=0.7, color=['blue', 'green', 'orange', 'purple'])
        axes[1, 2].set_title('Evaluation Metrics Summary')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # add values to the bars
        for bar, metric in zip(bars, metrics):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{metric:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_table(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        summary_data = []
        
        # validity
        validity = evaluation_results['validity']
        summary_data.append({
            'Metric': 'Validity Rate',
            'Value': f"{validity['validity_rate']:.3f}",
            'Description': f"{validity['valid_molecules']}/{validity['total_molecules']} molecules are chemically valid"
        })
        
        # uniqueness
        uniqueness = evaluation_results['uniqueness']
        summary_data.append({
            'Metric': 'Uniqueness Rate',
            'Value': f"{uniqueness['uniqueness_rate']:.3f}",
            'Description': f"{uniqueness['unique_molecules']}/{uniqueness['total_molecules']} molecules are unique"
        })
        
        # novelty
        novelty = evaluation_results['novelty']
        summary_data.append({
            'Metric': 'Novelty Rate',
            'Value': f"{novelty['novelty_rate']:.3f}",
            'Description': f"{novelty['novel_molecules']}/{novelty['total_generated']} molecules are novel"
        })
        
        # scaffold diversity
        scaffold = evaluation_results['scaffold_diversity']
        summary_data.append({
            'Metric': 'Scaffold Diversity',
            'Value': f"{scaffold['scaffold_diversity']:.3f}",
            'Description': f"{scaffold['unique_scaffolds']}/{scaffold['total_molecules']} unique scaffolds"
        })
        
        # functional groups
        functional = evaluation_results['functional_groups']
        summary_data.append({
            'Metric': 'Functional Groups',
            'Value': f"{functional['unique_groups']}",
            'Description': f"{functional['unique_groups']} different functional groups detected"
        })
        
        return pd.DataFrame(summary_data)

def run_chemical_evaluation(generated_smiles: List[str],
                           reference_smiles: Optional[List[str]] = None,
                           output_dir: str = 'experiments/outputs') -> Dict[str, Any]:
    evaluator = ChemicalEvaluator(reference_smiles)
    
    # comprehensive evaluation
    evaluation_results = evaluator.comprehensive_evaluation(generated_smiles)
    
    # generate summary table
    summary_table = evaluator.generate_summary_table(evaluation_results)
    
    # save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    summary_table.to_csv(f"{output_dir}/chemical_evaluation_summary.csv", index=False)
    
    # save detailed results
    import json
    with open(f"{output_dir}/chemical_evaluation_detailed.json", 'w') as f:
        # convert unserializable objects
        json_results = {}
        for key, value in evaluation_results.items():
            if key == 'scaffold_diversity':
                json_results[key] = {
                    'total_molecules': value['total_molecules'],
                    'unique_scaffolds': value['unique_scaffolds'],
                    'scaffold_diversity': value['scaffold_diversity'],
                    'gini_coefficient': value['gini_coefficient'],
                    'most_common_scaffolds': value['most_common_scaffolds']
                }
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    # plot results
    plot_path = f"{output_dir}/chemical_evaluation_plots.png"
    evaluator.plot_chemical_evaluation(evaluation_results, plot_path)
    
    return {
        'evaluation_results': evaluation_results,
        'summary_table': summary_table
    }


if __name__ == "__main__":
    test_smiles = [
        'CCO',  # ethanol
        'CC(C)O',  # isopropanol
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CCc1ccccc1',  # ethylbenzene
        'CCO',  # duplicate
        'CC(C)O',  # duplicate
    ]
    
    evaluator = ChemicalEvaluator()
    results = evaluator.comprehensive_evaluation(test_smiles)
    
    print("Chemical evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")
