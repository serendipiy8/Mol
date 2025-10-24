import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class MolecularPropertyEvaluator:
    
    def __init__(self):
        self.property_functions = {
            'qed': self._calculate_qed,
            'sa_score': self._calculate_sa_score,
            'logp': self._calculate_logp,
            'molecular_weight': self._calculate_molecular_weight,
            'num_atoms': self._calculate_num_atoms,
            'num_rings': self._calculate_num_rings,
            'num_aromatic_rings': self._calculate_num_aromatic_rings,
            'num_rotatable_bonds': self._calculate_num_rotatable_bonds,
            'tpsa': self._calculate_tpsa,
            'hbd': self._calculate_hbd,
            'hba': self._calculate_hba,
            'lipinski_violations': self._calculate_lipinski_violations
        }
    
    def evaluate_molecules(self, smiles_list: List[str], 
                          properties: List[str] = None) -> pd.DataFrame:
        if properties is None:
            properties = list(self.property_functions.keys())
        
        results = []
        valid_smiles = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
                mol_props = {'smiles': smiles, 'index': i}
                
                for prop in properties:
                    if prop in self.property_functions:
                        try:
                            value = self.property_functions[prop](mol)
                            mol_props[prop] = value
                        except Exception as e:
                            mol_props[prop] = np.nan
                            print(f"Error calculating {prop} for {smiles}: {e}")
                
                results.append(mol_props)
            else:
                print(f"Invalid SMILES: {smiles}")
        
        return pd.DataFrame(results)
    
    # Compare with reference molecular
    def compare_with_reference(self, generated_df: pd.DataFrame, reference_df: pd.DataFrame,
                             properties: List[str] = None) -> Dict[str, Any]:
        if properties is None:
            properties = [col for col in generated_df.columns 
                         if col not in ['smiles', 'index']]
        
        comparison_results = {}
        
        for prop in properties:
            if prop in generated_df.columns and prop in reference_df.columns:
                gen_values = generated_df[prop].dropna()
                ref_values = reference_df[prop].dropna()
                
                if len(gen_values) > 0 and len(ref_values) > 0:
                    comparison_results[prop] = {
                        'generated_mean': gen_values.mean(),
                        'generated_std': gen_values.std(),
                        'reference_mean': ref_values.mean(),
                        'reference_std': ref_values.std(),
                        'ks_statistic': stats.ks_2samp(gen_values, ref_values)[0],
                        'ks_p_value': stats.ks_2samp(gen_values, ref_values)[1],
                        'mann_whitney_u': stats.mannwhitneyu(gen_values, ref_values)[1],
                        'effect_size': self._calculate_effect_size(gen_values, ref_values)
                    }
        
        return comparison_results
    
    # Quantitative Estimate of Drug-likeness
    def _calculate_qed(self, mol: Chem.Mol) -> float:
        return QED.qed(mol)
    
    # Synthetic Accessibility Score(未完成)
    def _calculate_sa_score(self, mol: Chem.Mol) -> float:

        try:
            from rdkit.Chem import rdMolDescriptors
            return rdMolDescriptors.CalcNumRotatableBonds(mol) 
        except:
            return np.nan
    
    # LogP
    def _calculate_logp(self, mol: Chem.Mol) -> float:
        return Crippen.MolLogP(mol)
    
    # Molecular Weight
    def _calculate_molecular_weight(self, mol: Chem.Mol) -> float:
        return Descriptors.MolWt(mol)
    
    # Number of Atoms
    def _calculate_num_atoms(self, mol: Chem.Mol) -> int:
        return mol.GetNumAtoms()
    
    # Number of Rings
    def _calculate_num_rings(self, mol: Chem.Mol) -> int:
        return Descriptors.RingCount(mol)
    
    # Number of Aromatic Rings
    def _calculate_num_aromatic_rings(self, mol: Chem.Mol) -> int:
        return Descriptors.NumAromaticRings(mol)
    
    # Number of Rotatable Bonds
    def _calculate_num_rotatable_bonds(self, mol: Chem.Mol) -> int:
        return Descriptors.NumRotatableBonds(mol)
    
    # Topological Polar Surface Area
    def _calculate_tpsa(self, mol: Chem.Mol) -> float:
        return Descriptors.TPSA(mol)
    
    # Hydrogen Bond Donor
    def _calculate_hbd(self, mol: Chem.Mol) -> int:
        return Descriptors.NumHDonors(mol)
    
    # Hydrogen Bond Acceptor
    def _calculate_hba(self, mol: Chem.Mol) -> int:
        return Descriptors.NumHAcceptors(mol)
    
    # Lipinski Rule Violations (RO5): HBD>5, HBA>10, MW>500, logP>5
    def _calculate_lipinski_violations(self, mol: Chem.Mol) -> int:
        try:
            hbd = Lipinski.NumHDonors(mol)
        except Exception:
            from rdkit.Chem import rdMolDescriptors
            hbd = rdMolDescriptors.CalcNumHBD(mol)
        try:
            hba = Lipinski.NumHAcceptors(mol)
        except Exception:
            from rdkit.Chem import rdMolDescriptors
            hba = rdMolDescriptors.CalcNumHBA(mol)
        try:
            mw = Descriptors.MolWt(mol)
        except Exception:
            mw = 0.0
        try:
            logp = Crippen.MolLogP(mol)
        except Exception:
            logp = 0.0
        violations = int(hbd > 5) + int(hba > 10) + int(mw > 500.0) + int(logp > 5.0)
        return violations

    # Effect Size(Cohen's d)
    def _calculate_effect_size(self, group1: np.ndarray, group2: np.ndarray) -> float:
        n1, n2 = len(group1), len(group2)
        s1, s2 = group1.std(), group2.std()
        s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        return (group1.mean() - group2.mean()) / s_pooled
    
    def plot_property_distributions(self, generated_df: pd.DataFrame,
                                   reference_df: pd.DataFrame,
                                   properties: List[str] = None,
                                   save_path: Optional[str] = None):
        if properties is None:
            properties = [col for col in generated_df.columns 
                         if col not in ['smiles', 'index']]
        
        n_props = len(properties)
        n_cols = min(3, n_props)
        n_rows = (n_props + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_props == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, prop in enumerate(properties):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if prop in generated_df.columns and prop in reference_df.columns:
                gen_values = generated_df[prop].dropna()
                ref_values = reference_df[prop].dropna()
                
                ax.hist(ref_values, bins=30, alpha=0.7, label='Reference', 
                       color='blue', density=True)
                ax.hist(gen_values, bins=30, alpha=0.7, label='Generated', 
                       color='red', density=True)
                
                ax.set_xlabel(prop)
                ax.set_ylabel('Density')
                ax.set_title(f'{prop} Distribution')
                ax.legend()
        
        for i in range(n_props, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_table(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        summary_data = []
        
        for prop, results in comparison_results.items():
            summary_data.append({
                'Property': prop,
                'Generated_Mean': f"{results['generated_mean']:.3f}",
                'Generated_Std': f"{results['generated_std']:.3f}",
                'Reference_Mean': f"{results['reference_mean']:.3f}",
                'Reference_Std': f"{results['reference_std']:.3f}",
                'KS_Statistic': f"{results['ks_statistic']:.3f}",
                'KS_P_Value': f"{results['ks_p_value']:.3f}",
                'Effect_Size': f"{results['effect_size']:.3f}"
            })
        
        return pd.DataFrame(summary_data)

def run_molecular_properties_evaluation(generated_smiles: List[str],
                                       reference_smiles: List[str],
                                       output_dir: str = 'experiments/outputs',
                                       properties: List[str] = None) -> Dict[str, Any]:
    evaluator = MolecularPropertyEvaluator()
    
    print("Evaluating generated molecules...")
    generated_df = evaluator.evaluate_molecules(generated_smiles, properties)
    
    print("Evaluating reference molecules...")
    reference_df = evaluator.evaluate_molecules(reference_smiles, properties)
    
    print("Comparing generated and reference molecules...")
    comparison_results = evaluator.compare_with_reference(
        generated_df, reference_df, properties
    )
    
    summary_table = evaluator.generate_summary_table(comparison_results)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generated_df.to_csv(f"{output_dir}/generated_molecular_properties.csv", index=False)
    reference_df.to_csv(f"{output_dir}/reference_molecular_properties.csv", index=False)
    summary_table.to_csv(f"{output_dir}/molecular_properties_summary.csv", index=False)
    
    plot_path = f"{output_dir}/molecular_properties_distributions.png"
    evaluator.plot_property_distributions(generated_df, reference_df, 
                                        properties, plot_path)
    
    return {
        'generated_properties': generated_df,
        'reference_properties': reference_df,
        'comparison_results': comparison_results,
        'summary_table': summary_table
    }