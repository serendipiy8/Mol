import numpy as np
import pandas as pd
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import json

class DockingEvaluator:
    
    def __init__(self, vina_path: str = 'vina', 
                 high_affinity_threshold: float = -7.0,
                 success_rate_threshold: float = -6.0):
        self.vina_path = vina_path
        self.high_affinity_threshold = high_affinity_threshold
        self.success_rate_threshold = success_rate_threshold
    
    # The function is prepared for SMILES, if 3D-Generation we can omit it.
    def prepare_ligand(self, smiles: str, output_path: str) -> bool:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            mol = Chem.AddHs(mol)
            
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            
            with Chem.PDBWriter(output_path) as writer:
                writer.write(mol)
            
            return True
        except Exception as e:
            print(f"Error preparing ligand {smiles}: {e}")
            return False

    def prepare_ligand_from_mol(self, mol: Chem.Mol, output_path: str,
                                sanitize: bool = True, add_h: bool = True,
                                embed_if_missing: bool = True, optimize: bool = True) -> bool:
        try:
            if mol is None:
                return False

            if sanitize:
                Chem.SanitizeMol(mol)
            if add_h:
                mol = Chem.AddHs(mol, addCoords=True)

            has_conf = mol.GetNumConformers() > 0
            if not has_conf and embed_if_missing:
                try:
                    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                    has_conf = mol.GetNumConformers() > 0
                except Exception:
                    has_conf = mol.GetNumConformers() > 0

            if optimize and has_conf:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass

            with Chem.PDBWriter(output_path) as writer:
                writer.write(mol)

            return True
        except Exception as e:
            print(f"Error preparing ligand from Mol: {e}")
            return False
    
    def run_vina_docking(self, ligand_path: str, receptor_path: str,
                        center_x: float, center_y: float, center_z: float,
                        size_x: float = 20.0, size_y: float = 20.0, size_z: float = 20.0,
                        exhaustiveness: int = 32, num_modes: int = 9) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as temp_out:
                output_path = temp_out.name
            
            cmd = [
                self.vina_path,
                '--receptor', receptor_path,
                '--ligand', ligand_path,
                '--out', output_path,
                '--center_x', str(center_x),
                '--center_y', str(center_y),
                '--center_z', str(center_z),
                '--size_x', str(size_x),
                '--size_y', str(size_y),
                '--size_z', str(size_z),
                '--exhaustiveness', str(exhaustiveness),
                '--num_modes', str(num_modes)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Vina error: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            docking_results = self._parse_vina_output(output_path, result.stdout)
            
            os.unlink(output_path)
            
            return docking_results
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Docking timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _parse_vina_output(self, output_path: str, stdout: str) -> Dict[str, Any]:
        results = {
            'success': True,
            'binding_affinities': [],
            'best_affinity': None,
            'mean_affinity': None,
            'std_affinity': None,
            'num_modes': 0
        }
        
        lines = stdout.split('\n')
        affinities = []
        
        for line in lines:
            if 'REMARK VINA RESULT:' in line:
                try:
                    parts = line.split()
                    affinity = float(parts[3])
                    affinities.append(affinity)
                except (IndexError, ValueError):
                    continue
        
        if affinities:
            results['binding_affinities'] = affinities
            results['best_affinity'] = min(affinities) 
            results['mean_affinity'] = np.mean(affinities)
            results['std_affinity'] = np.std(affinities)
            results['num_modes'] = len(affinities)
        
        return results
    
    def evaluate_molecules(self, smiles_list: List[str], 
                          receptor_paths: List[str],
                          binding_sites: List[Dict[str, float]],
                          exhaustiveness: int = 32) -> pd.DataFrame:
        results = []
        
        for i, smiles in enumerate(smiles_list):
            print(f"docking molecular {i+1}/{len(smiles_list)}: {smiles}")
            
            with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as temp_ligand:
                ligand_path = temp_ligand.name
            
            if not self.prepare_ligand(smiles, ligand_path):
                results.append({
                    'smiles': smiles,
                    'index': i,
                    'success': False,
                    'error': 'Failed to prepare ligand'
                })
                continue
            
            best_affinity = float('inf')
            all_affinities = []
            successful_dockings = 0
            
            for j, (receptor_path, binding_site) in enumerate(zip(receptor_paths, binding_sites)):
                docking_result = self.run_vina_docking(
                    ligand_path, receptor_path,
                    binding_site['center_x'], binding_site['center_y'], binding_site['center_z'],
                    binding_site.get('size_x', 20.0), binding_site.get('size_y', 20.0), 
                    binding_site.get('size_z', 20.0),
                    exhaustiveness=exhaustiveness
                )
                
                if docking_result['success'] and docking_result['binding_affinities']:
                    all_affinities.extend(docking_result['binding_affinities'])
                    best_affinity = min(best_affinity, min(docking_result['binding_affinities']))
                    successful_dockings += 1
            
            os.unlink(ligand_path)
            
            if all_affinities:
                results.append({
                    'smiles': smiles,
                    'index': i,
                    'success': True,
                    'best_affinity': best_affinity,
                    'mean_affinity': np.mean(all_affinities),
                    'std_affinity': np.std(all_affinities),
                    'num_dockings': len(all_affinities),
                    'successful_receptors': successful_dockings,
                    'high_affinity': best_affinity <= self.high_affinity_threshold,
                    'successful_docking': best_affinity <= self.success_rate_threshold
                })
            else:
                results.append({
                    'smiles': smiles,
                    'index': i,
                    'success': False,
                    'error': 'No successful dockings'
                })
        
        return pd.DataFrame(results)

    def evaluate_molecules_from_mols(self, mol_list: List[Chem.Mol], receptor_paths: List[str],
                                     binding_sites: List[Dict[str, float]], exhaustiveness: int = 32) -> pd.DataFrame:
        results = []
        
        for i, mol in enumerate(mol_list):
            print(f"docking molecular {i+1}/{len(mol_list)} (Mol)")
            
            with tempfile.NamedTemporaryFile(suffix='.pdbqt', delete=False) as temp_ligand:
                ligand_path = temp_ligand.name
            
            if not self.prepare_ligand_from_mol(mol, ligand_path):
                results.append({
                    'index': i,
                    'success': False,
                    'error': 'Failed to prepare ligand from Mol'
                })
                continue
            
            best_affinity = float('inf')
            all_affinities = []
            successful_dockings = 0
            
            for j, (receptor_path, binding_site) in enumerate(zip(receptor_paths, binding_sites)):
                docking_result = self.run_vina_docking(
                    ligand_path, receptor_path,
                    binding_site['center_x'], binding_site['center_y'], binding_site['center_z'],
                    binding_site.get('size_x', 20.0), binding_site.get('size_y', 20.0), 
                    binding_site.get('size_z', 20.0),
                    exhaustiveness=exhaustiveness
                )
                
                if docking_result['success'] and docking_result['binding_affinities']:
                    all_affinities.extend(docking_result['binding_affinities'])
                    best_affinity = min(best_affinity, min(docking_result['binding_affinities']))
                    successful_dockings += 1
            
            os.unlink(ligand_path)
            
            if all_affinities:
                results.append({
                    'index': i,
                    'success': True,
                    'best_affinity': best_affinity,
                    'mean_affinity': np.mean(all_affinities),
                    'std_affinity': np.std(all_affinities),
                    'num_dockings': len(all_affinities),
                    'successful_receptors': successful_dockings,
                    'high_affinity': best_affinity <= self.high_affinity_threshold,
                    'successful_docking': best_affinity <= self.success_rate_threshold
                })
            else:
                results.append({
                    'index': i,
                    'success': False,
                    'error': 'No successful dockings'
                })
        
        return pd.DataFrame(results)
    
    def calculate_docking_metrics(self, docking_results: pd.DataFrame) -> Dict[str, Any]:
        if len(docking_results) == 0:
            return {}
        
        successful_results = docking_results[docking_results['success'] == True]
        
        if len(successful_results) == 0:
            return {
                'total_molecules': len(docking_results),
                'successful_dockings': 0,
                'success_rate': 0.0,
                'high_affinity_rate': 0.0,
                'mean_affinity': None,
                'std_affinity': None
            }
        
        metrics = {
            'total_molecules': len(docking_results),
            'successful_dockings': len(successful_results),
            'success_rate': len(successful_results) / len(docking_results),
            'high_affinity_rate': successful_results['high_affinity'].sum() / len(successful_results),
            'successful_docking_rate': successful_results['successful_docking'].sum() / len(successful_results),
            'mean_affinity': successful_results['best_affinity'].mean(),
            'std_affinity': successful_results['best_affinity'].std(),
            'median_affinity': successful_results['best_affinity'].median(),
            'min_affinity': successful_results['best_affinity'].min(),
            'max_affinity': successful_results['best_affinity'].max()
        }
        
        return metrics
    
    def compare_with_baselines(self, our_results: pd.DataFrame,
                             baseline_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        comparison = {}
        
        our_metrics = self.calculate_docking_metrics(our_results)
        
        for baseline_name, baseline_df in baseline_results.items():
            baseline_metrics = self.calculate_docking_metrics(baseline_df)
            
            comparison[baseline_name] = {
                'our_metrics': our_metrics,
                'baseline_metrics': baseline_metrics,
                'improvement': {
                    'success_rate': our_metrics.get('success_rate', 0) - baseline_metrics.get('success_rate', 0),
                    'high_affinity_rate': our_metrics.get('high_affinity_rate', 0) - baseline_metrics.get('high_affinity_rate', 0),
                    'mean_affinity': baseline_metrics.get('mean_affinity', 0) - our_metrics.get('mean_affinity', 0)  # 越低越好
                }
            }
        
        return comparison
    
    def plot_docking_results(self, docking_results: pd.DataFrame,
                           comparison_results: Dict[str, Any] = None,
                           save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        successful_results = docking_results[docking_results['success'] == True]
        if len(successful_results) > 0:
            axes[0, 0].hist(successful_results['best_affinity'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(self.high_affinity_threshold, color='red', linestyle='--', 
                             label=f'High Affinity Threshold ({self.high_affinity_threshold})')
            axes[0, 0].axvline(self.success_rate_threshold, color='orange', linestyle='--',
                             label=f'Success Threshold ({self.success_rate_threshold})')
            axes[0, 0].set_xlabel('Binding Affinity (kcal/mol)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Binding Affinity Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率比较
        if comparison_results:
            # 计算当前结果的整体指标以便对比
            our_metrics = self.calculate_docking_metrics(docking_results)
            models = list(comparison_results.keys()) + ['Our Model']
            success_rates = [comparison_results[model]['baseline_metrics'].get('success_rate', 0) 
                           for model in comparison_results.keys()]
            success_rates.append(our_metrics.get('success_rate', 0))
            
            axes[0, 1].bar(models, success_rates, alpha=0.7)
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title('Docking Success Rate Comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 高亲和力率比较
        if comparison_results:
            high_affinity_rates = [comparison_results[model]['baseline_metrics'].get('high_affinity_rate', 0) 
                                 for model in comparison_results.keys()]
            high_affinity_rates.append(our_metrics.get('high_affinity_rate', 0))
            
            axes[1, 0].bar(models, high_affinity_rates, alpha=0.7, color='green')
            axes[1, 0].set_ylabel('High Affinity Rate')
            axes[1, 0].set_title('High Affinity Rate Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 平均结合亲和力比较
        if comparison_results:
            mean_affinities = [comparison_results[model]['baseline_metrics'].get('mean_affinity', 0) 
                             for model in comparison_results.keys()]
            mean_affinities.append(our_metrics.get('mean_affinity', 0))
            
            axes[1, 1].bar(models, mean_affinities, alpha=0.7, color='purple')
            axes[1, 1].set_ylabel('Mean Binding Affinity (kcal/mol)')
            axes[1, 1].set_title('Mean Binding Affinity Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def run_docking_evaluation(smiles_list: List[str],
                          receptor_paths: List[str],
                          binding_sites: List[Dict[str, float]],
                          output_dir: str = 'experiments/outputs',
                          high_affinity_threshold: float = -7.0,
                          success_rate_threshold: float = -6.0) -> Dict[str, Any]:
    """运行对接评估"""
    evaluator = DockingEvaluator(
        high_affinity_threshold=high_affinity_threshold,
        success_rate_threshold=success_rate_threshold
    )
    
    print("开始分子对接评估...")
    docking_results = evaluator.evaluate_molecules(
        smiles_list, receptor_paths, binding_sites
    )
    
    # 计算指标
    metrics = evaluator.calculate_docking_metrics(docking_results)
    
    # 保存结果
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    docking_results.to_csv(f"{output_dir}/docking_evaluation_results.csv", index=False)
    
    # 保存指标
    with open(f"{output_dir}/docking_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # 绘制结果
    plot_path = f"{output_dir}/docking_evaluation_plots.png"
    evaluator.plot_docking_results(docking_results, save_path=plot_path)
    
    return {
        'docking_results': docking_results,
        'metrics': metrics
    }

if __name__ == "__main__":
    # 测试代码
    test_smiles = [
        'CCO',  # ethanol
        'CC(C)O',  # isopropanol
        'c1ccccc1',  # benzene
    ]
    
    # 假设的受体和结合位点
    receptor_paths = ['receptor1.pdbqt', 'receptor2.pdbqt']
    binding_sites = [
        {'center_x': 0.0, 'center_y': 0.0, 'center_z': 0.0},
        {'center_x': 1.0, 'center_y': 1.0, 'center_z': 1.0}
    ]
    
    evaluator = DockingEvaluator()
    print("对接评估器初始化完成")
