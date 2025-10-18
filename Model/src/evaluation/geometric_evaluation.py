import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Geometry import rdGeometry

# Geometric Evaluator
class GeometricEvaluator:

    
    def __init__(self, bins: int = 50):
        self.bins = bins
        self.bond_length_bins = np.linspace(0.5, 3.0, bins)
        self.bond_angle_bins = np.linspace(0, np.pi, bins)
        self.torsion_angle_bins = np.linspace(-np.pi, np.pi, bins)
    
    # evaluate bond distance
    def evaluate_bond_distances(self, molecules: List[Any]) -> Dict[str, np.ndarray]:
        bond_distances = [] 
        
        for mol in molecules:
            if isinstance(mol, str):  # SMILES
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
                continue
            
            elif hasattr(mol, 'protein_pos') and hasattr(mol, 'ligand_pos'):
                ligand_pos = mol.ligand_pos
                if ligand_pos is not None:
                    distances = self._calculate_bond_distances_from_positions(ligand_pos)
                    bond_distances.extend(distances)
        
        return {
            'bond_distances': np.array(bond_distances),
            'histogram': np.histogram(bond_distances, bins=self.bond_length_bins)[0]
        }
    
    # evaluate bond angles
    def evaluate_bond_angles(self, molecules: List[Any]) -> Dict[str, np.ndarray]:
        bond_angles = []
        
        for mol in molecules:
            if isinstance(mol, str):  # SMILES
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
                continue
            
            elif hasattr(mol, 'protein_pos') and hasattr(mol, 'ligand_pos'):
                ligand_pos = mol.ligand_pos
                if ligand_pos is not None:
                    angles = self._calculate_bond_angles_from_positions(ligand_pos)
                    bond_angles.extend(angles)
        
        return {
            'bond_angles': np.array(bond_angles),
            'histogram': np.histogram(bond_angles, bins=self.bond_angle_bins)[0]
        }
    
    # evaluate torsion angles
    def evaluate_torsion_angles(self, molecules: List[Any]) -> Dict[str, np.ndarray]:
        torsion_angles = []
        
        for mol in molecules:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
                continue
            
            elif hasattr(mol, 'protein_pos') and hasattr(mol, 'ligand_pos'):
                ligand_pos = mol.ligand_pos
                if ligand_pos is not None:
                    torsions = self._calculate_torsion_angles_from_positions(ligand_pos)
                    torsion_angles.extend(torsions)
        
        return {
            'torsion_angles': np.array(torsion_angles),
            'histogram': np.histogram(torsion_angles, bins=self.torsion_angle_bins)[0]
        }
    
    # claculate bond distances from positions
    def _calculate_bond_distances_from_positions(self, positions: torch.Tensor) -> List[float]:
        distances = []
        pos_np = positions.numpy()
        
        for i in range(len(pos_np)):
            for j in range(i + 1, len(pos_np)):
                dist = np.linalg.norm(pos_np[i] - pos_np[j])
                # only consider reasonable bond length range (0.5-3.0 Å)
                if 0.5 <= dist <= 3.0:
                    distances.append(dist)
        
        return distances
    
    # claculate bond angles from positions
    def _calculate_bond_angles_from_positions(self, positions: torch.Tensor) -> List[float]:
        angles = []
        pos_np = positions.numpy()
        

        for i in range(len(pos_np)):
            distances = [np.linalg.norm(pos_np[i] - pos_np[j]) for j in range(len(pos_np)) if j != i]
            if len(distances) >= 2:
                nearest_indices = np.argsort(distances)[:2]
                if len(nearest_indices) >= 2:
                    j, k = nearest_indices[0], nearest_indices[1]
                    if j != k:
                        v1 = pos_np[j] - pos_np[i]
                        v2 = pos_np[k] - pos_np[i]
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
        
        return angles
    
    def _calculate_torsion_angles_from_positions(self, positions: torch.Tensor) -> List[float]:
        torsions = []
        pos_np = positions.numpy()
        
        for i in range(len(pos_np) - 3):
            for j in range(i + 1, len(pos_np) - 2):
                for k in range(j + 1, len(pos_np) - 1):
                    for l in range(k + 1, len(pos_np)):
                        torsion = self._calculate_dihedral_angle(
                            pos_np[i], pos_np[j], pos_np[k], pos_np[l]
                        )
                        if not np.isnan(torsion):
                            torsions.append(torsion)
        
        return torsions
    
    def _calculate_dihedral_angle(self, p1: np.ndarray, p2: np.ndarray, 
                                 p3: np.ndarray, p4: np.ndarray) -> float:
        try:
            v1 = p2 - p1
            v2 = p3 - p2
            v3 = p4 - p3
            
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            
            n1_norm = np.linalg.norm(n1)
            n2_norm = np.linalg.norm(n2)
            
            if n1_norm == 0 or n2_norm == 0:
                return np.nan
            
            n1 = n1 / n1_norm
            n2 = n2 / n2_norm
            
            cos_angle = np.dot(n1, n2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            sign = np.sign(np.dot(np.cross(n1, n2), v2))
            angle = sign * np.arccos(cos_angle)
            
            return angle
        except:
            return np.nan
    
    # calculate Jensen-Shannon divergence
    def calculate_js_divergence(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        p = dist1 / np.sum(dist1) if np.sum(dist1) > 0 else dist1
        q = dist2 / np.sum(dist2) if np.sum(dist2) > 0 else dist2
        
        p = p + 1e-10
        q = q + 1e-10
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        
        js_div = 0.5 * (kl_pq + kl_qp)
        
        return js_div
    
    def compare_geometric_distributions(self, generated_molecules: List[Any],
                                      reference_molecules: List[Any]) -> Dict[str, Any]:
        results = {}
        
        # bond distance distribution
        gen_bond_dist = self.evaluate_bond_distances(generated_molecules)
        ref_bond_dist = self.evaluate_bond_distances(reference_molecules)
        
        if len(gen_bond_dist['bond_distances']) > 0 and len(ref_bond_dist['bond_distances']) > 0:
            bond_js_div = self.calculate_js_divergence(
                gen_bond_dist['histogram'], ref_bond_dist['histogram']
            )
            
            results['bond_distance'] = {
                'generated_histogram': gen_bond_dist['histogram'],
                'reference_histogram': ref_bond_dist['histogram'],
                'js_divergence': bond_js_div,
                'generated_mean': np.mean(gen_bond_dist['bond_distances']),
                'generated_std': np.std(gen_bond_dist['bond_distances']),
                'reference_mean': np.mean(ref_bond_dist['bond_distances']),
                'reference_std': np.std(ref_bond_dist['bond_distances'])
            }
        
        # bond angle distribution
        gen_bond_angles = self.evaluate_bond_angles(generated_molecules)
        ref_bond_angles = self.evaluate_bond_angles(reference_molecules)
        
        if len(gen_bond_angles['bond_angles']) > 0 and len(ref_bond_angles['bond_angles']) > 0:
            angle_js_div = self.calculate_js_divergence(
                gen_bond_angles['histogram'], ref_bond_angles['histogram']
            )
            
            results['bond_angle'] = {
                'generated_histogram': gen_bond_angles['histogram'],
                'reference_histogram': ref_bond_angles['histogram'],
                'js_divergence': angle_js_div,
                'generated_mean': np.mean(gen_bond_angles['bond_angles']),
                'generated_std': np.std(gen_bond_angles['bond_angles']),
                'reference_mean': np.mean(ref_bond_angles['bond_angles']),
                'reference_std': np.std(ref_bond_angles['bond_angles'])
            }
        
        # torsion angle distribution
        gen_torsion = self.evaluate_torsion_angles(generated_molecules)
        ref_torsion = self.evaluate_torsion_angles(reference_molecules)
        
        if len(gen_torsion['torsion_angles']) > 0 and len(ref_torsion['torsion_angles']) > 0:
            torsion_js_div = self.calculate_js_divergence(
                gen_torsion['histogram'], ref_torsion['histogram']
            )
            
            results['torsion_angle'] = {
                'generated_histogram': gen_torsion['histogram'],
                'reference_histogram': ref_torsion['histogram'],
                'js_divergence': torsion_js_div,
                'generated_mean': np.mean(gen_torsion['torsion_angles']),
                'generated_std': np.std(gen_torsion['torsion_angles']),
                'reference_mean': np.mean(ref_torsion['torsion_angles']),
                'reference_std': np.std(ref_torsion['torsion_angles'])
            }
        
        return results
    
    def plot_geometric_distributions(self, comparison_results: Dict[str, Any],
                                   save_path: Optional[str] = None):
        n_plots = len(comparison_results)
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for i, (geom_type, results) in enumerate(comparison_results.items()):
            ax = axes[i]
            
            if geom_type == 'bond_distance':
                bins = self.bond_length_bins
                xlabel = 'Bond Distance (Å)'
            elif geom_type == 'bond_angle':
                bins = self.bond_angle_bins
                xlabel = 'Bond Angle (radians)'
            elif geom_type == 'torsion_angle':
                bins = self.torsion_angle_bins
                xlabel = 'Torsion Angle (radians)'
            else:
                continue
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            ax.plot(bin_centers, results['reference_histogram'], 
                   label='Reference', linewidth=2, alpha=0.8)
            ax.plot(bin_centers, results['generated_histogram'], 
                   label='Generated', linewidth=2, alpha=0.8)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Count')
            ax.set_title(f'{geom_type.replace("_", " ").title()} Distribution\n'
                        f'JS Divergence: {results["js_divergence"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_geometric_summary(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        summary_data = []
        
        for geom_type, results in comparison_results.items():
            summary_data.append({
                'Geometric_Property': geom_type.replace('_', ' ').title(),
                'JS_Divergence': f"{results['js_divergence']:.4f}",
                'Generated_Mean': f"{results['generated_mean']:.3f}",
                'Generated_Std': f"{results['generated_std']:.3f}",
                'Reference_Mean': f"{results['reference_mean']:.3f}",
                'Reference_Std': f"{results['reference_std']:.3f}"
            })
        
        return pd.DataFrame(summary_data)

def run_geometric_evaluation(generated_molecules: List[Any],
                           reference_molecules: List[Any],
                           output_dir: str = 'experiments/outputs',
                           bins: int = 50) -> Dict[str, Any]:
    evaluator = GeometricEvaluator(bins=bins)
    
    print("Evaluating geometric distributions...")
    comparison_results = evaluator.compare_geometric_distributions(
        generated_molecules, reference_molecules
    )
    
    summary_table = evaluator.generate_geometric_summary(comparison_results)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    summary_table.to_csv(f"{output_dir}/geometric_evaluation_summary.csv", index=False)
    

    plot_path = f"{output_dir}/geometric_distributions.png"
    evaluator.plot_geometric_distributions(comparison_results, plot_path)
    
    import json
    with open(f"{output_dir}/geometric_evaluation_detailed.json", 'w') as f:
        json_results = {}
        for key, value in comparison_results.items():
            json_results[key] = {
                'js_divergence': float(value['js_divergence']),
                'generated_mean': float(value['generated_mean']),
                'generated_std': float(value['generated_std']),
                'reference_mean': float(value['reference_mean']),
                'reference_std': float(value['reference_std'])
            }
        json.dump(json_results, f, indent=2)
    
    return {
        'comparison_results': comparison_results,
        'summary_table': summary_table
    }

if __name__ == "__main__":
    # 测试代码
    evaluator = GeometricEvaluator()
    
    # 创建测试数据
    test_positions = torch.randn(10, 3)  # 10个原子的3D坐标
    
    # 测试键长计算
    distances = evaluator._calculate_bond_distances_from_positions(test_positions)
    print(f"计算得到 {len(distances)} 个键长")
    print(distances)
    
    # 测试键角计算
    angles = evaluator._calculate_bond_angles_from_positions(test_positions)
    print(f"计算得到 {len(angles)} 个键角")
    print(angles)
    
    # 测试二面角计算
    torsions = evaluator._calculate_torsion_angles_from_positions(test_positions)
    print(f"计算得到 {len(torsions)} 个二面角")
