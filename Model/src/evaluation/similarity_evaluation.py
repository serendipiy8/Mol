import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem import DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class SimilarityEvaluator:
    
    def __init__(self, fingerprint_radius: int = 2, fingerprint_bits: int = 2048):
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
    
    def compute_molecular_fingerprints(self, smiles_list: List[str]) -> List[Any]:
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                        mol, self.fingerprint_radius, nBits=self.fingerprint_bits
                    )
                    fingerprints.append(fp)
                    valid_smiles.append(smiles)
                except Exception as e:
                    print(f"Error computing fingerprint for {smiles}: {e}")
                    fingerprints.append(None)
            else:
                fingerprints.append(None)
        
        return fingerprints, valid_smiles
    
    def compute_tanimoto_similarity(self, fingerprints: List[Any]) -> np.ndarray:
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if fingerprints[i] is not None and fingerprints[j] is not None:
                    similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                else:
                    similarity_matrix[i, j] = 0.0
                    similarity_matrix[j, i] = 0.0
        
        return similarity_matrix
    
    def compute_molecular_descriptors(self, smiles_list: List[str]) -> pd.DataFrame:
        descriptors = []
        valid_smiles = []
        
        descriptor_functions = {
            'molecular_weight': Descriptors.MolWt,
            'logp': Descriptors.MolLogP,
            'tpsa': Descriptors.TPSA,
            'num_atoms': Descriptors.HeavyAtomCount,
            'num_rings': Descriptors.RingCount,
            'num_aromatic_rings': Descriptors.NumAromaticRings,
            'num_rotatable_bonds': Descriptors.NumRotatableBonds,
            'num_hbd': Descriptors.NumHDonors,
            'num_hba': Descriptors.NumHAcceptors,
            'formal_charge': Descriptors.FormalCharge,
            'num_heteroatoms': Descriptors.NumHeteroatoms,
            'fraction_csp3': Descriptors.FractionCsp3,
            'num_saturated_rings': Descriptors.NumSaturatedRings,
            'num_aliphatic_rings': Descriptors.NumAliphaticRings
        }
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_desc = {'smiles': smiles}
                for desc_name, desc_func in descriptor_functions.items():
                    try:
                        mol_desc[desc_name] = desc_func(mol)
                    except:
                        mol_desc[desc_name] = np.nan
                descriptors.append(mol_desc)
                valid_smiles.append(smiles)
        
        return pd.DataFrame(descriptors), valid_smiles
    
    def evaluate_internal_similarity(self, smiles_list: List[str]) -> Dict[str, Any]:
        fingerprints, valid_smiles = self.compute_molecular_fingerprints(smiles_list)
        
        if len(valid_smiles) < 2:
            return {
                'mean_similarity': 0.0,
                'std_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'similarity_matrix': None
            }
        
        similarity_matrix = self.compute_tanimoto_similarity(fingerprints)
        
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarities = similarity_matrix[upper_tri_indices]
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'similarity_matrix': similarity_matrix,
            'valid_smiles': valid_smiles
        }
    
    def evaluate_external_similarity(self, generated_smiles: List[str], 
                                   reference_smiles: List[str]) -> Dict[str, Any]:
        gen_fingerprints, gen_valid = self.compute_molecular_fingerprints(generated_smiles)
        ref_fingerprints, ref_valid = self.compute_molecular_fingerprints(reference_smiles)
        
        if len(gen_valid) == 0 or len(ref_valid) == 0:
            return {
                'mean_similarity': 0.0,
                'std_similarity': 0.0,
                'similarity_distribution': None
            }
        
        similarities = []
        for gen_fp in gen_fingerprints:
            if gen_fp is not None:
                gen_similarities = []
                for ref_fp in ref_fingerprints:
                    if ref_fp is not None:
                        sim = DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
                        gen_similarities.append(sim)
                if gen_similarities:
                    similarities.append(np.mean(gen_similarities))
        
        similarities = np.array(similarities)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'similarity_distribution': similarities,
            'generated_valid': gen_valid,
            'reference_valid': ref_valid
        }
    
    def evaluate_diversity_metrics(self, smiles_list: List[str]) -> Dict[str, Any]:
        fingerprints, valid_smiles = self.compute_molecular_fingerprints(smiles_list)
        
        if len(valid_smiles) < 2:
            return {
                'internal_diversity': 0.0,
                'avg_pairwise_similarity': 0.0,
                'diversity_index': 0.0
            }
        
        similarity_matrix = self.compute_tanimoto_similarity(fingerprints)
        
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        pairwise_similarities = similarity_matrix[upper_tri_indices]
        avg_pairwise_similarity = np.mean(pairwise_similarities)
        
        internal_diversity = 1 - avg_pairwise_similarity
        
        similarity_hist, _ = np.histogram(pairwise_similarities, bins=20, range=(0, 1))
        similarity_hist = similarity_hist / np.sum(similarity_hist)
        similarity_hist = similarity_hist[similarity_hist > 0]  # 移除零值
        diversity_index = -np.sum(similarity_hist * np.log(similarity_hist))
        
        return {
            'internal_diversity': internal_diversity,
            'avg_pairwise_similarity': avg_pairwise_similarity,
            'diversity_index': diversity_index,
            'similarity_distribution': pairwise_similarities
        }
    
    def compare_with_baselines(self, our_results: Dict[str, Any],
                             baseline_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        comparison = {}
        
        for baseline_name, baseline_data in baseline_results.items():
            comparison[baseline_name] = {
                'our_internal_diversity': our_results.get('internal_diversity', 0),
                'baseline_internal_diversity': baseline_data.get('internal_diversity', 0),
                'our_external_similarity': our_results.get('external_similarity', {}).get('mean_similarity', 0),
                'baseline_external_similarity': baseline_data.get('external_similarity', {}).get('mean_similarity', 0),
                'improvement_in_diversity': our_results.get('internal_diversity', 0) - baseline_data.get('internal_diversity', 0),
                'improvement_in_similarity': our_results.get('external_similarity', {}).get('mean_similarity', 0) - baseline_data.get('external_similarity', {}).get('mean_similarity', 0)
            }
        
        return comparison
    
    def plot_similarity_analysis(self, similarity_results: Dict[str, Any],
                                save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        if 'internal_similarity' in similarity_results and similarity_results['internal_similarity']['similarity_matrix'] is not None:
            sim_matrix = similarity_results['internal_similarity']['similarity_matrix']
            im = axes[0, 0].imshow(sim_matrix, cmap='viridis', aspect='auto')
            axes[0, 0].set_title('Internal Similarity Matrix')
            axes[0, 0].set_xlabel('Molecule Index')
            axes[0, 0].set_ylabel('Molecule Index')
            plt.colorbar(im, ax=axes[0, 0])
        
        if 'external_similarity' in similarity_results and similarity_results['external_similarity']['similarity_distribution'] is not None:
            sim_dist = similarity_results['external_similarity']['similarity_distribution']
            axes[0, 1].hist(sim_dist, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('External Similarity Distribution')
            axes[0, 1].set_xlabel('Tanimoto Similarity')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True, alpha=0.3)
        
        if 'diversity_metrics' in similarity_results:
            metrics = similarity_results['diversity_metrics']
            metric_names = ['Internal Diversity', 'Avg Pairwise Similarity', 'Diversity Index']
            metric_values = [
                metrics['internal_diversity'],
                metrics['avg_pairwise_similarity'],
                metrics['diversity_index']
            ]
            
            bars = axes[1, 0].bar(metric_names, metric_values, alpha=0.7, 
                                 color=['blue', 'green', 'orange'])
            axes[1, 0].set_title('Diversity Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        if 'baseline_comparison' in similarity_results:
            baseline_data = similarity_results['baseline_comparison']
            models = list(baseline_data.keys()) + ['Our Model']
            
            diversity_values = [baseline_data[model]['baseline_internal_diversity'] 
                              for model in baseline_data.keys()]
            diversity_values.append(similarity_results['diversity_metrics']['internal_diversity'])
            
            axes[1, 1].bar(models, diversity_values, alpha=0.7, color='purple')
            axes[1, 1].set_title('Internal Diversity Comparison')
            axes[1, 1].set_ylabel('Internal Diversity')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_similarity_summary(self, similarity_results: Dict[str, Any]) -> pd.DataFrame:
        summary_data = []
        
        if 'internal_similarity' in similarity_results:
            internal = similarity_results['internal_similarity']
            summary_data.append({
                'Metric': 'Internal Similarity (Mean)',
                'Value': f"{internal['mean_similarity']:.3f}",
                'Description': f"Average pairwise similarity among generated molecules"
            })
            
            summary_data.append({
                'Metric': 'Internal Similarity (Std)',
                'Value': f"{internal['std_similarity']:.3f}",
                'Description': f"Standard deviation of pairwise similarities"
            })
        
        if 'external_similarity' in similarity_results:
            external = similarity_results['external_similarity']
            summary_data.append({
                'Metric': 'External Similarity (Mean)',
                'Value': f"{external['mean_similarity']:.3f}",
                'Description': f"Average similarity to reference molecules"
            })
        
        if 'diversity_metrics' in similarity_results:
            diversity = similarity_results['diversity_metrics']
            summary_data.append({
                'Metric': 'Internal Diversity',
                'Value': f"{diversity['internal_diversity']:.3f}",
                'Description': f"1 - average pairwise similarity"
            })
            
            summary_data.append({
                'Metric': 'Diversity Index',
                'Value': f"{diversity['diversity_index']:.3f}",
                'Description': f"Entropy-based diversity measure"
            })
        
        return pd.DataFrame(summary_data)

def run_similarity_evaluation(generated_smiles: List[str],
                            reference_smiles: Optional[List[str]] = None,
                            baseline_results: Optional[Dict[str, Dict[str, Any]]] = None,
                            output_dir: str = 'experiments/outputs') -> Dict[str, Any]:
    evaluator = SimilarityEvaluator()
    
    print("Evaluating internal similarity...")
    internal_similarity = evaluator.evaluate_internal_similarity(generated_smiles)
    
    print("Evaluating external similarity...")
    external_similarity = evaluator.evaluate_external_similarity(generated_smiles, reference_smiles or [])
    
    print("Evaluating diversity metrics...")
    diversity_metrics = evaluator.evaluate_diversity_metrics(generated_smiles)
    
    similarity_results = {
        'internal_similarity': internal_similarity,
        'external_similarity': external_similarity,
        'diversity_metrics': diversity_metrics
    }
    
    if baseline_results:
        print("Comparing with baselines...")
        baseline_comparison = evaluator.compare_with_baselines(similarity_results, baseline_results)
        similarity_results['baseline_comparison'] = baseline_comparison
    
    summary_table = evaluator.generate_similarity_summary(similarity_results)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    summary_table.to_csv(f"{output_dir}/similarity_evaluation_summary.csv", index=False)
    
    plot_path = f"{output_dir}/similarity_evaluation_plots.png"
    evaluator.plot_similarity_analysis(similarity_results, plot_path)
    
    return {
        'similarity_results': similarity_results,
        'summary_table': summary_table
    }

if __name__ == "__main__":
    # 测试代码
    test_smiles = [
        'CCO',  # ethanol
        'CC(C)O',  # isopropanol
        'CC(=O)O',  # acetic acid
        'c1ccccc1',  # benzene
        'CCc1ccccc1'  # ethylbenzene
    ]
    
    reference_smiles = [
        'CCO',  # ethanol
        'CC(C)O',  # isopropanol
    ]
    
    evaluator = SimilarityEvaluator()
    
    # 测试内部相似性
    internal_results = evaluator.evaluate_internal_similarity(test_smiles)
    print("内部相似性结果:", internal_results)
    
    # 测试外部相似性
    external_results = evaluator.evaluate_external_similarity(test_smiles, reference_smiles)
    print("外部相似性结果:", external_results)
    
    # 测试多样性指标
    diversity_results = evaluator.evaluate_diversity_metrics(test_smiles)
    print("多样性指标结果:", diversity_results)
