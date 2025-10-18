import os
import json
import argparse
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

from .experiment_config import ExperimentConfig, create_experiment_parser
from .molecular_properties import run_molecular_properties_evaluation
from .geometric_evaluation import run_geometric_evaluation
from .docking_evaluation import run_docking_evaluation
from .chemical_evaluation import run_chemical_evaluation
from .similarity_evaluation import run_similarity_evaluation


# Experiment Runner
class ExperimentRunner:
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.output_dir = config.base_config['output_dir']
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_molecular_properties_experiment(self, generated_smiles: List[str],
                                          reference_smiles: List[str]) -> Dict[str, Any]:
        """运行分子属性实验"""
        print("=" * 60)
        print("🧪 运行分子属性评估实验")
        print("=" * 60)
        
        properties = list(self.config.evaluation_config['molecular_properties'].keys())
        properties = [prop for prop in properties if self.config.evaluation_config['molecular_properties'][prop]]
        
        results = run_molecular_properties_evaluation(
            generated_smiles=generated_smiles,
            reference_smiles=reference_smiles,
            output_dir=os.path.join(self.output_dir, 'molecular_properties'),
            properties=properties
        )
        
        self.results['molecular_properties'] = results
        return results
    
    def run_geometric_evaluation_experiment(self, generated_molecules: List[Any],
                                          reference_molecules: List[Any]) -> Dict[str, Any]:
        """运行几何评估实验"""
        print("=" * 60)
        print("📐 运行几何评估实验")
        print("=" * 60)
        
        bins = self.config.evaluation_config['geometric_evaluation'].get('bins', 50)
        
        results = run_geometric_evaluation(
            generated_molecules=generated_molecules,
            reference_molecules=reference_molecules,
            output_dir=os.path.join(self.output_dir, 'geometric_evaluation'),
            bins=bins
        )
        
        self.results['geometric_evaluation'] = results
        return results
    
    def run_docking_evaluation_experiment(self, generated_smiles: List[str],
                                        receptor_paths: List[str],
                                        binding_sites: List[Dict[str, float]]) -> Dict[str, Any]:
        """运行对接评估实验"""
        print("=" * 60)
        print("⚗️ 运行分子对接评估实验")
        print("=" * 60)
        
        high_affinity_threshold = self.config.evaluation_config['docking_evaluation'].get(
            'high_affinity_threshold', -7.0
        )
        success_rate_threshold = self.config.evaluation_config['docking_evaluation'].get(
            'success_rate_threshold', -6.0
        )
        
        results = run_docking_evaluation(
            smiles_list=generated_smiles,
            receptor_paths=receptor_paths,
            binding_sites=binding_sites,
            output_dir=os.path.join(self.output_dir, 'docking_evaluation'),
            high_affinity_threshold=high_affinity_threshold,
            success_rate_threshold=success_rate_threshold
        )
        
        self.results['docking_evaluation'] = results
        return results
    
    def run_chemical_evaluation_experiment(self, generated_smiles: List[str],
                                         reference_smiles: Optional[List[str]] = None) -> Dict[str, Any]:
        """运行化学评估实验"""
        print("=" * 60)
        print("🧬 运行化学评估实验")
        print("=" * 60)
        
        results = run_chemical_evaluation(
            generated_smiles=generated_smiles,
            reference_smiles=reference_smiles,
            output_dir=os.path.join(self.output_dir, 'chemical_evaluation')
        )
        
        self.results['chemical_evaluation'] = results
        return results
    
    def run_similarity_evaluation_experiment(self, generated_smiles: List[str],
                                           reference_smiles: Optional[List[str]] = None,
                                           baseline_results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行相似性评估实验"""
        print("=" * 60)
        print("🔍 运行相似性评估实验")
        print("=" * 60)
        
        results = run_similarity_evaluation(
            generated_smiles=generated_smiles,
            reference_smiles=reference_smiles,
            baseline_results=baseline_results,
            output_dir=os.path.join(self.output_dir, 'similarity_evaluation')
        )
        
        self.results['similarity_evaluation'] = results
        return results
    
    def run_full_evaluation(self, generated_smiles: List[str],
                          generated_molecules: Optional[List[Any]] = None,
                          reference_smiles: Optional[List[str]] = None,
                          reference_molecules: Optional[List[Any]] = None,
                          receptor_paths: Optional[List[str]] = None,
                          binding_sites: Optional[List[Dict[str, float]]] = None,
                          baseline_results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """运行完整评估"""
        print("🚀 开始完整评估实验")
        print(f"📊 生成分子数量: {len(generated_smiles)}")
        print(f"📊 参考分子数量: {len(reference_smiles) if reference_smiles else 0}")
        
        all_results = {}
        
        # 1. 分子属性评估
        if reference_smiles:
            all_results['molecular_properties'] = self.run_molecular_properties_experiment(
                generated_smiles, reference_smiles
            )
        
        # 2. 几何评估
        if generated_molecules and reference_molecules:
            all_results['geometric_evaluation'] = self.run_geometric_evaluation_experiment(
                generated_molecules, reference_molecules
            )
        
        # 3. 对接评估
        if receptor_paths and binding_sites:
            all_results['docking_evaluation'] = self.run_docking_evaluation_experiment(
                generated_smiles, receptor_paths, binding_sites
            )
        
        # 4. 化学评估
        all_results['chemical_evaluation'] = self.run_chemical_evaluation_experiment(
            generated_smiles, reference_smiles
        )
        
        # 5. 相似性评估
        all_results['similarity_evaluation'] = self.run_similarity_evaluation_experiment(
            generated_smiles, reference_smiles, baseline_results
        )
        
        # 生成综合报告
        self.generate_comprehensive_report(all_results)
        
        return all_results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]):
        """生成综合评估报告"""
        print("=" * 60)
        print("📋 生成综合评估报告")
        print("=" * 60)
        
        report_data = {
            'timestamp': self.timestamp,
            'experiment_config': self.config.base_config,
            'evaluation_results': {}
        }
        
        # 收集所有评估结果
        for experiment_type, result in results.items():
            if 'summary_table' in result:
                # 转换为字典格式以便JSON序列化
                summary_dict = result['summary_table'].to_dict('records')
                report_data['evaluation_results'][experiment_type] = {
                    'summary': summary_dict
                }
        
        # 保存详细报告
        report_path = os.path.join(self.output_dir, f'comprehensive_report_{self.timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 生成HTML报告
        self.generate_html_report(report_data)
        
        print(f"✅ 综合报告已保存到: {report_path}")
    
    def generate_html_report(self, report_data: Dict[str, Any]):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DecompDiff实验评估报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .value {{ font-weight: bold; color: #2c3e50; }}
                .description {{ color: #7f8c8d; font-size: 0.9em; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 DecompDiff实验评估报告</h1>
                <p>生成时间: {report_data['timestamp']}</p>
                <p>实验配置: {report_data['experiment_config']['dataset_name']}</p>
            </div>
        """
        
        # 添加各个实验的结果
        for experiment_type, result in report_data['evaluation_results'].items():
            html_content += f"""
            <div class="section">
                <h2>📊 {experiment_type.replace('_', ' ').title()}</h2>
            """
            
            if 'summary' in result:
                html_content += "<table><tr><th>Metric</th><th>Value</th><th>Description</th></tr>"
                for metric in result['summary']:
                    html_content += f"""
                    <tr>
                        <td>{metric.get('Metric', metric.get('Property', metric.get('Geometric_Property', 'N/A')))}</td>
                        <td class="value">{metric.get('Value', 'N/A')}</td>
                        <td class="description">{metric.get('Description', 'N/A')}</td>
                    </tr>
                    """
                html_content += "</table>"
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # 保存HTML报告
        html_path = os.path.join(self.output_dir, f'comprehensive_report_{self.timestamp}.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML报告已保存到: {html_path}")

def run_experiment_from_config(args: argparse.Namespace):
    """从配置运行实验"""
    # 加载配置
    config = ExperimentConfig()
    
    # 更新配置
    config.base_config.update({
        'dataset_path': args.dataset_path,
        'output_dir': args.output_dir,
        'num_samples_per_test': args.num_samples,
        'batch_size': args.batch_size,
        'device': args.device,
        'random_seed': args.random_seed
    })
    
    # 创建实验运行器
    runner = ExperimentRunner(config)
    
    # 这里需要加载生成的数据和参考数据
    # 由于模型部分还未实现，这里使用占位符
    print("⚠️ 注意: 模型部分还未实现，使用占位符数据")
    
    # 占位符数据（实际使用时需要替换为真实的生成结果）
    generated_smiles = [
        'CCO', 'CC(C)O', 'CC(=O)O', 'c1ccccc1', 'CCc1ccccc1',
        'CCN(CC)CC', 'CC(=O)Nc1ccccc1', 'CCc1ccc(C)cc1', 'CC(C)(C)O', 'CCc1ccccc1O'
    ] * 100  # 重复以模拟更多样本
    
    reference_smiles = [
        'CCO', 'CC(C)O', 'CC(=O)O', 'c1ccccc1', 'CCc1ccccc1'
    ] * 50
    
    # 运行实验
    if args.experiment_type == 'full_evaluation':
        results = runner.run_full_evaluation(
            generated_smiles=generated_smiles,
            reference_smiles=reference_smiles
        )
    else:
        # 运行特定实验
        if args.experiment_type == 'molecular_properties':
            results = runner.run_molecular_properties_experiment(
                generated_smiles, reference_smiles
            )
        elif args.experiment_type == 'chemical_evaluation':
            results = runner.run_chemical_evaluation_experiment(
                generated_smiles, reference_smiles
            )
        elif args.experiment_type == 'similarity_evaluation':
            results = runner.run_similarity_evaluation_experiment(
                generated_smiles, reference_smiles
            )
        else:
            print(f"❌ 未知的实验类型: {args.experiment_type}")
            return
    
    print("🎉 实验完成！")

def main():
    """主函数"""
    parser = create_experiment_parser()
    args = parser.parse_args()
    
    print("🚀 开始DecompDiff实验评估")
    print(f"📋 实验类型: {args.experiment_type}")
    print(f"📁 输出目录: {args.output_dir}")
    
    run_experiment_from_config(args)

if __name__ == "__main__":
    main()
