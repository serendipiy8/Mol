#!/usr/bin/env python3
"""
运行DecompDiff实验评估
主入口脚本
"""

import sys
import os
sys.path.append('src')

from src.evaluation import (
    ExperimentConfig, 
    ExperimentRunner, 
    run_experiment_from_config,
    create_experiment_parser
)

def main():
    """主函数"""
    print("🧪 DecompDiff实验评估系统")
    print("=" * 50)
    
    # 解析命令行参数
    parser = create_experiment_parser()
    args = parser.parse_args()
    
    # 显示配置信息
    print(f"📋 实验类型: {args.experiment_type}")
    print(f"📁 数据集路径: {args.dataset_path}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🔢 样本数量: {args.num_samples}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"💻 计算设备: {args.device}")
    print(f"🎲 随机种子: {args.random_seed}")
    
    if args.enable_visualization:
        print("📊 启用可视化")
    
    if args.save_intermediate:
        print("💾 保存中间结果")
    
    if args.run_baselines:
        print("📈 运行基线模型对比")
    
    print("=" * 50)
    
    try:
        # 运行实验
        run_experiment_from_config(args)
        
        print("\n🎉 实验评估完成！")
        print(f"📁 结果保存在: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n💥 实验过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
