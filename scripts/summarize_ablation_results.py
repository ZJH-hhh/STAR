#!/usr/bin/env python3
"""
消融实验结果汇总脚本

功能：
1. 自动扫描消融实验的日志文件
2. 提取每个实验的最佳AUROC分数
3. 生成LaTeX表格和可视化图表
4. 计算性能提升百分比
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def extract_best_score(log_file_path: str) -> float:
    """从训练日志中提取最佳分数"""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        pattern = r'best score:\s+([\d.]+)'
        matches = re.findall(pattern, content)
        if matches:
            return float(matches[-1])
        else:
            print(f"⚠️  未在 {log_file_path} 中找到best score")
            return 0.0
    except FileNotFoundError:
        print(f"❌ 日志文件不存在: {log_file_path}")
        return 0.0
    except Exception as e:
        print(f"❌ 解析日志文件时出错 {log_file_path}: {e}")
        return 0.0


def scan_ablation_results(log_base_dir: str, dataset: str, n_support: int) -> Dict[str, float]:
    """扫描所有消融实验的结果"""
    results = {}
    exp_prefix = f"ablation_{dataset}_{n_support}shot"
    
    experiments = {
        "Exp-1: Baseline": f"exp1_baseline/{exp_prefix}_exp1",
        "Exp-2: +KRGTS": f"exp2_krgts/{exp_prefix}_exp2",
        "Exp-3: +LLM Sampler": f"exp3_llm_sampler/{exp_prefix}_exp3",
        "Exp-4: +L-TAFM": f"exp4_ltafm/{exp_prefix}_exp4",
        "Exp-5: KRGTS+LLM": f"exp5_krgts_llm/{exp_prefix}_exp5",
        "Exp-6: KRGTS+L-TAFM": f"exp6_krgts_ltafm/{exp_prefix}_exp6",
        "Exp-7: LLM+L-TAFM": f"exp7_llm_ltafm/{exp_prefix}_exp7",
        "Exp-8: Full Model": f"exp8_full_model/{exp_prefix}_exp8",
    }
    
    print(f"\n正在扫描日志目录: {log_base_dir}")
    print("=" * 70)
    
    for exp_name, exp_path in experiments.items():
        log_file = os.path.join(log_base_dir, exp_path, "train.log")
        score = extract_best_score(log_file)
        results[exp_name] = score
        status = "✅" if score > 0 else "❌"
        print(f"{status} {exp_name:25s} | AUROC: {score:.5f}")
    
    print("=" * 70)
    return results


def calculate_improvements(results: Dict[str, float]) -> Dict[str, float]:
    """计算相对于baseline的性能提升百分比"""
    baseline_score = results.get("Exp-1: Baseline", 0.0)
    if baseline_score == 0:
        print("⚠️  Baseline分数为0，无法计算提升百分比")
        return {}
    
    improvements = {}
    for exp_name, score in results.items():
        improvement = ((score - baseline_score) / baseline_score) * 100
        improvements[exp_name] = improvement
    return improvements


def generate_latex_table(results: Dict[str, float], improvements: Dict[str, float]) -> str:
    """生成LaTeX格式的结果表格"""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Ablation Study Results}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Configuration & KRGTS & LLM Sampler & L-TAFM & AUROC & Improvement (\\%) \\\\",
        "\\midrule",
    ]
    
    config_map = {
        "Exp-1: Baseline": ("", "", ""),
        "Exp-2: +KRGTS": ("✓", "", ""),
        "Exp-3: +LLM Sampler": ("", "✓", ""),
        "Exp-4: +L-TAFM": ("", "", "✓"),
        "Exp-5: KRGTS+LLM": ("✓", "✓", ""),
        "Exp-6: KRGTS+L-TAFM": ("✓", "", "✓"),
        "Exp-7: LLM+L-TAFM": ("", "✓", "✓"),
        "Exp-8: Full Model": ("✓", "✓", "✓"),
    }
    
    for exp_name in results.keys():
        krgts, llm, ltafm = config_map[exp_name]
        score = results[exp_name]
        improvement = improvements.get(exp_name, 0.0)
        row = f"{exp_name:20s} & {krgts:5s} & {llm:5s} & {ltafm:5s} & {score:.5f} & {improvement:+.2f}\\% \\\\"
        latex.append(row)
    
    latex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(latex)


def plot_ablation_results(results: Dict[str, float], output_path: str):
    """绘制消融实验结果的可视化图表"""
    exp_names = list(results.keys())
    scores = [results[name] for name in exp_names]
    short_names = ["Baseline", "+KRGTS", "+LLM", "+L-TAFM", "KRGTS+LLM", "KRGTS+TAFM", "LLM+TAFM", "Full Model"]
    colors = ['gray'] + ['steelblue'] * 6 + ['crimson']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(short_names, scores, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{score:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    baseline_score = scores[0]
    plt.axhline(y=baseline_score, color='red', linestyle='--', linewidth=1.5, 
                label=f'Baseline: {baseline_score:.4f}')
    
    plt.ylabel('AUROC Score', fontsize=12, fontweight='bold')
    plt.xlabel('Experiment Configuration', fontsize=12, fontweight='bold')
    plt.title('Ablation Study: Component Contribution Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(min(scores) - 0.02, max(scores) + 0.02)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 图表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="消融实验结果汇总工具")
    parser.add_argument("--log_dir", type=str, default="logs/ablation_0305", help="消融实验日志目录")
    parser.add_argument("--dataset", type=str, default="tox21", help="数据集名称")
    parser.add_argument("--n_support", type=int, default=10, help="Support set大小")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="输出结果目录")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = scan_ablation_results(args.log_dir, args.dataset, args.n_support)
    
    if not results or all(score == 0 for score in results.values()):
        print("\n❌ 未找到有效的实验结果，请检查日志目录")
        return
    
    improvements = calculate_improvements(results)
    latex_table = generate_latex_table(results, improvements)
    latex_file = os.path.join(args.output_dir, "ablation_table.tex")
    with open(latex_file, 'w') as f:
        f.write(latex_table)
    print(f"\n📝 LaTeX表格已保存: {latex_file}")
    
    bar_chart_path = os.path.join(args.output_dir, "ablation_bar_chart.png")
    plot_ablation_results(results, bar_chart_path)
    
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)
    print(f"数据集: {args.dataset}")
    print(f"Shot数: {args.n_support}")
    print(f"Baseline AUROC: {results['Exp-1: Baseline']:.5f}")
    print(f"最佳配置: Exp-8: Full Model")
    print(f"最佳AUROC: {results['Exp-8: Full Model']:.5f}")
    print(f"相对提升: {improvements['Exp-8: Full Model']:.2f}%")
    print("=" * 70)
    
    print("\n组件独立贡献分析:")
    print(f"  KRGTS:       +{improvements['Exp-2: +KRGTS']:.2f}%")
    print(f"  LLM Sampler: +{improvements['Exp-3: +LLM Sampler']:.2f}%")
    print(f"  L-TAFM:      +{improvements['Exp-4: +L-TAFM']:.2f}%")
    
    print("\n组件协同效果分析:")
    print(f"  KRGTS + LLM:     +{improvements['Exp-5: KRGTS+LLM']:.2f}%")
    print(f"  KRGTS + L-TAFM:  +{improvements['Exp-6: KRGTS+L-TAFM']:.2f}%")
    print(f"  LLM + L-TAFM:    +{improvements['Exp-7: LLM+L-TAFM']:.2f}%")
    print(f"  完整模型:        +{improvements['Exp-8: Full Model']:.2f}%")
    print("=" * 70)
    print(f"\n✅ 所有结果已保存到: {args.output_dir}/")


if __name__ == "__main__":
    main()
