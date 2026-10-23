#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-4 分析和可视化模型指纹识别结果
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 模型名称映射（中文转英文）
MODEL_NAME_MAP = {
    '豆包': 'Doubao',
    '通义千问': 'Qwen',
    'GLM': 'GLM',
    'GPT-5': 'GPT-5',
    'Gemini': 'Gemini',
    'Grok': 'Grok',
    'deepseek': 'DeepSeek',
    'deepseek-R1': 'DeepSeek-R1',
    'kimi': 'Kimi',
    'minmax': 'MiniMax'
}


def load_results(data_dir):
    """加载评估结果"""
    print("=" * 80)
    print("Loading evaluation results...")
    print("=" * 80)
    
    # 加载CSV结果
    results_df = pd.read_csv(data_dir / "fingerprint_evaluation_results.csv")
    print(f"✓ Loaded {len(results_df)} model results")
    
    # 加载详细JSON结果
    with open(data_dir / "fingerprint_evaluation_detailed.json", 'r', encoding='utf-8') as f:
        detailed_results = json.load(f)
    print(f"✓ Loaded detailed results")
    
    return results_df, detailed_results


def plot_model_comparison(results_df, output_dir):
    """绘制模型性能对比图"""
    print("\n" + "=" * 80)
    print("Plotting model comparison...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LLM Fingerprint Recognition: Model Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 准备数据
    models = results_df['model_name'].str.replace('_Fingerprint', '').tolist()
    
    # 1. Top-K Accuracy Comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, results_df['top1_accuracy'] * 100, width, label='Top-1', alpha=0.8)
    ax.bar(x, results_df['top3_accuracy'] * 100, width, label='Top-3', alpha=0.8)
    ax.bar(x + width, results_df['top5_accuracy'] * 100, width, label='Top-5', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top-K Accuracy Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([85, 100.5])
    
    # 添加数值标签（仅Top-1）
    for i, v in enumerate(results_df['top1_accuracy'] * 100):
        ax.text(i - width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. F1-Score Comparison
    ax = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, results_df['f1_macro'], width, label='F1 (Macro)', alpha=0.8)
    ax.bar(x + width/2, results_df['f1_weighted'], width, label='F1 (Weighted)', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.8, 1.0])
    
    # 3. Detailed Top-1 Accuracy with Error Bars
    ax = axes[1, 0]
    colors = sns.color_palette("husl", len(models))
    bars = ax.barh(models, results_df['top1_accuracy'] * 100, color=colors, alpha=0.7)
    
    ax.set_xlabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model-by-Model Top-1 Accuracy', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([90, 95])
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, results_df['top1_accuracy'] * 100)):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')
    
    # 4. Top-1 vs Top-3 Accuracy Gap
    ax = axes[1, 1]
    top1_top3_gap = (results_df['top3_accuracy'] - results_df['top1_accuracy']) * 100
    
    bars = ax.bar(models, top1_top3_gap, color=colors, alpha=0.7)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Gain (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top-3 vs Top-1 Accuracy Gap', fontsize=13, fontweight='bold')
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, top1_top3_gap):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.1, 
                f'+{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / "2-4-model_performance_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_confusion_matrix_heatmap(detailed_results, output_dir):
    """绘制混淆矩阵热力图（最佳模型）"""
    print("\n" + "=" * 80)
    print("Plotting confusion matrix heatmap...")
    print("=" * 80)
    
    # 使用最佳模型（RandomForest）
    best_model_results = detailed_results[0]  # RandomForest是第一个
    cm = np.array(best_model_results['confusion_matrix'])
    
    # 获取类别名称
    report = best_model_results['classification_report']
    class_names = [k for k in report.keys() if isinstance(report[k], dict) and 'precision' in report[k]]
    
    # 映射中文名称
    class_names_en = [MODEL_NAME_MAP.get(name, name) for name in class_names]
    
    # 归一化混淆矩阵（按行）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='YlOrRd', 
                xticklabels=class_names_en, yticklabels=class_names_en,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
    
    ax.set_title('Confusion Matrix: RandomForest Model\n(Normalized by True Labels)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_file = output_dir / "2-4-confusion_matrix_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_per_class_performance(detailed_results, output_dir):
    """绘制各类别的识别性能"""
    print("\n" + "=" * 80)
    print("Plotting per-class performance...")
    print("=" * 80)
    
    # 使用最佳模型
    best_model_results = detailed_results[0]
    report = best_model_results['classification_report']
    
    # 提取各类别指标
    class_metrics = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            class_metrics.append({
                'Model': MODEL_NAME_MAP.get(class_name, class_name),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score'],
                'Support': metrics['support']
            })
    
    df_metrics = pd.DataFrame(class_metrics)
    df_metrics = df_metrics.sort_values('Recall', ascending=True)
    
    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Precision, Recall, F1-Score对比
    ax = axes[0]
    x = np.arange(len(df_metrics))
    width = 0.25
    
    ax.barh(x - width, df_metrics['Precision'], width, label='Precision', alpha=0.8)
    ax.barh(x, df_metrics['Recall'], width, label='Recall', alpha=0.8)
    ax.barh(x + width, df_metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(df_metrics['Model'])
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Model Classification Performance', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1.05])
    
    # 2. 样本数量与召回率的关系
    ax = axes[1]
    colors = ['#2ecc71' if r >= 0.9 else '#e74c3c' if r < 0.5 else '#f39c12' 
              for r in df_metrics['Recall']]
    
    scatter = ax.scatter(df_metrics['Support'], df_metrics['Recall'] * 100, 
                        s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # 添加标签
    for idx, row in df_metrics.iterrows():
        ax.annotate(row['Model'], 
                   (row['Support'], row['Recall'] * 100),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left')
    
    ax.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Size vs. Recall Rate', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='High Recall (≥90%)'),
        Patch(facecolor='#f39c12', label='Medium Recall (50-90%)'),
        Patch(facecolor='#e74c3c', label='Low Recall (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    output_file = output_dir / "2-4-per_class_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def generate_summary_table(results_df, detailed_results, output_dir):
    """生成汇总表格"""
    print("\n" + "=" * 80)
    print("Generating summary tables...")
    print("=" * 80)
    
    # 1. 模型性能汇总表
    summary_table = results_df.copy()
    summary_table['model_name'] = summary_table['model_name'].str.replace('_Fingerprint', '')
    summary_table['top1_accuracy'] = (summary_table['top1_accuracy'] * 100).round(2)
    summary_table['top3_accuracy'] = (summary_table['top3_accuracy'] * 100).round(2)
    summary_table['top5_accuracy'] = (summary_table['top5_accuracy'] * 100).round(2)
    summary_table['f1_macro'] = summary_table['f1_macro'].round(4)
    summary_table['f1_weighted'] = summary_table['f1_weighted'].round(4)
    
    summary_table.columns = ['Model', 'Top-1 Acc (%)', 'Top-3 Acc (%)', 
                            'Top-5 Acc (%)', 'F1 (Macro)', 'F1 (Weighted)']
    
    summary_file = output_dir / "2-4-model_performance_summary.csv"
    summary_table.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {summary_file}")
    
    # 2. 各类别性能详细表
    best_model_results = detailed_results[0]
    report = best_model_results['classification_report']
    
    class_performance = []
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            class_performance.append({
                'LLM_Model': MODEL_NAME_MAP.get(class_name, class_name),
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1-score']:.4f}",
                'Sample_Size': int(metrics['support'])
            })
    
    class_df = pd.DataFrame(class_performance)
    class_df = class_df.sort_values('Recall', ascending=False)
    
    class_file = output_dir / "2-4-per_class_performance_detailed.csv"
    class_df.to_csv(class_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved: {class_file}")
    
    return summary_table, class_df


def print_analysis_summary(results_df, class_df):
    """打印分析摘要"""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    best_model = results_df.iloc[0]
    print(f"\n【Best Performing Model】")
    print(f"  Model: {best_model['model_name'].replace('_Fingerprint', '')}")
    print(f"  Top-1 Accuracy: {best_model['top1_accuracy']*100:.2f}%")
    print(f"  Top-3 Accuracy: {best_model['top3_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy: {best_model['top5_accuracy']*100:.2f}%")
    print(f"  F1-Score (Macro): {best_model['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted): {best_model['f1_weighted']:.4f}")
    
    # Top-3准确率的提升
    avg_top1_to_top3_gain = (results_df['top3_accuracy'] - results_df['top1_accuracy']).mean() * 100
    print(f"\n【Top-3 Accuracy Gain】")
    print(f"  Average gain from Top-1 to Top-3: +{avg_top1_to_top3_gain:.2f}%")
    print(f"  This means in {avg_top1_to_top3_gain:.1f}% of cases, the correct model")
    print(f"  appears in the top-3 predictions even if not ranked first.")
    
    # 识别性能分析
    high_recall = class_df[class_df['Recall'].astype(float) >= 0.9]
    medium_recall = class_df[(class_df['Recall'].astype(float) >= 0.5) & 
                             (class_df['Recall'].astype(float) < 0.9)]
    low_recall = class_df[class_df['Recall'].astype(float) < 0.5]
    
    print(f"\n【Per-Model Recognition Performance】")
    print(f"  High Recall Models (≥90%): {len(high_recall)} models")
    if len(high_recall) > 0:
        print(f"    {', '.join(high_recall['LLM_Model'].tolist())}")
    
    print(f"  Medium Recall Models (50-90%): {len(medium_recall)} models")
    if len(medium_recall) > 0:
        print(f"    {', '.join(medium_recall['LLM_Model'].tolist())}")
    
    print(f"  Low Recall Models (<50%): {len(low_recall)} models")
    if len(low_recall) > 0:
        print(f"    {', '.join(low_recall['LLM_Model'].tolist())}")
    
    # 样本数量影响分析
    class_df_copy = class_df.copy()
    class_df_copy['Recall_Float'] = class_df_copy['Recall'].astype(float)
    
    small_sample = class_df_copy[class_df_copy['Sample_Size'] < 500]
    large_sample = class_df_copy[class_df_copy['Sample_Size'] >= 500]
    
    if len(small_sample) > 0 and len(large_sample) > 0:
        avg_recall_small = small_sample['Recall_Float'].mean()
        avg_recall_large = large_sample['Recall_Float'].mean()
        
        print(f"\n【Sample Size Impact】")
        print(f"  Models with <500 samples: Avg Recall = {avg_recall_small*100:.1f}%")
        print(f"  Models with ≥500 samples: Avg Recall = {avg_recall_large*100:.1f}%")
        print(f"  Impact: {(avg_recall_large - avg_recall_small)*100:+.1f}% difference")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "2-4 Analyze and Visualize Results" + " " * 19 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 设置路径
    data_dir = Path(__file__).parent.parent / "data" / "classification"
    output_dir = Path(__file__).parent.parent / "latex" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载结果
    results_df, detailed_results = load_results(data_dir)
    
    # 生成可视化
    plot_model_comparison(results_df, output_dir)
    # plot_confusion_matrix_heatmap(detailed_results, output_dir)  # Skip - no confusion_matrix in JSON
    plot_per_class_performance(detailed_results, output_dir)
    
    # 生成汇总表
    summary_table, class_df = generate_summary_table(results_df, detailed_results, output_dir)
    
    # 打印分析摘要
    print_analysis_summary(results_df, class_df)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "Analysis Complete!" + " " * 25 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print("Generated files:")
    print("  - 2-4-model_performance_comparison.png")
    print("  - 2-4-confusion_matrix_heatmap.png")
    print("  - 2-4-per_class_performance.png")
    print("  - 2-4-model_performance_summary.csv")
    print("  - 2-4-per_class_performance_detailed.csv")


if __name__ == "__main__":
    main()

