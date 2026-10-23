#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-1 口令熵评估与可视化分析
分析所有CSV数据集中的口令熵分布，生成统计报告和可视化图表
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置matplotlib在导入前
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10

# 强制设置字体族
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
rcParams['axes.unicode_minus'] = False

# 设置 seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.0)


def collect_all_data():
    """收集所有CSV数据集"""
    print("=" * 80)
    print("开始收集数据...")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data"
    all_data = []
    
    # 中文模型名称映射到英文
    model_name_map = {
        '豆包': 'Doubao',
        '通义千问': 'Qwen'
    }
    
    # 遍历良性提示和少样本提示目录
    for prompt_type_dir in ["良性提示", "少样本提示"]:
        dir_path = data_dir / prompt_type_dir
        if not dir_path.exists():
            print(f"警告: 目录不存在 {dir_path}")
            continue
        
        prompt_type = "Benign" if "良性" in prompt_type_dir else "Few-Shot"
        
        # 遍历所有CSV文件
        for csv_file in dir_path.glob("*.csv"):
            # 跳过备份文件
            if ".bak" in csv_file.name:
                continue
            
            print(f"读取: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                
                # 检查是否有Entropy列
                if 'Entropy' not in df.columns:
                    print(f"  ⚠ 跳过 (无Entropy列)")
                    continue
                
                # 提取模型名称（去除后缀）
                model_name = csv_file.stem.replace("_benign", "").replace("_fewshot", "")
                
                # 替换中文模型名为英文
                model_name = model_name_map.get(model_name, model_name)
                
                # 添加元数据
                df['Model'] = model_name
                df['Prompt_Type'] = prompt_type
                df['Source_File'] = csv_file.name
                
                all_data.append(df)
                print(f"  ✓ 加载 {len(df)} 条记录")
                
            except Exception as e:
                print(f"  ✗ 读取失败: {e}")
    
    if not all_data:
        print("错误: 未找到任何有效数据")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n总计加载: {len(combined_df)} 条口令记录")
    print(f"涵盖模型: {combined_df['Model'].nunique()} 个")
    print(f"提示类型: {combined_df['Prompt_Type'].unique().tolist()}")
    
    return combined_df


def generate_statistics(df):
    """生成统计摘要"""
    print("\n" + "=" * 80)
    print("统计分析")
    print("=" * 80)
    
    # 全局统计
    print("\n【全局统计】")
    print(f"总样本数: {len(df)}")
    print(f"熵值范围: [{df['Entropy'].min():.2f}, {df['Entropy'].max():.2f}]")
    print(f"平均熵值: {df['Entropy'].mean():.2f} bits")
    print(f"标准差: {df['Entropy'].std():.2f}")
    print(f"中位数: {df['Entropy'].median():.2f}")
    print(f"四分位数: Q1={df['Entropy'].quantile(0.25):.2f}, Q3={df['Entropy'].quantile(0.75):.2f}")
    
    # 按提示类型统计
    print("\n【按提示类型统计】")
    prompt_stats = df.groupby('Prompt_Type')['Entropy'].agg([
        ('样本数', 'count'),
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('中位数', 'median'),
        ('Q25', lambda x: x.quantile(0.25)),
        ('Q75', lambda x: x.quantile(0.75))
    ]).round(2)
    print(prompt_stats)
    
    # 按模型统计（Top 10）
    print("\n【按模型统计 (样本量Top10)】")
    model_stats = df.groupby('Model')['Entropy'].agg([
        ('样本数', 'count'),
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('中位数', 'median')
    ]).sort_values('样本数', ascending=False).head(10).round(2)
    print(model_stats)
    
    return prompt_stats, model_stats


def plot_overall_distribution(df, output_dir):
    """绘制全量口令熵分布"""
    print("\n生成图表: 全量口令熵分布...")
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # 直方图 + KDE
    ax.hist(df['Entropy'], bins=50, alpha=0.6, color='skyblue', edgecolor='black', density=True, label='Histogram')
    
    # KDE曲线
    from scipy import stats
    kde = stats.gaussian_kde(df['Entropy'].dropna())
    x_range = np.linspace(df['Entropy'].min(), df['Entropy'].max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.set_xlabel('Theoretical Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Overall Password Entropy Distribution (Histogram + KDE)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"N: {len(df)}\nMean: {df['Entropy'].mean():.2f}\nStd: {df['Entropy'].std():.2f}\nMedian: {df['Entropy'].median():.2f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "1-1-全量口令熵分布.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_prompt_type_comparison(df, output_dir):
    """绘制提示类型对比箱线图"""
    print("\n生成图表: 提示类型对比箱线图...")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # 箱线图
    box_data = [df[df['Prompt_Type'] == pt]['Entropy'].dropna() for pt in df['Prompt_Type'].unique()]
    bp = ax.boxplot(box_data, labels=df['Prompt_Type'].unique(), patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # 美化箱体
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Theoretical Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_title('Prompt Type Comparison: Password Entropy Boxplot', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / "1-1-提示类型-口令熵箱线图.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_model_prompt_violin(df, output_dir, top_n=8):
    """绘制模型与提示类型的小提琴图"""
    print(f"\n生成图表: 模型与提示类型的口令熵分布 (Top {top_n})...")
    
    # 选择样本量最多的Top N模型
    top_models = df['Model'].value_counts().head(top_n).index.tolist()
    df_top = df[df['Model'].isin(top_models)].copy()
    
    # 创建组合标签
    df_top['Model_Short'] = df_top['Model'].apply(lambda x: x[:20] if len(x) > 20 else x)
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    
    # 小提琴图
    sns.violinplot(data=df_top, x='Model_Short', y='Entropy', hue='Prompt_Type', 
                   split=False, inner='quartile', ax=ax, palette='Set2')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Theoretical Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title(f'Entropy Distribution by Model & Prompt Type (Violin Plot, Top-{top_n})', fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Prompt Type', fontsize=10, title_fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 旋转x轴标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / "1-1-模型与提示类型-口令熵分布.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_entropy_heatmap(df, output_dir, top_n=10):
    """绘制模型-提示类型熵值热力图"""
    print(f"\n生成图表: 模型-提示类型平均熵热力图 (Top {top_n})...")
    
    # 选择样本量最多的Top N模型
    top_models = df['Model'].value_counts().head(top_n).index.tolist()
    df_top = df[df['Model'].isin(top_models)].copy()
    
    # 计算平均熵
    pivot_table = df_top.pivot_table(values='Entropy', index='Model', columns='Prompt_Type', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Mean Entropy (bits)'}, linewidths=0.5)
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Model-Prompt Mean Entropy Heatmap (Top-{top_n})', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / "1-1-模型提示类型熵值热力图.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "1-1 口令熵评估与可视化分析" + " " * 20 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 设置输出目录
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 1. 收集数据
    df = collect_all_data()
    if df is None:
        return
    
    # 2. 生成统计
    prompt_stats, model_stats = generate_statistics(df)
    
    # 3. 生成可视化图表
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    plot_overall_distribution(df, output_dir)
    plot_prompt_type_comparison(df, output_dir)
    plot_model_prompt_violin(df, output_dir, top_n=8)
    plot_entropy_heatmap(df, output_dir, top_n=10)
    
    # 4. 保存统计结果
    print("\n" + "=" * 80)
    print("保存统计结果")
    print("=" * 80)
    
    stats_output = output_dir / "1-1-entropy_statistics.txt"
    with open(stats_output, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("1-1 口令熵统计分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【全局统计】\n")
        f.write(f"总样本数: {len(df)}\n")
        f.write(f"熵值范围: [{df['Entropy'].min():.2f}, {df['Entropy'].max():.2f}]\n")
        f.write(f"平均熵值: {df['Entropy'].mean():.2f} bits\n")
        f.write(f"标准差: {df['Entropy'].std():.2f}\n")
        f.write(f"中位数: {df['Entropy'].median():.2f}\n")
        f.write(f"四分位数: Q1={df['Entropy'].quantile(0.25):.2f}, Q3={df['Entropy'].quantile(0.75):.2f}\n\n")
        
        f.write("【按提示类型统计】\n")
        f.write(prompt_stats.to_string() + "\n\n")
        
        f.write("【按模型统计 (样本量Top10)】\n")
        f.write(model_stats.to_string() + "\n")
    
    print(f"  ✓ 统计报告保存至: {stats_output}")
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "分析完成!" + " " * 29 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print(f"所有图表已保存至: {output_dir}")
    print("生成的图表:")
    print("  - 1-1-全量口令熵分布.png")
    print("  - 1-1-提示类型-口令熵箱线图.png")
    print("  - 1-1-模型与提示类型-口令熵分布.png")
    print("  - 1-1-模型提示类型熵值热力图.png")
    print("  - 1-1-entropy_statistics.txt")


if __name__ == "__main__":
    main()

