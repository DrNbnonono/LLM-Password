#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-2 zxcvbn 破解难度评估与"理论熵—实测熵"失配分析
使用 zxcvbn 工具评估口令的实际破解难度，并与理论熵进行对比
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

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

try:
    from zxcvbn import zxcvbn
except ImportError:
    print("错误: 未安装 zxcvbn 库，请运行: pip install zxcvbn")
    sys.exit(1)


def safe_zxcvbn(password):
    """安全地调用zxcvbn，处理异常情况"""
    try:
        if len(password) > 72:
            # zxcvbn 限制最大72字符，超过的截断处理
            result = zxcvbn(password[:72])
            result['truncated'] = True
        else:
            result = zxcvbn(password)
            result['truncated'] = False
        return result
    except Exception as e:
        # 返回默认弱口令结果
        return {
            'score': 0,
            'guesses': 1,
            'guesses_log10': 0,
            'crack_times_seconds': {'offline_slow_hashing_1e4_per_second': 0.0001},
            'crack_times_display': {'offline_slow_hashing_1e4_per_second': '< 1秒'},
            'error': str(e),
            'truncated': False
        }


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
                
                # 检查必要列
                if 'Password' not in df.columns or 'Entropy' not in df.columns:
                    print(f"  ⚠ 跳过 (缺少必要列)")
                    continue
                
                # 提取模型名称
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
    
    return combined_df


def evaluate_with_zxcvbn(df, sample_size=None):
    """使用zxcvbn评估口令强度（优先使用CSV中已有的zxcvbn列）"""
    print("\n" + "=" * 80)
    print("读取 zxcvbn 评估结果")
    print("=" * 80)
    
    # 如果指定了采样大小，进行随机采样
    if sample_size and sample_size < len(df):
        print(f"采样策略: 从 {len(df)} 条记录中随机采样 {sample_size} 条")
        df_sample = df.sample(n=sample_size, random_state=42).copy()
    else:
        df_sample = df.copy()
        print(f"处理全部 {len(df_sample)} 条记录")
    
    # 检查是否已有zxcvbn列
    if 'zxcvbn_log2_guesses' in df_sample.columns and 'zxcvbn_score' in df_sample.columns:
        print("✓ CSV文件已包含zxcvbn评估结果，直接使用")
        
        # 确保数据类型正确
        df_sample['zxcvbn_score'] = pd.to_numeric(df_sample['zxcvbn_score'], errors='coerce').fillna(0).astype(int)
        df_sample['zxcvbn_guesses_log10'] = pd.to_numeric(df_sample['zxcvbn_guesses_log10'], errors='coerce').fillna(0.0)
        df_sample['zxcvbn_log2_guesses'] = pd.to_numeric(df_sample['zxcvbn_log2_guesses'], errors='coerce').fillna(0.0)
        df_sample['zxcvbn_crack_time_seconds'] = pd.to_numeric(df_sample['zxcvbn_crack_time_seconds'], errors='coerce').fillna(0.0)
        
        if 'zxcvbn_crack_time_display' not in df_sample.columns:
            df_sample['zxcvbn_crack_time_display'] = '未知'
        
        print(f"✓ 数据类型转换完成")
        return df_sample
    
    # 如果没有zxcvbn列，则进行评估
    print("⚠ CSV文件不包含zxcvbn结果，开始重新评估...")
    
    # 初始化结果列
    df_sample['zxcvbn_score'] = 0
    df_sample['zxcvbn_guesses'] = 0.0
    df_sample['zxcvbn_guesses_log10'] = 0.0
    df_sample['zxcvbn_log2_guesses'] = 0.0  # 等效熵位数
    df_sample['zxcvbn_crack_time_display'] = ''
    df_sample['zxcvbn_crack_time_seconds'] = 0.0
    
    print("\n开始评估...")
    for idx in tqdm(df_sample.index, desc="评估进度"):
        password = str(df_sample.at[idx, 'Password'])
        result = safe_zxcvbn(password)
        
        df_sample.at[idx, 'zxcvbn_score'] = result['score']
        df_sample.at[idx, 'zxcvbn_guesses'] = float(result['guesses'])
        df_sample.at[idx, 'zxcvbn_guesses_log10'] = float(result['guesses_log10'])
        
        # 计算 log2(guesses) 作为等效熵
        if result['guesses'] > 0:
            df_sample.at[idx, 'zxcvbn_log2_guesses'] = np.log2(result['guesses'])
        else:
            df_sample.at[idx, 'zxcvbn_log2_guesses'] = 0.0
        
        df_sample.at[idx, 'zxcvbn_crack_time_display'] = \
            result['crack_times_display']['offline_slow_hashing_1e4_per_second']
        df_sample.at[idx, 'zxcvbn_crack_time_seconds'] = \
            result['crack_times_seconds']['offline_slow_hashing_1e4_per_second']
    
    print(f"\n✓ 评估完成!")
    return df_sample


def analyze_entropy_mismatch(df):
    """分析理论熵与实测熵的失配"""
    print("\n" + "=" * 80)
    print("理论熵与实测熵失配分析")
    print("=" * 80)
    
    # 计算失配度 (理论熵 - 等效熵)
    df['entropy_mismatch'] = df['Entropy'] - df['zxcvbn_log2_guesses']
    df['entropy_mismatch_ratio'] = df['entropy_mismatch'] / df['Entropy']
    
    # 全局统计
    print("\n【全局统计】")
    print(f"理论熵均值: {df['Entropy'].mean():.2f} bits")
    print(f"等效熵均值 (log2 guesses): {df['zxcvbn_log2_guesses'].mean():.2f} bits")
    print(f"熵失配度均值: {df['entropy_mismatch'].mean():.2f} bits")
    print(f"熵失配比例均值: {df['entropy_mismatch_ratio'].mean():.2%}")
    
    # 按提示类型统计
    print("\n【按提示类型统计】")
    prompt_stats = df.groupby('Prompt_Type').agg({
        'Entropy': 'mean',
        'zxcvbn_log2_guesses': 'mean',
        'entropy_mismatch': 'mean',
        'entropy_mismatch_ratio': 'mean'
    }).round(2)
    print(prompt_stats)
    
    # zxcvbn score 分布
    print("\n【zxcvbn 强度评分分布】")
    score_dist = df.groupby(['Prompt_Type', 'zxcvbn_score']).size().unstack(fill_value=0)
    score_dist['Total'] = score_dist.sum(axis=1)
    for col in range(5):
        if col in score_dist.columns:
            score_dist[f'{col}_pct'] = (score_dist[col] / score_dist['Total'] * 100).round(2)
    print(score_dist)
    
    # 弱口令比例 (score <= 2)
    print("\n【弱口令比例 (zxcvbn score ≤ 2)】")
    weak_ratio = df.groupby('Prompt_Type').apply(
        lambda x: (x['zxcvbn_score'] <= 2).sum() / len(x) * 100
    ).round(2)
    print(weak_ratio)
    
    return df


def plot_score_distribution(df, output_dir):
    """绘制 zxcvbn 评分分布"""
    print("\n生成图表: zxcvbn 评分分布...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
    
    # 左图: 按提示类型的评分分布
    score_counts = df.groupby(['Prompt_Type', 'zxcvbn_score']).size().unstack(fill_value=0)
    score_counts_pct = score_counts.div(score_counts.sum(axis=1), axis=0) * 100
    
    score_counts_pct.T.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0].set_xlabel('zxcvbn Score (0=weakest, 4=strongest)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('zxcvbn Score Distribution by Prompt Type', fontsize=12, fontweight='bold')
    axes[0].legend(title='Prompt Type', fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # 右图: 堆叠条形图
    score_counts_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                          colormap='RdYlGn', alpha=0.8)
    axes[1].set_xlabel('Prompt Type', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('zxcvbn Score Stacked Distribution', fontsize=12, fontweight='bold')
    axes[1].legend(title='zxcvbn Score', fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / "1-2-zxcvbn评分分布.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_entropy_comparison(df, output_dir):
    """绘制理论熵与等效熵对比"""
    print("\n生成图表: 理论熵 vs 等效熵对比...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # 左图: 散点图 (采样)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    for prompt_type, color in [('Benign', '#3498db'), ('Few-Shot', '#e74c3c')]:
        data = df_sample[df_sample['Prompt_Type'] == prompt_type]
        axes[0].scatter(data['Entropy'], data['zxcvbn_log2_guesses'], 
                       alpha=0.3, s=10, label=prompt_type, color=color)
    
    # 添加 y=x 参考线
    max_val = max(df_sample['Entropy'].max(), df_sample['zxcvbn_log2_guesses'].max())
    axes[0].plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='Ideal (y=x)')
    
    axes[0].set_xlabel(r'Theoretical Entropy $E_{\mathregular{th}}$ (bits)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(r'Effective Entropy $\log_2(G)$ (bits)', fontsize=12, fontweight='bold')
    axes[0].set_title('Theoretical vs Effective Entropy Scatter Plot', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 右图: 失配度分布
    for prompt_type, color in [('Benign', '#3498db'), ('Few-Shot', '#e74c3c')]:
        data = df[df['Prompt_Type'] == prompt_type]['entropy_mismatch']
        axes[1].hist(data, bins=50, alpha=0.6, label=prompt_type, color=color, density=True)
    
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='No Mismatch')
    axes[1].set_xlabel(r'Entropy Mismatch $(E_{\mathregular{th}} - \log_2 G)$ (bits)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1].set_title('Entropy Mismatch Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "1-2-理论熵与等效熵对比.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_crack_time_analysis(df, output_dir):
    """绘制破解时间分析"""
    print("\n生成图表: 破解时间分析...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # 左图: 破解时间对数分布
    df_plot = df[df['zxcvbn_crack_time_seconds'] > 0].copy()
    df_plot['log_crack_time'] = np.log10(df_plot['zxcvbn_crack_time_seconds'])
    
    for prompt_type, color in [('Benign', '#3498db'), ('Few-Shot', '#e74c3c')]:
        data = df_plot[df_plot['Prompt_Type'] == prompt_type]['log_crack_time']
        axes[0].hist(data, bins=50, alpha=0.6, label=prompt_type, color=color)
    
    axes[0].set_xlabel('Crack Time (log10 seconds)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Estimated Crack Time Distribution (offline slow hashing)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 添加时间参考线
    time_refs = [
        (0, '1s'), (3, '1k s'), (6, '1M s\n(~12d)'),
        (9, '1B s\n(~32y)'), (12, '1T s\n(~30ky)')
    ]
    for log_val, label in time_refs:
        axes[0].axvline(x=log_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        axes[0].text(log_val, axes[0].get_ylim()[1]*0.95, label, 
                    rotation=90, va='top', ha='right', fontsize=7, alpha=0.7)
    
    # 右图: 箱线图对比
    box_data = []
    labels = []
    colors_box = []
    for prompt_type in ['Benign', 'Few-Shot']:
        data = df_plot[df_plot['Prompt_Type'] == prompt_type]['log_crack_time']
        box_data.append(data)
        labels.append(prompt_type)
        colors_box.append('#3498db' if prompt_type == 'Benign' else '#e74c3c')
    
    bp = axes[1].boxplot(box_data, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_ylabel('Crack Time (log10 seconds)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    axes[1].set_title('Crack Time Comparison (Boxplot)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "1-2-破解时间分析.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_model_comparison_heatmap(df, output_dir, top_n=10):
    """绘制模型弱口令比例热力图"""
    print("\n生成图表: 模型弱口令比例热力图...")
    
    # 选择样本量最多的Top N模型
    top_models = df['Model'].value_counts().head(top_n).index.tolist()
    df_top = df[df['Model'].isin(top_models)].copy()
    
    # 计算各模型在不同评分下的比例
    pivot_data = []
    for model in top_models:
        for prompt_type in ['Benign', 'Few-Shot']:
            data = df_top[(df_top['Model'] == model) & (df_top['Prompt_Type'] == prompt_type)]
            if len(data) > 0:
                weak_ratio = (data['zxcvbn_score'] <= 2).sum() / len(data) * 100
                pivot_data.append({
                    'Model': model,
                    'Prompt_Type': prompt_type,
                    'Weak_Ratio': weak_ratio
                })
    
    pivot_df = pd.DataFrame(pivot_data)
    pivot_table = pivot_df.pivot(index='Model', columns='Prompt_Type', values='Weak_Ratio')
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Weak Password Ratio (%)'}, linewidths=0.5,
                vmin=0, vmax=100)
    
    ax.set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Weak Password Ratio Heatmap (zxcvbn score <= 2, Top-{top_n})', 
                fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = output_dir / "1-2-模型弱口令比例热力图.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def save_results(df, output_dir):
    """保存评估结果"""
    print("\n" + "=" * 80)
    print("保存评估结果")
    print("=" * 80)
    
    # 保存完整评估结果
    output_csv = output_dir.parent / "data" / "zxcvbn_evaluation_results.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 完整结果保存至: {output_csv}")
    
    # 保存统计摘要
    stats_file = output_dir / "1-2-zxcvbn_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("1-2 zxcvbn 破解难度评估统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【全局统计】\n")
        f.write(f"评估样本数: {len(df)}\n")
        f.write(f"理论熵均值: {df['Entropy'].mean():.2f} bits\n")
        f.write(f"等效熵均值: {df['zxcvbn_log2_guesses'].mean():.2f} bits\n")
        f.write(f"熵失配度均值: {df['entropy_mismatch'].mean():.2f} bits\n")
        f.write(f"熵失配比例: {df['entropy_mismatch_ratio'].mean():.2%}\n\n")
        
        f.write("【按提示类型统计】\n")
        prompt_stats = df.groupby('Prompt_Type').agg({
            'Entropy': 'mean',
            'zxcvbn_log2_guesses': 'mean',
            'entropy_mismatch': 'mean',
            'zxcvbn_score': 'mean'
        }).round(2)
        f.write(prompt_stats.to_string() + "\n\n")
        
        f.write("【弱口令比例 (score ≤ 2)】\n")
        weak_ratio = df.groupby('Prompt_Type').apply(
            lambda x: (x['zxcvbn_score'] <= 2).sum() / len(x) * 100
        ).round(2)
        f.write(weak_ratio.to_string() + "\n\n")
        
        f.write("【zxcvbn 评分分布】\n")
        score_dist = df.groupby(['Prompt_Type', 'zxcvbn_score']).size().unstack(fill_value=0)
        f.write(score_dist.to_string() + "\n")
    
    print(f"  ✓ 统计报告保存至: {stats_file}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "1-2 zxcvbn 破解难度评估与失配分析" + " " * 15 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 设置输出目录
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 1. 收集数据
    df = collect_all_data()
    if df is None:
        return
    
    # 2. 使用 zxcvbn 评估 (可选择采样以加快速度)
    # 对于大数据集，可以设置 sample_size 参数，例如 sample_size=20000
    df_evaluated = evaluate_with_zxcvbn(df, sample_size=None)  # None 表示评估全部
    
    # 3. 分析失配
    df_analyzed = analyze_entropy_mismatch(df_evaluated)
    
    # 4. 生成可视化
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    plot_score_distribution(df_analyzed, output_dir)
    plot_entropy_comparison(df_analyzed, output_dir)
    plot_crack_time_analysis(df_analyzed, output_dir)
    plot_model_comparison_heatmap(df_analyzed, output_dir, top_n=10)
    
    # 5. 保存结果
    save_results(df_analyzed, output_dir)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "分析完成!" + " " * 29 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print(f"所有图表已保存至: {output_dir}")
    print("生成的图表:")
    print("  - 1-2-zxcvbn评分分布.png")
    print("  - 1-2-理论熵与等效熵对比.png")
    print("  - 1-2-破解时间分析.png")
    print("  - 1-2-模型弱口令比例热力图.png")
    print("  - 1-2-zxcvbn_statistics.txt")
    print(f"\n完整评估结果已保存至: {output_dir.parent}/data/zxcvbn_evaluation_results.csv")


if __name__ == "__main__":
    main()

