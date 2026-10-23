#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-3 倾向性指标分析：字符分布偏好与模式识别
分析不同LLM生成的口令在字符使用、位置偏好、长度分布等方面的倾向性
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from collections import Counter, defaultdict
from scipy import stats
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))

# 配置matplotlib
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']

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
    
    for prompt_type_dir in ["良性提示", "少样本提示"]:
        dir_path = data_dir / prompt_type_dir
        if not dir_path.exists():
            continue
        
        prompt_type = "Benign" if "良性" in prompt_type_dir else "Few-Shot"
        
        for csv_file in dir_path.glob("*.csv"):
            if ".bak" in csv_file.name:
                continue
            
            print(f"读取: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                
                if 'Password' not in df.columns:
                    continue
                
                model_name = csv_file.stem.replace("_benign", "").replace("_fewshot", "")
                model_name = model_name_map.get(model_name, model_name)
                
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
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n总计加载: {len(combined_df)} 条口令记录")
    
    return combined_df


def analyze_character_distribution(df):
    """分析字符分布偏好"""
    print("\n" + "=" * 80)
    print("1. 字符类型分布分析")
    print("=" * 80)
    
    # 定义字符类别
    def categorize_chars(password):
        """统计口令中各类字符的数量"""
        if pd.isna(password):
            return {'lowercase': 0, 'uppercase': 0, 'digit': 0, 'special': 0}
        
        pwd_str = str(password)
        return {
            'lowercase': sum(1 for c in pwd_str if c.islower()),
            'uppercase': sum(1 for c in pwd_str if c.isupper()),
            'digit': sum(1 for c in pwd_str if c.isdigit()),
            'special': sum(1 for c in pwd_str if not c.isalnum())
        }
    
    # 统计各类字符
    char_stats = df['Password'].apply(categorize_chars)
    for cat in ['lowercase', 'uppercase', 'digit', 'special']:
        df[f'count_{cat}'] = char_stats.apply(lambda x: x[cat])
    
    # 计算总字符数
    df['total_chars'] = df[['count_lowercase', 'count_uppercase', 'count_digit', 'count_special']].sum(axis=1)
    
    # 计算各类字符的比例
    for cat in ['lowercase', 'uppercase', 'digit', 'special']:
        df[f'ratio_{cat}'] = df[f'count_{cat}'] / df['total_chars'].replace(0, 1)
    
    # 按模型和提示类型统计
    print("\n【按模型与提示类型的字符类型比例】")
    summary_stats = df.groupby(['Model', 'Prompt_Type']).agg({
        'ratio_lowercase': 'mean',
        'ratio_uppercase': 'mean',
        'ratio_digit': 'mean',
        'ratio_special': 'mean'
    }).round(3)
    
    print(summary_stats)
    
    return df


def calculate_character_frequency(df):
    """计算字符频率分布"""
    print("\n" + "=" * 80)
    print("2. 字符频率分析")
    print("=" * 80)
    
    # 统计每个模型-提示类型组合的字符频率
    char_freq_by_group = {}
    
    for (model, prompt_type), group in df.groupby(['Model', 'Prompt_Type']):
        all_chars = ''.join(group['Password'].dropna().astype(str))
        char_freq = Counter(all_chars)
        total_chars = sum(char_freq.values())
        
        # 归一化频率
        char_freq_normalized = {char: count / total_chars for char, count in char_freq.items()}
        char_freq_by_group[(model, prompt_type)] = char_freq_normalized
    
    # 找出最常见的字符
    print("\n【各组Top-10最常见字符】")
    for (model, prompt_type), freq_dict in list(char_freq_by_group.items())[:6]:
        top_chars = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n{model} - {prompt_type}:")
        for char, freq in top_chars:
            char_display = char if char.isprintable() else repr(char)
            print(f"  '{char_display}': {freq:.4f}")
    
    return char_freq_by_group


def analyze_position_bias(df, sample_size=10000):
    """分析位置相关的字符偏好"""
    print("\n" + "=" * 80)
    print("3. 位置偏好分析")
    print("=" * 80)
    
    # 限制样本大小以提高性能
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # 分析首字符和尾字符的分布
    df_sample['first_char'] = df_sample['Password'].astype(str).str[0]
    df_sample['last_char'] = df_sample['Password'].astype(str).str[-1]
    
    # 统计首字符类型
    def char_type(c):
        if pd.isna(c) or c == '':
            return 'empty'
        if c.islower():
            return 'lowercase'
        if c.isupper():
            return 'uppercase'
        if c.isdigit():
            return 'digit'
        return 'special'
    
    df_sample['first_char_type'] = df_sample['first_char'].apply(char_type)
    df_sample['last_char_type'] = df_sample['last_char'].apply(char_type)
    
    # 按提示类型统计
    print("\n【首字符类型分布 (按提示类型)】")
    first_char_dist = df_sample.groupby('Prompt_Type')['first_char_type'].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(first_char_dist.round(2))
    
    print("\n【尾字符类型分布 (按提示类型)】")
    last_char_dist = df_sample.groupby('Prompt_Type')['last_char_type'].value_counts(normalize=True).unstack(fill_value=0) * 100
    print(last_char_dist.round(2))
    
    return df_sample


def analyze_length_distribution(df):
    """分析长度分布"""
    print("\n" + "=" * 80)
    print("4. 长度分布分析")
    print("=" * 80)
    
    # 计算口令长度
    df['length'] = df['Password'].astype(str).apply(len)
    
    # 全局统计
    print("\n【全局长度统计】")
    print(f"平均长度: {df['length'].mean():.2f}")
    print(f"中位数: {df['length'].median():.0f}")
    print(f"标准差: {df['length'].std():.2f}")
    print(f"范围: [{df['length'].min()}, {df['length'].max()}]")
    
    # 按提示类型统计
    print("\n【按提示类型的长度统计】")
    length_stats = df.groupby('Prompt_Type')['length'].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print(length_stats)
    
    # 按模型统计 (Top 10)
    print("\n【按模型的长度统计 (Top-10)】")
    model_length = df.groupby('Model')['length'].agg(['count', 'mean', 'std']).sort_values('count', ascending=False).head(10)
    print(model_length.round(2))
    
    return df


def calculate_uniformity_metrics(df, char_freq_by_group):
    """计算字符分布均匀性指标"""
    print("\n" + "=" * 80)
    print("5. 字符分布均匀性量化")
    print("=" * 80)
    
    results = []
    
    for (model, prompt_type), freq_dict in char_freq_by_group.items():
        # 准备数据
        chars = list(freq_dict.keys())
        observed_freq = np.array(list(freq_dict.values()))
        
        if len(chars) == 0:
            continue
        
        # 1. 计算熵 (Shannon Entropy)
        # H = -∑ p_i * log2(p_i)
        observed_freq_nonzero = observed_freq[observed_freq > 0]
        entropy = -np.sum(observed_freq_nonzero * np.log2(observed_freq_nonzero))
        max_entropy = np.log2(len(chars))  # 完全均匀分布的最大熵
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 2. 卡方检验 (Chi-Square Test)
        # 检验观察频率是否符合均匀分布
        expected_freq = np.ones(len(chars)) / len(chars)
        total_count = len(''.join(df[(df['Model'] == model) & (df['Prompt_Type'] == prompt_type)]['Password'].dropna().astype(str)))
        observed_count = observed_freq * total_count
        expected_count = expected_freq * total_count
        
        if total_count > 0:
            chi_square_stat, chi_square_p = stats.chisquare(observed_count, expected_count)
        else:
            chi_square_stat, chi_square_p = 0, 1
        
        # 3. Gini系数 (衡量不平等程度)
        sorted_freq = np.sort(observed_freq)
        n = len(sorted_freq)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_freq) / (n * np.sum(sorted_freq))) - (n + 1) / n
        
        results.append({
            'Model': model,
            'Prompt_Type': prompt_type,
            'Unique_Chars': len(chars),
            'Entropy': entropy,
            'Normalized_Entropy': normalized_entropy,
            'Chi_Square_Stat': chi_square_stat,
            'Chi_Square_P_Value': chi_square_p,
            'Gini_Coefficient': gini
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n【均匀性指标汇总】")
    print(results_df.round(4))
    
    print("\n【指标解释】")
    print("- Normalized_Entropy: 接近1表示分布均匀，接近0表示高度集中")
    print("- Chi_Square_P_Value: < 0.05表示显著偏离均匀分布")
    print("- Gini_Coefficient: 0表示完全均匀，1表示完全不均匀")
    
    return results_df


def calculate_js_divergence(char_freq_by_group):
    """计算不同模型间的JS散度"""
    print("\n" + "=" * 80)
    print("6. 模型间字符分布相似度 (JS Divergence)")
    print("=" * 80)
    
    # 获取所有可能的字符
    all_chars = set()
    for freq_dict in char_freq_by_group.values():
        all_chars.update(freq_dict.keys())
    all_chars = sorted(list(all_chars))
    
    # 构建分布矩阵
    groups = list(char_freq_by_group.keys())
    distributions = []
    
    for group in groups:
        freq_dict = char_freq_by_group[group]
        dist = np.array([freq_dict.get(c, 0) for c in all_chars])
        # 平滑处理，避免零概率
        dist = (dist + 1e-10) / (dist.sum() + len(all_chars) * 1e-10)
        distributions.append(dist)
    
    # 计算JS散度矩阵
    n_groups = len(groups)
    js_matrix = np.zeros((n_groups, n_groups))
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            js_div = jensenshannon(distributions[i], distributions[j])
            js_matrix[i, j] = js_div
            js_matrix[j, i] = js_div
    
    # 转换为DataFrame
    group_labels = [f"{model}_{ptype}" for model, ptype in groups]
    js_df = pd.DataFrame(js_matrix, index=group_labels, columns=group_labels)
    
    # 只显示部分（良性提示 vs 少样本提示的对比）
    print("\n【良性提示 vs 少样本提示的JS散度 (部分模型)】")
    sample_models = ['deepseek', 'GLM', 'kimi', 'Doubao', 'Qwen']
    
    for model in sample_models:
        benign_key = f"{model}_Benign"
        fewshot_key = f"{model}_Few-Shot"
        if benign_key in js_df.index and fewshot_key in js_df.index:
            js_value = js_df.loc[benign_key, fewshot_key]
            print(f"{model}: {js_value:.4f}")
    
    print("\n【解释】JS散度范围 [0, 1]，0表示完全相同，1表示完全不同")
    
    return js_df


def plot_character_type_distribution(df, output_dir):
    """绘制字符类型分布图"""
    print("\n生成图表: 字符类型分布...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
    
    # 按提示类型汇总
    prompt_data = df.groupby('Prompt_Type')[['ratio_lowercase', 'ratio_uppercase', 'ratio_digit', 'ratio_special']].mean()
    
    # 1. 提示类型对比 - 堆叠条形图
    prompt_data.plot(kind='bar', stacked=True, ax=axes[0, 0], 
                     color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8)
    axes[0, 0].set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Character Type Ratio', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Character Type Distribution by Prompt Type', fontsize=13, fontweight='bold')
    axes[0, 0].legend(['Lowercase', 'Uppercase', 'Digit', 'Special'], fontsize=10)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. 模型对比 (Top-8) - 分组条形图
    top_models = df['Model'].value_counts().head(8).index
    model_data = df[df['Model'].isin(top_models)].groupby('Model')[['ratio_lowercase', 'ratio_uppercase', 'ratio_digit', 'ratio_special']].mean()
    
    model_data.plot(kind='bar', ax=axes[0, 1], 
                    color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8)
    axes[0, 1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Character Type Ratio', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Character Type Distribution by Model (Top-8)', fontsize=13, fontweight='bold')
    axes[0, 1].legend(['Lowercase', 'Uppercase', 'Digit', 'Special'], fontsize=9)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 小写字母比例箱线图
    benign_lower = df[df['Prompt_Type'] == 'Benign']['ratio_lowercase']
    fewshot_lower = df[df['Prompt_Type'] == 'Few-Shot']['ratio_lowercase']
    
    box_data = [benign_lower, fewshot_lower]
    bp = axes[1, 0].boxplot(box_data, labels=['Benign', 'Few-Shot'], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_ylabel('Lowercase Ratio', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Lowercase Letter Usage Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 特殊字符比例箱线图
    benign_special = df[df['Prompt_Type'] == 'Benign']['ratio_special']
    fewshot_special = df[df['Prompt_Type'] == 'Few-Shot']['ratio_special']
    
    box_data = [benign_special, fewshot_special]
    bp = axes[1, 1].boxplot(box_data, labels=['Benign', 'Few-Shot'], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 1].set_ylabel('Special Character Ratio', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Prompt Type', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Special Character Usage Comparison', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "1-3-字符类型分布.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_length_distribution(df, output_dir):
    """绘制长度分布图"""
    print("\n生成图表: 长度分布...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    
    # 左图: 长度分布直方图
    for prompt_type, color in [('Benign', '#3498db'), ('Few-Shot', '#e74c3c')]:
        data = df[df['Prompt_Type'] == prompt_type]['length']
        axes[0].hist(data, bins=range(0, int(data.max()) + 2), alpha=0.6, 
                    label=prompt_type, color=color, density=True)
    
    axes[0].set_xlabel('Password Length', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('Password Length Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 30)
    
    # 右图: 模型长度对比箱线图 (Top-8)
    top_models = df['Model'].value_counts().head(8).index
    df_top = df[df['Model'].isin(top_models)].copy()
    
    # 创建Model_Short用于显示
    df_top['Model_Short'] = df_top['Model'].apply(lambda x: x[:12] if len(x) > 12 else x)
    
    sns.boxplot(data=df_top, x='Model_Short', y='length', hue='Prompt_Type', 
                ax=axes[1], palette=['#3498db', '#e74c3c'])
    
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Password Length', fontsize=12, fontweight='bold')
    axes[1].set_title('Password Length by Model (Top-8)', fontsize=13, fontweight='bold')
    axes[1].legend(title='Prompt Type', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / "1-3-口令长度分布.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_position_bias(df_sample, output_dir):
    """绘制位置偏好图"""
    print("\n生成图表: 位置偏好...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
    # 左图: 首字符类型分布
    first_char_data = df_sample.groupby(['Prompt_Type', 'first_char_type']).size().unstack(fill_value=0)
    first_char_pct = first_char_data.div(first_char_data.sum(axis=1), axis=0) * 100
    
    first_char_pct.T.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0].set_xlabel('First Character Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('First Character Type Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(title='Prompt Type', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # 右图: 尾字符类型分布
    last_char_data = df_sample.groupby(['Prompt_Type', 'last_char_type']).size().unstack(fill_value=0)
    last_char_pct = last_char_data.div(last_char_data.sum(axis=1), axis=0) * 100
    
    last_char_pct.T.plot(kind='bar', ax=axes[1], color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[1].set_xlabel('Last Character Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Last Character Type Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(title='Prompt Type', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_path = output_dir / "1-3-位置偏好分析.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def plot_uniformity_heatmap(uniformity_df, output_dir):
    """绘制均匀性指标热力图"""
    print("\n生成图表: 均匀性指标热力图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
    
    # 准备数据 (Top-10模型)
    top_models = uniformity_df.groupby('Model')['Unique_Chars'].sum().sort_values(ascending=False).head(10).index
    df_top = uniformity_df[uniformity_df['Model'].isin(top_models)]
    
    # 1. Normalized Entropy热力图
    entropy_pivot = df_top.pivot(index='Model', columns='Prompt_Type', values='Normalized_Entropy')
    sns.heatmap(entropy_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0],
                cbar_kws={'label': 'Normalized Entropy'}, linewidths=0.5, vmin=0, vmax=1)
    axes[0].set_title('Normalized Entropy (higher = more uniform)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Prompt Type', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Model', fontsize=11, fontweight='bold')
    
    # 2. Gini Coefficient热力图
    gini_pivot = df_top.pivot(index='Model', columns='Prompt_Type', values='Gini_Coefficient')
    sns.heatmap(gini_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[1],
                cbar_kws={'label': 'Gini Coefficient'}, linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Gini Coefficient (lower = more uniform)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Prompt Type', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Model', fontsize=11, fontweight='bold')
    
    # 3. Chi-Square P-Value热力图 (log scale for visualization)
    chi_pivot = df_top.pivot(index='Model', columns='Prompt_Type', values='Chi_Square_P_Value')
    # 对p值取log以便可视化
    chi_pivot_log = np.log10(chi_pivot + 1e-100)  # 避免log(0)
    sns.heatmap(chi_pivot_log, annot=chi_pivot, fmt='.2e', cmap='coolwarm', ax=axes[2],
                cbar_kws={'label': 'log10(P-Value)'}, linewidths=0.5)
    axes[2].set_title('Chi-Square P-Value (> 0.05 = uniform)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Prompt Type', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Model', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "1-3-均匀性指标热力图.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存至: {output_path}")


def save_results(df, uniformity_df, js_df, output_dir):
    """保存分析结果"""
    print("\n" + "=" * 80)
    print("保存分析结果")
    print("=" * 80)
    
    # 保存完整数据
    output_csv = output_dir.parent / "data" / "bias_analysis_results.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 完整结果保存至: {output_csv}")
    
    # 保存均匀性指标
    uniformity_csv = output_dir / "1-3-uniformity_metrics.csv"
    uniformity_df.to_csv(uniformity_csv, index=False, encoding='utf-8-sig')
    print(f"  ✓ 均匀性指标保存至: {uniformity_csv}")
    
    # 保存JS散度矩阵
    js_csv = output_dir / "1-3-js_divergence_matrix.csv"
    js_df.to_csv(js_csv, encoding='utf-8-sig')
    print(f"  ✓ JS散度矩阵保存至: {js_csv}")
    
    # 保存统计摘要
    stats_file = output_dir / "1-3-bias_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("1-3 倾向性指标分析统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【字符类型比例统计】\n")
        char_stats = df.groupby('Prompt_Type')[['ratio_lowercase', 'ratio_uppercase', 'ratio_digit', 'ratio_special']].mean()
        f.write(char_stats.to_string() + "\n\n")
        
        f.write("【长度统计】\n")
        length_stats = df.groupby('Prompt_Type')['length'].agg(['mean', 'median', 'std', 'min', 'max'])
        f.write(length_stats.to_string() + "\n\n")
        
        f.write("【均匀性指标汇总 (部分)】\n")
        f.write(uniformity_df.head(10).to_string(index=False) + "\n\n")
        
        f.write("【关键发现】\n")
        # 识别最不均匀的模型
        least_uniform = uniformity_df.nsmallest(5, 'Normalized_Entropy')[['Model', 'Prompt_Type', 'Normalized_Entropy']]
        f.write("最不均匀的5个模型-提示组合:\n")
        f.write(least_uniform.to_string(index=False) + "\n\n")
        
        # 识别最均匀的模型
        most_uniform = uniformity_df.nlargest(5, 'Normalized_Entropy')[['Model', 'Prompt_Type', 'Normalized_Entropy']]
        f.write("最均匀的5个模型-提示组合:\n")
        f.write(most_uniform.to_string(index=False) + "\n")
    
    print(f"  ✓ 统计报告保存至: {stats_file}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "1-3 倾向性指标分析：字符分布与模式识别" + " " * 15 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 设置输出目录
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # 1. 收集数据
    df = collect_all_data()
    if df is None:
        return
    
    # 2. 字符分布分析
    df = analyze_character_distribution(df)
    
    # 3. 字符频率分析
    char_freq_by_group = calculate_character_frequency(df)
    
    # 4. 位置偏好分析
    df_sample = analyze_position_bias(df, sample_size=10000)
    
    # 5. 长度分布分析
    df = analyze_length_distribution(df)
    
    # 6. 均匀性指标计算
    uniformity_df = calculate_uniformity_metrics(df, char_freq_by_group)
    
    # 7. JS散度计算
    js_df = calculate_js_divergence(char_freq_by_group)
    
    # 8. 生成可视化
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    plot_character_type_distribution(df, output_dir)
    plot_length_distribution(df, output_dir)
    plot_position_bias(df_sample, output_dir)
    plot_uniformity_heatmap(uniformity_df, output_dir)
    
    # 9. 保存结果
    save_results(df, uniformity_df, js_df, output_dir)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "分析完成!" + " " * 29 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print(f"所有图表已保存至: {output_dir}")
    print("生成的图表:")
    print("  - 1-3-字符类型分布.png")
    print("  - 1-3-口令长度分布.png")
    print("  - 1-3-位置偏好分析.png")
    print("  - 1-3-均匀性指标热力图.png")
    print("  - 1-3-bias_statistics.txt")


if __name__ == "__main__":
    main()

