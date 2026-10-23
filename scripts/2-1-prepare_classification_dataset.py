#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-1 构建LLM口令指纹识别数据集
目标：识别口令是否由LLM生成（二分类）以及来自哪个模型/提示类型（多分类）
使用所有口令数据进行LLM生成模式的指纹识别
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))


def collect_llm_passwords():
    """收集阶段一的所有LLM生成口令数据"""
    print("=" * 80)
    print("收集LLM生成的口令数据...")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data"
    all_data = []
    
    # 中文模型名称映射
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
            
            try:
                df = pd.read_csv(csv_file)
                
                # 检查必要列
                if 'Password' not in df.columns:
                    print(f"跳过 {csv_file.name}: 缺少Password列")
                    continue
                
                # 提取模型名称
                model_name = csv_file.stem.replace("_benign", "").replace("_fewshot", "")
                model_name = model_name_map.get(model_name, model_name)
                
                # 添加元数据
                df['Model'] = model_name
                df['Prompt_Type'] = prompt_type
                df['Source'] = 'LLM'
                df['Model_Prompt'] = f"{model_name}_{prompt_type}"
                
                # 保留有用的列
                cols_to_keep = ['Password', 'Model', 'Prompt_Type', 'Source', 'Model_Prompt']
                if 'zxcvbn_score' in df.columns:
                    cols_to_keep.append('zxcvbn_score')
                if 'Entropy' in df.columns:
                    cols_to_keep.append('Entropy')
                
                all_data.append(df[cols_to_keep])
                print(f"✓ {csv_file.name}: {len(df)} 条记录")
                
            except Exception as e:
                print(f"✗ 读取失败 {csv_file.name}: {e}")
    
    if not all_data:
        print("错误: 未找到任何有效数据")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n总计加载: {len(combined_df)} 条LLM生成口令")
    
    # 显示模型分布
    print("\n【模型分布】")
    model_dist = combined_df['Model'].value_counts()
    print(model_dist.head(10))
    
    print("\n【提示类型分布】")
    prompt_dist = combined_df['Prompt_Type'].value_counts()
    print(prompt_dist)
    
    return combined_df


def load_human_passwords():
    """加载真实人类创建的口令数据（从RockYou等数据集）"""
    print("\n" + "=" * 80)
    print("尝试加载人类口令数据...")
    print("=" * 80)
    
    # 检查是否有RockYou或其他真实口令数据
    data_dir = Path(__file__).parent.parent / "data"
    human_data = []
    
    # 尝试加载可能的人类口令数据源
    possible_files = [
        data_dir / "rockyou_sample.txt",
        data_dir / "human_passwords.txt",
        data_dir / "real_passwords.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            print(f"找到人类口令数据: {file_path.name}")
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    if 'password' in df.columns:
                        passwords = df['password'].tolist()
                    elif 'Password' in df.columns:
                        passwords = df['Password'].tolist()
                    else:
                        passwords = df.iloc[:, 0].tolist()
                else:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        passwords = [line.strip() for line in f if line.strip()]
                
                human_data.extend(passwords)
                print(f"✓ 加载 {len(passwords)} 条记录")
            except Exception as e:
                print(f"✗ 读取失败: {e}")
    
    if not human_data:
        print("⚠ 未找到人类口令数据，将从LLM数据中模拟对照组")
        return None
    
    # 创建DataFrame
    human_df = pd.DataFrame({
        'Password': human_data,
        'Model': 'Human',
        'Prompt_Type': 'N/A',
        'Source': 'Human',
        'Model_Prompt': 'Human_N/A'
    })
    
    print(f"\n总计加载: {len(human_df)} 条人类口令")
    
    return human_df


def create_binary_labels(df_llm, df_human=None):
    """创建二分类标签：LLM生成 vs 人类创建"""
    print("\n" + "=" * 80)
    print("创建二分类标签（LLM vs Human）...")
    print("=" * 80)
    
    # 标签：1 = LLM生成，0 = 人类创建
    df_llm['binary_label'] = 1
    df_llm['binary_label_name'] = 'LLM-Generated'
    
    if df_human is not None:
        df_human['binary_label'] = 0
        df_human['binary_label_name'] = 'Human-Created'
        
        # 合并数据
        df_combined = pd.concat([df_llm, df_human], ignore_index=True)
        
        print(f"\nLLM生成口令: {len(df_llm)} 条")
        print(f"人类创建口令: {len(df_human)} 条")
        print(f"总计: {len(df_combined)} 条")
    else:
        # 如果没有人类口令数据，使用LLM数据的提示类型作为代理
        print("\n⚠ 使用提示类型作为二分类标签的代理")
        print("  - 良性提示（Benign）视为'类人类'口令")
        print("  - 少样本提示（Few-Shot）视为'明显LLM特征'口令")
        
        df_llm['binary_label'] = df_llm['Prompt_Type'].apply(lambda x: 0 if x == 'Benign' else 1)
        df_llm['binary_label_name'] = df_llm['binary_label'].map({0: 'Benign-Like', 1: 'Few-Shot-Pattern'})
        
        df_combined = df_llm
        
        print(f"\n类人类口令（Benign）: {(df_combined['binary_label']==0).sum()} 条")
        print(f"明显LLM特征（Few-Shot）: {(df_combined['binary_label']==1).sum()} 条")
    
    return df_combined


def create_multiclass_labels(df):
    """创建多分类标签：模型-提示类型组合"""
    print("\n" + "=" * 80)
    print("创建多分类标签（模型指纹识别）...")
    print("=" * 80)
    
    # 使用Model_Prompt作为多分类标签
    unique_classes = df['Model_Prompt'].unique()
    print(f"\n检测到 {len(unique_classes)} 个不同的类别（模型-提示组合）")
    
    # 创建标签映射
    label_map = {cls: idx for idx, cls in enumerate(sorted(unique_classes))}
    df['multiclass_label'] = df['Model_Prompt'].map(label_map)
    
    # 显示类别分布（Top 10）
    print("\n【类别分布（Top-10）】")
    class_dist = df['Model_Prompt'].value_counts().head(10)
    print(class_dist)
    
    # 保存标签映射
    label_map_df = pd.DataFrame(list(label_map.items()), columns=['Model_Prompt', 'Label_ID'])
    
    return df, label_map_df


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, stratify_by='binary_label'):
    """划分训练集、验证集、测试集"""
    print("\n" + "=" * 80)
    print("划分数据集...")
    print("=" * 80)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 首先分离出测试集
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=42, 
        stratify=df[stratify_by] if stratify_by in df.columns else None
    )
    
    # 再从训练+验证集中分离出验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_ratio_adjusted, 
        random_state=42, 
        stratify=train_val_df[stratify_by] if stratify_by in train_val_df.columns else None
    )
    
    print(f"\n训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    if 'binary_label' in train_df.columns:
        print(f"  - 标签0: {(train_df['binary_label']==0).sum()}")
        print(f"  - 标签1: {(train_df['binary_label']==1).sum()}")
    
    print(f"\n验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    if 'binary_label' in val_df.columns:
        print(f"  - 标签0: {(val_df['binary_label']==0).sum()}")
        print(f"  - 标签1: {(val_df['binary_label']==1).sum()}")
    
    print(f"\n测试集: {len(test_df)} 条 ({len(test_df)/len(df)*100:.1f}%)")
    if 'binary_label' in test_df.columns:
        print(f"  - 标签0: {(test_df['binary_label']==0).sum()}")
        print(f"  - 标签1: {(test_df['binary_label']==1).sum()}")
    
    return train_df, val_df, test_df


def save_datasets(train_df, val_df, test_df, label_map_df, output_dir):
    """保存数据集"""
    print("\n" + "=" * 80)
    print("保存数据集...")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整数据集
    train_df.to_csv(output_dir / "train_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 训练集保存至: {output_dir / 'train_dataset.csv'}")
    
    val_df.to_csv(output_dir / "val_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 验证集保存至: {output_dir / 'val_dataset.csv'}")
    
    test_df.to_csv(output_dir / "test_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 测试集保存至: {output_dir / 'test_dataset.csv'}")
    
    # 保存标签映射
    label_map_df.to_csv(output_dir / "label_mapping.csv", index=False, encoding='utf-8-sig')
    print(f"✓ 标签映射保存至: {output_dir / 'label_mapping.csv'}")
    
    # 保存纯文本格式（仅口令，方便后续使用）
    for name, df_subset in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # 保存口令和二分类标签
        if 'binary_label' in df_subset.columns:
            df_subset[['Password', 'binary_label']].to_csv(
                output_dir / f"{name}_passwords_binary.txt", 
                index=False, 
                header=False, 
                sep='\t'
            )
        
        # 保存口令和多分类标签
        if 'multiclass_label' in df_subset.columns:
            df_subset[['Password', 'multiclass_label']].to_csv(
                output_dir / f"{name}_passwords_multiclass.txt", 
                index=False, 
                header=False, 
                sep='\t'
            )
    
    # 保存数据集统计信息
    stats_file = output_dir / "dataset_statistics.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM口令指纹识别数据集统计信息\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【任务描述】\n")
        f.write("- 二分类任务：识别口令是否由LLM生成\n")
        f.write("- 多分类任务：识别口令来自哪个模型-提示类型组合（LLM指纹溯源）\n\n")
        
        f.write("【数据集划分】\n")
        f.write(f"训练集: {len(train_df)} 条\n")
        f.write(f"验证集: {len(val_df)} 条\n")
        f.write(f"测试集: {len(test_df)} 条\n")
        f.write(f"总计: {len(train_df) + len(val_df) + len(test_df)} 条\n\n")
        
        if 'binary_label_name' in train_df.columns:
            f.write("【训练集二分类标签分布】\n")
            train_binary_dist = train_df['binary_label_name'].value_counts()
            f.write(train_binary_dist.to_string() + "\n\n")
        
        if 'Model_Prompt' in train_df.columns:
            f.write("【训练集多分类标签分布（Top-15）】\n")
            train_multi_dist = train_df['Model_Prompt'].value_counts().head(15)
            f.write(train_multi_dist.to_string() + "\n\n")
        
        f.write("【训练集来源分布】\n")
        if 'Prompt_Type' in train_df.columns and 'Source' in train_df.columns:
            train_source = train_df.groupby(['Source', 'Prompt_Type']).size()
            f.write(train_source.to_string() + "\n")
    
    print(f"✓ 统计信息保存至: {stats_file}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "2-1 构建LLM口令指纹识别数据集" + " " * 15 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 设置输出目录
    output_dir = Path(__file__).parent.parent / "data" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 收集LLM生成的口令数据
    df_llm = collect_llm_passwords()
    if df_llm is None:
        return
    
    # 2. 尝试加载人类口令数据
    df_human = load_human_passwords()
    
    # 3. 创建二分类标签
    df = create_binary_labels(df_llm, df_human)
    
    # 4. 创建多分类标签
    df, label_map_df = create_multiclass_labels(df)
    
    # 5. 数据清洗
    print("\n" + "=" * 80)
    print("数据清洗...")
    print("=" * 80)
    
    original_len = len(df)
    
    # 移除空口令
    df = df[df['Password'].notna()]
    df = df[df['Password'].astype(str).str.len() > 0]
    
    # 移除过长口令（超过100个字符）
    df = df[df['Password'].astype(str).str.len() <= 100]
    
    # 注意：对于模型指纹识别任务，我们不移除重复口令
    # 因为相同的口令可能来自不同的模型，这些信息对于识别模型指纹很重要
    # df = df.drop_duplicates(subset=['Password'], keep='first')  # 已禁用
    
    print(f"原始数据: {original_len} 条")
    print(f"清洗后: {len(df)} 条")
    print(f"移除: {original_len - len(df)} 条")
    
    # 6. 划分数据集
    train_df, val_df, test_df = split_dataset(
        df, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15,
        stratify_by='binary_label'
    )
    
    # 7. 保存数据集
    save_datasets(train_df, val_df, test_df, label_map_df, output_dir)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "数据集构建完成!" + " " * 27 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print(f"数据集保存至: {output_dir}")
    print("\n生成的文件:")
    print("  - train_dataset.csv               (训练集，包含所有字段)")
    print("  - val_dataset.csv                 (验证集，包含所有字段)")
    print("  - test_dataset.csv                (测试集，包含所有字段)")
    print("  - label_mapping.csv               (多分类标签映射)")
    print("  - train_passwords_binary.txt      (训练集，仅口令和二分类标签)")
    print("  - val_passwords_binary.txt        (验证集，仅口令和二分类标签)")
    print("  - test_passwords_binary.txt       (测试集，仅口令和二分类标签)")
    print("  - train_passwords_multiclass.txt  (训练集，仅口令和多分类标签)")
    print("  - val_passwords_multiclass.txt    (验证集，仅口令和多分类标签)")
    print("  - test_passwords_multiclass.txt   (测试集，仅口令和多分类标签)")
    print("  - dataset_statistics.txt          (数据集统计信息)")


if __name__ == "__main__":
    main()
