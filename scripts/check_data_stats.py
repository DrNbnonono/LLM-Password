#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据集统计信息"""

import pandas as pd
from pathlib import Path
from collections import defaultdict

data_root = Path("data")

# 统计各个CSV文件
stats = defaultdict(lambda: {'benign': 0, 'fewshot': 0})

print("=" * 80)
print("检查良性提示数据...")
print("=" * 80)

benign_dir = data_root / "良性提示"
for csv_file in benign_dir.glob("*.csv"):
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    model_name = csv_file.stem.replace('llm_passwords_', '').replace('_', ' ')
    stats[model_name]['benign'] = len(df)
    print(f"{csv_file.name}: {len(df)} 条")

print("\n" + "=" * 80)
print("检查少样本提示数据...")
print("=" * 80)

fewshot_dir = data_root / "少样本提示"
for csv_file in fewshot_dir.glob("*.csv"):
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    model_name = csv_file.stem.replace('llm_passwords_', '').replace('_', ' ')
    stats[model_name]['fewshot'] = len(df)
    print(f"{csv_file.name}: {len(df)} 条")

print("\n" + "=" * 80)
print("各模型数据汇总:")
print("=" * 80)
print(f"{'模型名称':<30} {'良性提示':<15} {'少样本提示':<15} {'总计':<15}")
print("-" * 80)

total_benign = 0
total_fewshot = 0
for model, counts in sorted(stats.items()):
    benign = counts['benign']
    fewshot = counts['fewshot']
    total = benign + fewshot
    total_benign += benign
    total_fewshot += fewshot
    print(f"{model:<30} {benign:<15} {fewshot:<15} {total:<15}")

print("-" * 80)
print(f"{'总计':<30} {total_benign:<15} {total_fewshot:<15} {total_benign + total_fewshot:<15}")

# 检查分类数据集
print("\n" + "=" * 80)
print("检查已构建的分类数据集:")
print("=" * 80)

classification_dir = data_root / "classification"
if (classification_dir / "train_with_features.csv").exists():
    train_df = pd.read_csv(classification_dir / "train_with_features.csv")
    val_df = pd.read_csv(classification_dir / "val_with_features.csv")
    test_df = pd.read_csv(classification_dir / "test_with_features.csv")
    
    print(f"训练集: {len(train_df)} 条")
    print(f"验证集: {len(val_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    print(f"总计: {len(train_df) + len(val_df) + len(test_df)} 条")
    
    print("\n模型分布:")
    model_counts = train_df['Model'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count} 条")
else:
    print("分类数据集尚未构建")

