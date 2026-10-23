#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-2 特征工程：提取LLM口令指纹特征
包括统计特征、序列模式特征和词汇特征
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))


def extract_statistical_features(password):
    """提取统计特征"""
    pwd = str(password)
    length = len(pwd)
    
    # 统计各类字符数量
    lowercase_count = sum(1 for c in pwd if c.islower())
    uppercase_count = sum(1 for c in pwd if c.isupper())
    digit_count = sum(1 for c in pwd if c.isdigit())
    special_count = sum(1 for c in pwd if not c.isalnum())
    
    # 计算比例
    lowercase_ratio = lowercase_count / length if length > 0 else 0
    uppercase_ratio = uppercase_count / length if length > 0 else 0
    digit_ratio = digit_count / length if length > 0 else 0
    special_ratio = special_count / length if length > 0 else 0
    
    # 字符集大小
    unique_chars = len(set(pwd))
    char_diversity = unique_chars / length if length > 0 else 0
    
    features = {
        'length': length,
        'lowercase_count': lowercase_count,
        'uppercase_count': uppercase_count,
        'digit_count': digit_count,
        'special_count': special_count,
        'lowercase_ratio': lowercase_ratio,
        'uppercase_ratio': uppercase_ratio,
        'digit_ratio': digit_ratio,
        'special_ratio': special_ratio,
        'unique_chars': unique_chars,
        'char_diversity': char_diversity
    }
    
    return features


def extract_sequence_features(password):
    """提取序列模式特征"""
    pwd = str(password)
    length = len(pwd)
    
    # 连续重复字符
    max_consecutive_same = 1
    current_consecutive = 1
    for i in range(1, length):
        if pwd[i] == pwd[i-1]:
            current_consecutive += 1
            max_consecutive_same = max(max_consecutive_same, current_consecutive)
        else:
            current_consecutive = 1
    
    # 连续数字
    consecutive_digits = len(re.findall(r'\d{2,}', pwd))
    max_consecutive_digits = max([len(m.group()) for m in re.finditer(r'\d+', pwd)], default=0)
    
    # 连续字母
    consecutive_letters = len(re.findall(r'[a-zA-Z]{3,}', pwd))
    max_consecutive_letters = max([len(m.group()) for m in re.finditer(r'[a-zA-Z]+', pwd)], default=0)
    
    # 键盘相邻模式检测（简化版）
    keyboard_patterns = [
        'qwer', 'asdf', 'zxcv', 'yuio', 'hjkl', 'bnm',
        '1234', '2345', '3456', '4567', '5678', '6789', '7890'
    ]
    keyboard_pattern_count = sum(1 for pattern in keyboard_patterns if pattern in pwd.lower())
    
    # 数字位置特征
    first_char_is_digit = 1 if length > 0 and pwd[0].isdigit() else 0
    last_char_is_digit = 1 if length > 0 and pwd[-1].isdigit() else 0
    first_char_is_upper = 1 if length > 0 and pwd[0].isupper() else 0
    first_char_is_lower = 1 if length > 0 and pwd[0].islower() else 0
    last_char_is_special = 1 if length > 0 and not pwd[-1].isalnum() else 0
    
    # 数字集中度（数字主要在开头、中间还是结尾）
    digit_positions = [i for i, c in enumerate(pwd) if c.isdigit()]
    if digit_positions:
        digit_concentration_start = sum(1 for pos in digit_positions if pos < length * 0.3)
        digit_concentration_end = sum(1 for pos in digit_positions if pos > length * 0.7)
    else:
        digit_concentration_start = 0
        digit_concentration_end = 0
    
    features = {
        'max_consecutive_same': max_consecutive_same,
        'consecutive_digits': consecutive_digits,
        'max_consecutive_digits': max_consecutive_digits,
        'consecutive_letters': consecutive_letters,
        'max_consecutive_letters': max_consecutive_letters,
        'keyboard_pattern_count': keyboard_pattern_count,
        'first_char_is_digit': first_char_is_digit,
        'last_char_is_digit': last_char_is_digit,
        'first_char_is_upper': first_char_is_upper,
        'first_char_is_lower': first_char_is_lower,
        'last_char_is_special': last_char_is_special,
        'digit_concentration_start': digit_concentration_start,
        'digit_concentration_end': digit_concentration_end
    }
    
    return features


def extract_ngram_features(password, n=2):
    """提取n-gram特征"""
    pwd = str(password)
    
    # 字符级n-gram
    char_ngrams = [pwd[i:i+n] for i in range(len(pwd)-n+1)]
    
    # 统计最常见的n-gram
    if char_ngrams:
        ngram_counts = Counter(char_ngrams)
        most_common_ngram_freq = ngram_counts.most_common(1)[0][1] if ngram_counts else 0
        unique_ngrams = len(ngram_counts)
        ngram_diversity = unique_ngrams / len(char_ngrams) if char_ngrams else 0
    else:
        most_common_ngram_freq = 0
        unique_ngrams = 0
        ngram_diversity = 0
    
    features = {
        f'{n}gram_count': len(char_ngrams),
        f'{n}gram_unique': unique_ngrams,
        f'{n}gram_diversity': ngram_diversity,
        f'{n}gram_most_common_freq': most_common_ngram_freq
    }
    
    return features


def extract_pattern_features(password):
    """提取模式特征"""
    pwd = str(password)
    
    # 常见口令模式
    has_word_digit = 1 if re.search(r'[a-zA-Z]+\d+', pwd) else 0
    has_digit_word = 1 if re.search(r'\d+[a-zA-Z]+', pwd) else 0
    has_word_special = 1 if re.search(r'[a-zA-Z]+[^a-zA-Z0-9]+', pwd) else 0
    
    # l33tspeak检测（如a->@, e->3, i->1, o->0, s->$）
    leetspeak_chars = {'@': 'a', '3': 'e', '1': 'i', '0': 'o', '$': 's', '!': 'i'}
    leetspeak_count = sum(1 for c in pwd if c in leetspeak_chars)
    
    # 重复单词检测（简化版）
    # 检测是否有重复的子串
    has_repeating_substring = 0
    for i in range(2, len(pwd)//2 + 1):
        for j in range(len(pwd) - 2*i + 1):
            substr = pwd[j:j+i]
            if substr in pwd[j+i:]:
                has_repeating_substring = 1
                break
        if has_repeating_substring:
            break
    
    # 年份模式检测
    has_year_pattern = 1 if re.search(r'(19|20)\d{2}', pwd) else 0
    
    # 月份/日期模式
    has_date_pattern = 1 if re.search(r'(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])', pwd) else 0
    
    features = {
        'has_word_digit': has_word_digit,
        'has_digit_word': has_digit_word,
        'has_word_special': has_word_special,
        'leetspeak_count': leetspeak_count,
        'has_repeating_substring': has_repeating_substring,
        'has_year_pattern': has_year_pattern,
        'has_date_pattern': has_date_pattern
    }
    
    return features


def extract_all_features(password):
    """提取所有特征"""
    all_features = {}
    
    # 统计特征
    all_features.update(extract_statistical_features(password))
    
    # 序列特征
    all_features.update(extract_sequence_features(password))
    
    # n-gram特征（2-gram和3-gram）
    all_features.update(extract_ngram_features(password, n=2))
    all_features.update(extract_ngram_features(password, n=3))
    
    # 模式特征
    all_features.update(extract_pattern_features(password))
    
    return all_features


def process_dataset(df, dataset_name='Dataset'):
    """处理数据集，提取所有特征"""
    print(f"\n处理 {dataset_name}...")
    print(f"  - 原始样本数: {len(df)}")
    
    # 提取特征
    print("  - 提取特征中...")
    features_list = []
    
    for idx, row in df.iterrows():
        password = row['Password']
        features = extract_all_features(password)
        features_list.append(features)
        
        if (idx + 1) % 5000 == 0:
            print(f"    进度: {idx + 1}/{len(df)}")
    
    # 创建特征DataFrame
    features_df = pd.DataFrame(features_list)
    
    # 合并原始数据和特征
    result_df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    
    print(f"  - 特征数量: {len(features_df.columns)}")
    print(f"  ✓ 完成")
    
    return result_df


def save_feature_description(output_dir):
    """保存特征说明文档"""
    desc_file = output_dir / "feature_description.txt"
    
    with open(desc_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("特征说明文档\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【统计特征 (11个)】\n")
        f.write("- length: 口令长度\n")
        f.write("- lowercase_count: 小写字母数量\n")
        f.write("- uppercase_count: 大写字母数量\n")
        f.write("- digit_count: 数字数量\n")
        f.write("- special_count: 特殊符号数量\n")
        f.write("- lowercase_ratio: 小写字母比例\n")
        f.write("- uppercase_ratio: 大写字母比例\n")
        f.write("- digit_ratio: 数字比例\n")
        f.write("- special_ratio: 特殊符号比例\n")
        f.write("- unique_chars: 不同字符数量\n")
        f.write("- char_diversity: 字符多样性（unique_chars/length）\n\n")
        
        f.write("【序列模式特征 (13个)】\n")
        f.write("- max_consecutive_same: 最长连续相同字符数\n")
        f.write("- consecutive_digits: 连续数字段数量\n")
        f.write("- max_consecutive_digits: 最长连续数字段长度\n")
        f.write("- consecutive_letters: 连续字母段数量\n")
        f.write("- max_consecutive_letters: 最长连续字母段长度\n")
        f.write("- keyboard_pattern_count: 键盘相邻模式数量\n")
        f.write("- first_char_is_digit: 首字符是否为数字\n")
        f.write("- last_char_is_digit: 尾字符是否为数字\n")
        f.write("- first_char_is_upper: 首字符是否为大写字母\n")
        f.write("- first_char_is_lower: 首字符是否为小写字母\n")
        f.write("- last_char_is_special: 尾字符是否为特殊符号\n")
        f.write("- digit_concentration_start: 数字集中在开头的数量\n")
        f.write("- digit_concentration_end: 数字集中在结尾的数量\n\n")
        
        f.write("【N-gram特征 (8个)】\n")
        f.write("- 2gram_count: 2-gram总数\n")
        f.write("- 2gram_unique: 不同2-gram数量\n")
        f.write("- 2gram_diversity: 2-gram多样性\n")
        f.write("- 2gram_most_common_freq: 最常见2-gram的频率\n")
        f.write("- 3gram_count: 3-gram总数\n")
        f.write("- 3gram_unique: 不同3-gram数量\n")
        f.write("- 3gram_diversity: 3-gram多样性\n")
        f.write("- 3gram_most_common_freq: 最常见3-gram的频率\n\n")
        
        f.write("【模式特征 (7个)】\n")
        f.write("- has_word_digit: 是否有'字母+数字'模式\n")
        f.write("- has_digit_word: 是否有'数字+字母'模式\n")
        f.write("- has_word_special: 是否有'字母+特殊符号'模式\n")
        f.write("- leetspeak_count: l33tspeak字符数量\n")
        f.write("- has_repeating_substring: 是否有重复子串\n")
        f.write("- has_year_pattern: 是否包含年份模式(19xx/20xx)\n")
        f.write("- has_date_pattern: 是否包含日期模式\n\n")
        
        f.write(f"总计特征数: 39个\n")
    
    print(f"  ✓ 特征说明保存至: {desc_file}")


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "2-2 特征工程：提取口令特征" + " " * 20 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 设置路径
    data_dir = Path(__file__).parent.parent / "data" / "classification"
    
    print("=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    # 加载数据集
    train_df = pd.read_csv(data_dir / "train_dataset.csv")
    val_df = pd.read_csv(data_dir / "val_dataset.csv")
    test_df = pd.read_csv(data_dir / "test_dataset.csv")
    
    print(f"✓ 训练集: {len(train_df)} 条")
    print(f"✓ 验证集: {len(val_df)} 条")
    print(f"✓ 测试集: {len(test_df)} 条")
    
    # 提取特征
    print("\n" + "=" * 80)
    print("提取特征...")
    print("=" * 80)
    
    train_df_with_features = process_dataset(train_df, "训练集")
    val_df_with_features = process_dataset(val_df, "验证集")
    test_df_with_features = process_dataset(test_df, "测试集")
    
    # 保存带特征的数据集
    print("\n" + "=" * 80)
    print("保存特征数据集...")
    print("=" * 80)
    
    train_df_with_features.to_csv(data_dir / "train_with_features.csv", index=False, encoding='utf-8-sig')
    print(f"✓ {data_dir / 'train_with_features.csv'}")
    
    val_df_with_features.to_csv(data_dir / "val_with_features.csv", index=False, encoding='utf-8-sig')
    print(f"✓ {data_dir / 'val_with_features.csv'}")
    
    test_df_with_features.to_csv(data_dir / "test_with_features.csv", index=False, encoding='utf-8-sig')
    print(f"✓ {data_dir / 'test_with_features.csv'}")
    
    # 保存特征说明
    save_feature_description(data_dir)
    
    # 特征统计
    print("\n" + "=" * 80)
    print("特征统计...")
    print("=" * 80)
    
    feature_cols = [col for col in train_df_with_features.columns 
                    if col not in ['Password', 'Model', 'Prompt_Type', 'Source', 'Model_Prompt', 
                                   'binary_label', 'binary_label_name', 'multiclass_label',
                                   'zxcvbn_score', 'Entropy']]
    
    print(f"\n总特征数: {len(feature_cols)}")
    print("\n特征列表:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")
    
    # 显示部分特征的统计信息
    print("\n【部分特征的统计信息（训练集）】")
    key_features = ['length', 'lowercase_ratio', 'digit_ratio', 'char_diversity', 
                    'max_consecutive_same', 'keyboard_pattern_count']
    print(train_df_with_features[key_features].describe().round(3))
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "特征提取完成!" + " " * 28 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print("生成的文件:")
    print("  - train_with_features.csv  (训练集 + 39个特征)")
    print("  - val_with_features.csv    (验证集 + 39个特征)")
    print("  - test_with_features.csv   (测试集 + 39个特征)")
    print("  - feature_description.txt  (特征说明文档)")


if __name__ == "__main__":
    main()

