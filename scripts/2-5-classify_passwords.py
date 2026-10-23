#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-5 口令分类验证脚本
支持从txt文件读取口令，使用训练好的模型进行分类预测
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))


def extract_statistical_features(password):
    """提取统计特征"""
    pwd = str(password)
    length = len(pwd)
    
    lowercase_count = sum(1 for c in pwd if c.islower())
    uppercase_count = sum(1 for c in pwd if c.isupper())
    digit_count = sum(1 for c in pwd if c.isdigit())
    special_count = sum(1 for c in pwd if not c.isalnum())
    
    lowercase_ratio = lowercase_count / length if length > 0 else 0
    uppercase_ratio = uppercase_count / length if length > 0 else 0
    digit_ratio = digit_count / length if length > 0 else 0
    special_ratio = special_count / length if length > 0 else 0
    
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
    import re
    
    pwd = str(password)
    length = len(pwd)
    
    max_consecutive_same = 1
    current_consecutive = 1
    for i in range(1, length):
        if pwd[i] == pwd[i-1]:
            current_consecutive += 1
            max_consecutive_same = max(max_consecutive_same, current_consecutive)
        else:
            current_consecutive = 1
    
    consecutive_digits = len(re.findall(r'\d{2,}', pwd))
    max_consecutive_digits = max([len(m.group()) for m in re.finditer(r'\d+', pwd)], default=0)
    
    consecutive_letters = len(re.findall(r'[a-zA-Z]{3,}', pwd))
    max_consecutive_letters = max([len(m.group()) for m in re.finditer(r'[a-zA-Z]+', pwd)], default=0)
    
    keyboard_patterns = [
        'qwer', 'asdf', 'zxcv', 'yuio', 'hjkl', 'bnm',
        '1234', '2345', '3456', '4567', '5678', '6789', '7890'
    ]
    keyboard_pattern_count = sum(1 for pattern in keyboard_patterns if pattern in pwd.lower())
    
    first_char_is_digit = 1 if length > 0 and pwd[0].isdigit() else 0
    last_char_is_digit = 1 if length > 0 and pwd[-1].isdigit() else 0
    first_char_is_upper = 1 if length > 0 and pwd[0].isupper() else 0
    first_char_is_lower = 1 if length > 0 and pwd[0].islower() else 0
    last_char_is_special = 1 if length > 0 and not pwd[-1].isalnum() else 0
    
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
    from collections import Counter
    
    pwd = str(password)
    
    char_ngrams = [pwd[i:i+n] for i in range(len(pwd)-n+1)]
    
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
    import re
    
    pwd = str(password)
    
    has_word_digit = 1 if re.search(r'[a-zA-Z]+\d+', pwd) else 0
    has_digit_word = 1 if re.search(r'\d+[a-zA-Z]+', pwd) else 0
    has_word_special = 1 if re.search(r'[a-zA-Z]+[^a-zA-Z0-9]+', pwd) else 0
    
    leetspeak_chars = {'@': 'a', '3': 'e', '1': 'i', '0': 'o', '$': 's', '!': 'i'}
    leetspeak_count = sum(1 for c in pwd if c in leetspeak_chars)
    
    has_repeating_substring = 0
    for i in range(2, len(pwd)//2 + 1):
        for j in range(len(pwd) - 2*i + 1):
            substr = pwd[j:j+i]
            if substr in pwd[j+i:]:
                has_repeating_substring = 1
                break
        if has_repeating_substring:
            break
    
    has_year_pattern = 1 if re.search(r'(19|20)\d{2}', pwd) else 0
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
    
    all_features.update(extract_statistical_features(password))
    all_features.update(extract_sequence_features(password))
    all_features.update(extract_ngram_features(password, n=2))
    all_features.update(extract_ngram_features(password, n=3))
    all_features.update(extract_pattern_features(password))
    
    return all_features


def load_passwords_from_txt(txt_file):
    """从txt文件加载口令"""
    print(f"\n加载口令文件: {txt_file}")
    
    txt_path = Path(txt_file)
    if not txt_path.exists():
        print(f"错误: 文件不存在 - {txt_file}")
        return None
    
    passwords = []
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                passwords.append(line)
    
    print(f"✓ 加载了 {len(passwords)} 条口令")
    return passwords


def load_model_and_preprocessor(model_name, models_dir):
    """加载指定的模型和预处理器"""
    print(f"\n加载模型: {model_name}")
    
    models_dir = Path(models_dir)
    
    # 加载scaler和特征列
    scaler = joblib.load(models_dir / "scaler.pkl")
    feature_cols = joblib.load(models_dir / "feature_columns.pkl")
    
    # 加载模型
    if model_name == 'DNN':
        from tensorflow import keras
        model = keras.models.load_model(models_dir / f"{model_name}.keras")
    else:
        model = joblib.load(models_dir / f"{model_name}.pkl")
    
    print(f"✓ 模型加载成功")
    
    return model, scaler, feature_cols


def extract_features_batch(passwords):
    """批量提取特征"""
    print(f"\n提取特征...")
    
    features_list = []
    for idx, password in enumerate(passwords):
        features = extract_all_features(password)
        features_list.append(features)
        
        if (idx + 1) % 1000 == 0:
            print(f"  进度: {idx + 1}/{len(passwords)}")
    
    features_df = pd.DataFrame(features_list)
    print(f"✓ 特征提取完成")
    
    return features_df


def predict_passwords(passwords, model, scaler, feature_cols, model_name):
    """预测口令类别"""
    print(f"\n使用 {model_name} 模型进行预测...")
    
    # 提取特征
    features_df = extract_features_batch(passwords)
    
    # 确保特征顺序一致
    X = features_df[feature_cols].values
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测
    if model_name == 'DNN':
        predictions_proba = model.predict(X_scaled, verbose=0)
        predictions = (predictions_proba > 0.5).astype(int).flatten()
        confidence = np.maximum(predictions_proba, 1 - predictions_proba).flatten()
    else:
        predictions = model.predict(X_scaled)
        if hasattr(model, 'predict_proba'):
            predictions_proba = model.predict_proba(X_scaled)
            confidence = np.max(predictions_proba, axis=1)
        else:
            confidence = np.ones(len(predictions))
    
    print(f"✓ 预测完成")
    
    return predictions, confidence


def generate_report(passwords, predictions, confidence, output_file=None):
    """生成分类报告"""
    print("\n" + "=" * 80)
    print("分类结果")
    print("=" * 80)
    
    # 统计
    benign_count = (predictions == 0).sum()
    fewshot_count = (predictions == 1).sum()
    
    print(f"\n总口令数: {len(passwords)}")
    print(f"类人类口令（Benign-Like）: {benign_count} ({benign_count/len(passwords)*100:.1f}%)")
    print(f"明显LLM特征（Few-Shot-Pattern）: {fewshot_count} ({fewshot_count/len(predictions)*100:.1f}%)")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Password': passwords,
        'Prediction': predictions,
        'Label': ['Benign-Like' if p == 0 else 'Few-Shot-Pattern' for p in predictions],
        'Confidence': confidence
    })
    
    # 显示一些示例
    print("\n【示例结果（前10条）】")
    print(results_df.head(10).to_string(index=False))
    
    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✓ 结果已保存至: {output_path}")
    
    return results_df


def main():
    """主函数"""
    import argparse
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "2-5 LLM口令指纹识别验证" + " " * 21 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='LLM口令指纹识别')
    parser.add_argument('input_file', type=str, help='输入的口令文件（txt格式，每行一个口令）')
    parser.add_argument('--model', type=str, default='RandomForest', 
                       choices=['RandomForest', 'GradientBoosting', 'LogisticRegression', 'DNN'],
                       help='使用的模型（默认: RandomForest）')
    parser.add_argument('--output', type=str, default=None, 
                       help='输出文件路径（CSV格式，默认: 不保存）')
    
    args = parser.parse_args()
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "data" / "classification" / "models"
    
    # 1. 加载口令
    passwords = load_passwords_from_txt(args.input_file)
    if passwords is None or len(passwords) == 0:
        print("错误: 未能加载任何口令")
        return
    
    # 2. 加载模型
    model, scaler, feature_cols = load_model_and_preprocessor(args.model, models_dir)
    
    # 3. 预测
    predictions, confidence = predict_passwords(passwords, model, scaler, feature_cols, args.model)
    
    # 4. 生成报告
    results_df = generate_report(passwords, predictions, confidence, args.output)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 30 + "分类完成!" + " " * 30 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()

