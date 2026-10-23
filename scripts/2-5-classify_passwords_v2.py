#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-5 LLM模型指纹识别验证脚本
功能：从txt文件读取口令，使用训练好的模型进行分类，输出预测结果和置信度
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

sys.path.append(str(Path(__file__).parent.parent))


def load_model_and_components(model_name, models_dir):
    """加载模型和必要的组件"""
    import joblib
    from tensorflow import keras
    
    models_dir = Path(models_dir)
    
    # 加载scaler、特征列和标签映射
    scaler = joblib.load(models_dir / "scaler.pkl")
    feature_cols = joblib.load(models_dir / "feature_columns.pkl")
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    label_mapping = joblib.load(models_dir / "label_mapping.pkl")
    
    # 加载模型
    if model_name.endswith('.keras'):
        model = keras.models.load_model(models_dir / model_name)
    else:
        model = joblib.load(models_dir / f"{model_name}.pkl")
    
    return model, scaler, feature_cols, label_encoder, label_mapping


def extract_password_features(password):
    """从口令提取39个特征（与2-2脚本保持一致）"""
    # 统计特征
    length = len(password)
    lower_count = sum(1 for c in password if c.islower())
    upper_count = sum(1 for c in password if c.isupper())
    digit_count = sum(1 for c in password if c.isdigit())
    special_count = sum(1 for c in password if not c.isalnum())
    unique_chars = len(set(password))
    char_diversity = unique_chars / length if length > 0 else 0
    
    lower_ratio = lower_count / length if length > 0 else 0
    upper_ratio = upper_count / length if length > 0 else 0
    digit_ratio = digit_count / length if length > 0 else 0
    special_ratio = special_count / length if length > 0 else 0
    
    # 序列模式特征
    max_consecutive_same = max((sum(1 for _ in g) for k, g in __import__('itertools').groupby(password)), default=0)
    
    max_consecutive_digits = 0
    max_consecutive_letters = 0
    current_digit_run = 0
    current_letter_run = 0
    
    for c in password:
        if c.isdigit():
            current_digit_run += 1
            max_consecutive_digits = max(max_consecutive_digits, current_digit_run)
            current_letter_run = 0
        elif c.isalpha():
            current_letter_run += 1
            max_consecutive_letters = max(max_consecutive_letters, current_letter_run)
            current_digit_run = 0
        else:
            current_digit_run = 0
            current_letter_run = 0
    
    # 键盘模式检测
    keyboard_patterns = [
        'qwerty', 'asdfgh', 'zxcvbn', '123456', '111111', '000000',
        'qwertyuiop', 'asdfghjkl', 'zxcvbnm', '1234567890'
    ]
    has_keyboard_pattern = any(pattern in password.lower() for pattern in keyboard_patterns)
    
    # 位置偏好
    first_is_upper = password[0].isupper() if len(password) > 0 else False
    first_is_digit = password[0].isdigit() if len(password) > 0 else False
    last_is_digit = password[-1].isdigit() if len(password) > 0 else False
    last_is_special = not password[-1].isalnum() if len(password) > 0 else False
    
    # 数字集中度
    digits_in_first_half = sum(1 for c in password[:len(password)//2] if c.isdigit())
    digits_in_last_half = sum(1 for c in password[len(password)//2:] if c.isdigit())
    digit_concentration_start = digits_in_first_half / (len(password)//2) if len(password) > 1 else 0
    digit_concentration_end = digits_in_last_half / (len(password) - len(password)//2) if len(password) > 1 else 0
    
    # N-gram特征
    bigrams = [password[i:i+2] for i in range(len(password)-1)]
    trigrams = [password[i:i+3] for i in range(len(password)-2)]
    
    bigram_count = len(bigrams)
    trigram_count = len(trigrams)
    unique_bigrams = len(set(bigrams))
    unique_trigrams = len(set(trigrams))
    
    bigram_diversity = unique_bigrams / bigram_count if bigram_count > 0 else 0
    trigram_diversity = unique_trigrams / trigram_count if trigram_count > 0 else 0
    
    from collections import Counter
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)
    most_common_bigram_freq = bigram_freq.most_common(1)[0][1] / bigram_count if bigram_count > 0 else 0
    most_common_trigram_freq = trigram_freq.most_common(1)[0][1] / trigram_count if trigram_count > 0 else 0
    
    # 模式识别特征
    import re
    has_word_digit = bool(re.search(r'[a-zA-Z]+\d+', password))
    has_digit_word = bool(re.search(r'\d+[a-zA-Z]+', password))
    has_word_special = bool(re.search(r'[a-zA-Z]+[^a-zA-Z\d]', password))
    
    # Leetspeak模式
    leetspeak_chars = {'4': 'a', '3': 'e', '1': 'i', '0': 'o', '5': 's', '7': 't', '@': 'a', '$': 's'}
    leetspeak_count = sum(1 for c in password if c in leetspeak_chars)
    
    # 重复子串
    repeating_substrings = sum(1 for i in range(len(password)-1) 
                               for j in range(i+2, len(password)) 
                               if password[i:j] in password[j:])
    
    # 年份模式
    has_year_pattern = bool(re.search(r'(19|20)\d{2}', password))
    has_date_pattern = bool(re.search(r'\d{1,2}[/-]\d{1,2}', password))
    
    # 返回特征向量
    features = [
        length, lower_count, upper_count, digit_count, special_count,
        unique_chars, char_diversity, lower_ratio, upper_ratio, digit_ratio, special_ratio,
        max_consecutive_same, max_consecutive_digits, max_consecutive_letters,
        int(has_keyboard_pattern), int(first_is_upper), int(first_is_digit),
        int(last_is_digit), int(last_is_special), digit_concentration_start, digit_concentration_end,
        bigram_count, trigram_count, unique_bigrams, unique_trigrams,
        bigram_diversity, trigram_diversity, most_common_bigram_freq, most_common_trigram_freq,
        int(has_word_digit), int(has_digit_word), int(has_word_special),
        leetspeak_count, repeating_substrings, int(has_year_pattern), int(has_date_pattern)
    ]
    
    return features


def classify_passwords(input_file, model_name, output_file, models_dir):
    """分类口令并输出结果"""
    print("=" * 80)
    print("LLM模型指纹识别验证")
    print("=" * 80)
    
    # 加载模型和组件
    print(f"\n正在加载模型: {model_name}...")
    model, scaler, feature_cols, label_encoder, label_mapping = load_model_and_components(model_name, models_dir)
    print(f"✓ 模型加载完成")
    print(f"✓ 类别数量: {len(label_mapping)}")
    print(f"✓ 特征数量: {len(feature_cols)}")
    
    # 读取口令
    print(f"\n正在读取口令: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        passwords = [line.strip() for line in f if line.strip()]
    print(f"✓ 读取到 {len(passwords)} 条口令")
    
    # 提取特征
    print("\n正在提取特征...")
    features_list = []
    for pwd in passwords:
        features = extract_password_features(pwd)
        features_list.append(features)
    
    X = np.array(features_list)
    print(f"✓ 特征提取完成: {X.shape}")
    
    # 标准化特征
    X_scaled = scaler.transform(X)
    
    # 预测
    print("\n正在进行预测...")
    is_keras = model_name.endswith('.keras')
    
    if is_keras:
        # Keras模型
        predictions_proba = model.predict(X_scaled, verbose=0)
        predictions = np.argmax(predictions_proba, axis=1)
    else:
        # sklearn模型
        predictions = model.predict(X_scaled)
        predictions_proba = model.predict_proba(X_scaled)
    
    # 准备结果
    results = []
    for i, pwd in enumerate(passwords):
        pred_label = predictions[i]
        pred_model = label_mapping[pred_label]
        confidence = predictions_proba[i][pred_label]
        
        # Top-3预测
        top3_indices = np.argsort(predictions_proba[i])[-3:][::-1]
        top3_models = [label_mapping[idx] for idx in top3_indices]
        top3_confidences = [predictions_proba[i][idx] for idx in top3_indices]
        
        results.append({
            'Password': pwd,
            'Predicted_Model': pred_model,
            'Confidence': f"{confidence:.4f}",
            'Top1': pred_model,
            'Top1_Conf': f"{top3_confidences[0]:.4f}",
            'Top2': top3_models[1] if len(top3_models) > 1 else '',
            'Top2_Conf': f"{top3_confidences[1]:.4f}" if len(top3_confidences) > 1 else '',
            'Top3': top3_models[2] if len(top3_models) > 2 else '',
            'Top3_Conf': f"{top3_confidences[2]:.4f}" if len(top3_confidences) > 2 else '',
        })
    
    # 创建DataFrame并保存
    df_results = pd.DataFrame(results)
    
    if output_file:
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ 结果已保存至: {output_file}")
    
    # 显示结果
    print("\n" + "=" * 80)
    print("预测结果（前10条）:")
    print("=" * 80)
    for i, row in df_results.head(10).iterrows():
        print(f"\n口令 #{i+1}: {row['Password']}")
        print(f"  预测模型: {row['Predicted_Model']}")
        print(f"  置信度: {row['Confidence']}")
        print(f"  Top-3: {row['Top1']} ({row['Top1_Conf']}), "
              f"{row['Top2']} ({row['Top2_Conf']}), "
              f"{row['Top3']} ({row['Top3_Conf']})")
    
    if len(df_results) > 10:
        print(f"\n... (共 {len(df_results)} 条结果，详见输出文件)")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    model_counts = df_results['Predicted_Model'].value_counts()
    for model, count in model_counts.items():
        avg_conf = df_results[df_results['Predicted_Model'] == model]['Confidence'].astype(float).mean()
        print(f"  {model}: {count} 条 (平均置信度: {avg_conf:.4f})")
    
    overall_avg_conf = df_results['Confidence'].astype(float).mean()
    print(f"\n整体平均置信度: {overall_avg_conf:.4f}")
    
    print("\n" + "=" * 80)
    print("验证完成!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='LLM模型指纹识别验证脚本')
    parser.add_argument('input_file', type=str, help='输入的口令文件 (.txt)')
    parser.add_argument('--model', type=str, default='DNN_Fingerprint.keras',
                       help='模型名称 (默认: DNN_Fingerprint.keras)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出CSV文件路径 (可选)')
    parser.add_argument('--models-dir', type=str, 
                       default=str(Path(__file__).parent.parent / "data" / "classification" / "fingerprint_models"),
                       help='模型目录')
    
    args = parser.parse_args()
    
    # 如果没有指定输出文件，自动生成
    if args.output is None:
        input_path = Path(args.input_file)
        output_dir = Path(__file__).parent.parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"{input_path.stem}_classification.csv")
    
    classify_passwords(args.input_file, args.model, args.output, args.models_dir)


if __name__ == "__main__":
    main()

