#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-6 单口令指纹识别工具
用法: python 2-6-predict_single_password.py <password>
示例: python 2-6-predict_single_password.py mnbvcxz
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Feature extraction functions
def extract_statistical_features(password):
    """提取统计特征"""
    features = {}
    features['length'] = len(password)
    features['lowercase_count'] = sum(1 for c in password if c.islower())
    features['uppercase_count'] = sum(1 for c in password if c.isupper())
    features['digit_count'] = sum(1 for c in password if c.isdigit())
    features['special_count'] = sum(1 for c in password if not c.isalnum())
    
    # Ratios
    total = len(password) if len(password) > 0 else 1
    features['lowercase_ratio'] = features['lowercase_count'] / total
    features['uppercase_ratio'] = features['uppercase_count'] / total
    features['digit_ratio'] = features['digit_count'] / total
    features['special_ratio'] = features['special_count'] / total
    
    # Diversity
    features['unique_chars'] = len(set(password))
    features['char_diversity'] = features['unique_chars'] / total if total > 0 else 0
    
    return features

def extract_sequence_features(password):
    """提取序列特征"""
    features = {}
    
    # Count digit first (will be used later)
    digit_count = sum(1 for c in password if c.isdigit())
    
    # Max consecutive same characters
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(password)):
        if password[i] == password[i-1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1
    features['max_consecutive_same'] = max_consecutive
    
    # Consecutive digits/letters
    features['consecutive_digits'] = sum(1 for i in range(len(password)-1) 
                                        if password[i].isdigit() and password[i+1].isdigit())
    features['consecutive_letters'] = sum(1 for i in range(len(password)-1) 
                                         if password[i].isalpha() and password[i+1].isalpha())
    
    # Keyboard patterns (simple check)
    keyboard_patterns = ['qwert', 'asdfg', 'zxcvb', '12345', 'qwerty', 'asdf']
    features['keyboard_pattern_count'] = sum(1 for pattern in keyboard_patterns 
                                             if pattern in password.lower())
    
    # Positional bias
    features['first_char_is_upper'] = int(password[0].isupper()) if len(password) > 0 else 0
    features['first_char_is_lower'] = int(password[0].islower()) if len(password) > 0 else 0
    features['first_char_is_digit'] = int(password[0].isdigit()) if len(password) > 0 else 0
    features['last_char_is_digit'] = int(password[-1].isdigit()) if len(password) > 0 else 0
    features['last_char_is_special'] = int(not password[-1].isalnum()) if len(password) > 0 else 0
    
    # Digit concentration
    if digit_count > 0:
        first_digit_pos = next((i for i, c in enumerate(password) if c.isdigit()), len(password))
        last_digit_pos = next((i for i, c in enumerate(reversed(password)) if c.isdigit()), len(password))
        features['digit_start_ratio'] = first_digit_pos / len(password)
        features['digit_end_ratio'] = last_digit_pos / len(password)
    else:
        features['digit_start_ratio'] = 0
        features['digit_end_ratio'] = 0
    
    return features

def extract_ngram_features(password):
    """提取N-gram特征"""
    features = {}
    
    # 2-grams
    bigrams = [password[i:i+2] for i in range(len(password)-1)]
    features['bigram_count'] = len(bigrams)
    features['unique_bigrams'] = len(set(bigrams))
    features['bigram_diversity'] = features['unique_bigrams'] / features['bigram_count'] if features['bigram_count'] > 0 else 0
    
    if bigrams:
        from collections import Counter
        bigram_freq = Counter(bigrams)
        features['most_common_bigram_freq'] = bigram_freq.most_common(1)[0][1]
    else:
        features['most_common_bigram_freq'] = 0
    
    # 3-grams
    trigrams = [password[i:i+3] for i in range(len(password)-2)]
    features['trigram_count'] = len(trigrams)
    features['unique_trigrams'] = len(set(trigrams))
    features['trigram_diversity'] = features['unique_trigrams'] / features['trigram_count'] if features['trigram_count'] > 0 else 0
    
    if trigrams:
        from collections import Counter
        trigram_freq = Counter(trigrams)
        features['most_common_trigram_freq'] = trigram_freq.most_common(1)[0][1]
    else:
        features['most_common_trigram_freq'] = 0
    
    return features

def extract_pattern_features(password):
    """提取模式识别特征"""
    features = {}
    pwd_lower = password.lower()
    
    # Word-digit patterns
    import re
    features['has_word_digit'] = int(bool(re.search(r'[a-zA-Z]+\d+', password)))
    features['has_digit_word'] = int(bool(re.search(r'\d+[a-zA-Z]+', password)))
    features['has_word_special'] = int(bool(re.search(r'[a-zA-Z]+[^a-zA-Z0-9]+', password)))
    
    # Leetspeak
    leetspeak_map = {'3': 'e', '1': 'i', '0': 'o', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's'}
    features['leetspeak_count'] = sum(1 for char in password if char in leetspeak_map)
    
    # Repeating substrings
    features['has_repeating_substring'] = 0
    for length in range(2, min(5, len(password)//2 + 1)):
        for i in range(len(password) - length):
            substring = password[i:i+length]
            if substring in password[i+length:]:
                features['has_repeating_substring'] = 1
                break
        if features['has_repeating_substring']:
            break
    
    # Year patterns
    features['has_year_pattern'] = int(bool(re.search(r'(19|20)\d{2}', password)))
    
    # Date patterns
    features['has_date_pattern'] = int(bool(re.search(r'\d{1,2}[/-]\d{1,2}', password)))
    
    # Common words (simple check)
    common_words = ['password', 'admin', 'user', 'welcome', 'login', 'test', 'qwerty']
    features['has_common_word'] = int(any(word in pwd_lower for word in common_words))
    
    # Sequential patterns
    features['has_sequential_digits'] = int(bool(re.search(r'(012|123|234|345|456|567|678|789)', password)))
    
    return features

def extract_all_features(password):
    """提取所有39个特征"""
    features = {}
    features.update(extract_statistical_features(password))
    features.update(extract_sequence_features(password))
    features.update(extract_ngram_features(password))
    features.update(extract_pattern_features(password))
    return features


def load_models_and_preprocessors(models_dir):
    """加载所有模型和预处理器"""
    try:
        scaler = joblib.load(models_dir / "scaler.pkl")
        feature_cols = joblib.load(models_dir / "feature_columns.pkl")
        label_encoder = joblib.load(models_dir / "label_encoder.pkl")
        label_mapping = joblib.load(models_dir / "label_mapping.pkl")
        
        models = {}
        try:
            models['RandomForest'] = joblib.load(models_dir / "RandomForest_Fingerprint.pkl")
        except:
            pass
        
        try:
            models['GradientBoosting'] = joblib.load(models_dir / "GradientBoosting_Fingerprint.pkl")
        except:
            pass
        
        return models, scaler, feature_cols, label_encoder, label_mapping
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None, None


def predict_password_source(password, models, scaler, feature_cols, label_mapping, top_k=3):
    """预测口令来源"""
    # Extract features
    features = extract_all_features(password)
    feature_df = pd.DataFrame([features])
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in feature_df.columns:
            feature_df[col] = 0
    
    # Reorder columns
    feature_df = feature_df[feature_cols]
    
    # Scale features
    features_scaled = scaler.transform(feature_df)
    
    # Predict with all models
    results = {}
    
    for model_name, model in models.items():
        probs = model.predict_proba(features_scaled)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        
        predictions = []
        for idx in top_k_indices:
            model_label = label_mapping[int(idx)]
            confidence = float(probs[idx])
            predictions.append({
                'model': model_label,
                'confidence': confidence,
                'rank': len(predictions) + 1
            })
        
        results[model_name] = predictions
    
    return results


def save_prediction_result(password, predictions, output_dir):
    """保存预测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to text file
    output_file = output_dir / f"prediction_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM Password Fingerprint Identification - Single Password Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Password: {password}\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 80 + "\n\n")
        
        for model_name, preds in predictions.items():
            f.write(f"【{model_name} Model】\n")
            for pred in preds:
                f.write(f"  Rank {pred['rank']}: {pred['model']:<25} (Confidence: {pred['confidence']*100:5.2f}%)\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Interpretation:\n")
        f.write("-" * 80 + "\n")
        
        # Get consensus prediction
        all_top1 = [preds[0]['model'] for preds in predictions.values()]
        all_top1_conf = [preds[0]['confidence'] for preds in predictions.values()]
        
        avg_confidence = sum(all_top1_conf) / len(all_top1_conf)
        
        if len(set(all_top1)) == 1:
            # All models agree
            consensus = all_top1[0]
            f.write(f"✓ All models agree: This password is likely generated by {consensus}\n")
            f.write(f"  Average confidence: {avg_confidence*100:.2f}%\n")
        else:
            f.write(f"⚠ Models disagree on the top prediction:\n")
            for model_name, preds in predictions.items():
                f.write(f"  - {model_name}: {preds[0]['model']} ({preds[0]['confidence']*100:.2f}%)\n")
        
        # Confidence interpretation
        if avg_confidence >= 0.7:
            f.write(f"\n✓ High confidence prediction (≥70%)\n")
            f.write(f"  Recommendation: Strong evidence of LLM generation\n")
        elif avg_confidence >= 0.4:
            f.write(f"\n⚠ Medium confidence prediction (40-70%)\n")
            f.write(f"  Recommendation: Consider Top-3 candidates for further analysis\n")
        else:
            f.write(f"\n✗ Low confidence prediction (<40%)\n")
            f.write(f"  Recommendation: Manual review recommended\n")
    
    return output_file


def main():
    """主函数"""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python 2-6-predict_single_password.py <password>")
        print("Example: python 2-6-predict_single_password.py mnbvcxz")
        sys.exit(1)
    
    password = sys.argv[1]
    
    print("\n" + "=" * 80)
    print("LLM Password Fingerprint Identification")
    print("=" * 80)
    print(f"\nPassword to analyze: {password}")
    print(f"Length: {len(password)} characters\n")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "data" / "classification" / "fingerprint_models"
    output_dir = project_root / "data" / "classification" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models, scaler, feature_cols, label_encoder, label_mapping = load_models_and_preprocessors(models_dir)
    
    if not models or len(models) == 0:
        print("\n❌ Error: No models found!")
        print(f"Please ensure models are trained and located in: {models_dir}")
        sys.exit(1)
    
    print(f"✓ Loaded {len(models)} model(s): {', '.join(models.keys())}\n")
    
    # Predict
    print("Analyzing password...")
    predictions = predict_password_source(password, models, scaler, feature_cols, 
                                         label_mapping, top_k=3)
    
    # Display results
    print("\n" + "=" * 80)
    print("Prediction Results:")
    print("=" * 80 + "\n")
    
    for model_name, preds in predictions.items():
        print(f"【{model_name} Model】")
        for pred in preds:
            bar_length = int(pred['confidence'] * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  Rank {pred['rank']}: {pred['model']:<25} {bar} {pred['confidence']*100:5.2f}%")
        print()
    
    # Save results
    output_file = save_prediction_result(password, predictions, output_dir)
    
    print("=" * 80)
    print(f"✓ Results saved to: {output_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

