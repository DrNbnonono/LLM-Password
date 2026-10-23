#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-6 演示：使用训练好的模型识别LLM生成的口令
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Feature extraction functions (same as in 2-2-feature_engineering.py)
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
    print("=" * 80)
    print("Loading trained models and preprocessors...")
    print("=" * 80)
    
    # Load scaler
    scaler = joblib.load(models_dir / "scaler.pkl")
    print(f"✓ Loaded StandardScaler")
    
    # Load feature columns
    feature_cols = joblib.load(models_dir / "feature_columns.pkl")
    print(f"✓ Loaded feature columns ({len(feature_cols)} features)")
    
    # Load label encoder and mapping
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    label_mapping = joblib.load(models_dir / "label_mapping.pkl")
    print(f"✓ Loaded label encoder ({len(label_mapping)} classes)")
    
    # Load models
    models = {}
    
    # Random Forest
    try:
        models['RandomForest'] = joblib.load(models_dir / "RandomForest_Fingerprint.pkl")
        print(f"✓ Loaded RandomForest model")
    except:
        print("✗ RandomForest model not found")
    
    # Gradient Boosting
    try:
        models['GradientBoosting'] = joblib.load(models_dir / "GradientBoosting_Fingerprint.pkl")
        print(f"✓ Loaded GradientBoosting model")
    except:
        print("✗ GradientBoosting model not found")
    
    # DNN
    try:
        models['DNN'] = keras.models.load_model(models_dir / "DNN_Fingerprint.keras")
        print(f"✓ Loaded DNN model")
    except Exception as e:
        print(f"✗ DNN model not found: {e}")
    
    return models, scaler, feature_cols, label_encoder, label_mapping


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
        if model_name == 'DNN':
            # DNN returns probabilities directly
            probs = model.predict(features_scaled, verbose=0)[0]
        else:
            # Traditional ML models
            probs = model.predict_proba(features_scaled)[0]
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        
        predictions = []
        for idx in top_k_indices:
            model_label = label_mapping[int(idx)]  # Use int key, not string
            confidence = float(probs[idx])
            predictions.append({
                'model': model_label,
                'confidence': confidence,
                'rank': len(predictions) + 1
            })
        
        results[model_name] = predictions
    
    return results


def generate_test_passwords():
    """生成测试口令集（模拟LLM生成的弱口令）"""
    print("\n" + "=" * 80)
    print("Generating test passwords (simulating LLM-generated weak passwords)...")
    print("=" * 80)
    
    # 模拟不同提示策略生成的口令
    test_passwords = [
        # 少样本提示风格（弱口令模式）
        {"password": "p@ssw0rd123", "description": "Few-shot style: common word + leetspeak + digits"},
        {"password": "admin12345", "description": "Few-shot style: admin + sequential digits"},
        {"password": "Welcome2024!", "description": "Few-shot style: capitalized word + year + special"},
        {"password": "user1234", "description": "Few-shot style: common word + digits"},
        {"password": "qwerty123456", "description": "Few-shot style: keyboard pattern + digits"},
        {"password": "iloveyou2024", "description": "Few-shot style: phrase + year"},
        {"password": "Test@123", "description": "Few-shot style: word + special + digits"},
        {"password": "password1!", "description": "Few-shot style: literal 'password' + digit + special"},
        
        # 良性提示风格（强口令模式）
        {"password": "Kx7#mQ2$nL9&pR4", "description": "Benign style: random chars, high entropy"},
        {"password": "G8!vZ#4mT@9qY$2", "description": "Benign style: mixed chars, high entropy"},
        {"password": "dF#8kL@2mN$7wQ!", "description": "Benign style: alternating patterns"},
        {"password": "7$Rt9#Lm3@Pk5&Xz", "description": "Benign style: high randomness"},
        
        # 混合风格
        {"password": "MyP@ssw0rd2024!", "description": "Mixed: personal + leetspeak + year"},
        {"password": "Secure#Pass123", "description": "Mixed: meaningful + special + digits"},
        {"password": "Winter2024@Snow", "description": "Mixed: seasonal + year + symbol + word"},
    ]
    
    print(f"\n✓ Generated {len(test_passwords)} test passwords")
    return test_passwords


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "LLM Password Fingerprint Identification Demo" + " " * 15 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "data" / "classification" / "fingerprint_models"
    output_dir = project_root / "data" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    models, scaler, feature_cols, label_encoder, label_mapping = load_models_and_preprocessors(models_dir)
    
    if not models:
        print("\n❌ No models found! Please train models first.")
        return
    
    # Generate test passwords
    test_passwords = generate_test_passwords()
    
    # Predict for each password
    print("\n" + "=" * 80)
    print("Prediction Results:")
    print("=" * 80)
    
    all_results = []
    
    for i, pwd_info in enumerate(test_passwords, 1):
        password = pwd_info['password']
        description = pwd_info['description']
        
        print(f"\n{'─' * 80}")
        print(f"Test #{i}: {password}")
        print(f"Description: {description}")
        print(f"{'─' * 80}")
        
        # Predict with all models
        predictions = predict_password_source(password, models, scaler, feature_cols, 
                                             label_mapping, top_k=3)
        
        # Store results
        result_entry = {
            'password': password,
            'description': description
        }
        
        # Print predictions for each model
        for model_name, preds in predictions.items():
            print(f"\n【{model_name} Model】")
            for pred in preds:
                print(f"  Rank {pred['rank']}: {pred['model']:<25} (Confidence: {pred['confidence']*100:5.2f}%)")
                
                # Store in result entry
                result_entry[f'{model_name}_rank{pred["rank"]}_model'] = pred['model']
                result_entry[f'{model_name}_rank{pred["rank"]}_confidence'] = pred['confidence']
        
        all_results.append(result_entry)
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = output_dir / "demo_fingerprint_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'=' * 80}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    
    for model_name in models.keys():
        top1_col = f'{model_name}_rank1_confidence'
        if top1_col in results_df.columns:
            avg_confidence = results_df[top1_col].mean()
            max_confidence = results_df[top1_col].max()
            min_confidence = results_df[top1_col].min()
            
            print(f"\n【{model_name}】")
            print(f"  Average Top-1 Confidence: {avg_confidence*100:.2f}%")
            print(f"  Max Confidence: {max_confidence*100:.2f}%")
            print(f"  Min Confidence: {min_confidence*100:.2f}%")
    
    # Generate human-readable report
    report_file = output_dir / "demo_fingerprint_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM Password Fingerprint Identification - Demo Report\n")
        f.write("=" * 80 + "\n\n")
        
        for i, pwd_info in enumerate(test_passwords, 1):
            password = pwd_info['password']
            description = pwd_info['description']
            
            f.write(f"Test #{i}: {password}\n")
            f.write(f"Description: {description}\n")
            f.write("-" * 80 + "\n")
            
            predictions = predict_password_source(password, models, scaler, feature_cols, 
                                                 label_mapping, top_k=3)
            
            for model_name, preds in predictions.items():
                f.write(f"\n{model_name} Model:\n")
                for pred in preds:
                    f.write(f"  Rank {pred['rank']}: {pred['model']:<25} ({pred['confidence']*100:5.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n✓ Detailed report saved to: {report_file}")
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "Demo Complete!" + " " * 28 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()

