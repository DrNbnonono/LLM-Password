#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-3 训练LLM模型指纹识别系统
目标：通过口令特征识别其生成模型（多分类任务）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime

# 机器学习相关库
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             top_k_accuracy_score)

# 深度学习相关库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings('ignore')

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent.parent))


def check_gpu():
    """检查GPU是否可用"""
    print("=" * 80)
    print("检查GPU状态...")
    print("=" * 80)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ 检测到 {len(gpus)} 个GPU:")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU内存动态增长已启用")
    else:
        print("⚠ 未检测到GPU，将使用CPU训练")
    
    print(f"TensorFlow版本: {tf.__version__}")
    print()


def load_data(data_dir):
    """加载带特征的数据集"""
    print("=" * 80)
    print("加载数据集...")
    print("=" * 80)
    
    train_df = pd.read_csv(data_dir / "train_with_features.csv")
    val_df = pd.read_csv(data_dir / "val_with_features.csv")
    test_df = pd.read_csv(data_dir / "test_with_features.csv")
    
    print(f"✓ 训练集: {len(train_df)} 条")
    print(f"✓ 验证集: {len(val_df)} 条")
    print(f"✓ 测试集: {len(test_df)} 条")
    print()
    
    return train_df, val_df, test_df


def create_model_labels(train_df, val_df, test_df):
    """创建纯模型标签（忽略提示类型）"""
    print("=" * 80)
    print("创建模型指纹识别标签...")
    print("=" * 80)
    
    # 提取纯模型名称（移除提示类型后缀）
    for df in [train_df, val_df, test_df]:
        df['Pure_Model'] = df['Model']
    
    # 统计模型分布
    model_dist = train_df['Pure_Model'].value_counts()
    print(f"\n检测到 {len(model_dist)} 个不同的模型")
    print("\n【训练集模型分布】")
    print(model_dist)
    
    # 创建标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['Pure_Model'])
    
    # 编码标签
    train_df['model_label'] = label_encoder.transform(train_df['Pure_Model'])
    val_df['model_label'] = label_encoder.transform(val_df['Pure_Model'])
    test_df['model_label'] = label_encoder.transform(test_df['Pure_Model'])
    
    # 创建标签映射
    label_mapping = {label: model for model, label in 
                     zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    
    print(f"\n标签映射:")
    for label, model in sorted(label_mapping.items()):
        count = (train_df['model_label'] == label).sum()
        print(f"  {label}: {model} ({count} 条)")
    
    return train_df, val_df, test_df, label_encoder, label_mapping


def prepare_features(train_df, val_df, test_df):
    """准备特征"""
    print("\n" + "=" * 80)
    print("准备特征...")
    print("=" * 80)
    
    # 定义特征列
    exclude_cols = ['Password', 'Model', 'Prompt_Type', 'Source', 'Model_Prompt', 
                    'binary_label', 'binary_label_name', 'multiclass_label',
                    'zxcvbn_score', 'Entropy', 'Pure_Model', 'model_label']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    
    # 提取特征
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # 提取标签
    y_train = train_df['model_label'].values
    y_val = val_df['model_label'].values
    y_test = test_df['model_label'].values
    
    n_classes = len(np.unique(y_train))
    
    print(f"类别数量: {n_classes}")
    print(f"训练集标签分布: {np.bincount(y_train)[:10]}...（前10个）")
    print()
    
    # 标准化特征
    print("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("✓ 完成")
    print()
    
    return (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, 
            scaler, feature_cols, n_classes)


def train_random_forest(X_train, y_train, X_val, y_val, n_classes, model_name="RandomForest_Fingerprint"):
    """训练随机森林模型"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    model = RandomForestClassifier(
        n_estimators=200,  # 增加树的数量以处理更多类别
        max_depth=30,      # 增加深度
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("开始训练...")
    model.fit(X_train, y_train)
    
    # 验证集评估
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Top-3 和 Top-5 准确率
    val_proba = model.predict_proba(X_val)
    top3_acc = top_k_accuracy_score(y_val, val_proba, k=3)
    top5_acc = top_k_accuracy_score(y_val, val_proba, k=5)
    
    print(f"✓ 训练完成")
    print(f"验证集 Top-1 准确率: {val_acc:.4f}")
    print(f"验证集 Top-3 准确率: {top3_acc:.4f}")
    print(f"验证集 Top-5 准确率: {top5_acc:.4f}")
    
    return model


def train_gradient_boosting(X_train, y_train, X_val, y_val, n_classes, model_name="GradientBoosting_Fingerprint"):
    """训练梯度提升模型"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=1
    )
    
    print("开始训练...")
    model.fit(X_train, y_train)
    
    # 验证集评估
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Top-3 和 Top-5 准确率
    val_proba = model.predict_proba(X_val)
    top3_acc = top_k_accuracy_score(y_val, val_proba, k=3)
    top5_acc = top_k_accuracy_score(y_val, val_proba, k=5)
    
    print(f"✓ 训练完成")
    print(f"验证集 Top-1 准确率: {val_acc:.4f}")
    print(f"验证集 Top-3 准确率: {top3_acc:.4f}")
    print(f"验证集 Top-5 准确率: {top5_acc:.4f}")
    
    return model


def build_dnn_model(input_dim, n_classes):
    """构建深度神经网络模型用于多分类"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                 keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    return model


def train_dnn(X_train, y_train, X_val, y_val, n_classes, model_name="DNN_Fingerprint"):
    """训练深度神经网络用于模型指纹识别"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    input_dim = X_train.shape[1]
    model = build_dnn_model(input_dim, n_classes)
    
    print(f"\n模型架构:")
    model.summary()
    
    # 回调函数
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=128,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 验证集评估
    val_results = model.evaluate(X_val, y_val, verbose=0)
    val_acc = val_results[1]  # accuracy
    top3_acc = val_results[2]  # top_3_accuracy
    top5_acc = val_results[3]  # top_5_accuracy
    
    print(f"\n✓ 训练完成")
    print(f"验证集 Top-1 准确率: {val_acc:.4f}")
    print(f"验证集 Top-3 准确率: {top3_acc:.4f}")
    print(f"验证集 Top-5 准确率: {top5_acc:.4f}")
    
    return model, history


def evaluate_model(model, X_test, y_test, label_mapping, model_name, is_keras=False):
    """评估模型性能"""
    print(f"\n{'='*80}")
    print(f"评估 {model_name} 模型...")
    print(f"{'='*80}")
    
    # 预测
    if is_keras:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Top-K 准确率
        top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Top-K 准确率
        top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    
    n_classes = len(np.unique(y_test))
    
    # 宏平均和加权平均
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n【测试集性能】")
    print(f"Top-1 准确率:  {accuracy:.4f}")
    print(f"Top-3 准确率:  {top3_acc:.4f}")
    print(f"Top-5 准确率:  {top5_acc:.4f}")
    print(f"\n【宏平均指标】")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall:    {recall_macro:.4f}")
    print(f"F1-Score:  {f1_macro:.4f}")
    print(f"\n【加权平均指标】")
    print(f"Precision: {precision_weighted:.4f}")
    print(f"Recall:    {recall_weighted:.4f}")
    print(f"F1-Score:  {f1_weighted:.4f}")
    
    # 混淆矩阵（显示部分）
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n【混淆矩阵对角线（正确预测数）】")
    for i in range(min(10, len(cm))):
        model_name_str = label_mapping[i]
        correct = cm[i, i]
        total = cm[i].sum()
        print(f"  {model_name_str}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # 分类报告（保存到结果中）
    report = classification_report(y_test, y_pred, 
                                   target_names=[label_mapping[i] for i in range(n_classes)],
                                   zero_division=0,
                                   output_dict=True)
    
    results = {
        'model_name': model_name,
        'top1_accuracy': float(accuracy),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'classification_report': report
    }
    
    return results


def save_models_and_results(models, results, scaler, feature_cols, label_encoder, label_mapping, output_dir):
    """保存模型和结果"""
    print("\n" + "=" * 80)
    print("保存模型和结果...")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    models_dir = output_dir / "fingerprint_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存scaler、特征列和标签编码器
    import joblib
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(feature_cols, models_dir / "feature_columns.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    joblib.dump(label_mapping, models_dir / "label_mapping.pkl")
    print(f"✓ Scaler、特征列和标签编码器保存至: {models_dir}")
    
    # 保存传统机器学习模型
    for model_name, model_obj in models.items():
        if 'DNN' not in model_name:
            model_path = models_dir / f"{model_name}.pkl"
            joblib.dump(model_obj, model_path)
            print(f"✓ {model_name} 模型保存至: {model_path}")
    
    # 保存深度学习模型
    for model_name, model_obj in models.items():
        if 'DNN' in model_name:
            model_path = models_dir / f"{model_name}.keras"
            model_obj.save(model_path)
            print(f"✓ {model_name} 模型保存至: {model_path}")
    
    # 保存评估结果
    results_summary = []
    for r in results:
        summary = {
            'model_name': r['model_name'],
            'top1_accuracy': r['top1_accuracy'],
            'top3_accuracy': r['top3_accuracy'],
            'top5_accuracy': r['top5_accuracy'],
            'f1_macro': r['f1_macro'],
            'f1_weighted': r['f1_weighted']
        }
        results_summary.append(summary)
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('top1_accuracy', ascending=False)
    
    results_path = output_dir / "fingerprint_evaluation_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✓ 评估结果保存至: {results_path}")
    
    # 保存详细JSON结果
    results_json_path = output_dir / "fingerprint_evaluation_detailed.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 详细评估结果(JSON)保存至: {results_json_path}")
    
    # 生成模型对比报告
    report_path = output_dir / "fingerprint_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM模型指纹识别性能对比报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"任务: 多分类 - 识别口令来自哪个具体的LLM模型\n")
        f.write(f"类别数量: {len(label_mapping)}\n\n")
        
        f.write("【模型性能排名（按Top-1准确率）】\n\n")
        for idx, row in results_df.iterrows():
            f.write(f"{idx + 1}. {row['model_name']}\n")
            f.write(f"   Top-1 Accuracy: {row['top1_accuracy']:.4f}\n")
            f.write(f"   Top-3 Accuracy: {row['top3_accuracy']:.4f}\n")
            f.write(f"   Top-5 Accuracy: {row['top5_accuracy']:.4f}\n")
            f.write(f"   F1-Score (Macro): {row['f1_macro']:.4f}\n")
            f.write(f"   F1-Score (Weighted): {row['f1_weighted']:.4f}\n\n")
        
        f.write("\n【最佳模型】\n")
        best_model = results_df.iloc[0]
        f.write(f"模型名称: {best_model['model_name']}\n")
        f.write(f"Top-1 Accuracy: {best_model['top1_accuracy']:.4f}\n")
        f.write(f"Top-3 Accuracy: {best_model['top3_accuracy']:.4f}\n")
    
    print(f"✓ 模型对比报告保存至: {report_path}")
    
    # 显示最佳模型
    print("\n" + "=" * 80)
    print(f"【最佳模型】{results_df.iloc[0]['model_name']}")
    print(f"Top-1 Accuracy: {results_df.iloc[0]['top1_accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results_df.iloc[0]['top3_accuracy']:.4f}")
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "2-3 训练LLM模型指纹识别系统" + " " * 18 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 检查GPU
    check_gpu()
    
    # 设置路径
    data_dir = Path(__file__).parent.parent / "data" / "classification"
    
    # 加载数据
    train_df, val_df, test_df = load_data(data_dir)
    
    # 创建模型标签（纯模型识别，不考虑提示类型）
    train_df, val_df, test_df, label_encoder, label_mapping = create_model_labels(
        train_df, val_df, test_df
    )
    
    # 准备特征
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, feature_cols, n_classes) = prepare_features(train_df, val_df, test_df)
    
    # 训练多个模型
    models = {}
    results = []
    
    # 1. 随机森林
    rf_model = train_random_forest(X_train, y_train, X_val, y_val, n_classes)
    models['RandomForest_Fingerprint'] = rf_model
    rf_results = evaluate_model(rf_model, X_test, y_test, label_mapping, 
                                'RandomForest_Fingerprint', is_keras=False)
    results.append(rf_results)
    
    # 2. 梯度提升
    gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val, n_classes)
    models['GradientBoosting_Fingerprint'] = gb_model
    gb_results = evaluate_model(gb_model, X_test, y_test, label_mapping,
                                'GradientBoosting_Fingerprint', is_keras=False)
    results.append(gb_results)
    
    # 3. 深度神经网络
    dnn_model, dnn_history = train_dnn(X_train, y_train, X_val, y_val, n_classes)
    models['DNN_Fingerprint'] = dnn_model
    dnn_results = evaluate_model(dnn_model, X_test, y_test, label_mapping,
                                'DNN_Fingerprint', is_keras=True)
    results.append(dnn_results)
    
    # 保存模型和结果
    save_models_and_results(models, results, scaler, feature_cols, 
                           label_encoder, label_mapping, data_dir)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 25 + "模型指纹识别训练完成!" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print("生成的文件:")
    print("  - fingerprint_models/RandomForest_Fingerprint.pkl")
    print("  - fingerprint_models/GradientBoosting_Fingerprint.pkl")
    print("  - fingerprint_models/DNN_Fingerprint.keras")
    print("  - fingerprint_models/scaler.pkl")
    print("  - fingerprint_models/feature_columns.pkl")
    print("  - fingerprint_models/label_encoder.pkl")
    print("  - fingerprint_models/label_mapping.pkl")
    print("  - fingerprint_evaluation_results.csv")
    print("  - fingerprint_evaluation_detailed.json")
    print("  - fingerprint_comparison_report.txt")


if __name__ == "__main__":
    main()

