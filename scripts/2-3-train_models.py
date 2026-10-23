#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2-3 训练LLM口令指纹识别模型
包括传统机器学习模型和深度学习模型
使用GPU加速训练
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime

# 机器学习相关库
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)

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
        
        # 设置GPU内存增长
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


def prepare_features(train_df, val_df, test_df, task='binary'):
    """准备特征和标签"""
    print("=" * 80)
    print(f"准备特征和标签（任务类型: {task}）...")
    print("=" * 80)
    
    # 定义特征列（排除非特征列）
    exclude_cols = ['Password', 'Model', 'Prompt_Type', 'Source', 'Model_Prompt', 
                    'binary_label', 'binary_label_name', 'multiclass_label',
                    'zxcvbn_score', 'Entropy']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"特征数量: {len(feature_cols)}")
    
    # 提取特征
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # 提取标签
    if task == 'binary':
        y_train = train_df['binary_label'].values
        y_val = val_df['binary_label'].values
        y_test = test_df['binary_label'].values
        n_classes = 2
    else:  # multiclass
        y_train = train_df['multiclass_label'].values
        y_val = val_df['multiclass_label'].values
        y_test = test_df['multiclass_label'].values
        n_classes = len(np.unique(y_train))
    
    print(f"类别数量: {n_classes}")
    print(f"训练集标签分布: {np.bincount(y_train)}")
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


def train_random_forest(X_train, y_train, X_val, y_val, model_name="RandomForest"):
    """训练随机森林模型"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
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
    
    print(f"✓ 训练完成")
    print(f"验证集准确率: {val_acc:.4f}")
    
    return model


def train_gradient_boosting(X_train, y_train, X_val, y_val, model_name="GradientBoosting"):
    """训练梯度提升模型"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
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
    
    print(f"✓ 训练完成")
    print(f"验证集准确率: {val_acc:.4f}")
    
    return model


def train_logistic_regression(X_train, y_train, X_val, y_val, n_classes, model_name="LogisticRegression"):
    """训练逻辑回归模型"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    if n_classes > 2:
        model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    
    print("开始训练...")
    model.fit(X_train, y_train)
    
    # 验证集评估
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"✓ 训练完成")
    print(f"验证集准确率: {val_acc:.4f}")
    
    return model


def build_dnn_model(input_dim, n_classes, task='binary'):
    """构建深度神经网络模型"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        
        layers.Dense(n_classes if n_classes > 2 else 1, 
                    activation='softmax' if n_classes > 2 else 'sigmoid')
    ])
    
    if task == 'binary':
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model


def train_dnn(X_train, y_train, X_val, y_val, n_classes, task='binary', model_name="DNN"):
    """训练深度神经网络"""
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型...")
    print(f"{'='*80}")
    
    input_dim = X_train.shape[1]
    model = build_dnn_model(input_dim, n_classes, task)
    
    print(f"\n模型架构:")
    model.summary()
    
    # 回调函数
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\n开始训练...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 验证集评估
    val_results = model.evaluate(X_val, y_val, verbose=0)
    val_acc = val_results[1]  # accuracy is the second metric
    
    print(f"\n✓ 训练完成")
    print(f"验证集准确率: {val_acc:.4f}")
    
    return model, history


def evaluate_model(model, X_test, y_test, model_name, is_keras=False):
    """评估模型性能"""
    print(f"\n{'='*80}")
    print(f"评估 {model_name} 模型...")
    print(f"{'='*80}")
    
    # 预测
    if is_keras:
        y_pred_proba = model.predict(X_test, verbose=0)
        if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
            # Binary classification
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    
    # 根据类别数决定average参数
    n_classes = len(np.unique(y_test))
    avg_method = 'binary' if n_classes == 2 else 'weighted'
    
    precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
    
    print(f"\n【测试集性能】")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # 混淆矩阵（仅对二分类显示详细信息）
    if n_classes == 2:
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n【混淆矩阵】")
        print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    # 分类报告
    print(f"\n【详细分类报告】")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist() if n_classes == 2 else None
    }
    
    return results


def save_models_and_results(models, results, scaler, feature_cols, output_dir):
    """保存模型和结果"""
    print("\n" + "=" * 80)
    print("保存模型和结果...")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存scaler和特征列
    import joblib
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(feature_cols, models_dir / "feature_columns.pkl")
    print(f"✓ Scaler和特征列保存至: {models_dir}")
    
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
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    results_path = output_dir / "model_evaluation_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✓ 评估结果保存至: {results_path}")
    
    # 保存JSON格式的结果
    results_json_path = output_dir / "model_evaluation_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 评估结果(JSON)保存至: {results_json_path}")
    
    # 生成模型对比报告
    report_path = output_dir / "model_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM口令指纹识别模型对比报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【模型性能排名（按F1-Score）】\n\n")
        for idx, row in results_df.iterrows():
            f.write(f"{row.name + 1}. {row['model_name']}\n")
            f.write(f"   Accuracy:  {row['accuracy']:.4f}\n")
            f.write(f"   Precision: {row['precision']:.4f}\n")
            f.write(f"   Recall:    {row['recall']:.4f}\n")
            f.write(f"   F1-Score:  {row['f1_score']:.4f}\n\n")
        
        f.write("\n【最佳模型】\n")
        best_model = results_df.iloc[0]
        f.write(f"模型名称: {best_model['model_name']}\n")
        f.write(f"F1-Score: {best_model['f1_score']:.4f}\n")
    
    print(f"✓ 模型对比报告保存至: {report_path}")
    
    # 显示最佳模型
    print("\n" + "=" * 80)
    print(f"【最佳模型】{results_df.iloc[0]['model_name']}")
    print(f"F1-Score: {results_df.iloc[0]['f1_score']:.4f}")
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 18 + "2-3 训练LLM口令指纹识别模型" + " " * 18 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    # 检查GPU
    check_gpu()
    
    # 设置路径
    data_dir = Path(__file__).parent.parent / "data" / "classification"
    
    # 加载数据
    train_df, val_df, test_df = load_data(data_dir)
    
    # 准备特征（二分类任务）
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     scaler, feature_cols, n_classes) = prepare_features(train_df, val_df, test_df, task='binary')
    
    # 训练多个模型
    models = {}
    results = []
    
    # 1. 随机森林
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    models['RandomForest'] = rf_model
    rf_results = evaluate_model(rf_model, X_test, y_test, 'RandomForest', is_keras=False)
    results.append(rf_results)
    
    # 2. 梯度提升
    gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val)
    models['GradientBoosting'] = gb_model
    gb_results = evaluate_model(gb_model, X_test, y_test, 'GradientBoosting', is_keras=False)
    results.append(gb_results)
    
    # 3. 逻辑回归
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val, n_classes)
    models['LogisticRegression'] = lr_model
    lr_results = evaluate_model(lr_model, X_test, y_test, 'LogisticRegression', is_keras=False)
    results.append(lr_results)
    
    # 4. 深度神经网络
    dnn_model, dnn_history = train_dnn(X_train, y_train, X_val, y_val, n_classes, task='binary')
    models['DNN'] = dnn_model
    dnn_results = evaluate_model(dnn_model, X_test, y_test, 'DNN', is_keras=True)
    results.append(dnn_results)
    
    # 保存模型和结果
    save_models_and_results(models, results, scaler, feature_cols, data_dir)
    
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 28 + "模型训练完成!" + " " * 28 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    print("生成的文件:")
    print("  - models/RandomForest.pkl         (随机森林模型)")
    print("  - models/GradientBoosting.pkl     (梯度提升模型)")
    print("  - models/LogisticRegression.pkl   (逻辑回归模型)")
    print("  - models/DNN/                     (深度神经网络模型)")
    print("  - models/scaler.pkl               (特征标准化器)")
    print("  - models/feature_columns.pkl      (特征列名)")
    print("  - model_evaluation_results.csv    (评估结果)")
    print("  - model_evaluation_results.json   (评估结果JSON)")
    print("  - model_comparison_report.txt     (模型对比报告)")


if __name__ == "__main__":
    main()

